"""Label-free causal history snapshots, retrieval, and raw H scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.models import HistoryNeighbor
from nurse_rcl.schemas import StrictFrozenModel

HISTORY_K: Literal[5] = 5
HISTORY_SCHEMA_VERSION: Literal[1] = 1


class HistoricalServiceScore(StrictFrozenModel):
    service: str = Field(min_length=1)
    score: float

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError("historical service score must be finite")
        return score


class HistoricalCase(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    evidence_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding: tuple[float, ...] = Field(min_length=1)
    frozen_scores: tuple[HistoricalServiceScore, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_embedding_and_scores(self) -> Self:
        _validate_embedding(self.embedding, name="historical embedding")
        services = [canonicalize_service_name(item.service) for item in self.frozen_scores]
        if len(services) != len(set(services)):
            raise ValueError("historical scores require unique canonical services")
        return self


class HistoryFailureTombstone(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    failure_code: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )


class HistoryGroup(StrictFrozenModel):
    schema_version: Literal[1] = HISTORY_SCHEMA_VERSION
    t0: int = Field(ge=0)
    cases: tuple[HistoricalCase, ...]
    tombstones: tuple[HistoryFailureTombstone, ...]

    @model_validator(mode="after")
    def require_one_complete_timestamp_group(self) -> Self:
        if any(case.t0 != self.t0 for case in self.cases):
            raise ValueError("historical case timestamp does not match its group")
        if any(tombstone.t0 != self.t0 for tombstone in self.tombstones):
            raise ValueError("history tombstone timestamp does not match its group")
        case_ids = [case.case_id for case in self.cases]
        tombstone_ids = [item.case_id for item in self.tombstones]
        all_ids = [*case_ids, *tombstone_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("history group case identities must be unique")
        return self


class HistoryGroupArtifact(StrictFrozenModel):
    group: HistoryGroup
    group_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        return self.group_hash == _model_hash(self.group)


class HistorySnapshot(StrictFrozenModel):
    schema_version: Literal[1] = HISTORY_SCHEMA_VERSION
    before_t0: int = Field(ge=0)
    group_hashes: tuple[str, ...]
    cases: tuple[HistoricalCase, ...]
    tombstones: tuple[HistoryFailureTombstone, ...]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_strictly_earlier_records(self) -> Self:
        if any(case.t0 >= self.before_t0 for case in self.cases):
            raise ValueError("history snapshot contains a same-time or future case")
        if any(item.t0 >= self.before_t0 for item in self.tombstones):
            raise ValueError("history snapshot contains a same-time or future tombstone")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"snapshot_hash"})
        return self.snapshot_hash == sha256_hex(canonical_json_bytes(unsigned))


class HistoryScoreContribution(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    historical_service: str = Field(min_length=1)
    similarity: float = Field(gt=0.0, le=1.0)
    historical_score: float
    weighted_score: float


class HistoryCandidateScore(StrictFrozenModel):
    service: str = Field(min_length=1)
    h: float
    contributions: tuple[HistoryScoreContribution, ...]


class HistoryScoreArtifact(StrictFrozenModel):
    schema_version: Literal[1] = HISTORY_SCHEMA_VERSION
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_embedding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    k: Literal[5] = HISTORY_K
    neighbors: tuple[HistoryNeighbor, ...]
    candidates: tuple[HistoryCandidateScore, ...] = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class CausalHistoryIndex:
    """Filesystem index with one immutable atomic artifact per timestamp group."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._groups = root / "groups"

    def commit_group(
        self,
        *,
        t0: int,
        cases: tuple[HistoricalCase, ...],
        tombstones: tuple[HistoryFailureTombstone, ...],
    ) -> HistoryGroupArtifact:
        group = HistoryGroup(
            t0=t0,
            cases=tuple(sorted(cases, key=lambda case: case.case_id)),
            tombstones=tuple(sorted(tombstones, key=lambda item: item.case_id)),
        )
        artifact = HistoryGroupArtifact(group=group, group_hash=_model_hash(group))
        path = self._group_path(t0)
        if path.exists():
            existing = _read_group_artifact(path)
            if existing != artifact:
                raise ValueError("history timestamp group is immutable")
            return existing
        _reject_duplicate_case_ids(
            artifacts=self._load_group_artifacts(),
            incoming_ids={
                *(case.case_id for case in group.cases),
                *(item.case_id for item in group.tombstones),
            },
        )
        atomic_write_bytes(path, canonical_json_bytes(artifact.model_dump(mode="json")))
        return artifact

    def snapshot_for(self, *, t0: int) -> HistorySnapshot:
        selected = tuple(
            artifact
            for artifact in self._load_group_artifacts()
            if artifact.group.t0 < t0
        )
        cases = tuple(
            sorted(
                (case for artifact in selected for case in artifact.group.cases),
                key=lambda case: (case.t0, case.case_id),
            )
        )
        tombstones = tuple(
            sorted(
                (item for artifact in selected for item in artifact.group.tombstones),
                key=lambda item: (item.t0, item.case_id),
            )
        )
        unsigned = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "before_t0": t0,
            "group_hashes": [artifact.group_hash for artifact in selected],
            "cases": [case.model_dump(mode="json") for case in cases],
            "tombstones": [item.model_dump(mode="json") for item in tombstones],
        }
        snapshot = HistorySnapshot(
            before_t0=t0,
            group_hashes=tuple(artifact.group_hash for artifact in selected),
            cases=cases,
            tombstones=tombstones,
            snapshot_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        if not snapshot.verify():  # pragma: no cover - construction invariant
            raise AssertionError("history snapshot verification failed")
        return snapshot

    def _load_group_artifacts(self) -> tuple[HistoryGroupArtifact, ...]:
        if not self._groups.exists():
            return ()
        artifacts = tuple(
            _read_group_artifact(path)
            for path in sorted(self._groups.glob("*.json"), key=lambda path: path.name)
        )
        timestamps = [artifact.group.t0 for artifact in artifacts]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("history index contains duplicate timestamp groups")
        return tuple(sorted(artifacts, key=lambda artifact: artifact.group.t0))

    def _group_path(self, t0: int) -> Path:
        if t0 < 0:
            raise ValueError("history timestamp must be non-negative")
        return self._groups / f"{t0:020d}.json"


def compute_history_scores(
    *,
    snapshot: HistorySnapshot,
    query_embedding: tuple[float, ...],
    candidate_services: tuple[str, ...],
) -> HistoryScoreArtifact:
    """Apply paper H as positive-cosine-weighted frozen historical scores."""

    if not snapshot.verify():
        raise ValueError("history snapshot hash verification failed")
    query_norm = _validate_embedding(query_embedding, name="query embedding")
    candidates = tuple(
        canonicalize_service_name(service) for service in candidate_services
    )
    if not candidates:
        raise ValueError("history scoring requires at least one candidate service")
    if len(candidates) != len(set(candidates)):
        raise ValueError("history scoring candidates must be canonically unique")

    similarities: list[tuple[float, HistoricalCase]] = []
    for case in snapshot.cases:
        if len(case.embedding) != len(query_embedding):
            raise ValueError("historical embedding dimension does not match the query")
        case_norm = _validate_embedding(case.embedding, name="historical embedding")
        similarity = math.fsum(
            current * historical
            for current, historical in zip(query_embedding, case.embedding, strict=True)
        ) / (query_norm * case_norm)
        similarity = min(1.0, max(-1.0, similarity))
        if similarity > 0.0:
            similarities.append((similarity, case))
    nearest = tuple(
        sorted(similarities, key=lambda item: (-item[0], item[1].case_id))[:HISTORY_K]
    )

    candidate_set = set(candidates)
    score_maps = {
        case.case_id: {
            canonicalize_service_name(score.service): score for score in case.frozen_scores
        }
        for _, case in nearest
    }
    neighbors = tuple(
        HistoryNeighbor(
            case_id=case.case_id,
            similarity=similarity,
            matched_services=tuple(sorted(candidate_set & set(score_maps[case.case_id]))),
        )
        for similarity, case in nearest
    )
    scored_candidates: list[HistoryCandidateScore] = []
    for service in sorted(candidates):
        contributions: list[HistoryScoreContribution] = []
        for similarity, case in nearest:
            score = score_maps[case.case_id].get(service)
            if score is None:
                continue
            contributions.append(
                HistoryScoreContribution(
                    case_id=case.case_id,
                    historical_service=canonicalize_service_name(score.service),
                    similarity=similarity,
                    historical_score=score.score,
                    weighted_score=similarity * score.score,
                )
            )
        scored_candidates.append(
            HistoryCandidateScore(
                service=service,
                h=math.fsum(item.weighted_score for item in contributions),
                contributions=tuple(contributions),
            )
        )

    query_hash = sha256_hex(canonical_json_bytes(list(query_embedding)))
    unsigned = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "snapshot_hash": snapshot.snapshot_hash,
        "query_embedding_hash": query_hash,
        "k": HISTORY_K,
        "neighbors": [neighbor.model_dump(mode="json") for neighbor in neighbors],
        "candidates": [candidate.model_dump(mode="json") for candidate in scored_candidates],
    }
    result = HistoryScoreArtifact(
        snapshot_hash=snapshot.snapshot_hash,
        query_embedding_hash=query_hash,
        neighbors=neighbors,
        candidates=tuple(scored_candidates),
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("history score artifact verification failed")
    return result


def _validate_embedding(embedding: tuple[float, ...], *, name: str) -> float:
    if not embedding or any(not math.isfinite(value) for value in embedding):
        raise ValueError(f"{name} must contain finite values")
    norm = math.sqrt(math.fsum(value * value for value in embedding))
    if norm == 0.0:
        raise ValueError(f"{name} must have a non-zero norm")
    return norm


def _read_group_artifact(path: Path) -> HistoryGroupArtifact:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        artifact = HistoryGroupArtifact.model_validate(document)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"history group artifact is unreadable: {path.name}") from error
    if not artifact.verify():
        raise ValueError(f"history group artifact hash is invalid: {path.name}")
    return artifact


def _reject_duplicate_case_ids(
    *,
    artifacts: tuple[HistoryGroupArtifact, ...],
    incoming_ids: set[str],
) -> None:
    existing_ids = {
        case.case_id
        for artifact in artifacts
        for case in artifact.group.cases
    } | {
        item.case_id
        for artifact in artifacts
        for item in artifact.group.tombstones
    }
    duplicates = existing_ids & incoming_ids
    if duplicates:
        raise ValueError(f"history case identity was already committed: {sorted(duplicates)}")


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
