"""Exact reference-ranking replay and non-causal model sensitivity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.ranking_schemas import (
    EvidenceTieGroup,
    ScoreDecomposition,
    TopKArtifact,
    build_top_k,
)
from nurse_rcl.v2.reference_ranking import (
    ReferenceRankingResult,
    ReferenceScoringConfig,
    _binary64,
    _build_candidate_decomposition,
    _build_evidence_tie_groups,
    _ordered_service_keys,
)

RANKING_REPLAY_RESULT_SCHEMA_VERSION: Literal["nurse.v2.ranking-replay-result/1"] = (
    "nurse.v2.ranking-replay-result/1"
)
SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION: Literal["nurse.v2.sensitivity-candidate-terms/1"] = (
    "nurse.v2.sensitivity-candidate-terms/1"
)
MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION: Literal["nurse.v2.model-sensitivity-request/1"] = (
    "nurse.v2.model-sensitivity-request/1"
)
CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION: Literal["nurse.v2.candidate-rank-sensitivity/1"] = (
    "nurse.v2.candidate-rank-sensitivity/1"
)
MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION: Literal["nurse.v2.model-sensitivity-result/1"] = (
    "nurse.v2.model-sensitivity-result/1"
)

_MAX_RANKING_CANDIDATES = 4_096


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> tuple[str, str]:
    digest = _digest(value)
    return f"{prefix}_{digest[:24]}", digest


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bounded_canonical_string_ids(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    if type(values) is tuple:
        tuple_values = cast(tuple[str, ...], values)
        count = tuple.__len__(tuple_values)
        exact_values: tuple[str, ...] | list[str] = tuple_values
    elif type(values) is list:
        list_values = cast(list[str], values)
        count = list.__len__(list_values)
        exact_values = list_values
    else:
        raise ValueError(f"{label} must use an exact tuple or list")
    if count > _MAX_RANKING_CANDIDATES:
        raise ValueError(f"{label} exceeds the replay limit")
    if any(type(item) is not str for item in exact_values):
        raise ValueError(f"{label} must contain exact strings")
    return tuple(sorted(set(exact_values)))


def _preflight_ranking(value: object) -> ReferenceRankingResult:
    if type(value) is not ReferenceRankingResult:
        raise ValueError("reference ranking must use the exact model type")
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or "scores" not in state:
        raise ValueError("reference ranking lacks its candidate scores")
    scores = dict.__getitem__(state, "scores")
    if type(scores) is not tuple:
        raise ValueError("reference ranking scores must use an exact tuple")
    if len(scores) > _MAX_RANKING_CANDIDATES:
        raise ValueError("reference ranking candidate count exceeds replay limit")
    if any(type(item) is not ScoreDecomposition for item in scores):
        raise ValueError("reference ranking uses a noncanonical score type")
    return value


def _rebuild_projection(
    *,
    config: ReferenceScoringConfig,
    opaque_case_id: str,
    candidate_universe_sha256: str,
    scores: Sequence[ScoreDecomposition],
) -> tuple[
    tuple[ScoreDecomposition, ...],
    tuple[EvidenceTieGroup, ...],
    tuple[str, ...],
    TopKArtifact,
]:
    canonical_scores = tuple(sorted(scores, key=lambda item: item.candidate_service_key))
    tie_groups = _build_evidence_tie_groups(canonical_scores)
    ordered_service_keys = _ordered_service_keys(
        scores=canonical_scores,
        tie_break_seed_sha256=config.tie_break_seed_sha256,
    )
    top_k = build_top_k(
        opaque_case_id=opaque_case_id,
        candidate_universe_sha256=candidate_universe_sha256,
        scores=canonical_scores,
        tie_groups=tie_groups,
        ordered_service_keys=ordered_service_keys,
        k=config.top_k,
        tie_break_seed_sha256=config.tie_break_seed_sha256,
    )
    return canonical_scores, tie_groups, ordered_service_keys, top_k


def _replay_scores(
    ranking: ReferenceRankingResult,
) -> tuple[ScoreDecomposition, ...]:
    return tuple(
        _build_candidate_decomposition(
            candidate_service_key=item.candidate_service_key,
            event_evidence_score=item.event_evidence_score,
            propagation_score=item.propagation_score,
            rule_score=item.rule_score,
            history_score=item.history_score,
            explicit_rules_enabled=ranking.explicit_rules_enabled,
            config=ranking.config,
            event_group_ids=item.event_group_ids,
            propagation_path_ids=item.propagation_path_ids,
            rule_ids=item.rule_ids,
            history_neighbor_ids=item.history_neighbor_ids,
        )
        for item in ranking.scores
    )


class RankingReplayResult(StrictFrozenModel):
    """Content-addressed receipt for one exact ranking replay."""

    schema_version: Literal["nurse.v2.ranking-replay-result/1"] = (
        RANKING_REPLAY_RESULT_SCHEMA_VERSION
    )
    replay_result_id: str = Field(pattern=r"^rreplay_[0-9a-f]{24}$")
    source_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1, le=_MAX_RANKING_CANDIDATES)
    candidate_service_keys: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    score_sha256s: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    evidence_tie_group_sha256s: tuple[str, ...] = Field(min_length=1)
    ordered_service_keys: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    top_k_ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_match: Literal[True] = True
    replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_receipt(self) -> Self:
        if self.candidate_service_keys != tuple(sorted(set(self.candidate_service_keys))):
            raise ValueError("replay candidate keys must be canonical and unique")
        if (
            self.candidate_count != len(self.candidate_service_keys)
            or len(self.score_sha256s) != self.candidate_count
            or len(set(self.score_sha256s)) != self.candidate_count
            or len(self.ordered_service_keys) != self.candidate_count
            or set(self.ordered_service_keys) != set(self.candidate_service_keys)
        ):
            raise ValueError("replay receipt does not exactly cover its candidates")
        if self.evidence_tie_group_sha256s != tuple(sorted(set(self.evidence_tie_group_sha256s))):
            raise ValueError("replay tie-group digests must be canonical and unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not RankingReplayResult:
                return False
            rebuilt = RankingReplayResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"replay_result_id", "replay_sha256"},
            )
            identifier, digest = _stable_id("rreplay", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            rebuilt == self
            and rebuilt.replay_result_id == identifier
            and rebuilt.replay_sha256 == digest
        )


def replay_reference_ranking(
    ranking: ReferenceRankingResult,
    *,
    expected_result_sha256: str,
    expected_authoritative_input_sha256: str,
) -> RankingReplayResult:
    """Independently rebuild a ranking under two external digest anchors."""

    trusted = _preflight_ranking(ranking)
    result_anchor = _require_sha256(
        expected_result_sha256,
        label="expected ranking result",
    )
    input_anchor = _require_sha256(
        expected_authoritative_input_sha256,
        label="expected ranking authoritative input",
    )
    if trusted.result_sha256 != result_anchor or trusted.authoritative_input_sha256 != input_anchor:
        raise ValueError("external ranking anchor does not match the supplied result")
    if not trusted.verify_structure():
        raise ValueError("reference ranking failed structural verification")
    replayed_scores = _replay_scores(trusted)
    scores, ties, order, top_k = _rebuild_projection(
        config=trusted.config,
        opaque_case_id=trusted.opaque_case_id,
        candidate_universe_sha256=trusted.candidate_universe_sha256,
        scores=replayed_scores,
    )
    if (
        scores != trusted.scores
        or ties != trusted.evidence_tie_groups
        or order != trusted.ordered_service_keys
        or top_k != trusted.top_k
    ):
        raise ValueError("reference ranking does not replay exactly")
    unsigned: dict[str, object] = {
        "schema_version": RANKING_REPLAY_RESULT_SCHEMA_VERSION,
        "source_result_sha256": trusted.result_sha256,
        "source_authoritative_input_sha256": trusted.authoritative_input_sha256,
        "candidate_count": trusted.candidate_count,
        "candidate_service_keys": tuple(item.candidate_service_key for item in scores),
        "score_sha256s": tuple(item.score_sha256 for item in scores),
        "evidence_tie_group_sha256s": tuple(item.tie_group_sha256 for item in ties),
        "ordered_service_keys": order,
        "top_k_ranking_sha256": top_k.ranking_sha256,
        "exact_match": True,
    }
    identifier, digest = _stable_id("rreplay", unsigned)
    result = RankingReplayResult.model_validate(
        {
            **unsigned,
            "replay_result_id": identifier,
            "replay_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("ranking replay receipt verification failed")
    return result


class SensitivityCandidateTerms(StrictFrozenModel):
    """Complete post-removal E/P/R/H inputs for one legal candidate."""

    schema_version: Literal["nurse.v2.sensitivity-candidate-terms/1"] = (
        SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION
    )
    candidate_service_key: str = Field(min_length=1)
    event_evidence_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    propagation_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rule_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    event_group_ids: tuple[str, ...] = Field(max_length=_MAX_RANKING_CANDIDATES)
    propagation_path_ids: tuple[str, ...] = Field(max_length=_MAX_RANKING_CANDIDATES)
    rule_ids: tuple[str, ...] = Field(max_length=_MAX_RANKING_CANDIDATES)
    history_neighbor_ids: tuple[str, ...] = Field(max_length=_MAX_RANKING_CANDIDATES)
    source_recomputation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terms_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_references(self) -> Self:
        for values, label in (
            (self.event_group_ids, "event group IDs"),
            (self.propagation_path_ids, "propagation path IDs"),
            (self.rule_ids, "rule IDs"),
            (self.history_neighbor_ids, "history neighbor IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"sensitivity {label} must be canonical and unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not SensitivityCandidateTerms:
                return False
            rebuilt = SensitivityCandidateTerms.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(mode="json", exclude={"terms_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt == self and rebuilt.terms_sha256 == _digest(unsigned)


def build_sensitivity_candidate_terms(
    *,
    candidate_service_key: str,
    event_evidence_score: float,
    propagation_score: float,
    rule_score: float,
    history_score: float,
    event_group_ids: Sequence[str],
    propagation_path_ids: Sequence[str],
    rule_ids: Sequence[str],
    history_neighbor_ids: Sequence[str],
    source_recomputation_sha256: str,
) -> SensitivityCandidateTerms:
    canonical_event_group_ids = _bounded_canonical_string_ids(
        event_group_ids,
        label="sensitivity event group IDs",
    )
    canonical_propagation_path_ids = _bounded_canonical_string_ids(
        propagation_path_ids,
        label="sensitivity propagation path IDs",
    )
    canonical_rule_ids = _bounded_canonical_string_ids(
        rule_ids,
        label="sensitivity rule IDs",
    )
    canonical_history_neighbor_ids = _bounded_canonical_string_ids(
        history_neighbor_ids,
        label="sensitivity history neighbor IDs",
    )
    unsigned: dict[str, object] = {
        "schema_version": SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "event_evidence_score": event_evidence_score,
        "propagation_score": propagation_score,
        "rule_score": rule_score,
        "history_score": history_score,
        "event_group_ids": canonical_event_group_ids,
        "propagation_path_ids": canonical_propagation_path_ids,
        "rule_ids": canonical_rule_ids,
        "history_neighbor_ids": canonical_history_neighbor_ids,
        "source_recomputation_sha256": _require_sha256(
            source_recomputation_sha256,
            label="sensitivity source recomputation",
        ),
    }
    result = SensitivityCandidateTerms.model_validate(
        {**unsigned, "terms_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("sensitivity candidate terms verification failed")
    return result


class ModelSensitivityRequest(StrictFrozenModel):
    """Content-addressed request for a model-behavior-only recomputation."""

    schema_version: Literal["nurse.v2.model-sensitivity-request/1"] = (
        MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION
    )
    scenario_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    removed_hypothesis_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    replacement_terms: tuple[SensitivityCandidateTerms, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    analysis_kind: Literal["model_sensitivity"] = "model_sensitivity"
    claim_scope: Literal["model_behavior_only"] = "model_behavior_only"
    is_causal_intervention: Literal[False] = False
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_request(self) -> Self:
        if self.removed_hypothesis_ids != tuple(sorted(set(self.removed_hypothesis_ids))):
            raise ValueError("removed hypothesis IDs must be canonical and unique")
        if self.replacement_terms != tuple(
            sorted(
                self.replacement_terms,
                key=lambda item: item.candidate_service_key,
            )
        ):
            raise ValueError("sensitivity replacement terms must be canonical")
        if len({item.candidate_service_key for item in self.replacement_terms}) != len(
            self.replacement_terms
        ):
            raise ValueError("sensitivity replacement candidates must be unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ModelSensitivityRequest or any(
                type(item) is not SensitivityCandidateTerms or not item.verify()
                for item in self.replacement_terms
            ):
                return False
            rebuilt = ModelSensitivityRequest.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(mode="json", exclude={"request_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt == self and rebuilt.request_sha256 == _digest(unsigned)


def build_model_sensitivity_request(
    *,
    scenario_key: str,
    removed_hypothesis_ids: Sequence[str],
    replacement_terms: Sequence[SensitivityCandidateTerms],
) -> ModelSensitivityRequest:
    canonical_removed_ids = _bounded_canonical_string_ids(
        removed_hypothesis_ids,
        label="removed hypothesis IDs",
    )
    if type(replacement_terms) is tuple:
        tuple_terms = cast(tuple[SensitivityCandidateTerms, ...], replacement_terms)
        replacement_count = tuple.__len__(tuple_terms)
        exact_terms: tuple[SensitivityCandidateTerms, ...] | list[SensitivityCandidateTerms] = (
            tuple_terms
        )
    elif type(replacement_terms) is list:
        list_terms = cast(list[SensitivityCandidateTerms], replacement_terms)
        replacement_count = list.__len__(list_terms)
        exact_terms = list_terms
    else:
        raise ValueError("sensitivity replacements must use an exact tuple or list")
    if replacement_count > _MAX_RANKING_CANDIDATES:
        raise ValueError("sensitivity replacement count exceeds the replay limit")
    if any(
        type(item) is not SensitivityCandidateTerms or not item.verify() for item in exact_terms
    ):
        raise ValueError("sensitivity replacement terms failed verification")
    ordered_terms = tuple(sorted(exact_terms, key=lambda item: item.candidate_service_key))
    unsigned: dict[str, object] = {
        "schema_version": MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION,
        "scenario_key": scenario_key,
        "removed_hypothesis_ids": canonical_removed_ids,
        "replacement_terms": tuple(item.model_dump(mode="json") for item in ordered_terms),
        "analysis_kind": "model_sensitivity",
        "claim_scope": "model_behavior_only",
        "is_causal_intervention": False,
    }
    result = ModelSensitivityRequest.model_validate(
        {**unsigned, "request_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("model sensitivity request verification failed")
    return result


class CandidateRankSensitivity(StrictFrozenModel):
    """Exact score and rank movement for one candidate."""

    schema_version: Literal["nurse.v2.candidate-rank-sensitivity/1"] = (
        CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION
    )
    candidate_service_key: str = Field(min_length=1)
    base_rank: int = Field(ge=1, le=_MAX_RANKING_CANDIDATES)
    sensitivity_rank: int = Field(ge=1, le=_MAX_RANKING_CANDIDATES)
    base_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensitivity_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_total_score: float = Field(allow_inf_nan=False)
    sensitivity_total_score: float = Field(allow_inf_nan=False)
    total_score_delta: float = Field(allow_inf_nan=False)
    rank_changed: bool
    score_changed: bool
    sensitivity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_exact_delta_flags(self) -> Self:
        expected_delta = _binary64(
            self.sensitivity_total_score - self.base_total_score,
            label="sensitivity total-score delta",
        )
        if self.total_score_delta != expected_delta:
            raise ValueError("sensitivity total-score delta is not exact")
        if self.rank_changed != (self.base_rank != self.sensitivity_rank):
            raise ValueError("sensitivity rank-change flag is inconsistent")
        if self.score_changed != (self.base_score_sha256 != self.sensitivity_score_sha256):
            raise ValueError("sensitivity score-change flag is inconsistent")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not CandidateRankSensitivity:
                return False
            rebuilt = CandidateRankSensitivity.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"sensitivity_sha256"},
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt == self and rebuilt.sensitivity_sha256 == _digest(unsigned)


def _build_candidate_delta(
    *,
    candidate_service_key: str,
    base_rank: int,
    sensitivity_rank: int,
    base_score: ScoreDecomposition,
    sensitivity_score: ScoreDecomposition,
) -> CandidateRankSensitivity:
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "base_rank": base_rank,
        "sensitivity_rank": sensitivity_rank,
        "base_score_sha256": base_score.score_sha256,
        "sensitivity_score_sha256": sensitivity_score.score_sha256,
        "base_total_score": base_score.total_score,
        "sensitivity_total_score": sensitivity_score.total_score,
        "total_score_delta": _binary64(
            sensitivity_score.total_score - base_score.total_score,
            label="sensitivity total-score delta",
        ),
        "rank_changed": base_rank != sensitivity_rank,
        "score_changed": base_score.score_sha256 != sensitivity_score.score_sha256,
    }
    result = CandidateRankSensitivity.model_validate(
        {**unsigned, "sensitivity_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("candidate rank sensitivity verification failed")
    return result


class ModelSensitivityResult(StrictFrozenModel):
    """Complete rerank after a declared model-input removal scenario."""

    schema_version: Literal["nurse.v2.model-sensitivity-result/1"] = (
        MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION
    )
    model_sensitivity_result_id: str = Field(pattern=r"^msens_[0-9a-f]{24}$")
    base_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ReferenceScoringConfig
    explicit_rules_enabled: bool
    candidate_count: int = Field(ge=1, le=_MAX_RANKING_CANDIDATES)
    base_ordered_service_keys: tuple[str, ...] = Field(min_length=1)
    scores: tuple[ScoreDecomposition, ...] = Field(min_length=1)
    evidence_tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1)
    ordered_service_keys: tuple[str, ...] = Field(min_length=1)
    top_k: TopKArtifact
    candidate_deltas: tuple[CandidateRankSensitivity, ...] = Field(min_length=1)
    top_1_changed: bool
    top_k_membership_changed: bool
    analysis_kind: Literal["model_sensitivity"] = "model_sensitivity"
    claim_scope: Literal["model_behavior_only"] = "model_behavior_only"
    is_causal_intervention: Literal[False] = False
    is_root_cause_probability: Literal[False] = False
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_projection(self) -> Self:
        if self.scores != tuple(sorted(self.scores, key=lambda item: item.candidate_service_key)):
            raise ValueError("sensitivity scores must be canonical")
        candidate_keys = tuple(item.candidate_service_key for item in self.scores)
        if (
            self.candidate_count != len(candidate_keys)
            or len(set(candidate_keys)) != self.candidate_count
            or len(self.base_ordered_service_keys) != self.candidate_count
            or set(self.base_ordered_service_keys) != set(candidate_keys)
            or len(self.ordered_service_keys) != self.candidate_count
            or set(self.ordered_service_keys) != set(candidate_keys)
        ):
            raise ValueError("sensitivity result does not exactly cover candidates")
        if (
            self.candidate_deltas
            != tuple(
                sorted(
                    self.candidate_deltas,
                    key=lambda item: item.candidate_service_key,
                )
            )
            or tuple(item.candidate_service_key for item in self.candidate_deltas) != candidate_keys
        ):
            raise ValueError("sensitivity deltas must canonically cover candidates")
        return self

    def verify_structure(self) -> bool:
        try:
            if type(self) is not ModelSensitivityResult:
                return False
            rebuilt = ModelSensitivityResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self or not rebuilt.config.verify():
                return False
            if any(
                type(item) is not ScoreDecomposition or not item.verify() for item in rebuilt.scores
            ) or any(
                type(item) is not CandidateRankSensitivity or not item.verify()
                for item in rebuilt.candidate_deltas
            ):
                return False
            for item in rebuilt.scores:
                exact_total = _binary64(
                    (1.0 - item.lambda_p) * item.event_evidence_score
                    + item.lambda_p * item.propagation_score
                    + item.eta_r * item.rule_score
                    + item.eta_h * item.history_score,
                    label="sensitivity total score",
                )
                if (
                    item.total_score != exact_total
                    or item.lambda_p != rebuilt.config.lambda_p
                    or item.eta_r
                    != (
                        rebuilt.config.eta_r_when_enabled if rebuilt.explicit_rules_enabled else 0.0
                    )
                    or item.eta_h != rebuilt.config.eta_h
                    or item.scoring_configuration_sha256 != rebuilt.config.config_sha256
                ):
                    return False
            expected_scores, expected_ties, expected_order, expected_top_k = _rebuild_projection(
                config=rebuilt.config,
                opaque_case_id=rebuilt.opaque_case_id,
                candidate_universe_sha256=(rebuilt.candidate_universe_sha256),
                scores=rebuilt.scores,
            )
            if (
                expected_scores != rebuilt.scores
                or expected_ties != rebuilt.evidence_tie_groups
                or expected_order != rebuilt.ordered_service_keys
                or expected_top_k != rebuilt.top_k
            ):
                return False
            base_rank_by_key = {
                candidate_service_key: rank
                for rank, candidate_service_key in enumerate(
                    rebuilt.base_ordered_service_keys,
                    start=1,
                )
            }
            sensitivity_rank_by_key = {
                candidate_service_key: rank
                for rank, candidate_service_key in enumerate(
                    rebuilt.ordered_service_keys,
                    start=1,
                )
            }
            sensitivity_score_by_key = {item.candidate_service_key: item for item in rebuilt.scores}
            for delta in rebuilt.candidate_deltas:
                sensitivity_score = sensitivity_score_by_key[delta.candidate_service_key]
                if (
                    delta.base_rank != base_rank_by_key[delta.candidate_service_key]
                    or delta.sensitivity_rank
                    != sensitivity_rank_by_key[delta.candidate_service_key]
                    or delta.sensitivity_score_sha256 != sensitivity_score.score_sha256
                    or delta.sensitivity_total_score != sensitivity_score.total_score
                ):
                    return False
            base_top_k = set(
                rebuilt.base_ordered_service_keys[
                    : min(rebuilt.config.top_k, rebuilt.candidate_count)
                ]
            )
            sensitivity_top_k = {item.service_entity_key for item in rebuilt.top_k.entries}
            if rebuilt.top_1_changed != (
                rebuilt.ordered_service_keys[0] != rebuilt.base_ordered_service_keys[0]
            ) or rebuilt.top_k_membership_changed != (sensitivity_top_k != base_top_k):
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"model_sensitivity_result_id", "result_sha256"},
            )
            identifier, digest = _stable_id("msens", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt.model_sensitivity_result_id == identifier and rebuilt.result_sha256 == digest

    def verify_against(
        self,
        ranking: ReferenceRankingResult,
        request: ModelSensitivityRequest,
    ) -> bool:
        try:
            if not self.verify_structure():
                return False
            trusted = _preflight_ranking(ranking)
            if type(request) is not ModelSensitivityRequest or not request.verify():
                return False
            replay_reference_ranking(
                trusted,
                expected_result_sha256=self.base_result_sha256,
                expected_authoritative_input_sha256=(self.base_authoritative_input_sha256),
            )
            if (
                self.base_result_sha256 != trusted.result_sha256
                or self.base_authoritative_input_sha256 != trusted.authoritative_input_sha256
                or self.request_sha256 != request.request_sha256
                or self.opaque_case_id != trusted.opaque_case_id
                or self.candidate_universe_sha256 != trusted.candidate_universe_sha256
                or self.config != trusted.config
                or self.explicit_rules_enabled != trusted.explicit_rules_enabled
                or self.base_ordered_service_keys != trusted.ordered_service_keys
            ):
                return False
            scores, ties, order, top_k = _sensitivity_projection(trusted, request)
            if (
                self.scores != scores
                or self.evidence_tie_groups != ties
                or self.ordered_service_keys != order
                or self.top_k != top_k
            ):
                return False
            expected_deltas = _candidate_deltas(
                trusted,
                sensitivity_scores=scores,
                sensitivity_order=order,
            )
            base_top_k = {item.service_entity_key for item in trusted.top_k.entries}
            sensitivity_top_k = {item.service_entity_key for item in top_k.entries}
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            self.candidate_deltas == expected_deltas
            and self.top_1_changed == (order[0] != trusted.ordered_service_keys[0])
            and self.top_k_membership_changed == (sensitivity_top_k != base_top_k)
        )

    def verify(self) -> bool:
        return self.verify_structure()


def _sensitivity_projection(
    ranking: ReferenceRankingResult,
    request: ModelSensitivityRequest,
) -> tuple[
    tuple[ScoreDecomposition, ...],
    tuple[EvidenceTieGroup, ...],
    tuple[str, ...],
    TopKArtifact,
]:
    candidate_keys = {item.candidate_service_key for item in ranking.scores}
    replacements = {item.candidate_service_key: item for item in request.replacement_terms}
    outside = set(replacements) - candidate_keys
    if outside:
        raise ValueError("sensitivity candidate is outside the frozen candidate universe")
    projected: list[ScoreDecomposition] = []
    for base in ranking.scores:
        replacement = replacements.get(base.candidate_service_key)
        if replacement is None:
            projected.append(base)
            continue
        projected.append(
            _build_candidate_decomposition(
                candidate_service_key=replacement.candidate_service_key,
                event_evidence_score=replacement.event_evidence_score,
                propagation_score=replacement.propagation_score,
                rule_score=replacement.rule_score,
                history_score=replacement.history_score,
                explicit_rules_enabled=ranking.explicit_rules_enabled,
                config=ranking.config,
                event_group_ids=replacement.event_group_ids,
                propagation_path_ids=replacement.propagation_path_ids,
                rule_ids=replacement.rule_ids,
                history_neighbor_ids=replacement.history_neighbor_ids,
            )
        )
    return _rebuild_projection(
        config=ranking.config,
        opaque_case_id=ranking.opaque_case_id,
        candidate_universe_sha256=ranking.candidate_universe_sha256,
        scores=projected,
    )


def _candidate_deltas(
    ranking: ReferenceRankingResult,
    *,
    sensitivity_scores: Sequence[ScoreDecomposition],
    sensitivity_order: Sequence[str],
) -> tuple[CandidateRankSensitivity, ...]:
    base_by_key = {item.candidate_service_key: item for item in ranking.scores}
    sensitivity_by_key = {item.candidate_service_key: item for item in sensitivity_scores}
    base_rank = {
        candidate_service_key: index
        for index, candidate_service_key in enumerate(
            ranking.ordered_service_keys,
            start=1,
        )
    }
    sensitivity_rank = {
        candidate_service_key: index
        for index, candidate_service_key in enumerate(sensitivity_order, start=1)
    }
    return tuple(
        _build_candidate_delta(
            candidate_service_key=candidate_service_key,
            base_rank=base_rank[candidate_service_key],
            sensitivity_rank=sensitivity_rank[candidate_service_key],
            base_score=base_by_key[candidate_service_key],
            sensitivity_score=sensitivity_by_key[candidate_service_key],
        )
        for candidate_service_key in sorted(base_by_key)
    )


def recompute_model_sensitivity(
    ranking: ReferenceRankingResult,
    request: ModelSensitivityRequest,
    *,
    expected_result_sha256: str,
    expected_authoritative_input_sha256: str,
) -> ModelSensitivityResult:
    """Recompute rank movement without claiming a real-world intervention."""

    trusted = _preflight_ranking(ranking)
    replay_reference_ranking(
        trusted,
        expected_result_sha256=expected_result_sha256,
        expected_authoritative_input_sha256=(expected_authoritative_input_sha256),
    )
    if type(request) is not ModelSensitivityRequest or not request.verify():
        raise ValueError("model sensitivity request failed verification")
    scores, ties, order, top_k = _sensitivity_projection(trusted, request)
    deltas = _candidate_deltas(
        trusted,
        sensitivity_scores=scores,
        sensitivity_order=order,
    )
    base_top_k = {item.service_entity_key for item in trusted.top_k.entries}
    sensitivity_top_k = {item.service_entity_key for item in top_k.entries}
    unsigned: dict[str, object] = {
        "schema_version": MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION,
        "base_result_sha256": trusted.result_sha256,
        "base_authoritative_input_sha256": trusted.authoritative_input_sha256,
        "request_sha256": request.request_sha256,
        "opaque_case_id": trusted.opaque_case_id,
        "candidate_universe_sha256": trusted.candidate_universe_sha256,
        "config": trusted.config.model_dump(mode="json"),
        "explicit_rules_enabled": trusted.explicit_rules_enabled,
        "candidate_count": trusted.candidate_count,
        "base_ordered_service_keys": trusted.ordered_service_keys,
        "scores": tuple(item.model_dump(mode="json") for item in scores),
        "evidence_tie_groups": tuple(item.model_dump(mode="json") for item in ties),
        "ordered_service_keys": order,
        "top_k": top_k.model_dump(mode="json"),
        "candidate_deltas": tuple(item.model_dump(mode="json") for item in deltas),
        "top_1_changed": order[0] != trusted.ordered_service_keys[0],
        "top_k_membership_changed": sensitivity_top_k != base_top_k,
        "analysis_kind": "model_sensitivity",
        "claim_scope": "model_behavior_only",
        "is_causal_intervention": False,
        "is_root_cause_probability": False,
    }
    identifier, digest = _stable_id("msens", unsigned)
    result = ModelSensitivityResult.model_validate(
        {
            **unsigned,
            "model_sensitivity_result_id": identifier,
            "result_sha256": digest,
        }
    )
    if not result.verify_against(trusted, request):  # pragma: no cover
        raise AssertionError("model sensitivity result verification failed")
    return result


__all__ = [
    "CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION",
    "MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION",
    "MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION",
    "RANKING_REPLAY_RESULT_SCHEMA_VERSION",
    "SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION",
    "CandidateRankSensitivity",
    "ModelSensitivityRequest",
    "ModelSensitivityResult",
    "RankingReplayResult",
    "SensitivityCandidateTerms",
    "build_model_sensitivity_request",
    "build_sensitivity_candidate_terms",
    "recompute_model_sensitivity",
    "replay_reference_ranking",
]
