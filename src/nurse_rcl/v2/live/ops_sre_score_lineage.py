"""Read-only D16 selection lineage from a terminal, already admitted core.

This is a trusted publication/import adapter, not a model tool or an independent
audit of the core. It never reads telemetry, invokes scoring, or changes core
bytes/timing. The independent artifact retains all terms once; candidate/family
records refer to them. Actual tool delivery grants report permissions later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.family_aggregation import (
    REFERENCE_KEY,
    FamilyAggregatorDefinition,
    resolve_declared_family_aggregator,
)
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.repair_local_scoring import RepairCandidateLocalScore
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_propagation import (
    RepairCandidatePropagation,
    RepairPropagationPath,
    RepairPropagationWitness,
)
from nurse_rcl.v2.live.repair_ranking import RepairScoreDecomposition
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_strengths import RepairDependencePool, RepairStrengthTerm
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.reference_ranking import ReferenceScoringConfig

LINEAGE_SCHEMA: Literal["nurse.v2.sre-score-lineage/1"] = "nurse.v2.sre-score-lineage/1"
T = TypeVar("T")


def _unique(values: tuple[T, ...], keys: tuple[str, ...]) -> dict[str, T]:
    result = dict(zip(keys, values, strict=True))
    if len(result) != len(values):
        raise ValueError("lineage duplicate source identity")
    return result


class ScoreSourcePins(StrictFrozenModel):
    evidence_sha256: SHA256
    ranking_sha256: SHA256
    ranking_semantics_sha256: SHA256
    propagation_sha256: SHA256
    propagation_semantics_sha256: SHA256
    local_sha256: SHA256
    local_semantics_sha256: SHA256
    strengths_sha256: SHA256
    strengths_semantics_sha256: SHA256
    mechanisms_sha256: SHA256


class CandidateScoreLineage(StrictFrozenModel):
    rank: int = Field(ge=1)
    score: RepairScoreDecomposition
    local: RepairCandidateLocalScore
    propagation: RepairCandidatePropagation
    final_term_ids: tuple[SHA256, ...]
    selection_meaning: Literal["aggregation_inputs_not_additive_contributions"] = (
        "aggregation_inputs_not_additive_contributions"
    )

    @model_validator(mode="after")
    def selection_binding(self) -> Self:
        score, local, prop = self.score, self.local, self.propagation
        families = _unique(local.families, tuple(f.primary_family for f in local.families))
        if not set(local.selected_family_keys) <= set(families):
            raise ValueError("lineage missing selected family")
        if (
            score.candidate_service_key != local.candidate_service_key
            or score.candidate_service_key != prop.candidate_service_key
            or score.selected_term_ids != local.pool_selected_term_ids
            or score.selected_family_keys != local.selected_family_keys
            or score.selected_path_ids != prop.selected_path_ids
            or score.propagation_reason_codes != prop.reason_codes
            or score.event_evidence_score != local.event_evidence_score
            or score.local_score_component != local.local_score_component
            or score.local_evidence_status != local.evidence_status
            or score.propagation_score != prop.propagation_score
            or score.unknown_term_ids != local.unknown_term_ids
            or score.dependent_unused_term_ids != local.dependent_unused_term_ids
            or self.final_term_ids
            != tuple(families[key].selected_term_id for key in local.selected_family_keys)
            or not set(self.final_term_ids) <= set(local.pool_selected_term_ids)
        ):
            raise ValueError("lineage saved selection or score channel differs")
        return self


class SreScoreLineage(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-score-lineage/1"] = LINEAGE_SCHEMA
    core_identity: OpsCoreIdentity
    core_result_digest: SHA256
    source_pins: ScoreSourcePins
    scoring_config: ReferenceScoringConfig
    aggregation_definition: FamilyAggregatorDefinition
    terms: tuple[RepairStrengthTerm, ...]
    pools: tuple[RepairDependencePool, ...]
    paths: tuple[RepairPropagationPath, ...]
    witnesses: tuple[RepairPropagationWitness, ...]
    candidates: tuple[CandidateScoreLineage, ...]
    unbound_term_ids: tuple[str, ...]
    score_authority: Literal[False] = False
    projection_sha256: SHA256

    @model_validator(mode="after")
    def source_closure(self) -> Self:
        terms = _unique(self.terms, tuple(t.term_id for t in self.terms))
        pools = _unique(self.pools, tuple(p.pool_key for p in self.pools))
        paths = _unique(self.paths, tuple(p.path_id for p in self.paths))
        witnesses = _unique(self.witnesses, tuple(w.witness_id for w in self.witnesses))
        keys = tuple(c.score.candidate_service_key for c in self.candidates)
        _unique(self.candidates, keys)
        if (
            not self.scoring_config.verify()
            or self.aggregation_definition != (resolve_declared_family_aggregator(REFERENCE_KEY))
            or sorted(c.rank for c in self.candidates) != list(range(1, len(keys) + 1))
        ):
            raise ValueError("lineage configuration or rank domain differs")
        # Cross-reference checks only: maxima, K3, aggregation and ranking are
        # never recalculated here. Original algorithms remain core-owned.
        retained: list[str] = []
        for pool in self.pools:
            retained.extend(pool.retained_term_ids)
            if not set(pool.retained_term_ids) <= set(terms) or any(
                terms[t].pool_key != pool.pool_key
                or terms[t].candidate_service_key != pool.candidate_service_key
                for t in pool.retained_term_ids
            ):
                raise ValueError("lineage pool source differs")
            if pool.selected_term_id is not None and (
                pool.selected_term_id not in pool.retained_term_ids
                or terms[pool.selected_term_id].local_strength is None
                or pool.local_strength != terms[pool.selected_term_id].local_strength
            ):
                raise ValueError("lineage selected pool term differs")
        if len(set(retained)) != len(retained) or set(retained) != set(terms):
            raise ValueError("lineage pools do not partition terms")
        if set(self.unbound_term_ids) != {
            t.term_id for t in self.terms if t.candidate_service_key is None
        } or any(
            t.candidate_service_key is not None and t.candidate_service_key not in keys
            for t in self.terms
        ):
            raise ValueError("lineage candidate or unbound source differs")
        for row in self.candidates:
            key = row.score.candidate_service_key
            candidate_terms = {t.term_id: t for t in self.terms if t.candidate_service_key == key}
            candidate_pools = tuple(
                p.pool_key for p in self.pools if p.candidate_service_key == key
            )
            winners = tuple(
                t.term_id
                for t in candidate_terms.values()
                if pools[t.pool_key].selected_term_id == t.term_id
            )
            if row.score.dependent_pool_keys != candidate_pools or winners != (
                row.local.pool_selected_term_ids
            ):
                raise ValueError("lineage candidate pools differ")
            family_terms: list[str] = []
            for family in row.local.families:
                ids = (
                    *family.numeric_term_ids,
                    *family.dependent_unused_term_ids,
                    *family.unknown_term_ids,
                )
                family_terms.extend(ids)
                if (
                    len(set(ids)) != len(ids)
                    or not set(ids) <= set(candidate_terms)
                    or any(candidate_terms[t].primary_family != family.primary_family for t in ids)
                ):
                    raise ValueError("lineage family partition differs")
                if (
                    set(family.numeric_term_ids) != set(ids).intersection(winners)
                    or any(
                        candidate_terms[t].local_strength is not None
                        for t in family.unknown_term_ids
                    )
                    or any(
                        candidate_terms[t].local_strength is None
                        for t in family.dependent_unused_term_ids
                    )
                ):
                    raise ValueError("lineage family selection status differs")
                if family.selected_term_id is not None:
                    if family.selected_term_id not in family.numeric_term_ids or family.value != (
                        candidate_terms[family.selected_term_id].local_strength
                    ):
                        raise ValueError("lineage family representative differs")
                elif family.value is not None or family.numeric_term_ids:
                    raise ValueError("lineage family representative missing")
                if set(family.noncontributing_term_ids) != (
                    set(family.numeric_term_ids) - {family.selected_term_id}
                ):
                    raise ValueError("lineage noncontributing terms differ")
            if len(set(family_terms)) != len(family_terms) or (
                set(family_terms) != set(candidate_terms)
            ):
                raise ValueError("lineage families do not partition candidate terms")
            if not set(row.propagation.selected_path_ids) <= set(row.propagation.retained_path_ids):
                raise ValueError("lineage selected path not retained")
            if any(
                p not in paths or paths[p].source_service_key != key
                for p in row.propagation.retained_path_ids
            ):
                raise ValueError("lineage candidate path source differs")
        for path in self.paths:
            if (
                path.source_term_id not in terms
                or path.target_term_id not in terms
                or (
                    path.source_witness_id not in witnesses
                    or path.target_witness_id not in witnesses
                )
            ):
                raise ValueError("lineage path witness source missing")
        if self.projection_sha256 != sha256_hex(
            canonical_json_bytes(self.model_dump(mode="json", exclude={"projection_sha256"}))
        ):
            raise ValueError("lineage projection digest differs")
        return self


@_module_call("sre_score_lineage")
def project_score_lineage(core: OpsCoreResult) -> SreScoreLineage:
    """Extract once at an authenticated publication/import boundary.

    Caller owns core admission. This narrow projection must not turn every tool
    access into a full source replay or grant facts before terminal publication.
    """
    if type(core) is not OpsCoreResult:
        raise ValueError("lineage requires admitted terminal core")
    ranking = core.evidence.ranking
    prop = ranking.propagation
    local = prop.local_scoring
    strengths = local.strengths
    keys = ranking.candidate_service_keys
    if any(
        domain != keys
        for domain in (
            local.candidate_service_keys,
            prop.candidate_service_keys,
            strengths.candidate_service_keys,
            tuple(s.candidate_service_key for s in local.candidates),
            tuple(s.candidate_service_key for s in prop.candidates),
            tuple(s.candidate_service_key for s in ranking.scores),
        )
    ):
        raise ValueError("lineage candidate domain differs")
    rows = []
    ranks = {key: index for index, key in enumerate(ranking.ordered_service_keys, 1)}
    for score, candidate, propagation in zip(
        ranking.scores, local.candidates, prop.candidates, strict=True
    ):
        families = {f.primary_family: f for f in candidate.families}
        if not set(candidate.selected_family_keys) <= set(families):
            raise ValueError("lineage selected family missing")
        rows.append(
            dict(
                rank=ranks[score.candidate_service_key],
                score=score.model_dump(mode="json"),
                local=candidate.model_dump(mode="json"),
                propagation=propagation.model_dump(mode="json"),
                final_term_ids=[
                    families[k].selected_term_id for k in candidate.selected_family_keys
                ],
                selection_meaning="aggregation_inputs_not_additive_contributions",
            )
        )
    body = dict(
        schema_version=LINEAGE_SCHEMA,
        core_identity=core.identity.model_dump(mode="json"),
        core_result_digest=core.core_result_digest,
        source_pins=dict(
            evidence_sha256=core.evidence.result_sha256,
            ranking_sha256=ranking.result_sha256,
            ranking_semantics_sha256=ranking.semantics_sha256,
            propagation_sha256=prop.result_sha256,
            propagation_semantics_sha256=prop.semantics_sha256,
            local_sha256=local.result_sha256,
            local_semantics_sha256=local.semantics_sha256,
            strengths_sha256=strengths.result_sha256,
            strengths_semantics_sha256=strengths.semantics_sha256,
            mechanisms_sha256=strengths.mechanisms.result_sha256,
        ),
        scoring_config=local.scoring_config.model_dump(mode="json"),
        aggregation_definition=resolve_declared_family_aggregator(REFERENCE_KEY).model_dump(
            mode="json"
        ),
        terms=[t.model_dump(mode="json") for t in strengths.terms],
        pools=[p.model_dump(mode="json") for p in strengths.pools],
        paths=[p.model_dump(mode="json") for p in prop.paths],
        witnesses=[w.model_dump(mode="json") for w in prop.witnesses],
        candidates=rows,
        unbound_term_ids=list(local.unbound_term_ids),
        score_authority=False,
    )
    body["projection_sha256"] = sha256_hex(canonical_json_bytes(body))
    raw = canonical_json_bytes(body)
    if len(raw) > REPAIR_MAX_JSON_BYTES:
        raise ValueError("lineage artifact exceeds core storage capacity")
    return SreScoreLineage.model_validate_json(raw)


def term_selection(row: CandidateScoreLineage, term_id: str) -> dict[str, str | bool]:
    """Expose distinct saved selection levels; no marginal contribution claim."""
    for family in row.local.families:
        if term_id in (
            *family.numeric_term_ids,
            *family.dependent_unused_term_ids,
            *family.unknown_term_ids,
        ):
            return dict(
                numeric_status="unknown" if term_id in family.unknown_term_ids else "numeric",
                pool_selected=term_id in family.numeric_term_ids,
                family_selected=term_id == family.selected_term_id,
                family_used_in_aggregation=family.primary_family in row.local.selected_family_keys,
                used_in_family_aggregation=term_id in row.final_term_ids,
            )
    raise ValueError("lineage term not in candidate")


def save_score_lineage(path: Path, projection: SreScoreLineage) -> None:
    require_plain_path(path.parent)
    raw = canonical_json_bytes(projection.model_dump(mode="json"))
    if len(raw) > REPAIR_MAX_JSON_BYTES:
        raise ValueError("lineage artifact exceeds core storage capacity")
    SreScoreLineage.model_validate_json(raw)
    if path.exists() or path.is_symlink():
        if _read_bytes(path, REPAIR_MAX_JSON_BYTES) != raw:
            raise ValueError("lineage destination already binds different content")
        return
    atomic_write_bytes(path, raw)


def load_score_lineage(
    path: Path, *, core: OpsCoreResult, max_bytes: int = REPAIR_MAX_JSON_BYTES,
) -> SreScoreLineage:
    """Restore at admission, authenticating the projection against its saved core."""
    raw = _read_bytes(path, max_bytes)
    projection = SreScoreLineage.model_validate_json(raw)
    expected = project_score_lineage(core)
    if raw != canonical_json_bytes(expected.model_dump(mode="json")):
        raise ValueError("lineage artifact differs from admitted core sources")
    return projection
