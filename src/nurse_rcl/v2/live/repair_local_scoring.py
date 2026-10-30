"""Complete local score channel, not a fabricated complete-ranking receipt.

The repair pools are dependent votes, NOT legacy event associations. Reuse only
the registered numerical family kernel. Raw-file authority, propagation roles,
negative correction and final ranking belong to the enclosing scorer. Unknown
propagation is not recorded here as a verified numerical zero.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Final, Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.family_aggregation import (
    _AGGREGATOR_IMPLEMENTATIONS,
    REFERENCE_KEY,
    resolve_declared_family_aggregator,
)
from nurse_rcl.v2.live.repair_mechanisms import _replayed_mechanism_context
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_strengths import (
    REPAIR_STRENGTH_SEMANTICS_SHA256,
    RepairStrengthEvidence,
    RepairStrengthTerm,
    _compile_strength_values,
)
from nurse_rcl.v2.reference_ranking import ReferenceScoringConfig, build_reference_scoring_config

REPAIR_LOCAL_SCORING_SCHEMA_VERSION: Final = "nurse.v2.live-repair-local-scoring/1"
REPAIR_LOCAL_SCORING_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-local-scoring/1",
            strengths=REPAIR_STRENGTH_SEMANTICS_SHA256,
            family=resolve_declared_family_aggregator(REFERENCE_KEY).definition_sha256,
            scoring=build_reference_scoring_config().config_sha256,
            within_family="max_of_one_selected_term_per_dependence_pool",
            tie="family_lexical_then_term_identity_not_additional_evidence",
            complete="all_frozen_C_all_terms_in_disjoint_numeric_selected_dependent_unknown",
            authority="local_channel_only_not_propagation_raw_file_or_final_ranking",
        )
    )
)


class RepairFamilyValue(RepairModel):
    primary_family: str
    value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    selected_term_id: SHA256 | None
    numeric_term_ids: tuple[str, ...]
    noncontributing_term_ids: tuple[str, ...]
    unknown_term_ids: tuple[str, ...]
    dependent_unused_term_ids: tuple[str, ...]


class RepairCandidateLocalScore(RepairModel):
    candidate_service_key: str
    evidence_status: Literal["computed", "unknown_only", "no_retained_evidence"]
    families: tuple[RepairFamilyValue, ...]
    selected_family_keys: tuple[str, ...]
    noncontributing_family_keys: tuple[str, ...]
    event_evidence_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    local_score_component: float = Field(ge=0, le=1, allow_inf_nan=False)
    pool_selected_term_ids: tuple[str, ...]
    dependent_unused_term_ids: tuple[str, ...]
    unknown_term_ids: tuple[str, ...]


class RepairLocalScoring(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-local-scoring/1"] = (
        REPAIR_LOCAL_SCORING_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    opaque_case_id: str
    strengths: RepairStrengthEvidence
    scoring_config: ReferenceScoringConfig
    candidate_service_keys: tuple[str, ...]
    candidates: tuple[RepairCandidateLocalScore, ...]
    unbound_term_ids: tuple[str, ...]
    propagation_status: Literal["not_compiled"] = "not_compiled"
    negative_correction_status: Literal["not_compiled"] = "not_compiled"
    raw_file_authority: Literal[False] = False
    ranking_ready: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (RepairLocalScoring, RepairStrengthEvidence)


def _candidate_value(
    candidate: str, terms: tuple[RepairStrengthTerm, ...], winners: set[str], lambda_p: float
) -> RepairCandidateLocalScore:
    groups: dict[str, list[RepairStrengthTerm]] = defaultdict(list)
    for term in terms:
        groups[term.primary_family].append(term)
    rows = []
    for family, members in sorted(groups.items()):
        numeric = tuple(t for t in members if t.term_id in winners)
        if any(t.local_strength is None for t in numeric):
            raise ValueError("unknown term cannot win a numerical dependence pool")
        selected = (
            min(numeric, key=lambda t: (-cast(float, t.local_strength), t.term_id))
            if numeric
            else None
        )
        rows.append(
            RepairFamilyValue(
                primary_family=family,
                value=selected.local_strength if selected else None,
                selected_term_id=selected.term_id if selected else None,
                numeric_term_ids=tuple(t.term_id for t in numeric),
                noncontributing_term_ids=tuple(
                    t.term_id
                    for t in numeric
                    if selected is not None and t.term_id != selected.term_id
                ),
                unknown_term_ids=tuple(t.term_id for t in members if t.local_strength is None),
                dependent_unused_term_ids=tuple(
                    t.term_id
                    for t in members
                    if t.local_strength is not None and t.term_id not in winners
                ),
            )
        )
    numeric_families = sorted(
        (f for f in rows if f.value is not None),
        key=lambda f: (-cast(float, f.value), f.primary_family),
    )
    implementation = _AGGREGATOR_IMPLEMENTATIONS[REFERENCE_KEY]
    count = implementation.definition.maximum_contributing_families
    selected_families = numeric_families[:count]
    event_score = implementation.aggregate(tuple(cast(float, f.value) for f in selected_families))
    return RepairCandidateLocalScore(
        candidate_service_key=candidate,
        evidence_status="computed"
        if numeric_families
        else "unknown_only"
        if terms
        else "no_retained_evidence",
        families=tuple(rows),
        selected_family_keys=tuple(f.primary_family for f in selected_families),
        noncontributing_family_keys=tuple(
            sorted(f.primary_family for f in numeric_families[count:])
        ),
        event_evidence_score=event_score,
        local_score_component=(1.0 - lambda_p) * event_score,
        pool_selected_term_ids=tuple(t.term_id for t in terms if t.term_id in winners),
        dependent_unused_term_ids=tuple(
            t.term_id for t in terms if t.local_strength is not None and t.term_id not in winners
        ),
        unknown_term_ids=tuple(t.term_id for t in terms if t.local_strength is None),
    )


def _compile_local_values(strengths: RepairStrengthEvidence) -> RepairLocalScoring:
    """Numerical body only, inside complete source/ledger and two-exit guards."""
    winners = {p.selected_term_id for p in strengths.pools if p.selected_term_id is not None}
    config = build_reference_scoring_config()
    by_candidate: dict[str, list[RepairStrengthTerm]] = defaultdict(list)
    unbound = []
    for term in strengths.terms:
        if term.candidate_service_key is None:
            unbound.append(term.term_id)
        elif term.candidate_service_key not in strengths.candidate_service_keys:
            raise ValueError("local score term outside frozen complete C")
        else:
            by_candidate[term.candidate_service_key].append(term)
    result = RepairLocalScoring(
        semantics_sha256=REPAIR_LOCAL_SCORING_SEMANTICS_SHA256,
        opaque_case_id=strengths.opaque_case_id,
        strengths=strengths,
        scoring_config=config,
        candidate_service_keys=strengths.candidate_service_keys,
        candidates=tuple(
            _candidate_value(c, tuple(by_candidate[c]), winners, config.lambda_p)
            for c in strengths.candidate_service_keys
        ),
        unbound_term_ids=tuple(unbound),
        result_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(result))
    unsigned.pop("result_sha256")
    return result.model_copy(update={"result_sha256": sha256_hex(canonical_json_bytes(unsigned))})


def build_repair_local_scoring(**source_arguments: Any) -> RepairLocalScoring:
    """No arbitrary scores/strengths accepted; replay the complete paid source context."""
    with ExitStack() as guards:
        mechanisms, session, summaries = guards.enter_context(
            _replayed_mechanism_context(**source_arguments)
        )
        strengths = _compile_strength_values(
            mechanisms=mechanisms,
            session=session,
            sources=source_arguments["runtime"].arguments["sources"],
            error_summaries=summaries,
        )
        guards.callback(require_unchanged, strengths, _ROOTS, snapshot(strengths, _ROOTS))
        result = _compile_local_values(strengths)
        canonical(result, _ROOTS)
        guards.callback(require_unchanged, result, _ROOTS, snapshot(result, _ROOTS))
        return result


def require_repair_local_scoring(
    *, scoring: RepairLocalScoring, expected_scoring_sha256: str, **source_arguments: Any
) -> RepairLocalScoring:
    if type(scoring) is not RepairLocalScoring:
        raise ValueError("exact repair local scoring artifact required")
    before = snapshot(scoring, _ROOTS)
    try:
        require_pin(expected_scoring_sha256)
        rebuilt = build_repair_local_scoring(**source_arguments)
        if rebuilt.result_sha256 != expected_scoring_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(scoring, _ROOTS):
            raise ValueError("repair local scoring differs from complete independent replay")
        return rebuilt
    finally:
        require_unchanged(scoring, _ROOTS, before)
