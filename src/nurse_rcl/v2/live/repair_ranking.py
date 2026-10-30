"""Full-C repair score projection, subordinate to complete raw-file audit.

The existing scalar formula/quantization/hash-order kernels are reused without
pretending dependence pools are legacy EventGroups or causal graph receipts.
No caller-authored scores or weights enter this public source-replay boundary.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Final, Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_propagation import (
    REPAIR_PROPAGATION_SEMANTICS_SHA256,
    RepairPropagationEvidence,
    _replayed_propagation_context,
)
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.ranking_schemas import _quantize, _stable_tie_break_sha256

REPAIR_RANKING_SCHEMA_VERSION: Final = "nurse.v2.live-repair-ranking/1"
REPAIR_RANKING_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-ranking/1",
            propagation=REPAIR_PROPAGATION_SEMANTICS_SHA256,
            score="fixed_reference_0.75_local_plus_0.25_dependency_symptom_support",
            history_and_rules="off",
            negative_correction="not_granted_by_dependency_symptom_relation",
            order="round_half_even_12_desc_score_E_P_then_fixed_reference_hash",
            evidence_ties="equal_quantized_score_E_P_hash_is_not_evidence",
            lineage="repair_dependence_pools_and_terms_not_legacy_event_groups",
            authority="complete_score_projection_requires_outer_raw_file_replay",
        )
    )
)


class RepairScoreDecomposition(RepairModel):
    candidate_service_key: str
    local_evidence_status: Literal["computed", "unknown_only", "no_retained_evidence"]
    event_evidence_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    propagation_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    local_score_component: float = Field(ge=0, le=1, allow_inf_nan=False)
    propagation_score_component: float = Field(ge=0, le=1, allow_inf_nan=False)
    rule_score_component: float = Field(default=0.0, ge=0, le=0, allow_inf_nan=False)
    history_score_component: float = Field(default=0.0, ge=0, le=0, allow_inf_nan=False)
    negative_correction_status: Literal["no_authorized_negative_role"] = (
        "no_authorized_negative_role"
    )
    total_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    quantized_total_score: str
    quantized_event_evidence_score: str
    quantized_propagation_score: str
    dependent_pool_keys: tuple[str, ...]
    selected_term_ids: tuple[str, ...]
    selected_family_keys: tuple[str, ...]
    selected_path_ids: tuple[str, ...]
    unknown_term_ids: tuple[str, ...]
    dependent_unused_term_ids: tuple[str, ...]
    propagation_reason_codes: tuple[str, ...]
    score_sha256: SHA256


class RepairEvidenceTieGroup(RepairModel):
    tie_group_id: SHA256
    quantized_total_score: str
    quantized_event_evidence_score: str
    quantized_propagation_score: str
    member_service_keys: tuple[str, ...]
    member_score_sha256s: tuple[str, ...]
    tie_break_is_evidence: Literal[False] = False


class RepairRanking(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-ranking/1"] = REPAIR_RANKING_SCHEMA_VERSION
    semantics_sha256: SHA256
    opaque_case_id: str
    propagation: RepairPropagationEvidence
    candidate_service_keys: tuple[str, ...]
    scores: tuple[RepairScoreDecomposition, ...]
    tie_groups: tuple[RepairEvidenceTieGroup, ...]
    ordered_service_keys: tuple[str, ...]
    top_k: int = Field(ge=1, le=100)
    top_k_service_keys: tuple[str, ...]
    is_root_cause_probability: Literal[False] = False
    raw_file_authority: Literal[False] = False
    report_ready: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (RepairRanking, RepairPropagationEvidence)


def _seal(value: Any, field: str) -> Any:
    document = cast(dict[str, object], plain(value))
    document.pop(field)
    return value.model_copy(update={field: sha256_hex(canonical_json_bytes(document))})


@_module_call("ranking")
def _compile_ranking_values(propagation: RepairPropagationEvidence) -> RepairRanking:
    """Pure numerical projection under the complete live-source context guards."""
    local = propagation.local_scoring
    config = local.scoring_config
    keys = local.candidate_service_keys
    if keys != propagation.candidate_service_keys or not keys:
        raise ValueError("repair ranking requires the complete nonempty frozen C")
    if (
        tuple(c.candidate_service_key for c in local.candidates) != keys
        or tuple(c.candidate_service_key for c in propagation.candidates) != keys
    ):
        raise ValueError("repair score channels do not exactly cover frozen C")
    if config.history_mode != "off" or config.lambda_p != 0.25:
        raise ValueError("repair ranking cannot change frozen weights or enable history")
    scores = []
    for left, right in zip(local.candidates, propagation.candidates, strict=True):
        contribution = config.lambda_p * right.propagation_score
        total = left.local_score_component + contribution
        draft = RepairScoreDecomposition(
            candidate_service_key=left.candidate_service_key,
            local_evidence_status=left.evidence_status,
            event_evidence_score=left.event_evidence_score,
            propagation_score=right.propagation_score,
            local_score_component=left.local_score_component,
            propagation_score_component=contribution,
            total_score=total,
            quantized_total_score=_quantize(total),
            quantized_event_evidence_score=_quantize(left.event_evidence_score),
            quantized_propagation_score=_quantize(right.propagation_score),
            dependent_pool_keys=tuple(
                p.pool_key
                for p in local.strengths.pools
                if p.candidate_service_key == left.candidate_service_key
            ),
            selected_term_ids=left.pool_selected_term_ids,
            selected_family_keys=left.selected_family_keys,
            selected_path_ids=right.selected_path_ids,
            unknown_term_ids=left.unknown_term_ids,
            dependent_unused_term_ids=left.dependent_unused_term_ids,
            propagation_reason_codes=right.reason_codes,
            score_sha256="0" * 64,
        )
        scores.append(_seal(draft, "score_sha256"))
    by_tie: dict[tuple[str, str, str], list[RepairScoreDecomposition]] = defaultdict(list)
    for score in scores:
        by_tie[
            (
                score.quantized_total_score,
                score.quantized_event_evidence_score,
                score.quantized_propagation_score,
            )
        ].append(score)
    ties = []
    for (total_q, event_q, propagation_q), members in sorted(by_tie.items()):
        members = sorted(members, key=lambda s: s.candidate_service_key)
        ties.append(
            _seal(
                RepairEvidenceTieGroup(
                    tie_group_id="0" * 64,
                    quantized_total_score=total_q,
                    quantized_event_evidence_score=event_q,
                    quantized_propagation_score=propagation_q,
                    member_service_keys=tuple(s.candidate_service_key for s in members),
                    member_score_sha256s=tuple(s.score_sha256 for s in members),
                ),
                "tie_group_id",
            )
        )
    ordered = tuple(
        s.candidate_service_key
        for s in sorted(
            scores,
            key=lambda s: (
                -Decimal(s.quantized_total_score),
                -Decimal(s.quantized_event_evidence_score),
                -Decimal(s.quantized_propagation_score),
                _stable_tie_break_sha256(
                    tie_break_seed_sha256=config.tie_break_seed_sha256,
                    candidate_service_key=s.candidate_service_key,
                ),
            ),
        )
    )
    return cast(
        RepairRanking,
        _seal(
            RepairRanking(
                semantics_sha256=REPAIR_RANKING_SEMANTICS_SHA256,
                opaque_case_id=local.opaque_case_id,
                propagation=propagation,
                candidate_service_keys=keys,
                scores=tuple(scores),
                tie_groups=tuple(sorted(ties, key=lambda t: t.tie_group_id)),
                ordered_service_keys=ordered,
                top_k=config.top_k,
                top_k_service_keys=ordered[: config.top_k],
                result_sha256="0" * 64,
            ),
            "result_sha256",
        ),
    )


def build_repair_ranking(**source_arguments: Any) -> RepairRanking:
    with _replayed_propagation_context(**source_arguments) as propagation:
        result = _compile_ranking_values(propagation)
        canonical(result, _ROOTS)
        return result


def require_repair_ranking(
    *, ranking: RepairRanking, expected_ranking_sha256: str, **source_arguments: Any
) -> RepairRanking:
    if type(ranking) is not RepairRanking:
        raise ValueError("exact current repair ranking required")
    before = snapshot(ranking, _ROOTS)
    try:
        require_pin(expected_ranking_sha256)
        rebuilt = build_repair_ranking(**source_arguments)
        if rebuilt.result_sha256 != expected_ranking_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(ranking, _ROOTS):
            raise ValueError("repair ranking differs from complete independent score replay")
        return rebuilt
    finally:
        require_unchanged(ranking, _ROOTS, before)
