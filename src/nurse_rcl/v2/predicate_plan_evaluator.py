"""Deterministically evaluate proposal predicates over frozen original observations.

Each leaf tests one referenced observation using the guard's fixed evidence
contract. Abnormal-strength predicates are existential (Strong Kleene OR), and
verified-normal predicates are universal (Strong Kleene AND) over those leaves.
These are tests within one pending predicate, not score aggregation or claims
that an entire service is normal. Exact repeated observation IDs are removed;
different observations with the same source fingerprint remain visible, including
conflicts. Strong Kleene composition never counts votes.

This module has no model measurements, query-derived observation store, labels,
or score authority. The hypothesis guard must still check candidate, event,
relation, and time bindings and the complete mechanism's prerequisites. Relation
tokens below record proposed references only; they do not authorize a relation.
Single-observation onset interval widths do not measure cross-service precedence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from nurse_rcl.v2.diagnostic_schemas import (
    MeasuredValue,
    PredicateResult,
    PredicateThreshold,
    build_predicate_result,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult, replay_event_grouping_result
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.observation_schemas import Observation
from nurse_rcl.v2.predicate_logic import (
    evaluate_predicate,
    strong_kleene_and,
    strong_kleene_or,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_proposals import PredicatePlan

PREDICATE_PLAN_EVALUATION_POLICY_VERSION = "original-observation-predicate-plan/1"


def _unavailable_reasons(observation: Observation) -> tuple[str, ...]:
    reasons: list[str] = []
    if not observation.formal_scoring_eligible:
        reasons.append("formal_scoring_ineligible")
    if observation.evidence_state in {"unknown", "degraded", "no_record"}:
        reasons.append(f"evidence_state_{observation.evidence_state}")
    # The producer preserves observed zero even when its coverage is insufficient.
    # Consume its declared quality decision instead of inventing a coverage cutoff.
    if "insufficient_window_coverage" in observation.quality_flags:
        reasons.append("insufficient_window_coverage")
    for name, summary in (
        ("baseline", observation.baseline_summary),
        ("fault", observation.fault_summary),
    ):
        if summary.record_state == "no_record":
            reasons.append(f"{name}_no_record")
        elif summary.valid_point_count < observation.minimum_valid_points:
            reasons.append(f"{name}_insufficient_samples")
    if observation.anomaly_strength is None:
        reasons.append("measurement_missing")
    return tuple(reasons)


def _pending(
    *,
    plan: PredicatePlan,
    observation_ids: tuple[str, ...],
    provenance_ids: tuple[str, ...],
) -> PredicateResult:
    # These two reasons have explicit pending semantics in both schema and guard.
    return build_predicate_result(
        predicate_kind=plan.predicate_kind,
        requirement=plan.requirement,
        truth_value="unknown",
        logical_operator="atomic",
        operand_states=(),
        observation_ids=observation_ids,
        verification_result_ids=(),
        measured_values=(),
        threshold=None,
        threshold_source=None,
        reason="pending_verification" if observation_ids else "no_bound_observations",
        provenance_ids=provenance_ids,
    )


def evaluate_predicate_plan(
    *, plan: PredicatePlan, event_grouping: EventGroupingResult
) -> PredicateResult:
    """Return a replayable predicate to submit for independent guard verification.

    Both complete model trees are preflighted before any verification, equality,
    or serialization can execute callbacks. Schema validation keeps the proposal's
    existing 64-reference limits and does not add a cap on valid source strings.
    Unsupported predicate kinds are static errors; unsupported evidence stays
    referenced in pending leaves so subsequent guards can reject invalid sources.
    """

    preflight_exact_pydantic_tree(
        (plan, event_grouping),
        root_model_types=(PredicatePlan, EventGroupingResult),
        label="predicate plan inputs",
        max_string_length=None,
        provenance_id_limit=None,
    )
    if type(plan) is not PredicatePlan or type(event_grouping) is not EventGroupingResult:
        raise ValueError("predicate plan evaluation requires exact typed inputs")
    validated_plan = PredicatePlan.model_validate(plan.model_dump(mode="python", warnings="error"))
    if validated_plan != plan:
        raise ValueError("predicate plan does not match its validated state")
    grouping = _prepare_predicate_grouping(event_grouping)
    return _evaluate_predicate_plan_values(plan=plan, grouping=grouping)


def _prepare_predicate_grouping(event_grouping: EventGroupingResult) -> EventGroupingResult:
    """Replay once inside the owning caller's preflight and unchanged-state boundary."""
    if not event_grouping.verify():
        raise ValueError("predicate plan event grouping failed verification")
    return replay_event_grouping_result(event_grouping)


@dataclass(frozen=True, slots=True)
class _PredicateValueIndex:
    """Call-local lookup data, not an authentication or validation receipt."""

    grouping: EventGroupingResult
    observations: Mapping[str, Observation]
    primary_families: Mapping[str, str]


def _build_predicate_value_index(grouping: EventGroupingResult) -> _PredicateValueIndex:
    """Index the owner's working grouping within its unchanged-state boundary."""
    return _PredicateValueIndex(
        grouping=grouping,
        observations=MappingProxyType(
            {item.observation_id: item for item in grouping.observations}
        ),
        primary_families=MappingProxyType({
            observation_id: group.primary_signal_family
            for group in grouping.event_groups
            for observation_id in group.observation_ids
        }),
    )


def _evaluate_predicate_plan_values(
    *, plan: PredicatePlan, grouping: EventGroupingResult,
    value_index: _PredicateValueIndex | None = None,
) -> PredicateResult:
    """Pure value compilation inside an owner's freshly authenticated scope.

    This helper grants no source authority. Independent callers must use the
    public evaluator; B0 owns its complete replay and working-object checks.
    """
    if value_index is not None and (
        type(value_index) is not _PredicateValueIndex or value_index.grouping is not grouping
    ):
        raise ValueError("predicate value index belongs to a different grouping")
    contract = _PREDICATE_EVIDENCE_CONTRACTS.get(plan.predicate_kind)
    if contract is None:
        raise ValueError("predicate kind has no fixed evidence contract")

    provenance = (
        f"predicate-plan-policy:{PREDICATE_PLAN_EVALUATION_POLICY_VERSION}",
        f"event-grouping:{grouping.result_sha256}",
        *(f"predicate-plan-relation:{item}" for item in sorted(set(plan.relation_ids))),
    )
    observation_ids = tuple(sorted(set(plan.observation_ids)))
    if not observation_ids:
        return _pending(
            plan=plan,
            observation_ids=(),
            provenance_ids=(*provenance, "predicate-plan-pending:no_bound_observations"),
        )

    index = value_index if value_index is not None else _build_predicate_value_index(grouping)
    observations = index.observations
    primary_families = index.primary_families
    leaves: list[PredicateResult] = []
    for observation_id in observation_ids:
        observation = observations.get(observation_id)
        leaf_provenance = (*provenance, f"observation:{observation_id}")
        reasons: list[str] = []
        if observation is None:
            reasons.append("observation_missing")
        else:
            leaf_provenance = (
                *leaf_provenance,
                *observation.provenance_ids,
                f"source-fingerprint:{observation.source_fingerprint}",
            )
            if (
                primary_families.get(observation_id) not in contract.primary_signal_families
                and observation.signal_family not in contract.source_signal_families
            ):
                reasons.append("signal_family_denied")
            reasons.extend(_unavailable_reasons(observation))
        if contract.measured_name == "onset_delay_unix_ns":
            reasons.append("cross_service_onset_unavailable")
        if reasons:
            leaves.append(
                _pending(
                    plan=plan,
                    observation_ids=(observation_id,),
                    provenance_ids=(
                        *leaf_provenance,
                        *(
                            f"predicate-plan-pending:{observation_id}:{reason}"
                            for reason in reasons
                        ),
                    ),
                )
            )
            continue
        # All currently measurable original-observation contracts use strength.
        # Never substitute a model-supplied value or an onset interval width.
        if contract.measured_name != "anomaly_strength":
            raise ValueError("predicate kind has no original-observation measurement")
        assert observation is not None and observation.anomaly_strength is not None
        leaves.append(
            evaluate_predicate(
                predicate_kind=plan.predicate_kind,
                requirement=plan.requirement,
                measured_values=(
                    MeasuredValue(
                        name=contract.measured_name,
                        value=observation.anomaly_strength,
                        unit=contract.threshold_unit,
                    ),
                ),
                threshold=PredicateThreshold.model_validate(
                    {
                        "comparator": contract.comparator,
                        "value": contract.threshold_value,
                        "unit": contract.threshold_unit,
                        "source": "mechanism_constant",
                    }
                ),
                threshold_source=contract.threshold_source,
                observation_ids=(observation_id,),
                verification_result_ids=(),
                provenance_ids=leaf_provenance,
            )
        )
    if len(leaves) == 1:
        return leaves[0]
    compose = strong_kleene_and if contract.comparator == "le" else strong_kleene_or
    return compose(
        predicate_kind=plan.predicate_kind, requirement=plan.requirement, operands=leaves
    )
