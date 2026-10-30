"""Private predicate projection after authoritative query source replay.

Session closure is NOT authenticity: a caller must independently replay the full
query session against its frozen inputs before using this helper. The hypothesis
guard and role replay remain mandatory before any diagnostic or score effect.
This module reads existing observations only and never executes a query.
"""

from __future__ import annotations

import json
import math

from nurse_rcl.v2.diagnostic_schemas import (
    MeasuredValue,
    PredicateResult,
    PredicateThreshold,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult, replay_event_grouping_result
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.predicate_logic import (
    evaluate_predicate,
    strong_kleene_and,
    strong_kleene_or,
)
from nurse_rcl.v2.predicate_plan_evaluator import (
    _pending,
    _unavailable_reasons,
    evaluate_predicate_plan,
)
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_proposals import PredicatePlan

QUERY_PREDICATE_EVALUATION_POLICY_VERSION = "query-observation-predicate-plan/1"


def _evaluate_query_predicate_plan(
    *,
    plan: PredicatePlan,
    event_grouping: EventGroupingResult,
    query_session: CaseQuerySessionResult,
) -> PredicateResult:
    """Evaluate fixed contracts over originals and already replayed query outputs.

    Every input tree is preflighted before verifiers, equality, or serialization.
    Internal session closure cannot authenticate source measurements or authority;
    the caller's full source replay must already have checked exact source/field
    consistency, permissions, and bindings. Independent guard/role replay is still
    required. A completed result's unknown truth is never treated as hypothesis
    truth; only an explicitly requested numeric measurement can decide a leaf.
    """

    try:
        preflight_exact_pydantic_tree(
            (plan, event_grouping, query_session),
            root_model_types=(PredicatePlan, EventGroupingResult, CaseQuerySessionResult),
            label="query predicate inputs",
            max_string_length=None,
            provenance_id_limit=None,
        )
    except (AttributeError, TypeError, ValueError):
        # Missing Pydantic slots are malformed just like hidden/subclass state.
        raise ValueError("query predicate inputs failed exact-tree preflight") from None
    if (
        type(plan) is not PredicatePlan
        or type(event_grouping) is not EventGroupingResult
        or type(query_session) is not CaseQuerySessionResult
    ):
        raise ValueError("query predicate evaluation requires exact typed inputs")
    validated_plan = PredicatePlan.model_validate(plan.model_dump(mode="python", warnings="error"))
    if validated_plan != plan:
        raise ValueError("query predicate plan does not match its validated state")
    if not query_session.verify():
        raise ValueError("query predicate session failed internal closure verification")
    if not event_grouping.verify():
        raise ValueError("query predicate event grouping failed verification")
    grouping = replay_event_grouping_result(event_grouping)
    if query_session.opaque_case_id != grouping.opaque_case_id:
        raise ValueError("query predicate session and grouping cases differ")
    contract = _PREDICATE_EVIDENCE_CONTRACTS.get(plan.predicate_kind)
    if contract is None:
        raise ValueError("predicate kind has no fixed evidence contract")

    ids = tuple(sorted(set(plan.observation_ids)))
    if not any(identifier.startswith("vob_") for identifier in ids):
        return evaluate_predicate_plan(plan=plan, event_grouping=grouping)

    batch = query_session.verification_result
    if batch is not None and batch.scope.event_grouping_result_sha256 != grouping.result_sha256:
        raise ValueError("query predicate session references a different event grouping")
    derived_by_id = (
        {}
        if batch is None
        else {item.derived_observation_id: item for item in batch.derived_observations}
    )
    requests = (
        {} if batch is None else {item.verification_request_id: item for item in batch.requests}
    )
    result_by_derived_id = (
        {}
        if batch is None
        else {
            identifier: item
            for item in batch.results
            for identifier in item.derived_observation_ids
        }
    )
    originals = {item.observation_id: item for item in grouping.observations}
    primary_families = {
        identifier: group.primary_signal_family
        for group in grouping.event_groups
        for identifier in group.observation_ids
    }
    provenance = (
        f"predicate-plan-policy:{QUERY_PREDICATE_EVALUATION_POLICY_VERSION}",
        f"event-grouping:{grouping.result_sha256}",
        f"query-session:{query_session.session_sha256}",
        *(f"predicate-plan-relation:{item}" for item in sorted(set(plan.relation_ids))),
    )
    leaves: list[PredicateResult] = []
    for identifier in ids:
        if not identifier.startswith("vob_"):
            leaves.append(
                evaluate_predicate_plan(
                    plan=plan.model_copy(update={"observation_ids": (identifier,)}),
                    event_grouping=grouping,
                )
            )
            continue
        observation = derived_by_id.get(identifier)
        leaf_provenance = (*provenance, f"derived-observation:{identifier}")
        reasons: list[str] = []
        measured: float | int | None = None
        result = result_by_derived_id.get(identifier)
        if observation is None:
            reasons.append("derived_observation_missing")
        else:
            request = requests.get(observation.verification_request_id)
            if (
                result is None
                or result.status != "completed"
                or request is None
                or result.verification_request_id != observation.verification_request_id
                or observation.opaque_case_id != request.opaque_case_id
                or observation.candidate_service_key != request.candidate_service_key
                or observation.operator != request.operator
                or observation.query_interval != request.query_interval
                or observation.decision_time_unix_ns != request.decision_time_unix_ns
                or observation.source_observation_ids != request.observation_ids
                or observation.source_relation_ids != request.relation_ids
            ):
                raise ValueError(
                    "derived observation is not linked to its actual completed request"
                )
            leaf_provenance = (
                *leaf_provenance,
                *observation.provenance_ids,
                f"verification-request:{request.verification_request_id}",
                f"verification-result:{result.verification_result_id}",
            )
            if len(observation.source_observation_ids) != 1:
                reasons.append("single_source_observation_required")
            for source_id in observation.source_observation_ids:
                source = originals.get(source_id)
                leaf_provenance = (*leaf_provenance, f"observation:{source_id}")
                if source is None:
                    reasons.append(f"{source_id}:source_observation_missing")
                    continue
                leaf_provenance = (
                    *leaf_provenance,
                    *source.provenance_ids,
                    f"source-fingerprint:{source.source_fingerprint}",
                )
                if (
                    primary_families.get(source_id) not in contract.primary_signal_families
                    and source.signal_family not in contract.source_signal_families
                ):
                    reasons.append(f"{source_id}:signal_family_denied")
                reasons.extend(f"{source_id}:{reason}" for reason in _unavailable_reasons(source))
            if contract.measured_name == "onset_delay_unix_ns":
                # This operator reports uncertainty width, never cross-service lag.
                reasons.append("cross_service_onset_unavailable")
            elif observation.operator != "compare_window_summary":
                reasons.append("measurement_operator_denied")
            elif contract.measured_name != "anomaly_strength":
                reasons.append("measurement_contract_unavailable")
            elif "anomaly_strength" not in request.requested_fields:
                reasons.append("measurement_not_requested")
            else:
                field = next(
                    (item for item in observation.values if item.field_name == "anomaly_strength"),
                    None,
                )
                value = None if field is None else json.loads(field.value_json)
                if value is None or type(value) not in (int, float) or not math.isfinite(value):
                    reasons.append("measurement_missing")
                else:
                    measured = value
        if reasons:
            # Pending schemas prohibit verification_result_ids; the actual result
            # ID is retained above in provenance without fabricating evaluated state.
            leaves.append(
                _pending(
                    plan=plan,
                    observation_ids=(identifier,),
                    provenance_ids=(
                        *leaf_provenance,
                        *(f"predicate-plan-pending:{identifier}:{reason}" for reason in reasons),
                    ),
                )
            )
            continue
        assert measured is not None and result is not None
        leaves.append(
            evaluate_predicate(
                predicate_kind=plan.predicate_kind,
                requirement=plan.requirement,
                measured_values=(
                    MeasuredValue(
                        name=contract.measured_name, value=measured, unit=contract.threshold_unit
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
                observation_ids=(identifier,),
                verification_result_ids=(result.verification_result_id,),
                provenance_ids=leaf_provenance,
            )
        )
    if len(leaves) == 1:
        return leaves[0]
    compose = strong_kleene_and if contract.comparator == "le" else strong_kleene_or
    return compose(
        predicate_kind=plan.predicate_kind, requirement=plan.requirement, operands=leaves
    )
