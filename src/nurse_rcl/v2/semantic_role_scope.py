"""True evidence witnesses for one semantic event/relation/interval role.

This narrows a proposal before the existing role compiler. It does not replace
accepted-guard replay, grant a score, or change the legacy compiler's behavior.
False/unknown branches remain auditable but cannot lend an event their support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis, PredicateResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, local_event_relation_id
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

SEMANTIC_ROLE_SCOPE_POLICY_VERSION: Literal["true-original-event-role-witness/1"] = (
    "true-original-event-role-witness/1"
)


def has_true_role_witness(
    *,
    hypothesis: DiagnosticHypothesis,
    proposal: RoleBindingProposal,
    predicates: Sequence[PredicateResult],
    event_grouping: EventGroupingResult,
    entity_registry: TemporalEntityRegistry,
) -> bool:
    """Require one SAME true source in all role bindings; original observations only.

    The caller must separately replay the accepted guard and frozen inputs. The
    final role compiler still enforces permitted roles and all refutation rules.
    """
    preflight_exact_pydantic_tree(
        (hypothesis, proposal, predicates, event_grouping, entity_registry),
        root_model_types=(
            DiagnosticHypothesis,
            RoleBindingProposal,
            PredicateResult,
            EventGroupingResult,
            TemporalEntityRegistry,
        ),
        label="semantic role scope",
        max_string_length=None,
        provenance_id_limit=None,
    )
    needed = set(hypothesis.necessary_predicate_result_ids)
    roots = {p.predicate_result_id: p for p in predicates if p.predicate_result_id in needed}
    if set(roots) != needed or any(
        not p.verify() or p.truth_value != "true" for p in roots.values()
    ):
        return False
    witnesses: set[str] = set()
    pending = list(roots.values())
    while pending:
        predicate = pending.pop()
        if predicate.truth_value != "true":
            continue
        if predicate.logical_operator == "atomic":
            witnesses.update(predicate.observation_ids)
        else:
            pending.extend(predicate.operand_states)
    group = next(
        (g for g in event_grouping.event_groups if g.event_group_id == proposal.event_group_id),
        None,
    )
    if group is None:
        return False
    witnesses.intersection_update(group.observation_ids)
    witnesses.intersection_update(
        r.observation.observation_id
        for r in entity_registry.resolutions
        if r.binding.status == "bound"
        and r.binding.service_entity_key == hypothesis.candidate_service_key
    )
    local_relation = local_event_relation_id(
        candidate_service_key=hypothesis.candidate_service_key,
        event_group_id=proposal.event_group_id,
    )
    if proposal.relation_id != local_relation:
        relation = next(
            (r for r in event_grouping.association_claims if r.claim_id == proposal.relation_id),
            None,
        )
        if relation is None:
            return False
        witnesses.intersection_update(relation.observation_ids)
    return any(
        o.observation_id in witnesses
        and o.interval.start_unix_ns < proposal.interval.end_unix_ns
        and proposal.interval.start_unix_ns < o.interval.end_unix_ns
        for o in event_grouping.observations
    )
