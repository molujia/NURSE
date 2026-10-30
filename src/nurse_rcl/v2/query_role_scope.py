"""Same-source role witnesses after authoritative query and guard replay.

This private-in-pipeline filter is not a source-authenticity or score gate. The
caller must replay the frozen sources and current session; the final competition
role compiler independently replays accepted guards before using any proposal.
"""

from __future__ import annotations

from collections.abc import Sequence

from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis, PredicateResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, local_event_relation_id
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

QUERY_ROLE_SCOPE_POLICY_VERSION = "true-query-source-event-role-witness/1"


def _overlaps(left: TimeInterval, right: TimeInterval) -> bool:
    return left.start_unix_ns < right.end_unix_ns and right.start_unix_ns < left.end_unix_ns


def has_true_query_role_witness(
    *,
    hypothesis: DiagnosticHypothesis,
    proposal: RoleBindingProposal,
    predicates: Sequence[PredicateResult],
    event_grouping: EventGroupingResult,
    entity_registry: TemporalEntityRegistry,
    query_session: CaseQuerySessionResult,
) -> bool:
    """Find one true, cited source satisfying every role binding simultaneously."""
    try:
        preflight_exact_pydantic_tree(
            (hypothesis, proposal, predicates, event_grouping, entity_registry, query_session),
            root_model_types=(
                DiagnosticHypothesis,
                RoleBindingProposal,
                PredicateResult,
                EventGroupingResult,
                TemporalEntityRegistry,
                CaseQuerySessionResult,
            ),
            label="query role scope",
            max_string_length=None,
            provenance_id_limit=None,
        )
    except AttributeError:
        raise ValueError("invalid_query_role_model_state") from None
    needed = set(hypothesis.necessary_predicate_result_ids)
    roots = {p.predicate_result_id: p for p in predicates if p.predicate_result_id in needed}
    if (
        not needed
        or set(roots) != needed
        or any(not p.verify() or p.truth_value != "true" for p in roots.values())
    ):
        return False
    group = next(
        (g for g in event_grouping.event_groups if g.event_group_id == proposal.event_group_id),
        None,
    )
    if group is None:
        return False
    allowed_sources = set(group.observation_ids)
    allowed_sources.intersection_update(
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
        allowed_sources.intersection_update(relation.observation_ids)
    observations = {o.observation_id: o for o in event_grouping.observations}
    batch = query_session.verification_result
    derived = (
        {} if batch is None else {d.derived_observation_id: d for d in batch.derived_observations}
    )
    results = {} if batch is None else {r.verification_result_id: r for r in batch.results}
    pending = list(roots.values())
    while pending:
        leaf = pending.pop()
        if leaf.truth_value != "true":
            continue
        if leaf.logical_operator != "atomic":
            pending.extend(leaf.operand_states)
            continue
        for identifier in leaf.observation_ids:
            if identifier not in hypothesis.observation_ids:
                continue
            sources: tuple[str, ...] = (identifier,)
            if identifier in derived:
                entry = derived[identifier]
                if (
                    entry.opaque_case_id != hypothesis.opaque_case_id
                    or entry.candidate_service_key != hypothesis.candidate_service_key
                    or entry.verification_request_id not in hypothesis.verification_request_ids
                    or not _overlaps(entry.query_interval, proposal.interval)
                    or not any(
                        r.status == "completed"
                        and r.verification_request_id == entry.verification_request_id
                        and identifier in r.derived_observation_ids
                        for result_id in leaf.verification_result_ids
                        if result_id in results
                        for r in (results[result_id],)
                    )
                ):
                    continue
                sources = entry.source_observation_ids
            for source in sources:
                observation = observations.get(source)
                if (
                    source in allowed_sources
                    and observation is not None
                    and observation.opaque_case_id == hypothesis.opaque_case_id
                    and _overlaps(observation.interval, proposal.interval)
                ):
                    return True
    return False
