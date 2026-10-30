"""Execute an initial semantic query plan through the existing case coordinator.

Receipts are private audit records, never self-authenticating evidence. Consumers
must independently pin and replay the current authoritative session against frozen
original sources with require_case_query_session_replay and require_current.
This bridge neither calls a model nor grants a second ranking request: revision
dependencies are only one gate of the separate follow-up budget decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import VerificationResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.semantic_context import DetailedEvidenceCard, SemanticContextBundle
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_query_plan import SemanticQueryPlan, plan_semantic_queries
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import VerificationExecutionPolicy


@dataclass(frozen=True, slots=True)
class SemanticQueryResultBinding:
    """Actual outcome at an original position, retaining private string ID/kind.

    A malformed non-string ID/kind is recoverable from plan.inspection raw bytes
    by index. Inspection refusals have no fabricated executor or result IDs.
    """

    index: int
    local_id: str | None = field(repr=False)
    kind: str | None = field(repr=False)
    normalized_request_id: str | None
    decision_id: str | None
    result_id: str | None
    status: Literal["denied", "completed", "reused", "refused", "rejected"]
    reason_code: str
    result: DetailedEvidenceCard | VerificationResult | None = field(repr=False)
    derived_observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticQueryExecution:
    """Private receipt; successful aliases do not themselves authorize revision."""

    plan: SemanticQueryPlan = field(repr=False)
    session: CaseQuerySessionResult | None = field(repr=False)
    bindings: tuple[SemanticQueryResultBinding, ...]
    successful_query_ids: tuple[str, ...] = field(repr=False)
    revision_dependency_proposal_ids: tuple[str, ...] = field(repr=False)


def _bind_results(
    plan: SemanticQueryPlan,
    session: CaseQuerySessionResult | None,
) -> tuple[tuple[SemanticQueryResultBinding, ...], tuple[str, ...], set[str]]:
    card_batch = None if session is None else session.card_expansion_result
    verification_batch = None if session is None else session.verification_result
    cards = (
        {} if card_batch is None else {c.detailed_card_sha256: c for c in card_batch.returned_cards}
    )
    card_decisions = (
        {}
        if card_batch is None
        else {d.request.expansion_request_id: d for d in card_batch.decisions}
    )
    verification_decisions = (
        {}
        if verification_batch is None
        else {d.verification_request_id: d for d in verification_batch.decisions}
    )
    verification_results = (
        {}
        if verification_batch is None
        else {r.verification_result_id: r for r in verification_batch.results}
    )
    derived_ids = (
        set()
        if verification_batch is None
        else {o.derived_observation_id for o in verification_batch.derived_observations}
    )
    bindings: list[SemanticQueryResultBinding] = []
    successful: list[str] = []
    new_evidence: set[str] = set()
    for binding in plan.bindings:
        status: Literal["denied", "completed", "reused", "refused", "rejected"] = "denied"
        reason = binding.reason_code
        decision_id = result_id = None
        result: DetailedEvidenceCard | VerificationResult | None = None
        outputs: tuple[str, ...] = ()
        request_id = binding.normalized_request_id
        usable = novel = False
        if request_id is not None and binding.kind == "card":
            decision = card_decisions[request_id]
            decision_id = decision.expansion_decision_id
            result_id = decision.detailed_card_sha256
            status, reason = decision.status, decision.reason_code
            result = None if result_id is None else cards[result_id]
            usable = result is not None and status in ("completed", "reused")
            # Initial Q reuse and metadata-only cards add no diagnostic observations.
            # Unknown or degraded observations still count; no score/quality filter.
            novel = status == "completed" and result is not None and bool(result.observations)
        elif request_id is not None and binding.kind == "verification":
            verification_decision = verification_decisions[request_id]
            decision_id = verification_decision.execution_decision_id
            result_id = verification_decision.verification_result_id
            result = verification_results[result_id]
            status, reason = result.status, verification_decision.reason_code
            if status == "completed":
                outputs = tuple(i for i in result.derived_observation_ids if i in derived_ids)
            usable = novel = bool(outputs)
        if binding.local_id is not None:
            if usable:
                successful.append(binding.local_id)
            if novel:
                new_evidence.add(binding.local_id)
        bindings.append(
            SemanticQueryResultBinding(
                index=binding.index,
                local_id=binding.local_id,
                kind=binding.kind,
                normalized_request_id=request_id,
                decision_id=decision_id,
                result_id=result_id,
                status=status,
                reason_code=reason,
                result=result,
                derived_observation_ids=outputs,
            )
        )
    return tuple(bindings), tuple(successful), new_evidence


def execute_semantic_queries(
    *,
    query_session_coordinator: CaseQuerySessionCoordinator,
    raw_response: bytes,
    prompt: SemanticPrompt,
    expected_prompt_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    semantic_context: SemanticContextBundle,
    card_expansion_policy: CardExpansionPolicy,
    verification_policy: VerificationExecutionPolicy,
    scope_interval: TimeInterval,
) -> SemanticQueryExecution:
    """Rebuild from raw sources before advancing the supplied authoritative head.

    Card requests always precede verification requests. Multiplicity survives
    submission, while phase ordering, deduplication, budget and prior ledger
    replay belong to the existing coordinator. No requests means no session call.
    """

    plan = plan_semantic_queries(
        raw_response=raw_response,
        prompt=prompt,
        expected_prompt_sha256=expected_prompt_sha256,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
        verification_policy=verification_policy,
        scope_interval=scope_interval,
    )
    if type(query_session_coordinator) is not CaseQuerySessionCoordinator:
        raise ValueError("semantic query execution requires the authoritative coordinator")
    session = None
    if plan.card_requests:
        session = query_session_coordinator.execute_card_expansion(
            requests=plan.card_requests,
            policy=card_expansion_policy,
            semantic_context=semantic_context,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
        )
    if plan.verification_requests:
        session = query_session_coordinator.execute_verification(
            requests=plan.verification_requests,
            policy=verification_policy,
            scope=plan.verification_scope,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
        )
    if session is not None:
        session = query_session_coordinator.require_current(session)
    bindings, successful, new_evidence = _bind_results(plan, session)
    dependencies = tuple(
        hypothesis.proposal_id
        for hypothesis in plan.inspection.eligible_hypotheses
        if new_evidence.intersection(hypothesis.query_ids)
    )
    return SemanticQueryExecution(plan, session, bindings, successful, dependencies)
