"""Independently replay a semantic response against the current query ledger.

The private execution receipt is reconstructed from original inputs. Caller
receipts, inspections and normalized plans never serve as evidence authority.
Replay audits existing results without submitting or completing new queries.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import (
    CaseQuerySessionCoordinator,
    CaseQuerySessionResult,
    require_case_query_session_replay,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_query_bridge import SemanticQueryExecution, _bind_results
from nurse_rcl.v2.semantic_query_plan import SemanticQueryPlan, plan_semantic_queries
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import VerificationExecutionPolicy


def _require_plan_requests(plan: SemanticQueryPlan, session: CaseQuerySessionResult) -> None:
    """Match complete requests and retain every submitted semantic alias."""

    if plan.card_requests:
        card = session.card_expansion_result
        if card is None:
            raise ValueError("semantic card requests are missing from the session")
        card_requests = {
            decision.request.expansion_request_id: decision.request for decision in card.decisions
        }
        if any(
            card_requests.get(request.expansion_request_id) != request
            for request in plan.card_requests
        ):
            raise ValueError("semantic card request does not match the session")
        card_counts = {
            item.expansion_request_id: item.submission_count for item in card.request_counts
        }
        if any(
            card_counts.get(identifier, 0) < count
            for identifier, count in Counter(
                request.expansion_request_id for request in plan.card_requests
            ).items()
        ):
            raise ValueError("semantic card request submission count is insufficient")
    if plan.verification_requests:
        verification = session.verification_result
        if verification is None:
            raise ValueError("semantic verification requests are missing from the session")
        verification_requests = {
            request.verification_request_id: request for request in verification.requests
        }
        if any(
            verification_requests.get(request.verification_request_id) != request
            for request in plan.verification_requests
        ):
            raise ValueError("semantic verification request does not match the session")
        verification_counts = {
            item.verification_request_id: item.submission_count
            for item in verification.request_counts
        }
        if any(
            verification_counts.get(identifier, 0) < count
            for identifier, count in Counter(
                request.verification_request_id for request in plan.verification_requests
            ).items()
        ):
            raise ValueError("semantic verification request submission count is insufficient")


def replay_semantic_queries(
    *,
    query_session_coordinator: CaseQuerySessionCoordinator,
    query_session: CaseQuerySessionResult,
    expected_session_sha256: str,
    expected_response_sha256: str,
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
    """Reinspect exact raw bytes and bind only source-replayed current requests.

    Complete callback-free preflight precedes hashing, comparisons, verification
    and coordinator access. The exact coordinator is trusted in-process state;
    its head is only read. The three digests must be independently supplied pins.
    An empty current plan may audit an existing historical session, but a missing
    session never constitutes evidence. Denied positions retain their decisions.
    """

    if (
        type(query_session_coordinator) is not CaseQuerySessionCoordinator
        or type(raw_response) is not bytes
        or type(prompt) is not SemanticPrompt
    ):
        raise ValueError("invalid_semantic_query_replay_input")
    try:
        payload = object.__getattribute__(prompt, "payload_json")
    except AttributeError:
        raise ValueError("invalid_semantic_prompt") from None
    if type(payload) is not bytes:
        raise ValueError("invalid_semantic_prompt")
    sources = (
        candidate_universe,
        frozen_candidates,
        entity_registry,
        event_grouping,
        semantic_context,
        card_expansion_policy,
        verification_policy,
        scope_interval,
        query_session,
    )
    source_types = (
        CandidateUniverse,
        FrozenCandidateUniverse,
        TemporalEntityRegistry,
        EventGroupingResult,
        SemanticContextBundle,
        CardExpansionPolicy,
        VerificationExecutionPolicy,
        TimeInterval,
        CaseQuerySessionResult,
    )
    if any(
        type(value) is not expected for value, expected in zip(sources, source_types, strict=True)
    ):
        raise ValueError("invalid_semantic_query_replay_source_type")
    pins = (expected_prompt_sha256, expected_response_sha256, expected_session_sha256)
    try:
        preflight_exact_pydantic_tree(
            (pins, sources),
            root_model_types=source_types,
            label="semantic query replay inputs",
            max_string_length=None,
            provenance_id_limit=None,
        )
    except AttributeError:
        raise ValueError("invalid_semantic_query_replay_model_state") from None
    if any(
        type(pin) is not str
        or len(pin) != 64
        or any(character not in "0123456789abcdef" for character in pin)
        for pin in pins
    ):
        raise ValueError("invalid_semantic_query_replay_pin")
    if hashlib.sha256(raw_response).hexdigest() != expected_response_sha256:
        raise ValueError("semantic_response_mismatch")
    if query_session.session_sha256 != expected_session_sha256:
        raise ValueError("semantic_query_session_mismatch")
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
    if (
        query_session.opaque_case_id != candidate_universe.opaque_case_id
        or query_session.candidate_freeze_sha256 != frozen_candidates.freeze_sha256
    ):
        raise ValueError("semantic query session does not match source case and freeze")
    verification = query_session.verification_result
    if verification is not None:
        if verification.policy != verification_policy:
            raise ValueError("semantic verification policy does not match independent policy")
        if verification.scope != plan.verification_scope:
            raise ValueError("semantic verification scope does not match independent scope")
    session = query_session_coordinator.require_current(query_session)
    session = require_case_query_session_replay(
        session=session,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    )
    _require_plan_requests(plan, session)
    bindings, successful, new_evidence = _bind_results(plan, session)
    dependencies = tuple(
        hypothesis.proposal_id
        for hypothesis in plan.inspection.eligible_hypotheses
        if new_evidence.intersection(hypothesis.query_ids)
    )
    return SemanticQueryExecution(plan, session, bindings, successful, dependencies)
