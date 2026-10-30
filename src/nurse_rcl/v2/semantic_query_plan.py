"""Pure initial-response query planning from independently supplied frozen sources.

These private in-process records are audit material, not persisted schemas or
outbound model payloads. Inspection eligibility never asserts executor success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import (
    CARD_EXPANSION_QUERY_KIND,
    CardExpansionPolicy,
    CardExpansionRequest,
    _validate_execution_inputs,
    build_card_expansion_request,
)
from nurse_rcl.v2.diagnostic_schemas import (
    VerificationOperator,
    VerificationRequest,
    build_verification_request,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.mechanism_library import build_mechanism_library
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import SemanticContextBundle, build_detailed_evidence_card
from nurse_rcl.v2.semantic_permissions import SemanticInspection, inspect_semantic_response
from nurse_rcl.v2.semantic_prompt import SEMANTIC_PROMPT_VERSION, SemanticPrompt
from nurse_rcl.v2.semantic_proposals import SemanticCardQuery
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import (
    VerificationExecutionPolicy,
    VerificationExecutionScope,
    build_verification_execution_scope,
)


@dataclass(frozen=True, slots=True)
class SemanticQueryBinding:
    """One positional decision, including malformed or duplicate local IDs.

    String IDs and kinds retain their submitted text privately, even when invalid.
    Non-string fields are represented by None; the exact original item remains
    recoverable from inspection.raw_response and index. Only an eligible decision
    with a normalized request ID can be submitted for execution.
    """

    index: int
    local_id: str | None = field(repr=False)
    kind: str | None = field(repr=False)
    disposition: Literal["eligible", "denied"]
    reason_code: str
    normalized_request_id: str | None


@dataclass(frozen=True, slots=True)
class SemanticQueryPlan:
    """Detached requests retain aliases; only the coordinator deduplicates them."""

    inspection: SemanticInspection = field(repr=False)
    bindings: tuple[SemanticQueryBinding, ...]
    card_requests: tuple[CardExpansionRequest, ...] = field(repr=False)
    verification_requests: tuple[VerificationRequest, ...] = field(repr=False)
    verification_scope: VerificationExecutionScope = field(repr=False)


def _require_source_projection(
    *,
    prompt: SemanticPrompt,
    expected_prompt_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    semantic_context: SemanticContextBundle,
    verification_policy: VerificationExecutionPolicy,
    scope_interval: TimeInterval,
) -> None:
    """Require an exact initial prompt and replay every initial detailed card."""

    if prompt.sha256 != expected_prompt_sha256:
        raise ValueError("semantic_prompt_mismatch")
    data = decode_json_object(prompt.payload_json, ExchangeLimits())
    identity_pin = data.get("identity_manifest_sha256")
    if (
        type(identity_pin) is not str
        or len(identity_pin) != 64
        or any(c not in "0123456789abcdef" for c in identity_pin)
    ):
        raise ValueError("semantic_prompt_source_mismatch")
    expected = {
        "prompt_version": SEMANTIC_PROMPT_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "source_context_sha256": semantic_context.bundle_sha256,
        # Identity is pinned by the independent pre-call prompt hash. No response
        # field is used as an authority or as a request/source hash.
        "identity_manifest_sha256": identity_pin,
        "decision_time_unix_ns": entity_registry.decision_time_unix_ns,
        "scope_interval": scope_interval.model_dump(mode="json"),
        "candidate_directory": semantic_context.candidate_directory.model_dump(mode="json"),
        "detailed_evidence_cards": [
            card.model_dump(mode="json") for card in semantic_context.detailed_evidence_cards
        ],
        "mechanism_library": build_mechanism_library().model_dump(mode="json"),
        "query_policy": verification_policy.model_dump(mode="json"),
        "visible_relation_ids": [],
    }
    # Canonical byte comparison also distinguishes JSON booleans from integers.
    if encode_json(data, ExchangeLimits()) != encode_json(expected, ExchangeLimits()):
        raise ValueError("semantic_prompt_source_mismatch")
    entries = semantic_context.candidate_directory.entries
    if (
        tuple(entry.service_entity_key for entry in entries)
        != frozen_candidates.candidate_service_keys
    ):
        raise ValueError("semantic context directory does not match source candidates")
    candidates = {item.service_entity_key: item for item in candidate_universe.candidates}
    if any(
        entry.evidence_state != candidates[entry.service_entity_key].evidence_state
        for entry in entries
    ):
        raise ValueError("semantic context evidence state does not match sources")
    for card in semantic_context.detailed_evidence_cards:
        rebuilt = build_detailed_evidence_card(
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            service_entity_key=card.card.service_entity_key,
            access_reason="focus_q",
        )
        if rebuilt != card or any(
            obs.interval.start_unix_ns < scope_interval.start_unix_ns
            or obs.interval.end_unix_ns > scope_interval.end_unix_ns
            for obs in card.observations
        ):
            raise ValueError("semantic context card does not match original sources")


def plan_semantic_queries(
    *,
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
) -> SemanticQueryPlan:
    """Reinspect raw bytes and normalize allowed queries without executing any.

    The full plain input tree is checked before any source comparison, verifier,
    serializer or prompt hashing. Exact bytes and the slotted prompt are checked
    separately because the callback-free Pydantic preflight excludes dataclasses
    and bytes. A malformed response can still contain independently eligible
    items; its original decisions and item positions are retained.
    """

    if type(raw_response) is not bytes or type(prompt) is not SemanticPrompt:
        raise ValueError("invalid_semantic_query_input")
    if type(object.__getattribute__(prompt, "payload_json")) is not bytes:
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
    )
    if any(
        type(value) is not expected for value, expected in zip(sources, source_types, strict=True)
    ):
        raise ValueError("invalid_semantic_query_source_type")
    preflight_exact_pydantic_tree(
        (expected_prompt_sha256, sources),
        root_model_types=source_types,
        label="semantic query sources",
        max_string_length=None,
        provenance_id_limit=None,
    )
    if type(expected_prompt_sha256) is not str:
        raise ValueError("invalid_semantic_prompt_pin")
    interval = TimeInterval.model_validate(
        scope_interval.model_dump(mode="python", warnings="error")
    )
    if interval != scope_interval:
        raise ValueError("invalid_semantic_scope")
    _validate_execution_inputs(
        policy=card_expansion_policy,
        semantic_context=semantic_context,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    if card_expansion_policy.query_limit > 8:
        raise ValueError("card policy exceeds the shared case query limit")
    scope = build_verification_execution_scope(
        policy=verification_policy,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        observation_ids=tuple(obs.observation_id for obs in event_grouping.observations),
        relation_ids=tuple(claim.claim_id for claim in event_grouping.association_claims),
        operator_permissions=verification_policy.operator_permissions,
        query_scope_interval=scope_interval,
    )
    _require_source_projection(
        prompt=prompt,
        expected_prompt_sha256=expected_prompt_sha256,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        verification_policy=verification_policy,
        scope_interval=scope_interval,
    )
    inspection = inspect_semantic_response(
        raw_response,
        prompt=prompt,
        expected_prompt_sha256=expected_prompt_sha256,
    )
    raw_queries = (
        decode_json_object(raw_response, ExchangeLimits())["queries"]
        if inspection.envelope_error is None
        else []
    )
    eligible = iter(inspection.eligible_queries)
    bindings: list[SemanticQueryBinding] = []
    cards: list[CardExpansionRequest] = []
    verifications: list[VerificationRequest] = []
    for decision in inspection.decisions:
        if decision.section != "queries":
            continue
        entry = raw_queries[decision.index]
        raw_id = entry.get("query_id") if type(entry) is dict else None
        raw_kind = entry.get("kind") if type(entry) is dict else None
        local_id = raw_id if type(raw_id) is str else None
        kind = raw_kind if type(raw_kind) is str else None
        request_id = None
        if decision.disposition == "eligible":
            proposal = next(eligible)
            if isinstance(proposal, SemanticCardQuery):
                card_request = build_card_expansion_request(
                    opaque_case_id=candidate_universe.opaque_case_id,
                    candidate_freeze_sha256=frozen_candidates.freeze_sha256,
                    semantic_context_bundle_sha256=semantic_context.bundle_sha256,
                    service_entity_key=proposal.candidate_service_key,
                    query_kind=CARD_EXPANSION_QUERY_KIND,
                    decision_time_unix_ns=entity_registry.decision_time_unix_ns,
                )
                cards.append(card_request)
                request_id = card_request.expansion_request_id
            else:
                verification_request = build_verification_request(
                    opaque_case_id=candidate_universe.opaque_case_id,
                    candidate_freeze_sha256=frozen_candidates.freeze_sha256,
                    verification_scope_sha256=scope.scope_sha256,
                    candidate_service_key=proposal.candidate_service_key,
                    operator=cast(VerificationOperator, proposal.operator),
                    signal_family=proposal.signal_family,
                    query_interval=TimeInterval.model_validate(
                        proposal.query_interval.model_dump()
                    ),
                    decision_time_unix_ns=entity_registry.decision_time_unix_ns,
                    observation_ids=proposal.observation_ids,
                    relation_ids=proposal.relation_ids,
                    requested_fields=proposal.requested_fields,
                )
                verifications.append(verification_request)
                request_id = verification_request.verification_request_id
        bindings.append(
            SemanticQueryBinding(
                decision.index,
                local_id,
                kind,
                decision.disposition,
                decision.reason_code,
                request_id,
            )
        )
    return SemanticQueryPlan(inspection, tuple(bindings), tuple(cards), tuple(verifications), scope)
