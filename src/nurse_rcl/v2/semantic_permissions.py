"""Offline proposal inspection; eligibility is not deterministic acceptance."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_proposals import (
    ProposalInterval,
    ResponseEnvelope,
    SemanticCardQuery,
    SemanticHypothesisProposal,
    SemanticQuery,
    SemanticVerificationQuery,
)


@dataclass(frozen=True, slots=True)
class ProposalDecision:
    section: Literal["hypotheses", "queries"]
    index: int
    local_id: str | None
    disposition: Literal["eligible", "denied"]
    reason_code: str


@dataclass(frozen=True, slots=True)
class SemanticInspection:
    structurally_valid: bool
    envelope_error: str | None
    raw_response: bytes | None = field(repr=False)
    response_sha256: str | None
    received_bytes: int | None
    prompt_sha256: str
    decisions: tuple[ProposalDecision, ...]
    eligible_hypotheses: tuple[SemanticHypothesisProposal, ...] = field(repr=False)
    eligible_queries: tuple[SemanticCardQuery | SemanticVerificationQuery, ...] = field(repr=False)


def _inside(interval: ProposalInterval, scope: dict[str, Any]) -> bool:
    return bool(
        scope["start_unix_ns"]
        <= interval.start_unix_ns
        < interval.end_unix_ns
        <= scope["end_unix_ns"]
    )


def _hypothesis_permission(
    item: SemanticHypothesisProposal,
    data: dict[str, Any],
    observation_ids: set[str],
    event_ids: set[str],
    hypothesis_ids: set[str],
    query_ids: set[str],
) -> str | None:
    if not set(item.observation_ids) <= observation_ids:
        return "observation_denied"
    if not set(item.event_group_ids) <= event_ids:
        return "event_denied"
    if not set(item.relation_ids) <= set(data["visible_relation_ids"]):
        return "relation_denied"
    if any(not _inside(interval, data["scope_interval"]) for interval in item.time_intervals):
        return "time_denied"
    definitions = {
        entry["mechanism_key"]: entry for entry in data["mechanism_library"]["definitions"]
    }
    mechanism = definitions.get(item.mechanism_key)
    if mechanism is None:
        return "mechanism_denied"
    required = {
        "necessary": "necessary_predicate_kinds",
        "optional": "optional_predicate_kinds",
        "refuting": "refuting_predicate_kinds",
    }
    if set(mechanism["necessary_predicate_kinds"]) != {
        plan.predicate_kind for plan in item.predicate_plans if plan.requirement == "necessary"
    } or any(
        plan.predicate_kind not in mechanism[required[plan.requirement]]
        for plan in item.predicate_plans
    ):
        return "mechanism_predicates_denied"
    for plan in item.predicate_plans:
        if not set(plan.observation_ids) <= set(item.observation_ids):
            return "observation_denied"
        if not set(plan.relation_ids) <= set(item.relation_ids):
            return "relation_denied"
    for role in item.role_proposals:
        if (
            role.role not in mechanism["allowed_roles"]
            or role.event_group_id not in item.event_group_ids
        ):
            return "role_denied"
        if not set(role.relation_ids) <= set(item.relation_ids):
            return "relation_denied"
        if not any(
            _inside(role.interval, interval.model_dump()) for interval in item.time_intervals
        ):
            return "time_denied"
    if (
        not set(item.alternative_proposal_ids) <= hypothesis_ids
        or item.proposal_id in item.alternative_proposal_ids
    ):
        return "proposal_reference_denied"
    if not set(item.query_ids) <= query_ids:
        return "query_reference_denied"
    lists = (
        item.observation_ids,
        item.event_group_ids,
        item.relation_ids,
        item.alternative_proposal_ids,
        item.query_ids,
        tuple((plan.requirement, plan.predicate_kind) for plan in item.predicate_plans),
        *(plan.observation_ids for plan in item.predicate_plans),
        *(plan.relation_ids for plan in item.predicate_plans),
    )
    if any(len(values) != len(set(values)) for values in lists):
        return "duplicate_reference"
    return None


def _query_permission(
    item: SemanticCardQuery | SemanticVerificationQuery,
    data: dict[str, Any],
    observations: dict[str, dict[str, Any]],
) -> str | None:
    if isinstance(item, SemanticCardQuery):
        return None
    operators = {
        entry["operator"]: set(entry["requested_fields"])
        for entry in data["query_policy"]["operator_permissions"]
    }
    if item.operator not in operators:
        return "operator_denied"
    if not set(item.requested_fields) <= operators[item.operator]:
        return "field_denied"
    if not _inside(item.query_interval, data["scope_interval"]):
        return "time_denied"
    if not set(item.observation_ids) <= set(observations):
        return "observation_denied"
    if not set(item.relation_ids) <= set(data["visible_relation_ids"]):
        return "relation_denied"
    for obs_id in item.observation_ids:
        obs = observations[obs_id]
        if obs["candidate_service_key"] != item.candidate_service_key:
            return "observation_candidate_denied"
        if item.signal_family is not None and obs["signal_family"] != item.signal_family:
            return "signal_family_denied"
    if any(
        len(values) != len(set(values))
        for values in (item.observation_ids, item.relation_ids, item.requested_fields)
    ):
        return "duplicate_reference"
    return None


def inspect_semantic_response(
    raw: bytes,
    *,
    prompt: SemanticPrompt,
    expected_prompt_sha256: str,
) -> SemanticInspection:
    """Inspect once without retries, dispatch, evaluation or primary selection.

    The prompt hash must come from trusted pre-call state. Returned raw data and
    eligible proposal text are private evidence, not automatically secret-free.
    """
    limits = ExchangeLimits()
    if type(prompt) is not SemanticPrompt or type(prompt.payload_json) is not bytes:
        raise ExchangeBoundaryError("invalid_semantic_prompt")
    data = decode_json_object(prompt.payload_json, limits)
    prompt_hash = prompt.sha256
    if prompt_hash != expected_prompt_sha256:
        raise ExchangeBoundaryError("semantic_prompt_mismatch")
    received = len(raw) if type(raw) is bytes else None
    retained = raw if received is not None and received <= limits.max_bytes else None
    digest = hashlib.sha256(retained).hexdigest() if retained is not None else None

    def failure(code: str) -> SemanticInspection:
        return SemanticInspection(False, code, retained, digest, received, prompt_hash, (), (), ())

    try:
        value = decode_json_object(raw, limits)
        envelope = ResponseEnvelope.model_validate_json(encode_json(value, limits), strict=True)
    except ExchangeBoundaryError as exc:
        return failure(exc.code)
    except ValidationError:
        return failure("envelope_schema_invalid")
    observations = {
        obs["observation_id"]: {**obs, "candidate_service_key": card["card"]["service_entity_key"]}
        for card in data["detailed_evidence_cards"]
        for obs in card["observations"]
    }
    events = {
        entry["event_group_id"]
        for card in data["detailed_evidence_cards"]
        for entry in card["event_groups"]
    }
    candidates = {entry["service_entity_key"] for entry in data["candidate_directory"]["entries"]}
    hypotheses: list[SemanticHypothesisProposal] = []
    queries: list[SemanticCardQuery | SemanticVerificationQuery] = []
    decisions: list[ProposalDecision] = []
    counts: dict[str, Counter[str]] = {
        section: Counter(
            item[key] for item in items if type(item) is dict and type(item.get(key)) is str
        )
        for section, key, items in (
            ("hypotheses", "proposal_id", envelope.hypotheses),
            ("queries", "query_id", envelope.queries),
        )
    }
    structurally_valid = True
    query_adapter = TypeAdapter[SemanticQuery](SemanticQuery)
    sections: tuple[Literal["hypotheses", "queries"], ...] = ("hypotheses", "queries")
    parsed: list[
        tuple[
            Literal["hypotheses", "queries"],
            int,
            SemanticHypothesisProposal | SemanticCardQuery | SemanticVerificationQuery | None,
        ]
    ] = []
    reference_ids: dict[str, set[str]] = {section: set() for section in sections}
    for section in sections:
        items = envelope.hypotheses if section == "hypotheses" else envelope.queries
        for index, entry in enumerate(items):
            try:
                encoded = encode_json(entry, limits)
                item = (
                    SemanticHypothesisProposal.model_validate_json(encoded, strict=True)
                    if section == "hypotheses"
                    else query_adapter.validate_json(encoded, strict=True)
                )
            except ValidationError:
                structurally_valid = False
                parsed.append((section, index, None))
                continue
            parsed.append((section, index, item))
            local_id = (
                item.proposal_id if isinstance(item, SemanticHypothesisProposal) else item.query_id
            )
            # Only unique, schema-valid records are unambiguous local targets.
            # Permission or predicate acceptance remains a separate later check.
            if counts[section][local_id] == 1:
                reference_ids[section].add(local_id)
    for section, index, parsed_item in parsed:
        if parsed_item is None:
            decisions.append(
                ProposalDecision(section, index, None, "denied", "item_schema_invalid")
            )
            continue
        local_id = (
            parsed_item.proposal_id
            if isinstance(parsed_item, SemanticHypothesisProposal)
            else parsed_item.query_id
        )
        code: str | None = None
        if counts[section][local_id] != 1:
            code = "duplicate_local_id"
        elif parsed_item.candidate_service_key not in candidates:
            code = "candidate_denied"
        elif isinstance(parsed_item, SemanticHypothesisProposal):
            code = _hypothesis_permission(
                parsed_item,
                data,
                set(observations),
                events,
                reference_ids["hypotheses"],
                reference_ids["queries"],
            )
        else:
            code = _query_permission(parsed_item, data, observations)
        decisions.append(
            ProposalDecision(
                section,
                index,
                local_id,
                "eligible" if code is None else "denied",
                code or "eligible_for_verification",
            )
        )
        if code is None:
            if isinstance(parsed_item, SemanticHypothesisProposal):
                hypotheses.append(parsed_item)
            else:
                queries.append(parsed_item)
    return SemanticInspection(
        structurally_valid,
        None,
        retained,
        digest,
        received,
        prompt_hash,
        tuple(decisions),
        tuple(hypotheses),
        tuple(queries),
    )
