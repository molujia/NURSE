"""Final-snapshot permissions; no new query execution or cross-round graph union."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.semantic_followup import SEMANTIC_FOLLOWUP_PROMPT_VERSION, SemanticFollowupPrompt
from nurse_rcl.v2.semantic_permissions import (
    ProposalDecision,
    SemanticInspection,
    _hypothesis_permission,
)
from nurse_rcl.v2.semantic_proposals import (
    ResponseEnvelope,
    SemanticCardQuery,
    SemanticHypothesisProposal,
    SemanticQuery,
    SemanticVerificationQuery,
)


def inspect_semantic_followup_response(
    raw: bytes,
    *,
    prompt: SemanticFollowupPrompt,
    expected_prompt_sha256: str,
) -> SemanticInspection:
    """Inspect against an independently pinned, replay-built second prompt.

    A pin must originate from trusted pre-call state, not the model or a rehashed
    payload. Eligibility is not evidence verification. Consumers of derived IDs
    still need the original source/session replay and deterministic compiler.
    """
    if type(prompt) is not SemanticFollowupPrompt or type(expected_prompt_sha256) is not str:
        raise ExchangeBoundaryError("invalid_semantic_followup_prompt")
    try:
        payload = object.__getattribute__(prompt, "payload_json")
    except AttributeError:
        raise ExchangeBoundaryError("invalid_semantic_followup_prompt") from None
    if type(payload) is not bytes:
        raise ExchangeBoundaryError("invalid_semantic_followup_prompt")
    limits = ExchangeLimits()
    prompt_hash = prompt.sha256
    if prompt_hash != expected_prompt_sha256:
        raise ExchangeBoundaryError("semantic_prompt_mismatch")
    data = decode_json_object(prompt.payload_json, limits)
    if (
        data.get("prompt_version") != SEMANTIC_FOLLOWUP_PROMPT_VERSION
        or type(data.get("round_number")) is not int
        or data["round_number"] != 2
        or data.get("replacement_policy") != "complete_snapshot"
        or data.get("new_queries_allowed") is not False
    ):
        raise ExchangeBoundaryError("invalid_semantic_followup_prompt")
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
    cards = data["detailed_evidence_cards"] + data["returned_cards"]
    observations = {obs["observation_id"] for card in cards for obs in card["observations"]}
    # Visibility only: do not recast a derivation as an original Observation.
    observations.update(item["derived_observation_id"] for item in data["derived_observations"])
    events = {event["event_group_id"] for card in cards for event in card["event_groups"]}
    candidates = {entry["service_entity_key"] for entry in data["candidate_directory"]["entries"]}
    aliases = {item["query_id"] for item in data["query_outcomes"]}
    counts = {
        section: Counter(
            item[key] for item in items if type(item) is dict and type(item.get(key)) is str
        )
        for section, key, items in (
            ("hypotheses", "proposal_id", envelope.hypotheses),
            ("queries", "query_id", envelope.queries),
        )
    }
    parsed: list[
        tuple[
            Literal["hypotheses", "queries"],
            int,
            SemanticHypothesisProposal | SemanticCardQuery | SemanticVerificationQuery | None,
        ]
    ] = []
    current_hypothesis_ids: set[str] = set()
    structurally_valid = True
    query_adapter = TypeAdapter[SemanticQuery](SemanticQuery)
    sections: tuple[Literal["hypotheses", "queries"], ...] = ("hypotheses", "queries")
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
            if (
                isinstance(item, SemanticHypothesisProposal)
                and counts[section][item.proposal_id] == 1
            ):
                current_hypothesis_ids.add(item.proposal_id)
    decisions: list[ProposalDecision] = []
    eligible: list[SemanticHypothesisProposal] = []
    for section, index, parsed_item in parsed:
        if parsed_item is None:
            decisions.append(
                ProposalDecision(section, index, None, "denied", "item_schema_invalid")
            )
            continue
        if not isinstance(parsed_item, SemanticHypothesisProposal):
            decisions.append(
                ProposalDecision(
                    section,
                    index,
                    parsed_item.query_id,
                    "denied",
                    "queries_closed_after_first_round",
                )
            )
            continue
        code: str | None = None
        if counts[section][parsed_item.proposal_id] != 1:
            code = "duplicate_local_id"
        elif parsed_item.candidate_service_key not in candidates:
            code = "candidate_denied"
        else:
            code = _hypothesis_permission(
                parsed_item, data, observations, events, current_hypothesis_ids, aliases
            )
        decisions.append(
            ProposalDecision(
                section,
                index,
                parsed_item.proposal_id,
                "eligible" if code is None else "denied",
                code or "eligible_for_verification",
            )
        )
        if code is None:
            eligible.append(parsed_item)
    return SemanticInspection(
        structurally_valid,
        None,
        retained,
        digest,
        received,
        prompt_hash,
        tuple(decisions),
        tuple(eligible),
        (),
    )
