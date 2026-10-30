"""Deterministic validation boundary for Semantic Fact Agent responses."""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact
from nurse_rcl.models import EvidenceRecord, GuardDecision, SemanticFact
from nurse_rcl.schemas import StrictFrozenModel

SEMANTIC_GUARD_VERSION: Literal["semantic_guard_v1"] = "semantic_guard_v1"

_ANSWER_KEY_FRAGMENTS = (
    "answerkey",
    "groundtruth",
    "injection",
    "knownrootcause",
    "label",
)
_RANK_EDIT_KEYS = frozenset(
    {
        "rank",
        "ranking",
        "score",
        "scores",
        "topk",
        "prediction",
        "predictions",
    }
)
_FREE_TEXT_EVIDENCE_KEYS = frozenset(
    {
        "claim",
        "description",
        "evidencedescription",
        "evidencetext",
        "explanation",
        "freetext",
        "rationale",
        "reasoning",
    }
)


class SemanticGuardArtifact(StrictFrozenModel):
    """Hash-bound audit record for one complete semantic-agent response."""

    guard_version: Literal["semantic_guard_v1"] = SEMANTIC_GUARD_VERSION
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_canonical_json: str = Field(min_length=1)
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parsed_facts: tuple[SemanticFact | None, ...]
    accepted_facts: tuple[SemanticFact, ...]
    decisions: tuple[GuardDecision, ...]
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify response, fact, decision, and artifact hash bindings."""

        try:
            response = json.loads(self.response_canonical_json)
            canonical_response = canonical_json_bytes(response)
        except (TypeError, ValueError):
            return False
        if canonical_response.decode("utf-8") != self.response_canonical_json:
            return False
        if sha256_hex(canonical_response) != self.response_hash:
            return False
        if not isinstance(response, dict) or response.get("payload_hash") != self.payload_hash:
            return False
        raw_facts = response.get("facts")
        if not isinstance(raw_facts, list):
            return False
        if len(raw_facts) != len(self.parsed_facts) or len(raw_facts) != len(self.decisions):
            return False
        for index, (raw_fact, decision) in enumerate(zip(raw_facts, self.decisions, strict=True)):
            if decision.fact_index != index:
                return False
            if decision.fact_hash != sha256_hex(canonical_json_bytes(raw_fact)):
                return False
        expected_accepted = tuple(
            fact
            for fact, decision in zip(self.parsed_facts, self.decisions, strict=True)
            if fact is not None and decision.accepted
        )
        if expected_accepted != self.accepted_facts:
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def validate_semantic_fact_response(
    *,
    artifact: SemanticPayloadArtifact,
    response: Mapping[str, Any],
) -> SemanticGuardArtifact:
    """Validate every submitted fact independently and retain every decision."""

    if not artifact.verify():
        raise ValueError("semantic payload hash verification failed")

    response_document = dict(response)
    response_bytes = canonical_json_bytes(response_document)
    if response_document.get("payload_hash") != artifact.payload_hash:
        raise ValueError("semantic fact response payload hash does not match")
    if set(response_document) != {"payload_hash", "facts"}:
        raise ValueError("semantic fact response schema is invalid")
    raw_facts = response_document.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("semantic fact response facts must be an array")

    payload = artifact.payload
    candidates = {card.service: card for card in payload.candidates}
    evidence = {record.evidence_id: record for record in payload.evidence}
    adjacency = {
        card.service: frozenset(card.neighbors)
        for card in payload.candidates
    }

    parsed_facts: list[SemanticFact | None] = []
    accepted_facts: list[SemanticFact] = []
    decisions: list[GuardDecision] = []
    accepted_hashes: set[str] = set()
    evidence_roles: dict[str, str] = {}

    for index, raw_fact in enumerate(raw_facts):
        fact_hash = sha256_hex(canonical_json_bytes(raw_fact))
        fact, code = _validate_fact(
            raw_fact=raw_fact,
            candidates=candidates,
            evidence=evidence,
            adjacency=adjacency,
            accepted_hashes=accepted_hashes,
            evidence_roles=evidence_roles,
            fact_hash=fact_hash,
        )
        parsed_facts.append(fact)
        accepted = code == "accepted"
        decisions.append(
            GuardDecision(
                fact_index=index,
                accepted=accepted,
                code=code,
                fact_hash=fact_hash,
            )
        )
        if accepted:
            if fact is None:  # pragma: no cover - protected by _validate_fact
                raise AssertionError("accepted guard decision requires a parsed fact")
            accepted_facts.append(fact)
            accepted_hashes.add(fact_hash)
            evidence_roles.update(_fact_evidence_roles(fact))

    parsed_fact_tuple = tuple(parsed_facts)
    accepted_fact_tuple = tuple(accepted_facts)
    decision_tuple = tuple(decisions)
    unsigned = {
        "guard_version": SEMANTIC_GUARD_VERSION,
        "payload_hash": artifact.payload_hash,
        "response_canonical_json": response_bytes.decode("utf-8"),
        "response_hash": sha256_hex(response_bytes),
        "parsed_facts": [
            fact.model_dump(mode="json") if fact is not None else None
            for fact in parsed_facts
        ],
        "accepted_facts": [fact.model_dump(mode="json") for fact in accepted_facts],
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    result = SemanticGuardArtifact(
        guard_version=SEMANTIC_GUARD_VERSION,
        payload_hash=artifact.payload_hash,
        response_canonical_json=response_bytes.decode("utf-8"),
        response_hash=sha256_hex(response_bytes),
        parsed_facts=parsed_fact_tuple,
        accepted_facts=accepted_fact_tuple,
        decisions=decision_tuple,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("semantic guard artifact verification failed")
    return result


def _validate_fact(
    *,
    raw_fact: Any,
    candidates: Mapping[str, Any],
    evidence: Mapping[str, EvidenceRecord],
    adjacency: Mapping[str, frozenset[str]],
    accepted_hashes: set[str],
    evidence_roles: Mapping[str, str],
    fact_hash: str,
) -> tuple[SemanticFact | None, str]:
    if _contains_answer_reference(raw_fact):
        return None, "reject_answer_leakage"
    if _contains_key(raw_fact, _RANK_EDIT_KEYS):
        return None, "reject_rank_editing"
    if _contains_key(raw_fact, _FREE_TEXT_EVIDENCE_KEYS):
        return None, "reject_free_text_evidence"

    try:
        fact = SemanticFact.model_validate(raw_fact)
    except ValidationError:
        return None, "reject_schema"
    if not _relation_shape_is_valid(fact):
        return fact, "reject_schema"

    services = (fact.service, fact.related_service)
    if any(service is not None and service not in candidates for service in services):
        return fact, "reject_candidate_scope"

    evidence_ids = (fact.evidence_id, fact.related_evidence_id)
    if any(evidence_id is not None and evidence_id not in evidence for evidence_id in evidence_ids):
        return fact, "reject_evidence_missing"
    if evidence[fact.evidence_id].service != fact.service:
        return fact, "reject_evidence_ownership"
    if (
        fact.related_evidence_id is not None
        and fact.related_service is not None
        and evidence[fact.related_evidence_id].service != fact.related_service
    ):
        return fact, "reject_evidence_ownership"

    if fact.related_service is not None:
        if not _is_reachable(fact.service, fact.related_service, adjacency):
            return fact, "reject_topology"
        if not _source_precedes_symptom(fact, evidence):
            return fact, "reject_time_order"

    if fact_hash in accepted_hashes:
        return fact, "reject_duplicate"
    if any(
        evidence_id in evidence_roles and evidence_roles[evidence_id] != role
        for evidence_id, role in _fact_evidence_roles(fact).items()
    ):
        return fact, "reject_conflict"
    return fact, "accepted"


def _relation_shape_is_valid(fact: SemanticFact) -> bool:
    related = (fact.related_service, fact.related_evidence_id)
    if fact.fact_type == "root_support":
        return related == (None, None)
    return all(value is not None for value in related) and fact.service != fact.related_service


def _fact_evidence_roles(fact: SemanticFact) -> dict[str, str]:
    if fact.fact_type == "root_support":
        return {fact.evidence_id: "root"}
    if fact.related_evidence_id is None:
        return {}
    if fact.fact_type == "source_support":
        return {
            fact.evidence_id: "source",
            fact.related_evidence_id: "symptom",
        }
    return {
        fact.evidence_id: "symptom",
        fact.related_evidence_id: "source",
    }


def _source_precedes_symptom(
    fact: SemanticFact,
    evidence: Mapping[str, EvidenceRecord],
) -> bool:
    if fact.related_evidence_id is None:
        return True
    primary_time = evidence[fact.evidence_id].first_anomalous_at
    related_time = evidence[fact.related_evidence_id].first_anomalous_at
    if primary_time is None or related_time is None:
        return False
    if fact.fact_type == "source_support":
        return primary_time < related_time
    return related_time < primary_time


def _is_reachable(
    source: str,
    target: str,
    adjacency: Mapping[str, frozenset[str]],
) -> bool:
    pending = deque([source])
    visited = {source}
    while pending:
        current = pending.popleft()
        for neighbor in adjacency.get(current, frozenset()):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return False


def _contains_answer_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _has_answer_fragment(str(key)) or _contains_answer_reference(nested)
            for key, nested in value.items()
        )
    if _is_fact_sequence(value):
        return any(_contains_answer_reference(item) for item in value)
    return isinstance(value, str) and _has_answer_fragment(value)


def _has_answer_fragment(value: str) -> bool:
    normalized = _normalized_token(value)
    return any(fragment in normalized for fragment in _ANSWER_KEY_FRAGMENTS)


def _contains_key(value: Any, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            _normalized_token(str(key)) in forbidden or _contains_key(nested, forbidden)
            for key, nested in value.items()
        )
    if _is_fact_sequence(value):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _is_fact_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
