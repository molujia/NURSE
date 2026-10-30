"""Allowlisted, deduplicated, and budgeted detailed-card expansion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse
from nurse_rcl.v2.semantic_context import (
    DetailedEvidenceCard,
    SemanticContextBundle,
    build_detailed_evidence_card,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

CARD_EXPANSION_POLICY_SCHEMA_VERSION: Literal["nurse.v2.card-expansion-policy/1"] = (
    "nurse.v2.card-expansion-policy/1"
)
CARD_EXPANSION_REQUEST_SCHEMA_VERSION: Literal["nurse.v2.card-expansion-request/1"] = (
    "nurse.v2.card-expansion-request/1"
)
CARD_EXPANSION_DECISION_SCHEMA_VERSION: Literal["nurse.v2.card-expansion-decision/1"] = (
    "nurse.v2.card-expansion-decision/1"
)
CARD_EXPANSION_BATCH_SCHEMA_VERSION: Literal["nurse.v2.card-expansion-batch/1"] = (
    "nurse.v2.card-expansion-batch/1"
)
CARD_EXPANSION_POLICY_VERSION: Literal["nurse-v2-card-expansion-policy/1"] = (
    "nurse-v2-card-expansion-policy/1"
)
CARD_TOKEN_ESTIMATOR_VERSION: Literal["canonical-utf8-bytes-ceil-div-4/1"] = (
    "canonical-utf8-bytes-ceil-div-4/1"
)
CARD_EXPANSION_QUERY_KIND: Literal["candidate_detailed_evidence_card"] = (
    "candidate_detailed_evidence_card"
)

ExpansionStatus: TypeAlias = Literal["completed", "reused", "refused"]
ExpansionReason: TypeAlias = Literal[
    "expansion_completed",
    "initial_card_reused",
    "request_scope_mismatch",
    "query_not_allowlisted",
    "candidate_outside_frozen_universe",
    "query_budget_exhausted",
    "token_budget_exhausted",
]


class CardExpansionPolicy(StrictFrozenModel):
    """Frozen allowlist and independent query/context-token budgets."""

    schema_version: Literal["nurse.v2.card-expansion-policy/1"] = (
        CARD_EXPANSION_POLICY_SCHEMA_VERSION
    )
    policy_version: Literal["nurse-v2-card-expansion-policy/1"] = CARD_EXPANSION_POLICY_VERSION
    allowed_query_kinds: tuple[str, ...] = (CARD_EXPANSION_QUERY_KIND,)
    query_limit: int = Field(ge=0)
    token_limit: int = Field(ge=0)
    token_estimator_version: Literal["canonical-utf8-bytes-ceil-div-4/1"] = (
        CARD_TOKEN_ESTIMATOR_VERSION
    )
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_frozen_allowlist(self) -> Self:
        if self.allowed_query_kinds != (CARD_EXPANSION_QUERY_KIND,):
            raise ValueError("card expansion allowlist differs from frozen policy")
        return self

    def verify(self) -> bool:
        try:
            validated = CardExpansionPolicy.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"policy_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self and validated.policy_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


def build_card_expansion_policy(
    *,
    query_limit: int = 8,
    token_limit: int = 12_000,
) -> CardExpansionPolicy:
    """Freeze the initial card-expansion allowlist and budgets."""

    unsigned: dict[str, object] = {
        "schema_version": CARD_EXPANSION_POLICY_SCHEMA_VERSION,
        "policy_version": CARD_EXPANSION_POLICY_VERSION,
        "allowed_query_kinds": (CARD_EXPANSION_QUERY_KIND,),
        "query_limit": query_limit,
        "token_limit": token_limit,
        "token_estimator_version": CARD_TOKEN_ESTIMATOR_VERSION,
    }
    policy = CardExpansionPolicy.model_validate(
        {
            **unsigned,
            "policy_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not policy.verify():  # pragma: no cover - construction invariant
        raise AssertionError("card expansion policy verification failed")
    return policy


class CardExpansionRequest(StrictFrozenModel):
    """One structured request; policy and frozen-scope checks happen at execution."""

    schema_version: Literal["nurse.v2.card-expansion-request/1"] = (
        CARD_EXPANSION_REQUEST_SCHEMA_VERSION
    )
    expansion_request_id: str = Field(pattern=r"^ceq_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_context_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    service_entity_key: str = Field(min_length=1)
    query_kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    decision_time_unix_ns: int = Field(ge=0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = CardExpansionRequest.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(
                mode="json",
                exclude={"expansion_request_id", "request_sha256"},
            )
            digest = sha256_hex(canonical_json_bytes(unsigned))
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.request_sha256 == digest
            and validated.expansion_request_id == f"ceq_{digest[:24]}"
        )


def build_card_expansion_request(
    *,
    opaque_case_id: str,
    candidate_freeze_sha256: str,
    semantic_context_bundle_sha256: str,
    service_entity_key: str,
    query_kind: str,
    decision_time_unix_ns: int,
) -> CardExpansionRequest:
    """Normalize a request so identical semantics share one request ID."""

    unsigned: dict[str, object] = {
        "schema_version": CARD_EXPANSION_REQUEST_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "semantic_context_bundle_sha256": semantic_context_bundle_sha256,
        "service_entity_key": service_entity_key,
        "query_kind": query_kind,
        "decision_time_unix_ns": decision_time_unix_ns,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    request = CardExpansionRequest.model_validate(
        {
            **unsigned,
            "expansion_request_id": f"ceq_{digest[:24]}",
            "request_sha256": digest,
        }
    )
    if not request.verify():  # pragma: no cover - construction invariant
        raise AssertionError("card expansion request verification failed")
    return request


class CardExpansionRequestCount(StrictFrozenModel):
    """Batch-local duplicate accounting for one normalized request."""

    expansion_request_id: str = Field(pattern=r"^ceq_[0-9a-f]{24}$")
    submission_count: int = Field(ge=1)


class CardExpansionDecision(StrictFrozenModel):
    """One unique request outcome with post-decision budget balances."""

    schema_version: Literal["nurse.v2.card-expansion-decision/1"] = (
        CARD_EXPANSION_DECISION_SCHEMA_VERSION
    )
    expansion_decision_id: str = Field(pattern=r"^ced_[0-9a-f]{24}$")
    request: CardExpansionRequest
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ExpansionStatus
    reason_code: ExpansionReason
    detailed_card_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    charged_queries: int = Field(ge=0, le=1)
    estimated_tokens: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)
    remaining_queries: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if self.status == "completed":
            if (
                self.reason_code != "expansion_completed"
                or self.detailed_card_sha256 is None
                or self.charged_queries != 1
                or self.charged_tokens < 1
                or self.estimated_tokens != self.charged_tokens
            ):
                raise ValueError("completed expansion requires one charged card")
        elif self.status == "reused":
            if (
                self.reason_code != "initial_card_reused"
                or self.detailed_card_sha256 is None
                or self.charged_queries != 0
                or self.estimated_tokens != 0
                or self.charged_tokens != 0
            ):
                raise ValueError("reused expansion must return an uncharged initial card")
        elif (
            self.reason_code in {"expansion_completed", "initial_card_reused"}
            or self.detailed_card_sha256 is not None
            or self.charged_queries != 0
            or self.charged_tokens != 0
        ):
            raise ValueError("refused expansion cannot return or charge a card")
        if self.status == "refused":
            if self.reason_code == "token_budget_exhausted":
                if self.estimated_tokens <= self.remaining_tokens:
                    raise ValueError("token refusal requires a recorded budget excess")
            elif self.estimated_tokens != 0:
                raise ValueError("non-token refusal cannot estimate an unread card")
        return self

    def verify(self) -> bool:
        try:
            validated = CardExpansionDecision.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated.request.verify():
                return False
            unsigned = validated.model_dump(
                mode="json",
                exclude={"expansion_decision_id", "decision_sha256"},
            )
            digest = sha256_hex(canonical_json_bytes(unsigned))
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated.decision_sha256 == digest
            and validated.expansion_decision_id == f"ced_{digest[:24]}"
        )


class CardExpansionBatchResult(StrictFrozenModel):
    """Canonical batch ledger retaining every unique decision and duplicate count."""

    schema_version: Literal["nurse.v2.card-expansion-batch/1"] = CARD_EXPANSION_BATCH_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_context_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: CardExpansionPolicy
    submitted_request_count: int = Field(ge=0)
    unique_request_count: int = Field(ge=0)
    duplicate_request_count: int = Field(ge=0)
    request_counts: tuple[CardExpansionRequestCount, ...]
    decisions: tuple[CardExpansionDecision, ...]
    returned_cards: tuple[DetailedEvidenceCard, ...]
    charged_query_count: int = Field(ge=0)
    charged_token_count: int = Field(ge=0)
    remaining_query_count: int = Field(ge=0)
    remaining_token_count: int = Field(ge=0)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = CardExpansionBatchResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated.policy.verify():
                return False
            if validated.request_counts != tuple(
                sorted(
                    validated.request_counts,
                    key=lambda item: item.expansion_request_id,
                )
            ):
                return False
            if validated.returned_cards != tuple(
                sorted(
                    validated.returned_cards,
                    key=lambda item: item.card.service_entity_key,
                )
            ):
                return False
            request_ids = tuple(item.expansion_request_id for item in validated.request_counts)
            decision_request_ids = tuple(
                item.request.expansion_request_id for item in validated.decisions
            )
            if (
                set(request_ids) != set(decision_request_ids)
                or len(request_ids) != len(set(request_ids))
                or len(decision_request_ids) != len(set(decision_request_ids))
            ):
                return False
            if validated.submitted_request_count != sum(
                item.submission_count for item in validated.request_counts
            ):
                return False
            if validated.unique_request_count != len(validated.request_counts):
                return False
            if validated.duplicate_request_count != sum(
                item.submission_count - 1 for item in validated.request_counts
            ):
                return False
            if any(not item.verify() for item in validated.decisions):
                return False
            if any(
                item.policy_sha256 != validated.policy.policy_sha256 for item in validated.decisions
            ):
                return False
            if any(not item.verify() for item in validated.returned_cards):
                return False
            charged_queries = 0
            charged_tokens = 0
            for decision in validated.decisions:
                charged_queries += decision.charged_queries
                charged_tokens += decision.charged_tokens
                if decision.remaining_queries != validated.policy.query_limit - charged_queries:
                    return False
                if decision.remaining_tokens != validated.policy.token_limit - charged_tokens:
                    return False
            if validated.charged_query_count != charged_queries:
                return False
            if validated.charged_token_count != charged_tokens:
                return False
            if validated.remaining_query_count != validated.policy.query_limit - charged_queries:
                return False
            if validated.remaining_token_count != validated.policy.token_limit - charged_tokens:
                return False
            cards_by_hash = {item.detailed_card_sha256: item for item in validated.returned_cards}
            returned_decisions = tuple(
                item for item in validated.decisions if item.detailed_card_sha256 is not None
            )
            if len(cards_by_hash) != len(validated.returned_cards):
                return False
            if set(cards_by_hash) != {item.detailed_card_sha256 for item in returned_decisions}:
                return False
            for decision in returned_decisions:
                card_sha256 = decision.detailed_card_sha256
                if card_sha256 is None:  # pragma: no cover - narrowed above
                    return False
                card = cards_by_hash[card_sha256]
                if card.card.service_entity_key != decision.request.service_entity_key:
                    return False
                expected_access_reason = (
                    "approved_expansion" if decision.status == "completed" else "focus_q"
                )
                if card.card.access_reason != expected_access_reason:
                    return False
                if decision.status == "completed" and decision.estimated_tokens != (
                    _card_token_cost(card)
                ):
                    return False
                if (
                    card.card.opaque_case_id != validated.opaque_case_id
                    or card.card.decision_time_unix_ns != decision.request.decision_time_unix_ns
                ):
                    return False
            unsigned = validated.model_dump(mode="json", exclude={"result_sha256"})
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return validated.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _build_decision(
    *,
    request: CardExpansionRequest,
    policy: CardExpansionPolicy,
    status: ExpansionStatus,
    reason_code: ExpansionReason,
    detailed_card_sha256: str | None,
    charged_queries: int,
    estimated_tokens: int,
    charged_tokens: int,
    remaining_queries: int,
    remaining_tokens: int,
) -> CardExpansionDecision:
    unsigned: dict[str, object] = {
        "schema_version": CARD_EXPANSION_DECISION_SCHEMA_VERSION,
        "request": request.model_dump(mode="json"),
        "policy_sha256": policy.policy_sha256,
        "status": status,
        "reason_code": reason_code,
        "detailed_card_sha256": detailed_card_sha256,
        "charged_queries": charged_queries,
        "estimated_tokens": estimated_tokens,
        "charged_tokens": charged_tokens,
        "remaining_queries": remaining_queries,
        "remaining_tokens": remaining_tokens,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    decision = CardExpansionDecision.model_validate(
        {
            **unsigned,
            "expansion_decision_id": f"ced_{digest[:24]}",
            "decision_sha256": digest,
        }
    )
    if not decision.verify():  # pragma: no cover - construction invariant
        raise AssertionError("card expansion decision verification failed")
    return decision


def _card_token_cost(card: DetailedEvidenceCard) -> int:
    payload_size = len(canonical_json_bytes(card.model_dump(mode="json")))
    return max(1, (payload_size + 3) // 4)


def _validate_execution_inputs(
    *,
    policy: CardExpansionPolicy,
    semantic_context: SemanticContextBundle,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> None:
    if not policy.verify():
        raise ValueError("card expansion policy failed verification")
    if not semantic_context.verify():
        raise ValueError("semantic context failed verification")
    require_frozen_candidate_universe(
        frozen=frozen_candidates,
        candidate_universe=candidate_universe,
    )
    if not entity_registry.verify():
        raise ValueError("entity registry failed verification")
    if not event_grouping.verify():
        raise ValueError("event grouping failed verification")
    if not (
        semantic_context.opaque_case_id
        == candidate_universe.opaque_case_id
        == frozen_candidates.opaque_case_id
        == entity_registry.opaque_case_id
        == event_grouping.opaque_case_id
    ):
        raise ValueError("card expansion inputs belong to different cases")
    if semantic_context.candidate_freeze_sha256 != frozen_candidates.freeze_sha256:
        raise ValueError("semantic context does not match candidate freeze")
    if (
        semantic_context.candidate_directory.candidate_universe_sha256
        != candidate_universe.universe_sha256
    ):
        raise ValueError("semantic context does not match candidate universe")
    if semantic_context.entity_registry_sha256 != entity_registry.registry_sha256:
        raise ValueError("semantic context does not match entity registry")
    if semantic_context.event_grouping_result_sha256 != event_grouping.result_sha256:
        raise ValueError("semantic context does not match event grouping")
    if entity_registry.decision_time_unix_ns != event_grouping.decision_time_unix_ns:
        raise ValueError("card expansion decision times differ")
    if (
        semantic_context.candidate_directory.decision_time_unix_ns
        != entity_registry.decision_time_unix_ns
    ):
        raise ValueError("semantic context decision time does not match evidence")
    if tuple(item.observation for item in entity_registry.resolutions) != (
        event_grouping.observations
    ):
        raise ValueError("registry and grouping observation membership differs")


def _canonical_requests(
    requests: Sequence[CardExpansionRequest],
    *,
    prior_result: CardExpansionBatchResult | None,
) -> tuple[tuple[CardExpansionRequest, ...], tuple[CardExpansionRequestCount, ...]]:
    request_by_id: dict[str, CardExpansionRequest] = {}
    counts: Counter[str] = Counter()
    prior_request_ids: set[str] = set()
    if prior_result is not None:
        for decision in prior_result.decisions:
            item = decision.request
            request_by_id[item.expansion_request_id] = item
            prior_request_ids.add(item.expansion_request_id)
        counts.update(
            {
                item.expansion_request_id: item.submission_count
                for item in prior_result.request_counts
            }
        )
    for request in requests:
        try:
            validated = CardExpansionRequest.model_validate(
                request.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("card expansion request failed validation") from error
        if validated != request or not validated.verify():
            raise ValueError("card expansion request failed verification")
        previous = request_by_id.setdefault(validated.expansion_request_id, validated)
        if previous != validated:
            raise ValueError("card expansion request ID collision")
        counts[validated.expansion_request_id] += 1
    ordered = tuple(
        sorted(
            (
                item
                for request_id, item in request_by_id.items()
                if request_id not in prior_request_ids
            ),
            key=lambda item: item.expansion_request_id,
        )
    )
    request_counts = tuple(
        CardExpansionRequestCount(
            expansion_request_id=request_id,
            submission_count=counts[request_id],
        )
        for request_id in sorted(request_by_id)
    )
    return ordered, request_counts


def _validate_prior_result(
    *,
    prior_result: CardExpansionBatchResult,
    policy: CardExpansionPolicy,
    semantic_context: SemanticContextBundle,
    frozen_candidates: FrozenCandidateUniverse,
) -> None:
    if not prior_result.verify():
        raise ValueError("prior card expansion result failed verification")
    if prior_result.policy != policy:
        raise ValueError("prior card expansion policy differs")
    if (
        prior_result.opaque_case_id != semantic_context.opaque_case_id
        or prior_result.candidate_freeze_sha256 != frozen_candidates.freeze_sha256
        or prior_result.semantic_context_bundle_sha256 != semantic_context.bundle_sha256
    ):
        raise ValueError("prior card expansion result belongs to another scope")


def execute_card_expansion_batch(
    *,
    requests: Sequence[CardExpansionRequest],
    policy: CardExpansionPolicy,
    semantic_context: SemanticContextBundle,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    prior_result: CardExpansionBatchResult | None = None,
) -> CardExpansionBatchResult:
    """Execute one deterministic expansion batch without filesystem or network access."""

    _validate_execution_inputs(
        policy=policy,
        semantic_context=semantic_context,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    if prior_result is not None:
        _validate_prior_result(
            prior_result=prior_result,
            policy=policy,
            semantic_context=semantic_context,
            frozen_candidates=frozen_candidates,
        )
    ordered_requests, request_counts = _canonical_requests(
        requests,
        prior_result=prior_result,
    )
    candidate_keys = set(frozen_candidates.candidate_service_keys)
    initial_cards = {
        item.card.service_entity_key: item for item in semantic_context.detailed_evidence_cards
    }
    remaining_queries = (
        policy.query_limit if prior_result is None else prior_result.remaining_query_count
    )
    remaining_tokens = (
        policy.token_limit if prior_result is None else prior_result.remaining_token_count
    )
    decisions = [] if prior_result is None else list(prior_result.decisions)
    returned_cards: dict[str, DetailedEvidenceCard] = (
        {}
        if prior_result is None
        else {item.detailed_card_sha256: item for item in prior_result.returned_cards}
    )
    for item in ordered_requests:
        status: ExpansionStatus
        reason: ExpansionReason
        card: DetailedEvidenceCard | None = None
        charged_queries = 0
        estimated_tokens = 0
        charged_tokens = 0
        scope_matches = (
            item.opaque_case_id == semantic_context.opaque_case_id
            and item.candidate_freeze_sha256 == frozen_candidates.freeze_sha256
            and item.semantic_context_bundle_sha256 == semantic_context.bundle_sha256
            and item.decision_time_unix_ns
            == semantic_context.candidate_directory.decision_time_unix_ns
        )
        if not scope_matches:
            status = "refused"
            reason = "request_scope_mismatch"
        elif item.query_kind not in policy.allowed_query_kinds:
            status = "refused"
            reason = "query_not_allowlisted"
        elif item.service_entity_key not in candidate_keys:
            status = "refused"
            reason = "candidate_outside_frozen_universe"
        elif item.service_entity_key in initial_cards:
            status = "reused"
            reason = "initial_card_reused"
            card = initial_cards[item.service_entity_key]
        elif remaining_queries < 1:
            status = "refused"
            reason = "query_budget_exhausted"
        else:
            proposed_card = build_detailed_evidence_card(
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                service_entity_key=item.service_entity_key,
                access_reason="approved_expansion",
                entity_registry=entity_registry,
                event_grouping=event_grouping,
            )
            token_cost = _card_token_cost(proposed_card)
            estimated_tokens = token_cost
            if token_cost > remaining_tokens:
                status = "refused"
                reason = "token_budget_exhausted"
            else:
                status = "completed"
                reason = "expansion_completed"
                card = proposed_card
                charged_queries = 1
                charged_tokens = token_cost
                remaining_queries -= charged_queries
                remaining_tokens -= charged_tokens
        if card is not None:
            returned_cards[card.detailed_card_sha256] = card
        decisions.append(
            _build_decision(
                request=item,
                policy=policy,
                status=status,
                reason_code=reason,
                detailed_card_sha256=(card.detailed_card_sha256 if card is not None else None),
                charged_queries=charged_queries,
                estimated_tokens=estimated_tokens,
                charged_tokens=charged_tokens,
                remaining_queries=remaining_queries,
                remaining_tokens=remaining_tokens,
            )
        )
    ordered_cards = tuple(
        sorted(returned_cards.values(), key=lambda item: item.card.service_entity_key)
    )
    charged_query_count = policy.query_limit - remaining_queries
    charged_token_count = policy.token_limit - remaining_tokens
    unsigned: dict[str, object] = {
        "schema_version": CARD_EXPANSION_BATCH_SCHEMA_VERSION,
        "opaque_case_id": semantic_context.opaque_case_id,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "semantic_context_bundle_sha256": semantic_context.bundle_sha256,
        "policy": policy.model_dump(mode="json"),
        "submitted_request_count": sum(item.submission_count for item in request_counts),
        "unique_request_count": len(request_counts),
        "duplicate_request_count": sum(item.submission_count - 1 for item in request_counts),
        "request_counts": [item.model_dump(mode="json") for item in request_counts],
        "decisions": [item.model_dump(mode="json") for item in decisions],
        "returned_cards": [item.model_dump(mode="json") for item in ordered_cards],
        "charged_query_count": charged_query_count,
        "charged_token_count": charged_token_count,
        "remaining_query_count": remaining_queries,
        "remaining_token_count": remaining_tokens,
    }
    result = CardExpansionBatchResult.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("card expansion batch verification failed")
    return result
