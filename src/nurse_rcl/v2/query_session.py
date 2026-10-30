"""Canonical per-case query ledger shared by expansion and verification."""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from threading import RLock
from typing import Literal, Protocol

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import (
    CardExpansionBatchResult,
    CardExpansionPolicy,
    CardExpansionRequest,
    execute_card_expansion_batch,
)
from nurse_rcl.v2.diagnostic_schemas import VerificationRequest
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import (
    INITIAL_VERIFICATION_QUERY_LIMIT,
    VerificationBatchResult,
    VerificationExecutionPolicy,
    VerificationExecutionScope,
    _reject_hidden_cyclic_or_deep,
    execute_verification_batch,
)

CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.case-query-session-result/1"] = (
    "nurse.v2.case-query-session-result/1"
)


def _digest(unsigned: object) -> str:
    return sha256_hex(canonical_json_bytes(unsigned))


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and str.__len__(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_id(prefix: str, unsigned: object) -> tuple[str, str]:
    digest = _digest(unsigned)
    return f"{prefix}_{digest[:24]}", digest


class CaseQuerySessionResult(StrictFrozenModel):
    """Authoritative per-case ledger shared by expansion and verification."""

    schema_version: Literal["nurse.v2.case-query-session-result/1"] = (
        CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION
    )
    query_session_id: str = Field(pattern=r"^qss_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: Literal["card_expansion", "verification"]
    sequence_number: int = Field(ge=1)
    prior_session_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_limit: Literal[8] = INITIAL_VERIFICATION_QUERY_LIMIT
    card_expansion_result: CardExpansionBatchResult | None = None
    verification_result: VerificationBatchResult | None = None
    source_result_sha256s: tuple[str, ...]
    charged_query_count: int = Field(ge=0, le=8)
    remaining_query_count: int = Field(ge=0, le=8)
    session_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(CaseQuerySessionResult,),
                label="case query session",
                max_nodes=20_000,
                max_string_length=None,
                provenance_id_limit=None,
            )
            _reject_hidden_cyclic_or_deep(self)
            value = CaseQuerySessionResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if value != self:
                return False
            card = value.card_expansion_result
            verification = value.verification_result
            if card is None and verification is None:
                return False
            if (value.sequence_number == 1) != (value.prior_session_sha256 is None):
                return False
            if card is not None and (
                not card.verify()
                or card.opaque_case_id != value.opaque_case_id
                or card.candidate_freeze_sha256 != value.candidate_freeze_sha256
                or card.charged_query_count > value.query_limit
            ):
                return False
            if verification is not None:
                if (
                    value.phase != "verification"
                    or not verification.verify()
                    or verification.opaque_case_id != value.opaque_case_id
                    or verification.scope.candidate_freeze_sha256 != value.candidate_freeze_sha256
                ):
                    return False
                expected_preexisting_count = 0 if card is None else card.charged_query_count
                expected_preexisting_sources = () if card is None else (card.result_sha256,)
                if (
                    verification.preexisting_query_count != expected_preexisting_count
                    or verification.preexisting_query_source_ids != expected_preexisting_sources
                ):
                    return False
                expected_charge = verification.total_charged_query_count
            else:
                if value.phase != "card_expansion" or card is None:
                    return False
                expected_charge = card.charged_query_count
            expected_sources = tuple(
                item
                for item in (
                    None if card is None else card.result_sha256,
                    None if verification is None else verification.batch_sha256,
                )
                if item is not None
            )
            if value.source_result_sha256s != expected_sources:
                return False
            if (
                value.charged_query_count != expected_charge
                or value.remaining_query_count != value.query_limit - expected_charge
            ):
                return False
            unsigned = value.model_dump(mode="json", exclude={"query_session_id", "session_sha256"})
            identifier, digest = _stable_id("qss", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return value.query_session_id == identifier and value.session_sha256 == digest


def _validated_session(value: object, *, name: str) -> CaseQuerySessionResult:
    try:
        if type(value) is not CaseQuerySessionResult:
            raise ValueError
        preflight_exact_pydantic_tree(
            value,
            root_model_types=(CaseQuerySessionResult,),
            label=name,
            max_nodes=20_000,
            max_string_length=None,
            provenance_id_limit=None,
        )
        _reject_hidden_cyclic_or_deep(value)
        session = CaseQuerySessionResult.model_validate(
            value.model_dump(mode="python", warnings="error")
        )
        if session != value or not session.verify():
            raise ValueError
        return session
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{name} failed exact typed verification") from None


def _counts_extend(
    prior_counts: dict[str, int],
    current_counts: dict[str, int],
) -> bool:
    return all(current_counts.get(key, 0) >= value for key, value in prior_counts.items())


def _card_result_extends(
    prior: CardExpansionBatchResult,
    current: CardExpansionBatchResult,
) -> bool:
    return (
        current.policy == prior.policy
        and current.opaque_case_id == prior.opaque_case_id
        and current.candidate_freeze_sha256 == prior.candidate_freeze_sha256
        and current.semantic_context_bundle_sha256 == prior.semantic_context_bundle_sha256
        and current.decisions[: len(prior.decisions)] == prior.decisions
        and all(item in current.returned_cards for item in prior.returned_cards)
        and _counts_extend(
            {item.expansion_request_id: item.submission_count for item in prior.request_counts},
            {item.expansion_request_id: item.submission_count for item in current.request_counts},
        )
    )


def _verification_result_extends(
    prior: VerificationBatchResult,
    current: VerificationBatchResult,
) -> bool:
    return (
        current.policy == prior.policy
        and current.scope == prior.scope
        and current.preexisting_query_count == prior.preexisting_query_count
        and current.preexisting_query_source_ids == prior.preexisting_query_source_ids
        and current.decisions[: len(prior.decisions)] == prior.decisions
        and all(item in current.results for item in prior.results)
        and all(item in current.derived_observations for item in prior.derived_observations)
        and _counts_extend(
            {item.verification_request_id: item.submission_count for item in prior.request_counts},
            {
                item.verification_request_id: item.submission_count
                for item in current.request_counts
            },
        )
    )


def _session_is_strict_successor(
    prior: CaseQuerySessionResult | None,
    current: CaseQuerySessionResult,
) -> bool:
    if prior is None:
        return current.sequence_number == 1 and current.prior_session_sha256 is None
    if (
        current.prior_session_sha256 != prior.session_sha256
        or current.sequence_number != prior.sequence_number + 1
        or current.charged_query_count < prior.charged_query_count
        or (prior.phase == "verification" and current.phase != "verification")
    ):
        return False
    if prior.card_expansion_result is None:
        if current.card_expansion_result is not None:
            return False
    elif current.card_expansion_result is None or not _card_result_extends(
        prior.card_expansion_result,
        current.card_expansion_result,
    ):
        return False
    if prior.verification_result is None:
        return True
    return current.verification_result is not None and _verification_result_extends(
        prior.verification_result,
        current.verification_result,
    )


def _build_case_query_session_result(
    *,
    card_expansion_result: CardExpansionBatchResult | None,
    verification_result: VerificationBatchResult | None,
    prior_session: CaseQuerySessionResult | None = None,
) -> CaseQuerySessionResult:
    """Seal compatible low-level results into the only authoritative query ledger."""

    try:
        if card_expansion_result is not None and (
            type(card_expansion_result) is not CardExpansionBatchResult
            or not card_expansion_result.verify()
        ):
            raise ValueError
        if verification_result is not None and (
            type(verification_result) is not VerificationBatchResult
            or not verification_result.verify()
        ):
            raise ValueError
        if card_expansion_result is None and verification_result is None:
            raise ValueError
        if prior_session is not None:
            prior_session = _validated_session(
                prior_session,
                name="prior case query session",
            )
        if verification_result is not None:
            opaque_case_id = verification_result.opaque_case_id
            candidate_freeze_sha256 = verification_result.scope.candidate_freeze_sha256
            charged_query_count = verification_result.total_charged_query_count
        elif card_expansion_result is not None:
            opaque_case_id = card_expansion_result.opaque_case_id
            candidate_freeze_sha256 = card_expansion_result.candidate_freeze_sha256
            charged_query_count = card_expansion_result.charged_query_count
        else:  # pragma: no cover - narrowed above
            raise ValueError
        if prior_session is not None and (
            prior_session.opaque_case_id != opaque_case_id
            or prior_session.candidate_freeze_sha256 != candidate_freeze_sha256
            or (prior_session.phase == "verification" and verification_result is None)
            or (
                prior_session.card_expansion_result is not None
                and (
                    card_expansion_result is None
                    or not _card_result_extends(
                        prior_session.card_expansion_result,
                        card_expansion_result,
                    )
                )
            )
            or (
                prior_session.verification_result is not None
                and (
                    verification_result is None
                    or not _verification_result_extends(
                        prior_session.verification_result,
                        verification_result,
                    )
                )
            )
        ):
            raise ValueError
        unsigned: dict[str, object] = {
            "schema_version": CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION,
            "opaque_case_id": opaque_case_id,
            "candidate_freeze_sha256": candidate_freeze_sha256,
            "phase": "verification" if verification_result is not None else "card_expansion",
            "sequence_number": 1 if prior_session is None else prior_session.sequence_number + 1,
            "prior_session_sha256": (
                None if prior_session is None else prior_session.session_sha256
            ),
            "query_limit": INITIAL_VERIFICATION_QUERY_LIMIT,
            "card_expansion_result": (
                None
                if card_expansion_result is None
                else card_expansion_result.model_dump(mode="json")
            ),
            "verification_result": (
                None if verification_result is None else verification_result.model_dump(mode="json")
            ),
            "source_result_sha256s": tuple(
                item
                for item in (
                    None if card_expansion_result is None else card_expansion_result.result_sha256,
                    None if verification_result is None else verification_result.batch_sha256,
                )
                if item is not None
            ),
            "charged_query_count": charged_query_count,
            "remaining_query_count": INITIAL_VERIFICATION_QUERY_LIMIT - charged_query_count,
        }
        identifier, digest = _stable_id("qss", unsigned)
        result = CaseQuerySessionResult.model_validate(
            {**unsigned, "query_session_id": identifier, "session_sha256": digest}
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("results cannot form a canonical query ledger") from None
    if not result.verify():
        raise ValueError("results cannot form a canonical query ledger")
    return result


def require_case_query_session_replay(
    *,
    session: CaseQuerySessionResult,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> CaseQuerySessionResult:
    """Audit a canonical session against the current frozen typed inputs."""

    session = _validated_session(session, name="case query session")
    card = session.card_expansion_result
    if card is not None:
        if semantic_context is None or card_expansion_policy is None:
            raise ValueError("card session replay requires semantic context and policy")
        replayed_card: CardExpansionBatchResult | None = None
        count_by_id = {
            item.expansion_request_id: item.submission_count for item in card.request_counts
        }
        for decision in card.decisions:
            replayed_card = execute_card_expansion_batch(
                requests=(decision.request,) * count_by_id[decision.request.expansion_request_id],
                policy=card_expansion_policy,
                semantic_context=semantic_context,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
                prior_result=replayed_card,
            )
        if replayed_card is None:
            replayed_card = execute_card_expansion_batch(
                requests=(),
                policy=card_expansion_policy,
                semantic_context=semantic_context,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
            )
        if replayed_card != card:
            raise ValueError("card query ledger does not replay from frozen inputs")
    verification = session.verification_result
    if verification is not None:
        replayed_verification = execute_verification_batch(
            requests=(),
            policy=verification.policy,
            scope=verification.scope,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            prior_result=verification,
            prior_card_expansion_result=card,
        )
        if replayed_verification != verification:
            raise ValueError("verification query ledger does not replay from frozen inputs")
    return session


def _execute_case_query_session_card_expansion(
    *,
    requests: Sequence[CardExpansionRequest],
    policy: CardExpansionPolicy,
    semantic_context: SemanticContextBundle,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    prior_session: CaseQuerySessionResult | None = None,
) -> CaseQuerySessionResult:
    """Advance the authoritative case ledger while it remains in card phase."""

    prior_card: CardExpansionBatchResult | None = None
    if prior_session is not None:
        session = require_case_query_session_replay(
            session=prior_session,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            semantic_context=semantic_context,
            card_expansion_policy=policy,
        )
        if session.phase == "verification":
            raise ValueError("card expansion is forbidden after verification phase")
        prior_card = session.card_expansion_result
    card_result = execute_card_expansion_batch(
        requests=requests,
        policy=policy,
        semantic_context=semantic_context,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        prior_result=prior_card,
    )
    return _build_case_query_session_result(
        card_expansion_result=card_result,
        verification_result=None,
        prior_session=prior_session,
    )


def _execute_case_query_session_verification(
    *,
    requests: Sequence[VerificationRequest],
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    prior_session: CaseQuerySessionResult | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> CaseQuerySessionResult:
    """Advance or enter verification using the same authoritative query ledger."""

    prior_card: CardExpansionBatchResult | None = None
    prior_verification: VerificationBatchResult | None = None
    if prior_session is not None:
        session = require_case_query_session_replay(
            session=prior_session,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
        )
        prior_card = session.card_expansion_result
        prior_verification = session.verification_result
    verification_result = execute_verification_batch(
        requests=requests,
        policy=policy,
        scope=scope,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        prior_result=prior_verification,
        prior_card_expansion_result=prior_card,
    )
    return _build_case_query_session_result(
        card_expansion_result=prior_card,
        verification_result=verification_result,
        prior_session=prior_session,
    )


class CaseQuerySessionHeadStore(Protocol):
    """Trusted durable or process-shared compare-and-swap boundary."""

    def read_head(
        self,
        *,
        opaque_case_id: str,
        candidate_freeze_sha256: str,
    ) -> CaseQuerySessionResult | None: ...

    def compare_and_swap(
        self,
        *,
        opaque_case_id: str,
        candidate_freeze_sha256: str,
        expected_session_sha256: str | None,
        new_session: CaseQuerySessionResult,
    ) -> bool: ...


class InMemoryCaseQuerySessionHeadStore:
    """Thread-safe process-local head store for one shared runtime."""

    def __init__(self) -> None:
        self._heads: dict[tuple[str, str], CaseQuerySessionResult] = {}
        self._lock = RLock()

    def read_head(
        self,
        *,
        opaque_case_id: str,
        candidate_freeze_sha256: str,
    ) -> CaseQuerySessionResult | None:
        with self._lock:
            return self._heads.get((opaque_case_id, candidate_freeze_sha256))

    def compare_and_swap(
        self,
        *,
        opaque_case_id: str,
        candidate_freeze_sha256: str,
        expected_session_sha256: str | None,
        new_session: CaseQuerySessionResult,
    ) -> bool:
        session = _validated_session(new_session, name="new case query session")
        key = (opaque_case_id, candidate_freeze_sha256)
        if (session.opaque_case_id, session.candidate_freeze_sha256) != key:
            return False
        with self._lock:
            current = self._heads.get(key)
            current_sha256 = None if current is None else current.session_sha256
            if current_sha256 != expected_session_sha256 or not _session_is_strict_successor(
                current,
                session,
            ):
                return False
            self._heads[key] = session
            return True


class CaseQuerySessionCoordinator:
    """Trusted in-process head store preventing reset, fork, and rollback per case."""

    def __init__(
        self,
        *,
        head_store: CaseQuerySessionHeadStore,
        pinned_temporal_contexts: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._head_store = head_store
        contexts = {} if pinned_temporal_contexts is None else pinned_temporal_contexts
        if type(contexts) is not dict:
            raise ValueError("pinned temporal contexts must use an exact dictionary")
        clean_contexts: dict[str, tuple[str, str]] = {}
        for opaque_case_id, hashes in dict.items(contexts):
            if (
                type(opaque_case_id) is not str
                or type(hashes) is not tuple
                or tuple.__len__(hashes) != 2
            ):
                raise ValueError("pinned temporal context is malformed")
            time_contract_sha256 = tuple.__getitem__(hashes, 0)
            clock_resolution_sha256 = tuple.__getitem__(hashes, 1)
            if not _is_lower_sha256(time_contract_sha256) or not _is_lower_sha256(
                clock_resolution_sha256
            ):
                raise ValueError("pinned temporal context contains an invalid SHA-256")
            dict.__setitem__(
                clean_contexts,
                opaque_case_id,
                (time_contract_sha256, clock_resolution_sha256),
            )
        self._pinned_temporal_contexts = clean_contexts

    def require_pinned_temporal_context(
        self,
        *,
        opaque_case_id: str,
        time_contract_sha256: str,
    ) -> str:
        """Verify a case-time digest and return its coordinator-pinned clock digest."""

        if type(opaque_case_id) is not str or not _is_lower_sha256(time_contract_sha256):
            raise ValueError("temporal input does not match pinned temporal context")
        pinned = self._pinned_temporal_contexts.get(opaque_case_id)
        if pinned is None or not hmac.compare_digest(pinned[0], time_contract_sha256):
            raise ValueError("temporal input does not match pinned temporal context")
        return pinned[1]

    @staticmethod
    def _key(
        *,
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
    ) -> tuple[str, str]:
        return candidate_universe.opaque_case_id, frozen_candidates.freeze_sha256

    def require_current(self, session: CaseQuerySessionResult) -> CaseQuerySessionResult:
        session = _validated_session(session, name="case query session")
        current = self._head_store.read_head(
            opaque_case_id=session.opaque_case_id,
            candidate_freeze_sha256=session.candidate_freeze_sha256,
        )
        if current != session:
            raise ValueError("session is not the current authoritative head")
        return session

    def execute_card_expansion(
        self,
        *,
        requests: Sequence[CardExpansionRequest],
        policy: CardExpansionPolicy,
        semantic_context: SemanticContextBundle,
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
    ) -> CaseQuerySessionResult:
        key = self._key(
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
        )
        prior = self._head_store.read_head(
            opaque_case_id=key[0],
            candidate_freeze_sha256=key[1],
        )
        result = _execute_case_query_session_card_expansion(
            requests=requests,
            policy=policy,
            semantic_context=semantic_context,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            prior_session=prior,
        )
        expected = None if prior is None else prior.session_sha256
        if not self._head_store.compare_and_swap(
            opaque_case_id=key[0],
            candidate_freeze_sha256=key[1],
            expected_session_sha256=expected,
            new_session=result,
        ):
            raise ValueError("authoritative query head changed concurrently")
        return result

    def execute_verification(
        self,
        *,
        requests: Sequence[VerificationRequest],
        policy: VerificationExecutionPolicy,
        scope: VerificationExecutionScope,
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
    ) -> CaseQuerySessionResult:
        key = self._key(
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
        )
        prior = self._head_store.read_head(
            opaque_case_id=key[0],
            candidate_freeze_sha256=key[1],
        )
        result = _execute_case_query_session_verification(
            requests=requests,
            policy=policy,
            scope=scope,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            prior_session=prior,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
        )
        expected = None if prior is None else prior.session_sha256
        if not self._head_store.compare_and_swap(
            opaque_case_id=key[0],
            candidate_freeze_sha256=key[1],
            expected_session_sha256=expected,
            new_session=result,
        ):
            raise ValueError("authoritative query head changed concurrently")
        return result


__all__ = [
    "CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION",
    "CaseQuerySessionCoordinator",
    "CaseQuerySessionHeadStore",
    "CaseQuerySessionResult",
    "InMemoryCaseQuerySessionHeadStore",
    "require_case_query_session_replay",
]
