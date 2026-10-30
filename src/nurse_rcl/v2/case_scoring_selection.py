"""Private deterministic scoring selection from complete guard replay.

The receipt proves only internal integrity until both independent pins and all
original inputs are replayed. Predicates remain an evidence pool, not additional
scoring authority. This module does not establish B0 origin or change any score.
"""

from __future__ import annotations

import hmac
import json
import math
import re
from dataclasses import dataclass, fields
from types import UnionType
from typing import Annotated, Literal, Union, cast, get_args, get_origin, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse, require_frozen_candidate_universe
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult, execute_hypothesis_guard
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    SemanticFocusSet,
    TimeInterval,
)
from nurse_rcl.v2.query_session import (
    CaseQuerySessionCoordinator,
    CaseQuerySessionResult,
    InMemoryCaseQuerySessionHeadStore,
    require_case_query_session_replay,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import DetailedEvidenceCard, SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

CASE_SCORING_SELECTION_SCHEMA_VERSION: Literal["nurse.v2.case-scoring-selection/1"] = (
    "nurse.v2.case-scoring-selection/1"
)
CASE_SCORING_SELECTION_POLICY_VERSION: Literal["accepted-guard-scoring-selection/1"] = (
    "accepted-guard-scoring-selection/1"
)
_SOURCE_SCHEMA = "nurse.v2.case-scoring-selection-source/1"
_MAX_BYTES = 64 * 1024 * 1024
_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_GuardId = Annotated[str, Field(pattern=r"^hgr_[0-9a-f]{24}$")]


@dataclass(frozen=True, slots=True)
class CaseGuardReplayInputs:
    """Private input carrier: no persistent schema and no self-reported authority."""

    time_contract: CaseTimeContract
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    focus_set: SemanticFocusSet
    semantic_context: SemanticContextBundle | None
    card_expansion_policy: CardExpansionPolicy | None
    query_sessions: tuple[CaseQuerySessionResult, ...]
    mechanism_library: MechanismLibrary
    scope_interval: TimeInterval
    role_predicates: tuple[PredicateResult, ...]
    role_guard_results: tuple[HypothesisGuardResult, ...]


_INPUT_TYPES = get_type_hints(CaseGuardReplayInputs)
_INPUT_ADAPTER = TypeAdapter(CaseGuardReplayInputs)
_ROOTS = (
    CaseTimeContract,
    TemporalEntityRegistry,
    EventGroupingResult,
    CandidateUniverse,
    FrozenCandidateUniverse,
    SemanticFocusSet,
    SemanticContextBundle,
    CardExpansionPolicy,
    CaseQuerySessionResult,
    MechanismLibrary,
    TimeInterval,
    PredicateResult,
    HypothesisGuardResult,
)


def _exact(value: object, annotation: object) -> bool:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _exact(value, args[0])
    if origin is UnionType or origin is Union:
        return any(_exact(value, expected) for expected in args)
    if origin is Literal:
        return any(type(value) is type(expected) and value == expected for expected in args)
    if origin is tuple or origin is list:
        if type(value) is not origin:
            return False
        items = cast(tuple[object, ...], value)
        if origin is list or (len(args) == 2 and args[1] is Ellipsis):
            return all(_exact(item, args[0]) for item in items)
        return len(items) == len(args) and all(
            _exact(item, expected) for item, expected in zip(items, args, strict=True)
        )
    if origin is dict:
        return type(value) is dict and all(
            _exact(key, args[0]) and _exact(item, args[1])
            for key, item in cast(dict[object, object], value).items()
        )
    return type(value) is annotation


def _plain(value: object) -> object:
    """Only traverse state after the entire tree passed callback-free preflight."""
    kind = type(value)
    if kind is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("selection requires finite values")
        return value
    if kind in (str, int, bool, type(None)):
        if kind is str and len(cast(str, value)) > _MAX_BYTES:
            raise ValueError("selection source exceeds byte budget")
        return value
    if kind is tuple or kind is list:
        return [_plain(item) for item in cast(tuple[object, ...], value)]
    if kind is dict:
        return {key: _plain(item) for key, item in cast(dict[str, object], value).items()}
    state = cast(dict[str, object], object.__getattribute__(value, "__dict__"))
    annotations = cast(type[BaseModel], kind).model_fields
    if any(not _exact(item, annotations[name].annotation) for name, item in state.items()):
        raise ValueError("selection model field has a non-canonical exact type")
    return {name: _plain(item) for name, item in state.items()}


def _preflight(value: object) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(*_ROOTS, CaseScoringSelection),
        label="case scoring selection",
        max_string_length=None,
        provenance_id_limit=None,
    )
    _plain(value)


def _measure_json(value: object) -> int:
    """Measure preflighted plain JSON before allocating any serialized content.

    No per-string or per-atom cap narrows the complete 64 MiB budget. Decimal
    integer length is checked from its binary magnitude before any formatting.
    """
    total = 0

    def add(count: int) -> None:
        nonlocal total
        total += count
        if total > _MAX_BYTES:
            raise ValueError("selection source exceeds byte budget")

    def string_size(text: str) -> None:
        add(2)
        for start in range(0, len(text), 4096):
            piece = text[start : start + 4096]
            try:
                size = len(piece) if piece.isascii() else len(piece.encode("utf-8"))
            except UnicodeError:
                raise ValueError("selection requires valid UTF-8 strings") from None
            size += piece.count('"') + piece.count("\\")
            if not piece.isprintable():
                for code in range(32):
                    size += piece.count(chr(code)) * (1 if code in (8, 9, 10, 12, 13) else 5)
            add(size)

    def integer_size(number: int) -> None:
        add(int(number < 0))
        bits = int.bit_length(number)
        # This rational is strictly below log10(2). Reject the guaranteed
        # minimum decimal length before abs(), exponentiation or str().
        digits = max(0, bits - 1) * 30102999566398119521373889472449 // 10**32 + 1
        add(digits)
        magnitude = abs(number)
        power = 10**digits
        while magnitude >= power:
            add(1)
            power *= 10

    stack = [value]
    while stack:
        current = stack.pop()
        kind = type(current)
        if kind is str:
            string_size(cast(str, current))
        elif current is None:
            add(4)
        elif kind is bool:
            add(4 if current else 5)
        elif kind is int:
            integer_size(cast(int, current))
        elif kind is float:
            if not math.isfinite(cast(float, current)):
                raise ValueError("selection requires finite values")
            add(len(repr(current)))
        elif kind is list or kind is tuple:
            sequence = cast(list[object], current)
            add(2 + max(0, len(sequence) - 1))
            stack.extend(sequence)
        elif kind is dict:
            mapping = cast(dict[str | int, object], current)
            add(2 + max(0, len(mapping) - 1) + len(mapping))
            for key, child in mapping.items():
                if type(key) is str:
                    string_size(key)
                elif type(key) is int:
                    add(2)
                    integer_size(key)
                else:
                    raise ValueError("selection requires exact string or integer JSON keys")
                stack.append(child)
        else:
            raise ValueError("selection requires preflighted plain JSON")
    return total


def _bytes(value: object) -> bytes:
    plain = _plain(value)
    measured_size = _measure_json(plain)
    chunks: list[bytes] = []
    size = 0
    for part in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).iterencode(plain):
        chunk = part.encode("utf-8")
        size += len(chunk)
        if size > _MAX_BYTES:
            raise ValueError("selection source exceeds byte budget")
        chunks.append(chunk)
    if size != measured_size:
        raise ValueError("selection canonical JSON measurement differs from serialized size")
    return b"".join(chunks)


def _input_values(inputs: CaseGuardReplayInputs) -> dict[str, object]:
    if type(inputs) is not CaseGuardReplayInputs:
        raise ValueError("selection requires exact CaseGuardReplayInputs")
    return {
        field.name: object.__getattribute__(inputs, field.name)
        for field in fields(CaseGuardReplayInputs)
    }


def _source_bytes(
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection | None = None,
    pins: tuple[str, ...] = (),
) -> bytes:
    if type(inputs) is not CaseGuardReplayInputs or (
        selection is not None and type(selection) is not CaseScoringSelection
    ):
        raise ValueError("selection requires exact input and receipt roots")
    for pin in pins:
        if type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
            raise ValueError("selection requires exact lowercase SHA256 pins")
    values = _input_values(inputs)
    _preflight((values, selection))
    if any(not _exact(value, _INPUT_TYPES[name]) for name, value in values.items()):
        raise ValueError("selection input field has a non-canonical exact type")
    for sequence, key in (
        (inputs.query_sessions, "sequence_number"),
        (inputs.role_predicates, "predicate_result_id"),
        (inputs.role_guard_results, "guard_result_id"),
    ):
        identifiers = tuple(getattr(item, key) for item in sequence)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("selection source records require canonical unique IDs")
    # Every root must finish its own capacity check before the first serializer.
    # The auxiliary tuple used by structural preflight is not an artifact budget.
    if selection is not None:
        _measure_json(_plain(selection))
    return _bytes({"schema_version": _SOURCE_SCHEMA, **values})


def _unchanged(
    inputs: CaseGuardReplayInputs,
    original: bytes,
    selection: CaseScoringSelection | None = None,
    receipt: bytes | None = None,
) -> None:
    if not hmac.compare_digest(_source_bytes(inputs, selection), original):
        raise ValueError("selection original inputs changed during replay")
    if selection is not None and (
        receipt is None or not hmac.compare_digest(_bytes(selection), receipt)
    ):
        raise ValueError("selection original receipt changed during replay")


class _GuardQuerySessionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    guard_result_id: _GuardId
    session_sha256: _SHA

    @field_validator("*", mode="before")
    @classmethod
    def exact_strings(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("selection binding requires exact strings")
        return value


class CaseScoringSelection(BaseModel):
    """Private immutable partition; verify() never establishes source authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.case-scoring-selection/1"] = (
        CASE_SCORING_SELECTION_SCHEMA_VERSION
    )
    policy_version: Literal["accepted-guard-scoring-selection/1"] = (
        CASE_SCORING_SELECTION_POLICY_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: _SHA
    source_records_sha256: _SHA
    eligible_guard_result_ids: tuple[_GuardId, ...]
    excluded_guard_result_ids: tuple[_GuardId, ...]
    guard_query_session_bindings: tuple[_GuardQuerySessionBinding, ...]
    selection_sha256: _SHA

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        if type(value) is not dict and type(value) is not cls:
            raise ValueError("selection requires exact typed state or dictionary")
        _preflight(value)
        return value

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        annotation = cls.model_fields[cast(str, info.field_name)].annotation
        if info.mode == "json" and get_origin(annotation) is tuple and type(value) is list:
            return tuple(cast(list[object], value))
        if not _exact(value, annotation):
            raise ValueError("selection fields require exact declared types")
        return value

    @model_validator(mode="after")
    def canonical_partition(self) -> Self:
        for ids in (self.eligible_guard_result_ids, self.excluded_guard_result_ids):
            if ids != tuple(sorted(set(ids))):
                raise ValueError("selection IDs require canonical unique ordering")
        if set(self.eligible_guard_result_ids) & set(self.excluded_guard_result_ids):
            raise ValueError("selection dispositions overlap")
        bound = tuple(row.guard_result_id for row in self.guard_query_session_bindings)
        if bound != tuple(sorted(set(bound))) or set(bound) - set(self.eligible_guard_result_ids):
            raise ValueError("selection session bindings are not closed and canonical")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not CaseScoringSelection:
                return False
            _preflight(self)
            original = _bytes(self)
            try:
                rebuilt = CaseScoringSelection.model_validate_json(original)
                unsigned = cast(dict[str, object], _plain(rebuilt))
                unsigned.pop("selection_sha256")
                return hmac.compare_digest(original, _bytes(rebuilt)) and hmac.compare_digest(
                    rebuilt.selection_sha256, sha256_hex(_bytes(unsigned))
                )
            finally:
                _preflight(self)
                if not hmac.compare_digest(original, _bytes(self)):
                    raise ValueError("selection changed during verification")
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class ReplayedCaseScoringInputs:
    """Private fresh snapshot, usable only within the original pinned replay run.

    Obtain mapping and coordinator together with fresh_query_runtime(); callers
    must pass that same pair to the existing compiler. No mutable store is kept.
    This snapshot cannot replace original inputs or either public API pin.
    """

    selection: CaseScoringSelection
    eligible_guards: tuple[HypothesisGuardResult, ...]
    predicates: tuple[PredicateResult, ...]
    guard_query_session_bindings: tuple[_GuardQuerySessionBinding, ...]
    query_sessions: tuple[CaseQuerySessionResult, ...]

    def fresh_query_runtime(
        self,
    ) -> tuple[dict[str, CaseQuerySessionResult], CaseQuerySessionCoordinator | None]:
        original = _runtime_snapshot_bytes(self)
        try:
            # Clone from the fixed complete bytes, never from live caller fields.
            # This checks snapshot integrity only; source authority still requires
            # the public require API with its original inputs and independent pins.
            snapshot = _REPLAY_ADAPTER.validate_json(original, strict=True)
            if (
                not hmac.compare_digest(_runtime_snapshot_bytes(snapshot), original)
                or not snapshot.selection.verify()
                or snapshot.guard_query_session_bindings
                != snapshot.selection.guard_query_session_bindings
                or tuple(guard.guard_result_id for guard in snapshot.eligible_guards)
                != snapshot.selection.eligible_guard_result_ids
                or any(
                    not guard.verify() or guard.final_disposition != "accepted"
                    for guard in snapshot.eligible_guards
                )
                or any(not predicate.verify() for predicate in snapshot.predicates)
            ):
                raise ValueError("selection replay snapshot is not canonical")
            sessions = tuple(
                CaseQuerySessionResult.model_validate_json(_bytes(item))
                for item in snapshot.query_sessions
            )
            coordinator = _head_coordinator(sessions)
            by_hash = {item.session_sha256: item for item in sessions}
            mapping = {
                row.guard_result_id: by_hash[row.session_sha256]
                for row in snapshot.guard_query_session_bindings
            }
            if any(session != sessions[-1] for session in mapping.values()) or any(
                (session.opaque_case_id, session.candidate_freeze_sha256)
                != (snapshot.selection.opaque_case_id, snapshot.selection.candidate_freeze_sha256)
                for session in sessions
            ):
                raise ValueError("selection runtime requires the final session head")
            return mapping, coordinator if mapping else None
        finally:
            try:
                unchanged = hmac.compare_digest(_runtime_snapshot_bytes(self), original)
            except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
                unchanged = False
            if not unchanged:
                raise ValueError("selection runtime snapshot changed during cloning")


_REPLAY_ADAPTER = TypeAdapter(ReplayedCaseScoringInputs)
_REPLAY_TYPES = get_type_hints(ReplayedCaseScoringInputs)


def _runtime_snapshot_bytes(snapshot: ReplayedCaseScoringInputs) -> bytes:
    if type(snapshot) is not ReplayedCaseScoringInputs:
        raise ValueError("selection runtime requires an exact replay snapshot")
    values = {
        field.name: object.__getattribute__(snapshot, field.name)
        for field in fields(ReplayedCaseScoringInputs)
    }
    _preflight(values)
    if any(not _exact(value, _REPLAY_TYPES[name]) for name, value in values.items()):
        raise ValueError("selection runtime snapshot fields require exact declared types")
    return _bytes(values)


def case_guard_replay_inputs_sha256(*, inputs: CaseGuardReplayInputs) -> str:
    """Compute the complete source digest after exact preflight, for trusted callers."""
    original = _source_bytes(inputs)
    try:
        return sha256_hex(original)
    finally:
        _unchanged(inputs, original)


def _snapshot(original: bytes) -> CaseGuardReplayInputs:
    source = cast(dict[str, object], json.loads(original))
    source.pop("schema_version")
    snapshot = _INPUT_ADAPTER.validate_json(canonical_json_bytes(source), strict=True)
    if not hmac.compare_digest(original, _source_bytes(snapshot)):
        raise ValueError("selection input schema is not canonical")
    return snapshot


def _verify_sources(inputs: CaseGuardReplayInputs) -> None:
    case_id = inputs.time_contract.opaque_case_id
    for value in (
        inputs.entity_registry,
        inputs.event_grouping,
        inputs.candidate_universe,
        inputs.frozen_candidates,
        inputs.focus_set,
    ):
        if value.opaque_case_id != case_id or not value.verify():
            raise ValueError("selection frozen sources are invalid or cross-case")
    if (
        not inputs.mechanism_library.verify()
        or inputs.time_contract.decision_time_unix_ns
        != inputs.entity_registry.decision_time_unix_ns
        or inputs.time_contract.decision_time_unix_ns != inputs.event_grouping.decision_time_unix_ns
    ):
        raise ValueError("selection source time or mechanism contract differs")
    observations = tuple(
        sorted(
            (item.observation for item in inputs.entity_registry.resolutions),
            key=lambda item: item.observation_id,
        )
    )
    if observations != inputs.event_grouping.observations:
        raise ValueError("selection registry observations differ from grouping")
    rebuilt = build_complete_candidate_universe(
        entity_registry=inputs.entity_registry,
        event_grouping=inputs.event_grouping,
        scope_interval=inputs.scope_interval,
    )
    if rebuilt != inputs.candidate_universe:
        raise ValueError("selection complete candidate universe replay differs")
    require_frozen_candidate_universe(
        frozen=inputs.frozen_candidates, candidate_universe=inputs.candidate_universe
    )
    if inputs.focus_set.candidate_universe_sha256 != inputs.candidate_universe.universe_sha256 or {
        d.service_entity_key for d in inputs.focus_set.decisions
    } != set(inputs.frozen_candidates.candidate_service_keys):
        raise ValueError("selection focus does not close to complete C")
    if any(not predicate.verify() for predicate in inputs.role_predicates):
        raise ValueError("selection predicate integrity failed")


def _card_closes(card: DetailedEvidenceCard, inputs: CaseGuardReplayInputs) -> bool:
    observations = {item.observation_id: item for item in inputs.event_grouping.observations}
    groups = {item.event_group_id: item for item in inputs.event_grouping.event_groups}
    bindings = {
        item.binding_id: item
        for item in (
            *inputs.entity_registry.timeline_bindings,
            *(r.binding for r in inputs.entity_registry.resolutions),
        )
    }
    return (
        card.verify()
        and card.card.opaque_case_id == inputs.time_contract.opaque_case_id
        and card.card.candidate_universe_sha256 == inputs.candidate_universe.universe_sha256
        and card.card.service_entity_key in inputs.frozen_candidates.candidate_service_keys
        and all(observations.get(item.observation_id) == item for item in card.observations)
        and all(groups.get(item.event_group_id) == item for item in card.event_groups)
        and all(bindings.get(item.binding_id) == item for item in card.entity_bindings)
    )


def _verify_context(inputs: CaseGuardReplayInputs) -> None:
    context, policy = inputs.semantic_context, inputs.card_expansion_policy
    if (context is None) != (policy is None):
        raise ValueError("selection semantic context and card policy must be paired")
    if policy is not None and not policy.verify():
        raise ValueError("selection card policy verification failed")
    if context is None:
        return
    directory = context.candidate_directory
    if not context.verify() or (
        context.opaque_case_id != inputs.time_contract.opaque_case_id
        or context.candidate_freeze_sha256 != inputs.frozen_candidates.freeze_sha256
        or context.entity_registry_sha256 != inputs.entity_registry.registry_sha256
        or context.event_grouping_result_sha256 != inputs.event_grouping.result_sha256
        or directory.candidate_universe_sha256 != inputs.candidate_universe.universe_sha256
        or tuple(item.service_entity_key for item in directory.entries)
        != inputs.frozen_candidates.candidate_service_keys
    ):
        raise ValueError("selection context does not close to frozen C")
    focus = {item.service_entity_key: item for item in inputs.focus_set.decisions}
    for entry in directory.entries:
        decision = focus[entry.service_entity_key]
        if (
            entry.in_initial_focus,
            entry.focus_selection_phase,
            entry.focus_selection_reasons,
            entry.focus_exclusion_reason,
        ) != (
            decision.selected,
            decision.selection_phase,
            decision.selection_reasons,
            decision.exclusion_reason,
        ):
            raise ValueError("selection context does not project complete Q")
    if any(not _card_closes(card, inputs) for card in context.detailed_evidence_cards):
        raise ValueError("selection context card differs from frozen evidence")


def _head_coordinator(sessions: tuple[CaseQuerySessionResult, ...]) -> CaseQuerySessionCoordinator:
    store = InMemoryCaseQuerySessionHeadStore()
    coordinator = CaseQuerySessionCoordinator(head_store=store)
    prior: str | None = None
    for session in sessions:
        if (
            not session.verify()
            or (
                sessions
                and (session.opaque_case_id, session.candidate_freeze_sha256)
                != (sessions[0].opaque_case_id, sessions[0].candidate_freeze_sha256)
            )
            or not store.compare_and_swap(
                opaque_case_id=session.opaque_case_id,
                candidate_freeze_sha256=session.candidate_freeze_sha256,
                expected_session_sha256=prior,
                new_session=session,
            )
        ):
            raise ValueError("selection query sessions do not form one authoritative chain")
        prior = session.session_sha256
    return coordinator


def _query_runtime(
    inputs: CaseGuardReplayInputs,
) -> tuple[dict[str, CaseQuerySessionResult], CaseQuerySessionCoordinator]:
    _verify_context(inputs)
    coordinator = _head_coordinator(inputs.query_sessions)
    for session in inputs.query_sessions:
        if (
            session.opaque_case_id != inputs.time_contract.opaque_case_id
            or session.candidate_freeze_sha256 != inputs.frozen_candidates.freeze_sha256
        ):
            raise ValueError("selection query session is cross-case or cross-freeze")
        if session.card_expansion_result is not None and (
            inputs.semantic_context is None
            or inputs.card_expansion_policy is None
            or any(
                not _card_closes(card, inputs)
                for card in session.card_expansion_result.returned_cards
            )
        ):
            raise ValueError("selection expanded cards do not close to complete source")
        require_case_query_session_replay(
            session=session,
            candidate_universe=inputs.candidate_universe,
            frozen_candidates=inputs.frozen_candidates,
            entity_registry=inputs.entity_registry,
            event_grouping=inputs.event_grouping,
            semantic_context=inputs.semantic_context,
            card_expansion_policy=inputs.card_expansion_policy,
        )
    sessions = {item.session_sha256: item for item in inputs.query_sessions}
    predicates = {item.predicate_result_id: item for item in inputs.role_predicates}
    mapping: dict[str, CaseQuerySessionResult] = {}
    for guard in inputs.role_guard_results:
        provenance = {item for decision in guard.decisions for item in decision.provenance_ids}
        if guard.score_impact is not None:
            provenance.update(guard.score_impact.provenance_ids)
        matched = provenance & set(sessions)
        if len(matched) > 1:
            raise ValueError("selection guard references multiple query session heads")
        hypothesis = guard.parsed_hypothesis
        ids = (
            ()
            if hypothesis is None
            else (
                *hypothesis.necessary_predicate_result_ids,
                *hypothesis.optional_predicate_result_ids,
                *hypothesis.refuting_predicate_result_ids,
            )
        )
        if any(item not in predicates for item in ids):
            raise ValueError("selection guard is missing its source predicate")
        result_ids = {
            result_id for item in ids for result_id in predicates[item].verification_result_ids
        }
        request_ids = set() if hypothesis is None else set(hypothesis.verification_request_ids)
        if (result_ids or request_ids) and not matched:
            raise ValueError("selection verified guard is missing its query session")
        if matched:
            session = sessions[next(iter(matched))]
            verification = session.verification_result
            if session != inputs.query_sessions[-1] or verification is None:
                raise ValueError("selection guard must reference final verification session head")
            if not request_ids <= {
                item.verification_request_id for item in verification.results
            } or not result_ids <= {item.verification_result_id for item in verification.results}:
                raise ValueError("selection guard query references are absent from session")
            mapping[guard.guard_result_id] = session
    return mapping, coordinator


def _replay(inputs: CaseGuardReplayInputs, source_sha256: str) -> ReplayedCaseScoringInputs:
    _verify_sources(inputs)
    mapping, coordinator = _query_runtime(inputs)
    predicates = {p.predicate_result_id: p for p in inputs.role_predicates}
    eligible: list[HypothesisGuardResult] = []
    excluded: list[str] = []
    for guard in inputs.role_guard_results:
        if not guard.verify():
            raise ValueError("selection guard integrity failed")
        hypothesis = guard.parsed_hypothesis
        ids = (
            ()
            if hypothesis is None
            else (
                *hypothesis.necessary_predicate_result_ids,
                *hypothesis.optional_predicate_result_ids,
                *hypothesis.refuting_predicate_result_ids,
            )
        )
        if any(i not in predicates for i in ids) or (
            hypothesis is not None
            and hypothesis.opaque_case_id != inputs.time_contract.opaque_case_id
        ):
            raise ValueError("selection guard sources are missing or cross-case")
        replayed = execute_hypothesis_guard(
            submitted_record=json.loads(guard.submitted_record_json),
            predicates=tuple(predicates[i] for i in sorted(set(ids))),
            candidate_universe=inputs.candidate_universe,
            frozen_candidates=inputs.frozen_candidates,
            entity_registry=inputs.entity_registry,
            event_grouping=inputs.event_grouping,
            mechanism_library=inputs.mechanism_library,
            scope_interval=inputs.scope_interval,
            query_session=mapping.get(guard.guard_result_id),
            query_session_coordinator=coordinator,
            semantic_context=inputs.semantic_context,
            card_expansion_policy=inputs.card_expansion_policy,
        )
        if not hmac.compare_digest(_bytes(guard), _bytes(replayed)):
            raise ValueError("selection guard failed complete source replay")
        if guard.final_disposition == "accepted":
            if hypothesis is None or guard.score_impact is None or not guard.score_impact.verify():
                raise ValueError("selection accepted guard is incomplete")
            eligible.append(replayed)
        else:
            excluded.append(guard.guard_result_id)
    bindings = tuple(
        _GuardQuerySessionBinding(
            guard_result_id=guard.guard_result_id,
            session_sha256=mapping[guard.guard_result_id].session_sha256,
        )
        for guard in eligible
        if guard.guard_result_id in mapping
    )
    selection = CaseScoringSelection(
        opaque_case_id=inputs.time_contract.opaque_case_id,
        candidate_freeze_sha256=inputs.frozen_candidates.freeze_sha256,
        source_records_sha256=source_sha256,
        eligible_guard_result_ids=tuple(g.guard_result_id for g in eligible),
        excluded_guard_result_ids=tuple(excluded),
        guard_query_session_bindings=bindings,
        selection_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], _plain(selection))
    unsigned.pop("selection_sha256")
    selection = selection.model_copy(update={"selection_sha256": sha256_hex(_bytes(unsigned))})
    return ReplayedCaseScoringInputs(
        selection, tuple(eligible), inputs.role_predicates, bindings, inputs.query_sessions
    )


def build_case_scoring_selection(
    *, inputs: CaseGuardReplayInputs, expected_source_records_sha256: str
) -> CaseScoringSelection:
    """Replay every original guard before deriving the accepted scoring subset."""
    original = _source_bytes(inputs, pins=(expected_source_records_sha256,))
    try:
        if not hmac.compare_digest(sha256_hex(original), expected_source_records_sha256):
            raise ValueError("selection fails independent source pin")
        return _replay(_snapshot(original), expected_source_records_sha256).selection
    finally:
        _unchanged(inputs, original)


def require_case_scoring_selection(
    *,
    selection: CaseScoringSelection,
    expected_selection_sha256: str,
    inputs: CaseGuardReplayInputs,
    expected_source_records_sha256: str,
) -> ReplayedCaseScoringInputs:
    """Independently pin and rebuild all bytes; return a fresh compiler snapshot."""
    if type(selection) is not CaseScoringSelection:
        raise ValueError("selection requires an exact CaseScoringSelection")
    original = _source_bytes(
        inputs, selection, (expected_source_records_sha256, expected_selection_sha256)
    )
    receipt = _bytes(selection)
    try:
        if not hmac.compare_digest(
            selection.selection_sha256, expected_selection_sha256
        ) or not hmac.compare_digest(sha256_hex(original), expected_source_records_sha256):
            raise ValueError("selection fails independent pins")
        snapshot = CaseScoringSelection.model_validate_json(receipt)
        if not snapshot.verify():
            raise ValueError("selection fails internal integrity")
        replay = _replay(_snapshot(original), expected_source_records_sha256)
        if not hmac.compare_digest(receipt, _bytes(replay.selection)):
            raise ValueError("selection differs from complete source replay")
        return replay
    finally:
        _unchanged(inputs, original, selection, receipt)
