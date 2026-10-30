"""Single-copy live B0 selection; no legacy authority or trusted cache escapes.

The wire carries each complete source component once. Reconstructed legacy
carriers are call-local views, never an extra serialized copy or a trust token.
"""
from __future__ import annotations

import hmac
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal, cast, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2.b0_rule_binding import B0RuleBinding
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.live.b0_guards import execute_live_b0_guards
from nurse_rcl.v2.live_capacity import LIVE_B0_CAPACITY_PROFILE, binding_capacity
from nurse_rcl.v2.no_role_compilation import _representation
from nurse_rcl.v2.observation_schemas import CaseTimeContract, SemanticFocusSet
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

LIVE_SELECTION_MAX_NODES = 6_000_000
LIVE_SELECTION_MAX_JSON_BYTES = 256 * 1024 * 1024
LIVE_SELECTION_CAPACITY_PROFILE = 'live-selection-6m-256m/1'


@dataclass(frozen=True, slots=True)
class LiveScoringSources:
    envelope: RuleHypothesisEnvelope
    binding: B0RuleBinding
    time_contract: CaseTimeContract
    focus_set: SemanticFocusSet
    guard_results: tuple[HypothesisGuardResult, ...]
    schema_version: Literal['nurse.v2.live-b0-scoring-sources/1'] = 'nurse.v2.live-b0-scoring-sources/1'
    capacity_profile: Literal['live-selection-6m-256m/1'] = 'live-selection-6m-256m/1'


class LiveScoringSelection(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-b0-scoring-selection/1'] = 'nurse.v2.live-b0-scoring-selection/1'
    capacity_profile: Literal['live-selection-6m-256m/1'] = 'live-selection-6m-256m/1'
    opaque_case_id: str = Field(pattern=r'^opq_[0-9a-f]{32}$')
    source_records_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    rule_set_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    binding_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_freeze_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    eligible_guard_result_ids: tuple[str, ...]
    excluded_guard_result_ids: tuple[str, ...]
    selection_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


@dataclass(frozen=True, slots=True)
class LiveScoringReplay:
    """Ordinary current-version results; not an authorization for old consumers."""
    selection: LiveScoringSelection
    eligible_guards: tuple[HypothesisGuardResult, ...]
    predicates: tuple[PredicateResult, ...]


_SOURCE_TYPES = get_type_hints(LiveScoringSources)
_SOURCE_ADAPTER = TypeAdapter(LiveScoringSources)
_ROOTS = (RuleHypothesisEnvelope, B0RuleBinding, CaseTimeContract,
          SemanticFocusSet, HypothesisGuardResult, LiveScoringSelection)


def _measure(value: object) -> int:
    """Exact canonical size before whole-tree encoding, on preflighted types."""
    total = 0

    def add(n: int) -> None:
        nonlocal total
        total += n
        if total > LIVE_SELECTION_MAX_JSON_BYTES:
            raise ValueError('live selection exceeds the serialization byte budget')

    def string(text: str) -> None:
        add(2)
        for start in range(0, len(text), 4096):
            piece = text[start:start + 4096]
            try:
                size = len(piece) if piece.isascii() else len(piece.encode('utf-8'))
            except UnicodeError:
                raise ValueError('live selection requires valid UTF-8') from None
            size += piece.count('"') + piece.count('\\')
            # Printable strings cannot contain any JSON control escape. Keep
            # quote/backslash accounting above, and scan controls only when
            # needed; every byte is still measured before complete encoding.
            if not piece.isprintable():
                for code in range(32):
                    size += piece.count(chr(code)) * (1 if code in (8, 9, 10, 12, 13) else 5)
            add(size)

    def integer(number: int) -> None:
        add(int(number < 0))
        bits = int.bit_length(number)
        digits = max(0, bits - 1) * 30102999566398119521373889472449 // 10**32 + 1
        add(digits)
        magnitude, power = abs(number), 10**digits
        while magnitude >= power:
            add(1)
            power *= 10

    stack = [value]
    while stack:
        current = stack.pop()
        kind = type(current)
        if kind is str:
            string(cast(str, current))
        elif current is None:
            add(4)
        elif kind is bool:
            add(4 if current else 5)
        elif kind is int:
            integer(cast(int, current))
        elif kind is float:
            if not math.isfinite(cast(float, current)):
                raise ValueError('live selection requires finite values')
            add(len(repr(current)))
        elif kind in (tuple, list):
            sequence = cast(tuple[object, ...], current)
            add(2 + max(0, len(sequence) - 1))
            stack.extend(sequence)
        else:
            mapping = cast(dict[str | int, object], current if kind is dict else
                           object.__getattribute__(current, '__dict__'))
            add(2 + max(0, len(mapping) - 1) + len(mapping))
            for key, child in mapping.items():
                if type(key) is str:
                    string(key)
                else:
                    add(2)
                    integer(cast(int, key))
                stack.append(child)
    return total


def _preflight(value: object) -> tuple[object, int]:
    preflight_exact_pydantic_tree(value, root_model_types=_ROOTS,
        label='live selection', max_nodes=LIVE_SELECTION_MAX_NODES,
        max_depth=64, max_string_length=None, provenance_id_limit=None)
    size = _measure(value)
    # Exact annotation and finite checks remain those of the legacy selection.
    # This helper does not invoke the legacy node/byte policy or source replay.
    # Ordinary serialization data only, with no source/score authority. A
    # snapshot consumes it immediately; no value survives into another call.
    return legacy._plain(value), size


def _encoded(value: object) -> bytes:
    # Callers must have preflighted the whole authoritative boundary first.
    raw = canonical_json_bytes(legacy._plain(value))
    if len(raw) != _measure(value):
        raise ValueError('live selection encoding differs from byte measurement')
    return raw


def _values(sources: LiveScoringSources) -> dict[str, object]:
    if type(sources) is not LiveScoringSources:
        raise ValueError('live selection requires exact LiveScoringSources')
    return {field.name: object.__getattribute__(sources, field.name)
            for field in fields(LiveScoringSources)}


def _source_snapshot(sources: LiveScoringSources) -> tuple[bytes, bytes]:
    values = _values(sources)
    plain, size = _preflight(values)
    if any(not legacy._exact(value, _SOURCE_TYPES[name]) for name, value in values.items()):
        raise ValueError('live source fields require exact declared types')
    ids = tuple(g.guard_result_id for g in sources.guard_results)
    if ids != tuple(sorted(set(ids))):
        raise ValueError('live guards require canonical unique IDs')
    raw = canonical_json_bytes(plain)
    del plain
    if len(raw) != size:
        raise ValueError('live selection encoding differs from byte measurement')
    return raw, _representation(values)


def _snapshot(value: object) -> tuple[bytes, bytes]:
    plain, size = _preflight(value)
    raw = canonical_json_bytes(plain)
    del plain
    if len(raw) != size:
        raise ValueError('live selection encoding differs from byte measurement')
    return raw, _representation(value)


def _unchanged(value: object, before: tuple[bytes, bytes], *, source: bool = False) -> None:
    try:
        now = _source_snapshot(cast(LiveScoringSources, value)) if source else _snapshot(value)
        same = all(hmac.compare_digest(a, b) for a, b in zip(now, before, strict=True))
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
        same = False
    if not same:
        raise ValueError('live selection input content or representation changed during replay')


def _pin(pin: str) -> None:
    if type(pin) is not str or re.fullmatch(r'[0-9a-f]{64}', pin) is None:
        raise ValueError('live selection requires exact independent SHA256 pins')


def live_scoring_sources_bytes(*, sources: LiveScoringSources) -> bytes:
    before = _source_snapshot(sources)
    try:
        return before[0]
    finally:
        _unchanged(sources, before, source=True)


def live_scoring_source_sha256(*, sources: LiveScoringSources) -> str:
    return sha256_hex(live_scoring_sources_bytes(sources=sources))


def _scan_json(raw: bytes) -> None:
    """Lexical structural budget before any Pydantic callbacks or allocations.

    Same node convention as exact-tree preflight: mapping keys do not count;
    values and containers do. Parsing and canonical roundtrip still follow.
    """
    whitespace, delimiters = b' \t\r\n', b' \t\r\n,:{}[]"'
    depth = nodes = index = 0
    def node() -> None:
        nonlocal nodes
        nodes += 1
        if nodes > LIVE_SELECTION_MAX_NODES or depth > 64:
            raise ValueError('live source JSON exceeds structural budget')
    while index < len(raw):
        token = raw[index]
        if token == 34:
            index += 1
            escaped = False
            while index < len(raw):
                char = raw[index]
                index += 1
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    break
            else:
                raise ValueError('live source JSON has an unterminated string')
            following = index
            while following < len(raw) and raw[following] in whitespace:
                following += 1
            if following == len(raw) or raw[following] != 58:
                node()
        elif token in (123, 91):
            node()
            depth += 1
            index += 1
        elif token in (125, 93):
            depth -= 1
            if depth < 0:
                raise ValueError('live source JSON has invalid nesting')
            index += 1
        elif token in whitespace or token in (44, 58):
            index += 1
        else:
            node()
            while index < len(raw) and raw[index] not in delimiters:
                index += 1
    if depth:
        raise ValueError('live source JSON has invalid nesting')


def load_live_scoring_sources(raw: bytes, *, expected_source_records_sha256: str) -> LiveScoringSources:
    """Load pinned source bytes, not a cached 'already verified' conclusion."""
    _pin(expected_source_records_sha256)
    if type(raw) is not bytes or len(raw) > LIVE_SELECTION_MAX_JSON_BYTES:
        raise ValueError('live source bytes exceed the exact byte budget')
    if not hmac.compare_digest(sha256_hex(raw), expected_source_records_sha256):
        raise ValueError('live selection source pin mismatch')
    _scan_json(raw)
    # The only inherited scope is the explicit B0 callback policy; other
    # selection consumers never inherit this independent live capacity.
    with binding_capacity(LIVE_B0_CAPACITY_PROFILE):
        # Respect each existing model's declared JSON parsing policy. Forcing
        # strict=True through B0's tuple-converting validator turns nested JSON
        # arrays into Python-mode tuple errors. Exact typed-state preflight and
        # byte-for-byte roundtrip below still reject coercion or ignored fields.
        fresh = _SOURCE_ADAPTER.validate_json(raw)
    if not hmac.compare_digest(_source_snapshot(fresh)[0], raw):
        raise ValueError('live source bytes are not canonical')
    return fresh


def _guard_inputs(sources: LiveScoringSources) -> legacy.CaseGuardReplayInputs:
    envelope = sources.envelope
    return legacy.CaseGuardReplayInputs(time_contract=sources.time_contract,
        entity_registry=envelope.entity_registry, event_grouping=envelope.event_grouping,
        candidate_universe=envelope.candidate_universe, frozen_candidates=envelope.frozen_candidates,
        focus_set=sources.focus_set, semantic_context=None, card_expansion_policy=None,
        query_sessions=(), mechanism_library=envelope.mechanism_library,
        scope_interval=envelope.scope_interval, role_predicates=sources.binding.predicates,
        role_guard_results=sources.guard_results)


def _verify_sources(sources: LiveScoringSources) -> None:
    legacy._verify_sources(_guard_inputs(sources))


def _run(*, sources: LiveScoringSources, expected_source_records_sha256: str,
         expected_rule_set_sha256: str, expected_binding_sha256: str,
         selection: LiveScoringSelection | None = None, expected_selection_sha256: str | None = None,
         progress: Callable[[dict[str, Any]], None] | None = None,
         _consume: Callable[[LiveScoringSources, LiveScoringReplay], Any] | None = None) -> Any:
    for pin in (expected_source_records_sha256, expected_rule_set_sha256, expected_binding_sha256):
        _pin(pin)
    if selection is not None:
        if type(selection) is not LiveScoringSelection:
            raise ValueError('live selection requires an exact current-version receipt')
        _pin(cast(str, expected_selection_sha256))
    # Count the complete actual input plus optional receipt, without serializing
    # a second carrier containing the same full sources and predicates.
    _preflight((_values(sources), selection))
    original = _source_snapshot(sources)
    receipt_before = None if selection is None else _snapshot(selection)
    try:
        if not hmac.compare_digest(sha256_hex(original[0]), expected_source_records_sha256):
            raise ValueError('live selection independent source pin mismatch')
        if selection is not None and not hmac.compare_digest(selection.selection_sha256,
                                                            cast(str, expected_selection_sha256)):
            raise ValueError('live selection independent receipt pin mismatch')
        fresh = load_live_scoring_sources(original[0], expected_source_records_sha256=expected_source_records_sha256)
        fresh_before = _source_snapshot(fresh)
        try:
            _verify_sources(fresh)
            guards = tuple(sorted(execute_live_b0_guards(envelope=fresh.envelope, binding=fresh.binding,
                expected_rule_set_sha256=expected_rule_set_sha256, expected_binding_sha256=expected_binding_sha256,
                progress=progress), key=lambda g: g.guard_result_id))
            _preflight(guards)
            if not hmac.compare_digest(_encoded(guards), _encoded(fresh.guard_results)):
                raise ValueError('live selection guards differ from complete B0 source replay')
            eligible = tuple(g for g in guards if g.final_disposition == 'accepted')
            unsigned = dict(schema_version='nurse.v2.live-b0-scoring-selection/1',
                capacity_profile=LIVE_SELECTION_CAPACITY_PROFILE,
                opaque_case_id=fresh.time_contract.opaque_case_id,
                source_records_sha256=expected_source_records_sha256,
                rule_set_sha256=expected_rule_set_sha256, binding_sha256=expected_binding_sha256,
                candidate_freeze_sha256=fresh.envelope.frozen_candidates.freeze_sha256,
                eligible_guard_result_ids=tuple(g.guard_result_id for g in eligible),
                excluded_guard_result_ids=tuple(g.guard_result_id for g in guards if g.final_disposition != 'accepted'))
            _preflight(unsigned)
            result = LiveScoringSelection.model_validate({**unsigned, 'selection_sha256': sha256_hex(_encoded(unsigned))})
            if receipt_before is not None and not hmac.compare_digest(receipt_before[0], _encoded(result)):
                raise ValueError('live selection receipt differs from complete replay')
            consumed = dict(selection=result, eligible_guards=eligible, predicates=fresh.binding.predicates)
            _preflight(consumed)
            replay = LiveScoringReplay(result, eligible, fresh.binding.predicates)
            if _consume is None:
                return replay
            # Only the private synchronous engine can extend this invocation.
            # The normal/exceptional source checks below cover the consumer too;
            # replayed outputs get their own snapshot (they are not source aliases).
            consumed_before = _snapshot(consumed)
            try:
                return _consume(fresh, replay)
            finally:
                _unchanged(dict(selection=replay.selection, eligible_guards=replay.eligible_guards,
                                predicates=replay.predicates), consumed_before)
        finally:
            _unchanged(fresh, fresh_before, source=True)
    finally:
        try:
            _unchanged(sources, original, source=True)
        finally:
            if selection is not None and receipt_before is not None:
                _unchanged(selection, receipt_before)


def build_live_scoring_selection(*, sources: LiveScoringSources, expected_source_records_sha256: str,
        expected_rule_set_sha256: str, expected_binding_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveScoringSelection:
    return cast(LiveScoringReplay, _run(sources=sources, expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256, expected_binding_sha256=expected_binding_sha256,
        progress=progress)).selection


def require_live_scoring_selection(*, sources: LiveScoringSources, expected_source_records_sha256: str,
        expected_rule_set_sha256: str, expected_binding_sha256: str,
        selection: LiveScoringSelection, expected_selection_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveScoringReplay:
    return cast(LiveScoringReplay, _run(sources=sources, expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256, expected_binding_sha256=expected_binding_sha256,
        selection=selection, expected_selection_sha256=expected_selection_sha256, progress=progress))
