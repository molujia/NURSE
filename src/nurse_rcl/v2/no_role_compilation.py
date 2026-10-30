"""Independent compilation receipt for a completely replayed empty role selection."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal, NoReturn, cast, get_origin

from annotated_types import MaxLen
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_scoring_selection as _selection
from nurse_rcl.v2.case_scoring_selection import (
    CaseGuardReplayInputs,
    CaseScoringSelection,
    require_case_scoring_selection,
)
from nurse_rcl.v2.diagnostic_schemas import RoleAssertion
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.no-role-compilation-result/1"] = (
    "nurse.v2.no-role-compilation-result/1"
)
NO_ROLE_POLICY_VERSION: Literal["nurse-v2-no-eligible-role-replay/1"] = (
    "nurse-v2-no-eligible-role-replay/1"
)
_POLICY_DOCUMENT = {
    "version": NO_ROLE_POLICY_VERSION,
    "activation": "complete-source-replay-and-empty-eligible-selection",
    "roles": "empty-guards-proposals-assertions",
    "event": "original-base-times-neutral-0.5-preserve-unknown",
    "propagation": "original-empty-role-bundle-aggregation",
    "candidates": "complete-frozen-C",
    "rules": "independent-original-replay",
    "history": "off",
    "ranking": "original-binary64-quantization-ties-top-k",
}
NO_ROLE_POLICY_SHA256 = sha256_hex(canonical_json_bytes(_POLICY_DOCUMENT))
_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)


def _json_bytes(value: str | bytes | bytearray) -> bytes:
    kind = type(value)
    if kind is not str and kind is not bytes and kind is not bytearray:
        raise ValueError("NoRole JSON requires exact text or bytes")
    if len(value) > _selection._MAX_BYTES:
        raise ValueError("NoRole JSON exceeds byte budget")
    if kind is str:
        chunks: list[bytes] = []
        size = 0
        for start in range(0, len(value), 4096):
            chunk = cast(str, value)[start : start + 4096].encode("utf-8", errors="strict")
            size += len(chunk)
            if size > _selection._MAX_BYTES:
                raise ValueError("NoRole JSON exceeds byte budget")
            chunks.append(chunk)
        return b"".join(chunks)
    return bytes(cast(bytes | bytearray, value))


def _scan_json(raw: bytes) -> None:
    """Use the audit scanner's lexical mechanism, with NoRole's own budgets."""
    whitespace, delimiters = b" \t\r\n", b' \t\r\n,:{}[]"'
    depth = nodes = index = 0

    def node() -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 500_000 or depth > 64:
            raise ValueError("NoRole JSON exceeds structural budget")

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
                raise ValueError("NoRole JSON has an unterminated string")
            following = index
            while following < len(raw) and raw[following] in whitespace:
                following += 1
            if following == len(raw) or raw[following] != 58:
                node()
        elif token == 123 or token == 91:
            node()
            depth += 1
            index += 1
        elif token == 125 or token == 93:
            depth -= 1
            if depth < 0:
                raise ValueError("NoRole JSON has invalid nesting")
            index += 1
        elif token in whitespace or token == 44 or token == 58:
            index += 1
        else:
            node()
            while index < len(raw) and raw[index] not in delimiters:
                index += 1
    if depth:
        raise ValueError("NoRole JSON has invalid nesting")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("NoRole JSON contains a duplicate key")
        value[key] = child
    return value


def _reject_json_number(token: str) -> NoReturn:
    # This receipt has no numeric fields. Reject before Python attempts a
    # potentially expensive decimal conversion, including on Python 3.10.2.
    raise ValueError("NoRole JSON does not permit number tokens")


def _preflight(value: object) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(*_selection._ROOTS, CaseScoringSelection, NoRoleCompilationResult),
        label="NoRole compilation",
        max_string_length=None,
        provenance_id_limit=None,
    )
    # Exact field annotations are checked only after the complete callback-free
    # traversal has rejected unknown runtime types, hidden state and cycles.
    _selection._plain(value)
    _model_capacities(value)


def _model_capacities(value: object) -> None:
    """Preserve source schemas' narrower capacities before any source callback."""
    stack = [value]
    while stack:
        current = stack.pop()
        kind = type(current)
        if kind is tuple or kind is list:
            stack.extend(cast(tuple[object, ...], current))
        elif kind is dict:
            stack.extend(cast(dict[str, object], current).values())
        elif kind is str or kind is int or kind is float or kind is bool or current is None:
            continue
        else:
            state = cast(dict[str, object], object.__getattribute__(current, "__dict__"))
            for name, field in cast(type[BaseModel], kind).model_fields.items():
                child = state[name]
                for constraint in field.metadata:
                    if (
                        type(constraint) is MaxLen
                        and len(cast(tuple[object, ...], child)) > constraint.max_length
                    ):
                        raise ValueError("NoRole source exceeds declared field capacity")
            if kind is CaseQuerySessionResult:
                preflight_exact_pydantic_tree(
                    current,
                    root_model_types=(CaseQuerySessionResult,),
                    label="NoRole query session",
                    max_nodes=20_000,
                    max_string_length=None,
                    provenance_id_limit=None,
                )
            stack.extend(state.values())


def _representation(value: object) -> bytes:
    """Fingerprint exact container/scalar kinds and Pydantic field markers."""
    digest = hashlib.sha256()

    def visit(current: object) -> None:
        kind = type(current)
        if kind is tuple or kind is list:
            digest.update(b"tuple[" if kind is tuple else b"list[")
            for child in cast(tuple[object, ...], current):
                visit(child)
            digest.update(b"]")
        elif kind is dict:
            digest.update(b"dict{")
            for key, child in sorted(cast(dict[str, object], current).items()):
                digest.update(key.encode("utf-8") + b"\0")
                visit(child)
            digest.update(b"}")
        elif kind is float:
            # Plain equality equates -0.0 and +0.0; canonical JSON does not.
            digest.update(b"float:" + repr(current).encode("ascii") + b";")
        elif kind is str or kind is int or kind is bool or current is None:
            digest.update(kind.__name__.encode("ascii") + b";")
        else:
            # All model classes here are code-owned and preflighted by identity.
            digest.update(kind.__name__.encode("ascii") + b"{")
            markers = object.__getattribute__(current, "__pydantic_fields_set__")
            for marker in sorted(markers):
                digest.update(marker.encode("utf-8") + b"\0")
            visit(object.__getattribute__(current, "__dict__"))
            digest.update(b"}")

    visit(value)
    return digest.digest()


def _prepare(value: object) -> tuple[dict[str, object], bytes]:
    _preflight(value)
    plain = cast(dict[str, object], _selection._plain(value))
    _selection._measure_json(plain)
    return plain, _representation(value)


def _encoded(value: dict[str, object]) -> bytes:
    return _selection._bytes(value)


@dataclass(frozen=True, slots=True)
class _Seal:
    source: dict[str, object]
    selection: dict[str, object]
    result: dict[str, object] | None
    representations: tuple[bytes, ...]


def _seal(
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    result: NoRoleCompilationResult | None = None,
    pins: tuple[str, ...] = (),
) -> _Seal:
    """Preflight every root and its complete JSON capacity before serialization."""
    if type(inputs) is not CaseGuardReplayInputs or type(selection) is not CaseScoringSelection:
        raise ValueError("NoRole requires exact input and selection roots")
    if result is not None and type(result) is not NoRoleCompilationResult:
        raise ValueError("NoRole requires an exact result root")
    if any(type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None for pin in pins):
        raise ValueError("NoRole requires exact lowercase SHA256 pins")
    values = _selection._input_values(inputs)
    _preflight(values)
    _preflight(selection)
    if result is not None:
        _preflight(result)
    if any(
        not _selection._exact(value, _selection._INPUT_TYPES[key]) for key, value in values.items()
    ):
        raise ValueError("NoRole input field has a non-canonical exact type")
    for predicate in inputs.role_predicates:
        pending = [predicate]
        records = 1
        while pending:
            operands = pending.pop().operand_states
            records += len(operands)
            if records > 4096:
                raise ValueError("NoRole predicate tree exceeds its record capacity")
            pending.extend(operands)
    source = {
        "schema_version": _selection._SOURCE_SCHEMA,
        **cast(dict[str, object], _selection._plain(values)),
    }
    receipt = cast(dict[str, object], _selection._plain(selection))
    result_plain = None if result is None else cast(dict[str, object], _selection._plain(result))
    # The source, selection and result are independent artifacts. An auxiliary
    # tuple must not combine their separate 64 MiB byte budgets.
    for artifact in (source, receipt, result_plain):
        if artifact is not None:
            _selection._measure_json(artifact)
    for sequence, key in (
        (inputs.query_sessions, "sequence_number"),
        (inputs.role_predicates, "predicate_result_id"),
        (inputs.role_guard_results, "guard_result_id"),
    ):
        ids = tuple(object.__getattribute__(row, key) for row in sequence)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("NoRole source records require canonical unique IDs")
    return _Seal(
        source,
        receipt,
        result_plain,
        (_representation(values), _representation(selection), _representation(result)),
    )


def _unchanged(original: _Seal, current: _Seal) -> None:
    # These detached snapshots contain only exact, preflighted JSON values.
    # Equality plus the scalar/container representation fingerprint is canonical
    # byte equality, including signed zero, without reopening a serializer
    # callback window after the final live-input snapshot.
    if original.representations != current.representations:
        raise ValueError("NoRole original field representations changed during replay")
    for before, after in (
        (original.source, current.source),
        (original.selection, current.selection),
        (original.result, current.result),
    ):
        if before is None or after is None:
            if before is not after:
                raise ValueError("NoRole original result changed during replay")
        elif before != after:
            raise ValueError("NoRole original bytes changed during replay")


class NoRoleCompilationResult(StrictFrozenModel):
    """Internal integrity is distinct from independently pinned source authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.no-role-compilation-result/1"] = (
        NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION
    )
    role_compilation_id: str = Field(pattern=r"^nrc_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    policy_sha256: _SHA
    guard_result_ids: tuple[str, ...] = Field(max_length=0)
    proposals: tuple[RoleBindingProposal, ...] = Field(max_length=0)
    assertions: tuple[RoleAssertion, ...] = Field(max_length=0)
    result_sha256: _SHA
    source_records_sha256: _SHA
    scoring_selection_sha256: _SHA
    candidate_freeze_sha256: _SHA
    no_role_policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    no_role_policy_sha256: _SHA
    absence_reason: Literal["no_guard_records", "all_guards_rejected"]

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if cls is not NoRoleCompilationResult:
            raise ValueError("NoRole JSON requires the exact result class")
        raw = _json_bytes(json_data)
        try:
            text = raw.decode("utf-8", errors="strict")
            _scan_json(raw)
            document = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_int=_reject_json_number,
                parse_float=_reject_json_number,
                parse_constant=_reject_json_number,
            )
            _preflight(document)
            _selection._measure_json(_selection._plain(document))
            return super().model_validate_json(
                raw,
                strict=strict,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        finally:
            if not hmac.compare_digest(raw, _json_bytes(json_data)):
                raise ValueError("NoRole JSON changed during validation")

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        if type(value) is not dict and type(value) is not NoRoleCompilationResult:
            raise ValueError("NoRole requires an exact model or dictionary")
        _preflight(value)
        _selection._measure_json(_selection._plain(value))
        return value

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        annotation = cls.model_fields[cast(str, info.field_name)].annotation
        if info.mode == "json" and get_origin(annotation) is tuple and type(value) is list:
            value = tuple(cast(list[object], value))
        if not _selection._exact(value, annotation):
            raise ValueError("NoRole fields require exact declared types")
        return value

    @model_validator(mode="after")
    def fixed_policy_and_empty_roles(self) -> Self:
        if (
            self.policy_sha256 != NO_ROLE_POLICY_SHA256
            or self.no_role_policy_sha256 != NO_ROLE_POLICY_SHA256
        ):
            raise ValueError("NoRole policy digest differs")
        if self.guard_result_ids or self.proposals or self.assertions:
            raise ValueError("NoRole roles must be strictly empty")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not NoRoleCompilationResult:
                return False
            original, representation = _prepare(self)
            try:
                return _valid_result(original)
            finally:
                current, current_representation = _prepare(self)
                if representation != current_representation or original != current:
                    raise ValueError("NoRole result changed during verification")
        except _ERRORS:
            return False

    def verify_structure(self) -> bool:
        return NoRoleCompilationResult.verify(self)

    def verify_inputs(
        self,
        *,
        inputs: CaseGuardReplayInputs,
        selection: CaseScoringSelection,
        expected_source_records_sha256: str,
        expected_selection_sha256: str,
        expected_result_sha256: str,
    ) -> bool:
        try:
            original = _seal(
                inputs,
                selection,
                self,
                (
                    expected_source_records_sha256,
                    expected_selection_sha256,
                    expected_result_sha256,
                ),
            )
            try:
                if (
                    original.result is None
                    or not hmac.compare_digest(
                        cast(str, original.result["result_sha256"]), expected_result_sha256
                    )
                    or not _valid_result(original.result)
                ):
                    return False
                rebuilt = _build(
                    original, expected_source_records_sha256, expected_selection_sha256
                )
                return hmac.compare_digest(
                    _encoded(original.result), _encoded(_prepare(rebuilt)[0])
                )
            finally:
                _unchanged(original, _seal(inputs, selection, self))
        except _ERRORS:
            return False


def _valid_result(plain: dict[str, object]) -> bool:
    encoded = _encoded(plain)
    rebuilt = NoRoleCompilationResult.model_validate_json(encoded)
    reconstructed = _prepare(rebuilt)[0]
    unsigned = {
        key: value
        for key, value in reconstructed.items()
        if key not in {"role_compilation_id", "result_sha256"}
    }
    digest = sha256_hex(_encoded(unsigned))
    return (
        hmac.compare_digest(encoded, _encoded(reconstructed))
        and rebuilt.result_sha256 == digest
        and rebuilt.role_compilation_id == "nrc_" + digest[:24]
    )


def _build(original: _Seal, source_pin: str, selection_pin: str) -> NoRoleCompilationResult:
    source_bytes = _encoded(original.source)
    if not hmac.compare_digest(sha256_hex(source_bytes), source_pin) or not hmac.compare_digest(
        cast(str, original.selection["selection_sha256"]), selection_pin
    ):
        raise ValueError("NoRole fails independent source or selection pin")
    inputs = _selection._snapshot(source_bytes)
    selection = CaseScoringSelection.model_validate_json(_encoded(original.selection))
    replay = require_case_scoring_selection(
        inputs=inputs,
        selection=selection,
        expected_source_records_sha256=source_pin,
        expected_selection_sha256=selection_pin,
    )
    if (
        replay.selection.eligible_guard_result_ids
        or replay.selection.guard_query_session_bindings
        or replay.eligible_guards
        or replay.guard_query_session_bindings
    ):
        raise ValueError("NoRole requires an empty eligible guard selection")
    if not inputs.frozen_candidates.candidate_service_keys:
        raise ValueError("NoRole requires a nonempty complete frozen candidate universe")
    unsigned: dict[str, object] = {
        "schema_version": NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
        "opaque_case_id": replay.selection.opaque_case_id,
        "policy_version": NO_ROLE_POLICY_VERSION,
        "policy_sha256": NO_ROLE_POLICY_SHA256,
        "guard_result_ids": (),
        "proposals": (),
        "assertions": (),
        "source_records_sha256": source_pin,
        "scoring_selection_sha256": selection_pin,
        "candidate_freeze_sha256": replay.selection.candidate_freeze_sha256,
        "no_role_policy_version": NO_ROLE_POLICY_VERSION,
        "no_role_policy_sha256": NO_ROLE_POLICY_SHA256,
        "absence_reason": "all_guards_rejected"
        if inputs.role_guard_results
        else "no_guard_records",
    }
    digest = sha256_hex(_encoded(unsigned))
    return NoRoleCompilationResult.model_validate(
        {
            **unsigned,
            "role_compilation_id": "nrc_" + digest[:24],
            "result_sha256": digest,
        }
    )


def build_no_role_compilation(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
) -> NoRoleCompilationResult:
    """Replay all original sources and independently pinned selection before activation."""
    original = _seal(
        inputs, selection, pins=(expected_source_records_sha256, expected_selection_sha256)
    )
    try:
        return _build(original, expected_source_records_sha256, expected_selection_sha256)
    finally:
        _unchanged(original, _seal(inputs, selection))
