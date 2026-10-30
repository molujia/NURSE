"""Independent replay-gated neutral event and family evidence.

Integrity receipts do not establish source authority. Each public compilation or
input verification replays the complete pinned source and empty role selection.
Only the original numeric value helpers are shared with the nonempty branch.
"""

from __future__ import annotations

import hmac
import json
import math
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Literal, TypeVar, Union, cast, get_args, get_origin

from annotated_types import MaxLen
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_scoring_selection as _selection
from nurse_rcl.v2 import event_strength as _event
from nurse_rcl.v2 import family_aggregation as _family
from nurse_rcl.v2 import no_role_compilation as _roles
from nurse_rcl.v2.case_scoring_selection import CaseGuardReplayInputs, CaseScoringSelection
from nurse_rcl.v2.event_grouping import PrimarySignalFamily
from nurse_rcl.v2.event_strength import CandidateEventRoleAdjustment, EventGroupBaseStrength
from nurse_rcl.v2.family_aggregation import CandidateFamilyAggregation, UnboundEventGroupAudit
from nurse_rcl.v2.no_role_compilation import (
    NO_ROLE_POLICY_SHA256,
    NO_ROLE_POLICY_VERSION,
    NoRoleCompilationResult,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.no-role-event-strength-result/1"
] = "nurse.v2.no-role-event-strength-result/1"
NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.no-role-family-aggregation-result/1"
] = "nurse.v2.no-role-family-aggregation-result/1"
_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)
_ResultT = TypeVar("_ResultT", bound=StrictFrozenModel)


def _preflight(value: object) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(NoRoleEventStrengthResult, NoRoleFamilyAggregationResult),
        label="NoRole evidence",
        max_string_length=None,
        provenance_id_limit=None,
    )
    _selection._plain(value)
    _roles._model_capacities(value)


def _prepare(value: object) -> tuple[dict[str, object], bytes]:
    _preflight(value)
    plain = cast(dict[str, object], _selection._plain(value))
    _selection._measure_json(plain)
    return plain, _roles._representation(value)


def _json_shape(value: object, annotation: object) -> bool:
    """JSON containers may encode declared tuples/models; scalar kinds stay exact."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _json_shape(value, args[0])
    if origin is UnionType or origin is Union:
        return any(_json_shape(value, child) for child in args)
    if origin is tuple:
        return type(value) is list and all(
            _json_shape(child, args[0]) for child in cast(list[object], value)
        )
    # Annotation types are code-owned; unknown value types were rejected by the
    # full callback-free traversal before entering this recursive shape check.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if type(value) is not dict:
            return False
        fields = annotation.model_fields
        for name, child in cast(dict[str, object], value).items():
            if name not in fields or not _json_shape(child, fields[name].annotation):
                return False
            for constraint in fields[name].metadata:
                if type(constraint) is MaxLen and len(cast(tuple[object, ...], child)) > (
                    constraint.max_length
                ):
                    return False
        return True
    return _selection._exact(value, annotation)


def _safe_state(cls: type[BaseModel], value: object, info: ValidationInfo) -> object:
    if cls is not NoRoleEventStrengthResult and cls is not NoRoleFamilyAggregationResult:
        raise ValueError("NoRole evidence requires an exact result class")
    if type(value) is not dict and type(value) is not cls:
        raise ValueError("NoRole evidence requires an exact model or dictionary")
    _preflight(value)
    _selection._measure_json(_selection._plain(value))
    if type(value) is dict:
        if info.mode == "json":
            valid = _json_shape(value, cls)
        else:
            valid = all(
                name in cls.model_fields
                and _selection._exact(child, cls.model_fields[name].annotation)
                for name, child in cast(dict[str, object], value).items()
            )
        if not valid:
            raise ValueError("NoRole evidence fields require exact declared representations")
        if info.mode == "json":
            return _json_containers(value, cls)
    return value


def _json_containers(value: object, annotation: object) -> object:
    """Decode only schema-declared JSON containers, after the whole shape passed."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _json_containers(value, args[0])
    if origin is UnionType or origin is Union:
        branch = next(child for child in args if _json_shape(value, child))
        return _json_containers(value, branch)
    if origin is tuple:
        return tuple(_json_containers(child, args[0]) for child in cast(list[object], value))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {
            name: _json_containers(child, annotation.model_fields[name].annotation)
            for name, child in cast(dict[str, object], value).items()
        }
    return value


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("NoRole evidence JSON requires finite floats")
    return value


def _validate_json(
    cls: type[_ResultT],
    json_data: str | bytes | bytearray,
    *,
    strict: bool | None,
    context: object | None,
    by_alias: bool | None,
    by_name: bool | None,
) -> _ResultT:
    if cls is not NoRoleEventStrengthResult and cls is not NoRoleFamilyAggregationResult:
        raise ValueError("NoRole evidence JSON requires the exact result class")
    raw = _roles._json_bytes(json_data)
    try:
        text = raw.decode("utf-8", errors="strict")
        _roles._scan_json(raw)
        document = json.loads(
            text,
            object_pairs_hook=_roles._unique_object,
            # All numeric fields in these artifacts are floats. An integer
            # token is a different representation, even if Pydantic accepts it.
            parse_int=_roles._reject_json_number,
            parse_float=_finite_json_float,
            parse_constant=_roles._reject_json_number,
        )
        _preflight(document)
        _selection._measure_json(_selection._plain(document))
        if not _json_shape(document, cls):
            raise ValueError("NoRole evidence JSON has a noncanonical field representation")
        return super(cls, cls).model_validate_json(
            raw, strict=True, context=context, by_alias=by_alias, by_name=by_name
        )
    finally:
        if not hmac.compare_digest(raw, _roles._json_bytes(json_data)):
            raise ValueError("NoRole evidence JSON changed during validation")


class NoRoleEventStrengthResult(StrictFrozenModel):
    """Neutral values plus internal integrity, without standalone source authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.no-role-event-strength-result/1"] = (
        NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION
    )
    event_strength_result_id: str = Field(pattern=r"^nesr_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-event-strength/1"] = _event.EVENT_STRENGTH_POLICY_VERSION
    policy_sha256: _SHA
    event_grouping_result_sha256: _SHA
    role_compilation_id: str = Field(pattern=r"^nrc_[0-9a-f]{24}$")
    role_compilation_result_sha256: _SHA
    group_strengths: tuple[EventGroupBaseStrength, ...]
    candidate_event_adjustments: tuple[CandidateEventRoleAdjustment, ...]
    retained_assertion_ids: tuple[str, ...] = Field(max_length=0)
    unbound_assertion_ids: tuple[str, ...] = Field(max_length=0)
    result_sha256: _SHA
    source_role_schema_version: Literal["nurse.v2.no-role-compilation-result/1"] = (
        _roles.NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION
    )
    source_records_sha256: _SHA
    scoring_selection_sha256: _SHA
    candidate_freeze_sha256: _SHA
    no_role_policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    no_role_policy_sha256: _SHA

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
        return _validate_json(
            cls, json_data, strict=strict, context=context, by_alias=by_alias, by_name=by_name
        )

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object, info: ValidationInfo) -> object:
        return _safe_state(cls, value, info)

    @model_validator(mode="after")
    def closed_neutral_result(self) -> Self:
        if self.policy_sha256 != _event.EVENT_STRENGTH_POLICY_SHA256:
            raise ValueError("NoRole event numerical policy differs")
        if self.no_role_policy_sha256 != NO_ROLE_POLICY_SHA256:
            raise ValueError("NoRole activation policy differs")
        groups = tuple(row.event_group_id for row in self.group_strengths)
        bindings = tuple(
            (row.candidate_service_key, row.event_group_id)
            for row in self.candidate_event_adjustments
        )
        if groups != tuple(sorted(set(groups))) or bindings != tuple(sorted(set(bindings))):
            raise ValueError("NoRole event groups and bindings must be sorted and unique")
        if self.retained_assertion_ids or self.unbound_assertion_ids:
            raise ValueError("NoRole assertions must remain empty")
        for row in self.candidate_event_adjustments:
            if (
                row.contributing_assertion_ids
                or row.retained_assertion_ids
                or row.deduplicated_assertion_ids
                or row.support_strength != 0.0
                or row.refutation_strength != 0.0
                or row.role_adjustment != 0.5
            ):
                raise ValueError("NoRole event adjustments require empty neutral roles")
        return self

    def verify(self) -> bool:
        return _verify_structure(self, NoRoleEventStrengthResult)

    def verify_structure(self) -> bool:
        return NoRoleEventStrengthResult.verify(self)

    def verify_inputs(
        self,
        *,
        inputs: CaseGuardReplayInputs,
        selection: CaseScoringSelection,
        expected_source_records_sha256: str,
        expected_selection_sha256: str,
        role_compilation: NoRoleCompilationResult,
        expected_role_compilation_sha256: str,
        expected_result_sha256: str,
    ) -> bool:
        if type(self) is not NoRoleEventStrengthResult:
            return False
        try:
            original = _seal(
                inputs,
                selection,
                role_compilation,
                result=self,
                pins=(
                    expected_source_records_sha256,
                    expected_selection_sha256,
                    expected_role_compilation_sha256,
                    expected_result_sha256,
                ),
            )
            try:
                rebuilt = _build_event(
                    original,
                    expected_source_records_sha256,
                    expected_selection_sha256,
                    expected_role_compilation_sha256,
                )
                return _matches(original.result, rebuilt, expected_result_sha256)
            finally:
                _unchanged(original, _seal(inputs, selection, role_compilation, result=self))
        except _ERRORS:
            return False


class NoRoleFamilyAggregationResult(StrictFrozenModel):
    """All frozen candidates, all retained families, and complete unbound evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.no-role-family-aggregation-result/1"] = (
        NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION
    )
    family_aggregation_result_id: str = Field(pattern=r"^nfar_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-family-aggregation/1"] = (
        _family.FAMILY_AGGREGATION_POLICY_VERSION
    )
    policy_sha256: _SHA
    aggregator_version: str = Field(min_length=1, max_length=128)
    aggregator_definition_sha256: _SHA
    event_grouping_result_sha256: _SHA
    event_strength_result_id: str = Field(pattern=r"^nesr_[0-9a-f]{24}$")
    event_strength_result_sha256: _SHA
    retained_event_group_ids: tuple[str, ...]
    aggregated_event_group_ids: tuple[str, ...]
    unbound_event_group_ids: tuple[str, ...]
    unbound_event_group_audits: tuple[UnboundEventGroupAudit, ...]
    candidates: tuple[CandidateFamilyAggregation, ...]
    provenance_ids: tuple[str, ...]
    result_sha256: _SHA
    source_event_schema_version: Literal["nurse.v2.no-role-event-strength-result/1"] = (
        NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION
    )
    source_records_sha256: _SHA
    scoring_selection_sha256: _SHA
    candidate_freeze_sha256: _SHA
    no_role_policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    no_role_policy_sha256: _SHA

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
        return _validate_json(
            cls, json_data, strict=strict, context=context, by_alias=by_alias, by_name=by_name
        )

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object, info: ValidationInfo) -> object:
        return _safe_state(cls, value, info)

    @model_validator(mode="after")
    def closed_family_result(self) -> Self:
        if self.policy_sha256 != _family.FAMILY_AGGREGATION_POLICY_SHA256:
            raise ValueError("NoRole family numerical policy differs")
        if self.no_role_policy_sha256 != NO_ROLE_POLICY_SHA256:
            raise ValueError("NoRole activation policy differs")
        _aggregator_key(self.aggregator_version)
        definition = _family.resolve_declared_family_aggregator(self.aggregator_version)
        if self.aggregator_definition_sha256 != definition.definition_sha256:
            raise ValueError("NoRole family aggregator definition differs")
        keys = tuple(row.candidate_service_key for row in self.candidates)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("NoRole candidates must be nonempty, sorted and unique")
        for values in (
            self.retained_event_group_ids,
            self.aggregated_event_group_ids,
            self.unbound_event_group_ids,
            self.provenance_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("NoRole family references must be sorted and unique")
        aggregated, unbound = (
            set(self.aggregated_event_group_ids),
            set(self.unbound_event_group_ids),
        )
        if aggregated & unbound or aggregated | unbound != set(self.retained_event_group_ids):
            raise ValueError("NoRole event categories must partition all retained events")
        audited = tuple(row.event_group_id for row in self.unbound_event_group_audits)
        if audited != tuple(sorted(set(audited))) or set(audited) != unbound:
            raise ValueError("NoRole unbound audits must exactly cover unbound events")
        candidate_events = {
            event
            for candidate in self.candidates
            for family in candidate.family_maxima
            for event in family.retained_event_group_ids
        }
        if candidate_events != aggregated:
            raise ValueError("NoRole candidate family audits must cover aggregated events")
        if any(
            row.aggregator_version != self.aggregator_version
            or row.aggregator_definition_sha256 != self.aggregator_definition_sha256
            for row in self.candidates
        ):
            raise ValueError("NoRole candidate aggregator declarations differ")
        expected_provenance = {value for row in self.candidates for value in row.provenance_ids} | {
            value for row in self.unbound_event_group_audits for value in row.provenance_ids
        }
        if set(self.provenance_ids) != expected_provenance:
            raise ValueError("NoRole family provenance must cover every retained event")
        return self

    def verify(self) -> bool:
        return _verify_structure(self, NoRoleFamilyAggregationResult)

    def verify_structure(self) -> bool:
        return NoRoleFamilyAggregationResult.verify(self)

    def verify_inputs(
        self,
        *,
        inputs: CaseGuardReplayInputs,
        selection: CaseScoringSelection,
        expected_source_records_sha256: str,
        expected_selection_sha256: str,
        role_compilation: NoRoleCompilationResult,
        expected_role_compilation_sha256: str,
        event_strength: NoRoleEventStrengthResult,
        expected_event_strength_sha256: str,
        aggregator_version: str,
        expected_result_sha256: str,
    ) -> bool:
        if type(self) is not NoRoleFamilyAggregationResult:
            return False
        try:
            _aggregator_key(aggregator_version)
            original = _seal(
                inputs,
                selection,
                role_compilation,
                event_strength,
                self,
                (
                    expected_source_records_sha256,
                    expected_selection_sha256,
                    expected_role_compilation_sha256,
                    expected_event_strength_sha256,
                    expected_result_sha256,
                ),
            )
            try:
                rebuilt = _build_family(
                    original,
                    expected_source_records_sha256,
                    expected_selection_sha256,
                    expected_role_compilation_sha256,
                    expected_event_strength_sha256,
                    aggregator_version,
                )
                return _matches(original.result, rebuilt, expected_result_sha256)
            finally:
                _unchanged(
                    original, _seal(inputs, selection, role_compilation, event_strength, self)
                )
        except _ERRORS:
            return False


def _valid_result(plain: dict[str, object], cls: type[_ResultT]) -> bool:
    encoded = _roles._encoded(plain)
    rebuilt = cls.model_validate_json(encoded)
    restored = _prepare(rebuilt)[0]
    if type(rebuilt) is NoRoleEventStrengthResult:
        event = rebuilt
        if any(not row.verify() for row in event.group_strengths):
            return False
        bases = {row.strength_id: row for row in event.group_strengths}
        for row in event.candidate_event_adjustments:
            base = bases.get(row.base_strength_id)
            if base is None or base.event_group_id != row.event_group_id:
                return False
            expected = _event._build_candidate_event_adjustment(
                candidate_service_key=row.candidate_service_key, base_strength=base, assertions=()
            )
            if _roles._encoded(_prepare(row)[0]) != _roles._encoded(_prepare(expected)[0]):
                return False
        id_field, prefix = "event_strength_result_id", "nesr_"
    else:
        family = cast(NoRoleFamilyAggregationResult, rebuilt)
        if any(not row.verify() for row in family.candidates) or any(
            not row.verify() for row in family.unbound_event_group_audits
        ):
            return False
        id_field, prefix = "family_aggregation_result_id", "nfar_"
    digest = sha256_hex(
        _roles._encoded(
            {
                key: value
                for key, value in restored.items()
                if key not in (id_field, "result_sha256")
            }
        )
    )
    return (
        hmac.compare_digest(encoded, _roles._encoded(restored))
        and restored["result_sha256"] == digest
        and restored[id_field] == prefix + digest[:24]
    )


def _verify_structure(result: object, cls: type[_ResultT]) -> bool:
    try:
        if type(result) is not cls:
            return False
        original, representation = _prepare(result)
        try:
            return _valid_result(original, cls)
        finally:
            current, current_representation = _prepare(result)
            if original != current or representation != current_representation:
                raise ValueError("NoRole evidence changed during structural verification")
    except _ERRORS:
        return False


@dataclass(frozen=True, slots=True)
class _Seal:
    roles: _roles._Seal
    event: dict[str, object] | None
    result: dict[str, object] | None
    representations: tuple[bytes | None, bytes | None]


def _seal(
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    role_compilation: NoRoleCompilationResult,
    event_strength: NoRoleEventStrengthResult | None = None,
    result: NoRoleEventStrengthResult | NoRoleFamilyAggregationResult | None = None,
    pins: tuple[str, ...] = (),
) -> _Seal:
    if type(role_compilation) is not NoRoleCompilationResult:
        raise ValueError("NoRole evidence requires an exact NoRole compilation")
    if event_strength is not None and type(event_strength) is not NoRoleEventStrengthResult:
        raise ValueError("NoRole family requires an exact NoRole event result")
    if result is not None:
        expected = (
            NoRoleEventStrengthResult if event_strength is None else NoRoleFamilyAggregationResult
        )
        if type(result) is not expected:
            raise ValueError("NoRole verification requires the exact branch result")
    roles = _roles._seal(inputs, selection, role_compilation, pins)
    event, event_repr = (None, None) if event_strength is None else _prepare(event_strength)
    result_plain, result_repr = (None, None) if result is None else _prepare(result)
    return _Seal(roles, event, result_plain, (event_repr, result_repr))


def _unchanged(original: _Seal, current: _Seal) -> None:
    _roles._unchanged(original.roles, current.roles)
    if (
        original.event != current.event
        or original.result != current.result
        or original.representations != current.representations
    ):
        raise ValueError("NoRole original evidence content or representation changed during replay")


def _aggregator_key(version: str) -> None:
    if type(version) is not str or version != _family.REFERENCE_KEY:
        raise ValueError("NoRole requires the declared reference family aggregator version key")


def _matches(plain: dict[str, object] | None, rebuilt: StrictFrozenModel, pin: str) -> bool:
    return (
        plain is not None
        and plain["result_sha256"] == pin
        and hmac.compare_digest(_roles._encoded(plain), _roles._encoded(_prepare(rebuilt)[0]))
    )


def _replay_roles(
    original: _Seal, source_pin: str, selection_pin: str, role_pin: str
) -> tuple[CaseGuardReplayInputs, NoRoleCompilationResult]:
    if original.roles.result is None or original.roles.result["result_sha256"] != role_pin:
        raise ValueError("NoRole fails independent role compilation pin")
    rebuilt = _roles._build(original.roles, source_pin, selection_pin)
    if not hmac.compare_digest(
        _roles._encoded(original.roles.result), _roles._encoded(_roles._prepare(rebuilt)[0])
    ):
        raise ValueError("NoRole compilation differs from full original source replay")
    return _selection._snapshot(_roles._encoded(original.roles.source)), rebuilt


def _lineage(roles: NoRoleCompilationResult) -> dict[str, object]:
    return {
        "opaque_case_id": roles.opaque_case_id,
        "source_records_sha256": roles.source_records_sha256,
        "scoring_selection_sha256": roles.scoring_selection_sha256,
        "candidate_freeze_sha256": roles.candidate_freeze_sha256,
        "no_role_policy_version": NO_ROLE_POLICY_VERSION,
        "no_role_policy_sha256": NO_ROLE_POLICY_SHA256,
    }


def _finish(
    cls: type[_ResultT], unsigned: dict[str, object], id_field: str, prefix: str
) -> _ResultT:
    digest = sha256_hex(_roles._encoded(cast(dict[str, object], _selection._plain(unsigned))))
    return cls.model_validate({**unsigned, id_field: prefix + digest[:24], "result_sha256": digest})


def _event_values(
    inputs: CaseGuardReplayInputs, roles: NoRoleCompilationResult
) -> NoRoleEventStrengthResult:
    grouping = inputs.event_grouping
    bases = _event.compile_event_group_base_strengths(event_grouping=grouping)
    base_by_event = {row.event_group_id: row for row in bases}
    legal_bindings = sorted(
        {
            (candidate.service_entity_key, event)
            for candidate in inputs.candidate_universe.candidates
            for event in candidate.event_group_ids
        }
    )
    adjustments = tuple(
        _event._build_candidate_event_adjustment(
            candidate_service_key=key, base_strength=base_by_event[event], assertions=()
        )
        for key, event in legal_bindings
    )
    return _finish(
        NoRoleEventStrengthResult,
        {
            **_lineage(roles),
            "schema_version": NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
            "policy_version": _event.EVENT_STRENGTH_POLICY_VERSION,
            "policy_sha256": _event.EVENT_STRENGTH_POLICY_SHA256,
            "event_grouping_result_sha256": grouping.result_sha256,
            "role_compilation_id": roles.role_compilation_id,
            "role_compilation_result_sha256": roles.result_sha256,
            "source_role_schema_version": _roles.NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
            "group_strengths": bases,
            "candidate_event_adjustments": adjustments,
            "retained_assertion_ids": (),
            "unbound_assertion_ids": (),
        },
        "event_strength_result_id",
        "nesr_",
    )


def _build_event(
    original: _Seal, source_pin: str, selection_pin: str, role_pin: str
) -> NoRoleEventStrengthResult:
    inputs, roles = _replay_roles(original, source_pin, selection_pin, role_pin)
    return _event_values(inputs, roles)


def _build_family(
    original: _Seal,
    source_pin: str,
    selection_pin: str,
    role_pin: str,
    event_pin: str,
    aggregator_version: str,
) -> NoRoleFamilyAggregationResult:
    inputs, roles = _replay_roles(original, source_pin, selection_pin, role_pin)
    event = _event_values(inputs, roles)
    if not _matches(original.event, event, event_pin):
        raise ValueError("NoRole event fails independent pin or complete source replay")
    definition = _family.resolve_declared_family_aggregator(aggregator_version)
    implementation = _family._AGGREGATOR_IMPLEMENTATIONS[definition.aggregator_version]
    groups = {row.event_group_id: row for row in inputs.event_grouping.event_groups}
    bases = {row.event_group_id: row for row in event.group_strengths}
    by_candidate: dict[str, list[tuple[PrimarySignalFamily, _family._EventValue]]] = {}
    for row in event.candidate_event_adjustments:
        group, base = groups[row.event_group_id], bases[row.event_group_id]
        by_candidate.setdefault(row.candidate_service_key, []).append(
            (
                group.primary_signal_family,
                _family._EventValue(
                    event_group_id=row.event_group_id,
                    value=row.adjusted_strength,
                    provenance_ids=tuple(
                        sorted(
                            set(group.provenance_ids)
                            | set(base.provenance_ids)
                            | set(row.provenance_ids)
                        )
                    ),
                ),
            )
        )
    candidates = tuple(
        _family._build_candidate_aggregation(
            candidate_service_key=key,
            events=by_candidate.get(key, ()),
            implementation=implementation,
        )
        for key in inputs.frozen_candidates.candidate_service_keys
    )
    retained = tuple(sorted(bases))
    aggregated = tuple(sorted({row.event_group_id for row in event.candidate_event_adjustments}))
    unbound = tuple(sorted(set(retained) - set(aggregated)))
    audits = tuple(
        _family._build_unbound_event_group_audit(
            event_group_id=key,
            primary_signal_family=groups[key].primary_signal_family,
            base_strength_id=bases[key].strength_id,
            source_fingerprints=tuple(
                sorted(set(groups[key].source_fingerprints) | set(bases[key].source_fingerprints))
            ),
            provenance_ids=tuple(
                sorted(set(groups[key].provenance_ids) | set(bases[key].provenance_ids))
            ),
        )
        for key in unbound
    )
    return _finish(
        NoRoleFamilyAggregationResult,
        {
            **_lineage(roles),
            "schema_version": NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
            "policy_version": _family.FAMILY_AGGREGATION_POLICY_VERSION,
            "policy_sha256": _family.FAMILY_AGGREGATION_POLICY_SHA256,
            "aggregator_version": definition.aggregator_version,
            "aggregator_definition_sha256": definition.definition_sha256,
            "event_grouping_result_sha256": inputs.event_grouping.result_sha256,
            "event_strength_result_id": event.event_strength_result_id,
            "event_strength_result_sha256": event.result_sha256,
            "source_event_schema_version": NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
            "retained_event_group_ids": retained,
            "aggregated_event_group_ids": aggregated,
            "unbound_event_group_ids": unbound,
            "unbound_event_group_audits": audits,
            "candidates": candidates,
            "provenance_ids": tuple(
                sorted(
                    {value for row in candidates for value in row.provenance_ids}
                    | {value for row in audits for value in row.provenance_ids}
                )
            ),
        },
        "family_aggregation_result_id",
        "nfar_",
    )


def compile_no_role_event_strengths(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
) -> NoRoleEventStrengthResult:
    """Replay original sources and the independent N1 pin before neutral values."""
    original = _seal(
        inputs,
        selection,
        role_compilation,
        pins=(
            expected_source_records_sha256,
            expected_selection_sha256,
            expected_role_compilation_sha256,
        ),
    )
    try:
        return _build_event(
            original,
            expected_source_records_sha256,
            expected_selection_sha256,
            expected_role_compilation_sha256,
        )
    finally:
        _unchanged(original, _seal(inputs, selection, role_compilation))


def compile_no_role_family_aggregation(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
    event_strength: NoRoleEventStrengthResult,
    expected_event_strength_sha256: str,
    aggregator_version: str,
) -> NoRoleFamilyAggregationResult:
    """Replay N1 and neutral events before resolving the declared family helper."""
    if type(event_strength) is not NoRoleEventStrengthResult:
        raise ValueError("NoRole family requires an exact NoRole event result")
    _aggregator_key(aggregator_version)
    original = _seal(
        inputs,
        selection,
        role_compilation,
        event_strength,
        pins=(
            expected_source_records_sha256,
            expected_selection_sha256,
            expected_role_compilation_sha256,
            expected_event_strength_sha256,
        ),
    )
    try:
        return _build_family(
            original,
            expected_source_records_sha256,
            expected_selection_sha256,
            expected_role_compilation_sha256,
            expected_event_strength_sha256,
            aggregator_version,
        )
    finally:
        _unchanged(original, _seal(inputs, selection, role_compilation, event_strength))
