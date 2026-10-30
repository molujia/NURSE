"""Private, replayable B0 rule binding over original observations only.

This receipt evaluates the complete frozen C x mechanism seeds using the existing
predicate evaluator and guard evidence contracts. It grants no guard decision,
role, ranking, query, or network authority and is not an inference payload.
Nonempty query sessions do not themselves imply LLM use; this component consumes
no derived observation IDs or query results. Complete B1 gating and derived-query
binding are separate work. Seed unknown reasons describe proposal-time knowledge;
current truth remains in the evaluated predicates and subsequent guard decisions.
"""

from __future__ import annotations

import hmac
import math
import re
from types import UnionType
from typing import Annotated, Literal, TypeAlias, Union, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import predicate_plan_evaluator
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateResult,
    build_diagnostic_hypothesis,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.live_capacity import (
    LIVE_B0_MAX_DEPTH,
    LIVE_B0_MAX_STRING_LENGTH,
    active_binding_capacity,
    active_binding_limits,
    binding_capacity,
    require_live_json_budget,
)
from nurse_rcl.v2.predicate_logic import strong_kleene_and, strong_kleene_or
from nurse_rcl.v2.predicate_plan_evaluator import (
    PREDICATE_PLAN_EVALUATION_POLICY_VERSION,
)
from nurse_rcl.v2.rule_hypotheses import RULE_GENERATOR_VERSION, RuleHypothesisEnvelope
from nurse_rcl.v2.safe_serialization import (
    DEFAULT_SERIALIZATION_MAX_NODES,
    preflight_exact_pydantic_tree,
)
from nurse_rcl.v2.semantic_proposals import PredicatePlan

B0_RULE_BINDING_SCHEMA_VERSION: Literal["nurse.v2.b0-rule-binding/1"] = "nurse.v2.b0-rule-binding/1"
B0_RULE_BINDING_POLICY_VERSION: Literal["b0-original-observation-rule-binding/1"] = (
    "b0-original-observation-rule-binding/1"
)
_SHA: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_PredicateId: TypeAlias = Annotated[str, Field(pattern=r"^prd_[0-9a-f]{24}$")]
_HypothesisId: TypeAlias = Annotated[str, Field(pattern=r"^hyp_[0-9a-f]{24}$")]
_COLLECTIONS = ("predicate_id_mapping", "hypothesis_id_mapping", "predicates", "hypotheses")


def _exact_field_type(value: object, annotation: object) -> bool:
    """Check code-owned annotations after callback-free whole-tree preflight."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _exact_field_type(value, arguments[0])
    if origin is UnionType or origin is Union:
        return any(_exact_field_type(value, item) for item in arguments)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in arguments)
    if origin is tuple:
        if type(value) is not tuple:
            return False
        items = cast(tuple[object, ...], value)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_exact_field_type(item, arguments[0]) for item in items)
        return len(items) == len(arguments) and all(
            _exact_field_type(item, expected)
            for item, expected in zip(items, arguments, strict=True)
        )
    return type(value) is annotation


def _plain(value: object) -> object:
    """Read only preflighted exact builtin/model state; never invoke model serializers."""
    value_type = type(value)
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("B0 rule binding requires finite values")
        return value
    if value_type in (str, int, bool, type(None)):
        return value
    if value_type in (list, tuple):
        return [_plain(child) for child in cast(tuple[object, ...], value)]
    if value_type is dict:
        return {key: _plain(child) for key, child in cast(dict[str, object], value).items()}
    state = cast(dict[str, object], object.__getattribute__(value, "__dict__"))
    fields = cast(type[BaseModel], value_type).model_fields
    if any(not _exact_field_type(child, fields[key].annotation) for key, child in state.items()):
        raise ValueError("B0 rule binding model field has a non-canonical exact type")
    return {key: _plain(child) for key, child in state.items()}


def _preflight_tree(value: object, roots: tuple[type[BaseModel], ...]) -> None:
    limits = active_binding_limits()
    live = limits is not None
    preflight_exact_pydantic_tree(
        value,
        root_model_types=roots,
        label="B0 rule binding",
        max_nodes=limits[0] if limits is not None else DEFAULT_SERIALIZATION_MAX_NODES,
        max_depth=LIVE_B0_MAX_DEPTH,
        max_string_length=LIVE_B0_MAX_STRING_LENGTH if live else None,
        provenance_id_limit=None,
    )
    if live:
        require_live_json_budget(value)
    # The shared preflight rejects subclasses, hidden fields, cycles and resource
    # excess before this callback-free field-type and finite-value scan traverses.
    _plain(value)


def _bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(_plain(value))


class B0RuleBinding(BaseModel):
    """Private source-pinned receipt; verify() establishes internal integrity only.

    Mapping keys are complete unique source IDs after authoritative replay. Multiple
    seeds may evaluate to the same predicate, including empty-observation seeds;
    the output predicate collection stores each complete evaluated value once.
    Only require_b0_rule_binding() establishes the external source relationship.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    schema_version: Literal["nurse.v2.b0-rule-binding/1"] = B0_RULE_BINDING_SCHEMA_VERSION
    policy_version: Literal["b0-original-observation-rule-binding/1"] = (
        B0_RULE_BINDING_POLICY_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_rule_set_sha256: _SHA
    predicate_evaluation_policy_version: Literal["original-observation-predicate-plan/1"] = (
        "original-observation-predicate-plan/1"
    )
    predicate_id_mapping: tuple[tuple[_PredicateId, _PredicateId], ...] = Field(min_length=1)
    hypothesis_id_mapping: tuple[tuple[_HypothesisId, _HypothesisId], ...] = Field(min_length=1)
    predicates: tuple[PredicateResult, ...] = Field(min_length=1)
    hypotheses: tuple[DiagnosticHypothesis, ...] = Field(min_length=1)
    binding_sha256: _SHA

    @model_validator(mode="before")
    @classmethod
    def reject_noncanonical_python_values(cls, value: object) -> object:
        # Python strings/containers and nested model subclasses must not be silently
        # normalized by Pydantic, even inside older permissive nested contracts.
        if type(value) is not dict and type(value) is not cls:
            raise ValueError("B0 rule binding requires exact typed state or an exact dictionary")
        _preflight_tree(value, (B0RuleBinding,))
        return value

    @field_validator("*", mode="before")
    @classmethod
    def exact_metadata(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name not in _COLLECTIONS and type(value) is not str:
            raise ValueError("B0 rule binding requires exact string metadata")
        return value

    @field_validator(*_COLLECTIONS, mode="before")
    @classmethod
    def exact_collections(cls, value: object, info: ValidationInfo) -> object:
        if type(value) is not tuple and not (info.mode == "json" and type(value) is list):
            raise ValueError("B0 rule binding requires exact tuple collections")
        items = cast(tuple[object, ...], value)
        if info.field_name in ("predicate_id_mapping", "hypothesis_id_mapping"):
            pairs = []
            for pair in items:
                if type(pair) is not tuple and not (info.mode == "json" and type(pair) is list):
                    raise ValueError("B0 rule binding requires exact tuple mapping pairs")
                pairs.append(tuple(cast(tuple[object, ...], pair)))
            return tuple(pairs)
        return tuple(items)

    @model_validator(mode="after")
    def closed_internal_references(self) -> Self:
        for pairs in (self.predicate_id_mapping, self.hypothesis_id_mapping):
            if pairs != tuple(sorted(pairs)) or len({a for a, _ in pairs}) != len(pairs):
                raise ValueError("B0 rule binding mappings require canonical unique source IDs")
        predicate_ids = tuple(p.predicate_result_id for p in self.predicates)
        if predicate_ids != tuple(sorted(set(predicate_ids))):
            raise ValueError("B0 rule binding predicates require canonical unique IDs")
        if {b for _, b in self.predicate_id_mapping} != set(predicate_ids):
            raise ValueError("B0 rule binding predicate mapping is not closed")
        hypothesis_ids = {h.hypothesis_id for h in self.hypotheses}
        if (
            len(hypothesis_ids) != len(self.hypotheses)
            or len(self.hypothesis_id_mapping) != len(self.hypotheses)
            or {b for _, b in self.hypothesis_id_mapping} != hypothesis_ids
        ):
            raise ValueError("B0 rule binding hypothesis mapping is not closed and unique")
        pairs = tuple((h.candidate_service_key, h.mechanism_key) for h in self.hypotheses)
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError(
                "B0 rule binding hypotheses require canonical candidate/mechanism pairs"
            )
        predicates = {p.predicate_result_id: p for p in self.predicates}
        referenced: set[str] = set()
        for hypothesis in self.hypotheses:
            if (
                hypothesis.opaque_case_id != self.opaque_case_id
                or hypothesis.origin != "deterministic_rule"
                or hypothesis.origin_version != RULE_GENERATOR_VERSION
            ):
                raise ValueError("B0 rule binding contains a foreign hypothesis")
            for requirement, ids in (
                ("necessary", hypothesis.necessary_predicate_result_ids),
                ("optional", hypothesis.optional_predicate_result_ids),
                ("refuting", hypothesis.refuting_predicate_result_ids),
            ):
                for predicate_id in ids:
                    if predicate_id not in predicates:
                        raise ValueError("B0 rule binding has an unbound hypothesis predicate")
                    if predicates[predicate_id].requirement != requirement:
                        raise ValueError("B0 rule binding predicate requirement differs")
                    referenced.add(predicate_id)
        if referenced != set(predicates):
            raise ValueError("B0 rule binding has unreferenced evaluated predicates")
        return self

    def verify(self, *, capacity_profile: str | None = None) -> bool:
        with binding_capacity(capacity_profile):
            return self._verify()

    def _verify(self) -> bool:
        """Check exact shape, canonical closure and child digests, never source authority."""
        try:
            if type(self) is not B0RuleBinding:
                return False
            _preflight_tree(self, (B0RuleBinding,))
            original = _bytes(self)
            rebuilt = B0RuleBinding.model_validate_json(original)
            unsigned = cast(dict[str, object], _plain(rebuilt))
            unsigned.pop("binding_sha256")
            return (
                hmac.compare_digest(original, _bytes(rebuilt))
                and all(p.verify() for p in rebuilt.predicates)
                and all(h.verify() for h in rebuilt.hypotheses)
                and hmac.compare_digest(
                    rebuilt.binding_sha256, sha256_hex(canonical_json_bytes(unsigned))
                )
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


def _preflight_inputs(
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    binding: B0RuleBinding | None = None,
    expected_binding_sha256: str | None = None,
) -> None:
    # All roots and independent pins are checked before any attribute, equality,
    # hash, verifier or serializer callback; only builtin exact-type checks run.
    if type(envelope) is not RuleHypothesisEnvelope:
        raise ValueError("B0 rule binding requires an exact RuleHypothesisEnvelope")
    if binding is not None and type(binding) is not B0RuleBinding:
        raise ValueError("B0 rule binding requires an exact B0RuleBinding")
    pins: tuple[str, ...] = (expected_rule_set_sha256,)
    if binding is not None:
        pins = (*pins, cast(str, expected_binding_sha256))
    for pin in pins:
        if type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
            raise ValueError("B0 rule binding requires exact lowercase SHA256 pins")
    _preflight_tree((envelope, binding), (RuleHypothesisEnvelope, B0RuleBinding))


def _representation(value: object) -> object:
    """Read only preflighted state; preserve markers/types that JSON omits.

    Local to the original binding's unrestricted-string acceptance domain;
    dynamic verification's stricter serialization policy must not leak here.
    Noncanonical extra/private slots are rejected by the preceding preflight.
    """
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return (kind.__name__, value)
    if kind in (tuple, list):
        return (kind.__name__, tuple(_representation(v) for v in cast(tuple[object, ...], value)))
    if kind is dict:
        return (
            "dict",
            tuple((k, _representation(v)) for k, v in cast(dict[str, object], value).items()),
        )
    return (
        kind.__module__,
        kind.__name__,
        tuple(sorted(object.__getattribute__(value, "__pydantic_fields_set__"))),
        _representation(object.__getattribute__(value, "__dict__")),
    )


def _snapshot(value: BaseModel, model_type: type[BaseModel]) -> tuple[bytes, object]:
    _preflight_tree(value, (model_type,))
    return _bytes(value), _representation(value)


def _require_unchanged(
    value: BaseModel, original: tuple[bytes, object], model_type: type[BaseModel]
) -> None:
    try:
        current = _snapshot(value, model_type)
        unchanged = hmac.compare_digest(current[0], original[0]) and current[1] == original[1]
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise ValueError("B0 rule binding input content or representation changed during replay")


def _project(source: RuleHypothesisEnvelope, *, grouping: EventGroupingResult) -> B0RuleBinding:
    seeds = source.rule_hypothesis_set
    predicate_mapping, hypothesis_mapping, predicates, hypotheses = _bind_rule_values(
        seed_predicates=seeds.predicates, seed_hypotheses=seeds.hypotheses, grouping=grouping
    )
    result = B0RuleBinding(
        opaque_case_id=seeds.opaque_case_id,
        source_rule_set_sha256=seeds.result_sha256,
        predicate_evaluation_policy_version=cast(
            Literal["original-observation-predicate-plan/1"],
            PREDICATE_PLAN_EVALUATION_POLICY_VERSION,
        ),
        predicate_id_mapping=predicate_mapping,
        hypothesis_id_mapping=hypothesis_mapping,
        predicates=predicates,
        hypotheses=hypotheses,
        binding_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], _plain(result))
    unsigned.pop("binding_sha256")
    return result.model_copy(update={"binding_sha256": sha256_hex(canonical_json_bytes(unsigned))})


def _bind_rule_values(
    *, seed_predicates: tuple[PredicateResult, ...],
    seed_hypotheses: tuple[DiagnosticHypothesis, ...], grouping: EventGroupingResult,
) -> tuple[
    tuple[tuple[str, str], ...], tuple[tuple[str, str], ...],
    tuple[PredicateResult, ...], tuple[DiagnosticHypothesis, ...],
]:
    """Unchanged numerical kernel; caller owns complete source replay and exit guards."""
    value_index = predicate_plan_evaluator._build_predicate_value_index(grouping)
    predicates: dict[str, PredicateResult] = {}
    predicate_mapping: dict[str, str] = {}
    for seed in seed_predicates:
        # B0 seeds have no proposal cap. Evaluate every single original observation
        # with the unchanged value kernel, then use its existing composition.
        # Each code-owned plan still obeys the proposal's per-plan reference cap.
        refs = tuple((observation_id,) for observation_id in seed.observation_ids) or ((),)
        leaves = tuple(
            predicate_plan_evaluator._evaluate_predicate_plan_values(
                plan=PredicatePlan(
                    predicate_kind=seed.predicate_kind,
                    requirement=seed.requirement,
                    observation_ids=ids,
                    relation_ids=(),
                ),
                grouping=grouping,
                value_index=value_index,
            )
            for ids in refs
        )
        if len(leaves) == 1:
            evaluated = leaves[0]
        else:
            contract = _PREDICATE_EVIDENCE_CONTRACTS[seed.predicate_kind]
            compose = strong_kleene_and if contract.comparator == "le" else strong_kleene_or
            evaluated = compose(
                predicate_kind=seed.predicate_kind, requirement=seed.requirement, operands=leaves
            )
        previous = predicates.setdefault(evaluated.predicate_result_id, evaluated)
        if not hmac.compare_digest(_bytes(previous), _bytes(evaluated)):
            raise ValueError("B0 rule binding evaluated predicate ID collision")
        predicate_mapping[seed.predicate_result_id] = evaluated.predicate_result_id

    hypotheses: list[DiagnosticHypothesis] = []
    hypothesis_mapping: list[tuple[str, str]] = []
    for seed_hypothesis in seed_hypotheses:
        rebound = build_diagnostic_hypothesis(
            opaque_case_id=seed_hypothesis.opaque_case_id,
            candidate_service_key=seed_hypothesis.candidate_service_key,
            mechanism_key=seed_hypothesis.mechanism_key,
            mechanism_definition_sha256=seed_hypothesis.mechanism_definition_sha256,
            origin=seed_hypothesis.origin,
            origin_version=seed_hypothesis.origin_version,
            observation_ids=seed_hypothesis.observation_ids,
            relation_ids=seed_hypothesis.relation_ids,
            event_group_ids=seed_hypothesis.event_group_ids,
            time_intervals=seed_hypothesis.time_intervals,
            necessary_predicate_result_ids=tuple(
                predicate_mapping[i] for i in seed_hypothesis.necessary_predicate_result_ids
            ),
            optional_predicate_result_ids=tuple(
                predicate_mapping[i] for i in seed_hypothesis.optional_predicate_result_ids
            ),
            refuting_predicate_result_ids=tuple(
                predicate_mapping[i] for i in seed_hypothesis.refuting_predicate_result_ids
            ),
            alternative_hypothesis_ids=seed_hypothesis.alternative_hypothesis_ids,
            verification_request_ids=seed_hypothesis.verification_request_ids,
            unknown_reasons=seed_hypothesis.unknown_reasons,
            provenance_ids=seed_hypothesis.provenance_ids,
            lifecycle_status=seed_hypothesis.lifecycle_status,
        )
        hypotheses.append(rebound)
        hypothesis_mapping.append((seed_hypothesis.hypothesis_id, rebound.hypothesis_id))
    return (
        tuple(sorted(predicate_mapping.items())), tuple(sorted(hypothesis_mapping)),
        tuple(predicates[i] for i in sorted(predicates)), tuple(hypotheses),
    )


def build_b0_rule_binding(
    *, envelope: RuleHypothesisEnvelope, expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> B0RuleBinding:
    """Select a registered invocation budget; omitted profile retains legacy limits."""
    with binding_capacity(capacity_profile):
        return _build_b0_rule_binding(
            envelope=envelope, expected_rule_set_sha256=expected_rule_set_sha256
        )


def _build_b0_rule_binding(
    *, envelope: RuleHypothesisEnvelope, expected_rule_set_sha256: str,
) -> B0RuleBinding:
    """Replay all six frozen source inputs, then bind original-observation predicates."""
    _preflight_inputs(envelope, expected_rule_set_sha256)
    original = _snapshot(envelope, RuleHypothesisEnvelope)
    try:
        source = RuleHypothesisEnvelope.model_validate_json(original[0])
        source_before = _snapshot(source, RuleHypothesisEnvelope)
        try:
            if not hmac.compare_digest(original[0], source_before[0]):
                raise ValueError("B0 rule binding source is not a canonical envelope")
            if not hmac.compare_digest(
                source.rule_hypothesis_set.result_sha256, expected_rule_set_sha256
            ):
                raise ValueError("B0 rule binding source fails independent rule set pin")
            if not source.verify():
                raise ValueError("B0 rule binding source failed complete frozen-input replay")
            grouping = predicate_plan_evaluator._prepare_predicate_grouping(source.event_grouping)
            grouping_before = _snapshot(grouping, EventGroupingResult)
            try:
                return _project(source, grouping=grouping)
            finally:
                _require_unchanged(grouping, grouping_before, EventGroupingResult)
        finally:
            _require_unchanged(source, source_before, RuleHypothesisEnvelope)
    finally:
        _require_unchanged(envelope, original, RuleHypothesisEnvelope)


def require_b0_rule_binding(
    *,
    binding: B0RuleBinding,
    expected_binding_sha256: str,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> B0RuleBinding:
    with binding_capacity(capacity_profile):
        return _require_b0_rule_binding(
            binding=binding, expected_binding_sha256=expected_binding_sha256,
            envelope=envelope, expected_rule_set_sha256=expected_rule_set_sha256,
        )


def _require_b0_rule_binding(
    *,
    binding: B0RuleBinding,
    expected_binding_sha256: str,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
) -> B0RuleBinding:
    """Require independent pins and exact complete replay; return a fresh bound receipt."""
    if type(binding) is not B0RuleBinding:
        raise ValueError("B0 rule binding requires an exact B0RuleBinding")
    _preflight_inputs(envelope, expected_rule_set_sha256, binding, expected_binding_sha256)
    original_envelope = _snapshot(envelope, RuleHypothesisEnvelope)
    original_binding = _snapshot(binding, B0RuleBinding)
    try:
        if not hmac.compare_digest(binding.binding_sha256, expected_binding_sha256):
            raise ValueError("B0 rule binding fails independent binding pin")
        snapshot = B0RuleBinding.model_validate_json(original_binding[0])
        profile = active_binding_capacity()
        valid = snapshot.verify() if profile is None else snapshot.verify(capacity_profile=profile)
        if not valid:
            raise ValueError("B0 rule binding fails internal integrity")
        rebuilt = build_b0_rule_binding(
            envelope=envelope, expected_rule_set_sha256=expected_rule_set_sha256,
            capacity_profile=active_binding_capacity(),
        )
        if not hmac.compare_digest(original_binding[0], _bytes(rebuilt)):
            raise ValueError("B0 rule binding differs from independent source replay")
        return rebuilt
    finally:
        try:
            _require_unchanged(envelope, original_envelope, RuleHypothesisEnvelope)
        finally:
            _require_unchanged(binding, original_binding, B0RuleBinding)
