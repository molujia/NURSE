"""Replay-gated signal-family maxima and declared evidence aggregators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.event_grouping import (
    EventGroupingResult,
    PrimarySignalFamily,
    replay_event_grouping_result,
)
from nurse_rcl.v2.event_strength import (
    CandidateEventRoleAdjustment, EventGroupBaseStrength, EventStrengthResult,
)
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

FAMILY_AGGREGATOR_DEFINITION_SCHEMA_VERSION: Literal["nurse.v2.family-aggregator-definition/1"] = (
    "nurse.v2.family-aggregator-definition/1"
)
SIGNAL_FAMILY_MAXIMUM_SCHEMA_VERSION: Literal["nurse.v2.signal-family-maximum/1"] = (
    "nurse.v2.signal-family-maximum/1"
)
CANDIDATE_FAMILY_AGGREGATION_SCHEMA_VERSION: Literal["nurse.v2.candidate-family-aggregation/1"] = (
    "nurse.v2.candidate-family-aggregation/1"
)
UNBOUND_EVENT_GROUP_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.unbound-event-group-audit/1"] = (
    "nurse.v2.unbound-event-group-audit/1"
)
FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.family-aggregation-result/2"] = (
    "nurse.v2.family-aggregation-result/2"
)
FAMILY_AGGREGATION_POLICY_VERSION: Literal["nurse-v2-family-aggregation/1"] = (
    "nurse-v2-family-aggregation/1"
)
REFERENCE_KEY: Literal["nurse-v2-top-two-f-gamma/1"] = "nurse-v2-top-two-f-gamma/1"


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


FAMILY_AGGREGATION_POLICY_SHA256 = _digest(
    {
        "version": FAMILY_AGGREGATION_POLICY_VERSION,
        "family_value": "max(adjusted_event_strength)",
        "family_tie_break": "ascending_event_group_id",
        "unknown": "retained_without_numeric_contribution",
        "candidate_membership": "event_strength_adjustments_only",
        "family_order": "descending_value_then_ascending_family_key",
    }
)


class FamilyAggregatorDefinition(StrictFrozenModel):
    """Content-addressed public declaration for one code-backed aggregator."""

    schema_version: Literal["nurse.v2.family-aggregator-definition/1"] = (
        FAMILY_AGGREGATOR_DEFINITION_SCHEMA_VERSION
    )
    aggregator_version: str = Field(min_length=1, max_length=128)
    aggregation_kind: Literal["top_two_f_gamma"]
    maximum_contributing_families: int = Field(gt=0)
    gamma: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_reference_parameters(self) -> Self:
        if self.aggregation_kind == "top_two_f_gamma" and (
            self.maximum_contributing_families != 2 or self.gamma != 0.25
        ):
            raise ValueError("top-two F_gamma requires two families and gamma=0.25")
        return self

    def verify(self) -> bool:
        try:
            validated = FamilyAggregatorDefinition.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"definition_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self and self.definition_sha256 == _digest(unsigned)


def _build_reference_definition() -> FamilyAggregatorDefinition:
    unsigned: dict[str, object] = {
        "schema_version": FAMILY_AGGREGATOR_DEFINITION_SCHEMA_VERSION,
        "aggregator_version": REFERENCE_KEY,
        "aggregation_kind": "top_two_f_gamma",
        "maximum_contributing_families": 2,
        "gamma": 0.25,
    }
    return FamilyAggregatorDefinition.model_validate(
        {**unsigned, "definition_sha256": _digest(unsigned)}
    )


class DeclaredFamilyAggregator(Protocol):
    """Internal code contract implemented only by registered aggregators."""

    @property
    def definition(self) -> FamilyAggregatorDefinition:
        """Return the immutable public declaration."""

    def aggregate(self, selected_values: tuple[float, ...]) -> float:
        """Aggregate already ordered, distinct family maxima."""


@dataclass(frozen=True)
class _TopTwoFGammaAggregator:
    definition: FamilyAggregatorDefinition

    def aggregate(self, selected_values: tuple[float, ...]) -> float:
        if not selected_values:
            return 0.0
        first = selected_values[0]
        if len(selected_values) == 1:
            return first
        second = selected_values[1]
        return first + self.definition.gamma * (1.0 - first) * second


_REFERENCE_DEFINITION = _build_reference_definition()
_AGGREGATOR_IMPLEMENTATIONS: Mapping[str, DeclaredFamilyAggregator] = MappingProxyType(
    {REFERENCE_KEY: _TopTwoFGammaAggregator(_REFERENCE_DEFINITION)}
)
FAMILY_AGGREGATOR_REGISTRY: Mapping[str, FamilyAggregatorDefinition] = MappingProxyType(
    {REFERENCE_KEY: _REFERENCE_DEFINITION}
)


def resolve_declared_family_aggregator(version_key: str) -> FamilyAggregatorDefinition:
    """Resolve a sealed declaration by exact immutable code-registry key."""

    if type(version_key) is not str:
        raise ValueError("family aggregator must be selected by a version key")
    definition = FAMILY_AGGREGATOR_REGISTRY.get(version_key)
    if definition is None:
        raise ValueError(f"unknown family aggregator version: {version_key}")
    if not definition.verify():  # pragma: no cover - immutable construction invariant
        raise AssertionError("registered family aggregator definition failed verification")
    return definition


class SignalFamilyMaximum(StrictFrozenModel):
    """One candidate/family maximum with a closed event audit partition."""

    schema_version: Literal["nurse.v2.signal-family-maximum/1"] = (
        SIGNAL_FAMILY_MAXIMUM_SCHEMA_VERSION
    )
    family_maximum_id: str = Field(pattern=r"^sfm_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    primary_signal_family: PrimarySignalFamily
    status: Literal["computed", "unknown_only"]
    family_value: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    contributing_event_group_ids: tuple[str, ...]
    retained_event_group_ids: tuple[str, ...] = Field(min_length=1)
    noncontributing_event_group_ids: tuple[str, ...]
    unknown_event_group_ids: tuple[str, ...]
    contributing_provenance_ids: tuple[str, ...]
    noncontributing_provenance_ids: tuple[str, ...]
    unknown_provenance_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    family_maximum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_event_partition(self) -> Self:
        for values, name in (
            (self.contributing_event_group_ids, "contributing event group IDs"),
            (self.retained_event_group_ids, "retained event group IDs"),
            (self.noncontributing_event_group_ids, "noncontributing event group IDs"),
            (self.unknown_event_group_ids, "unknown event group IDs"),
            (self.contributing_provenance_ids, "contributing provenance IDs"),
            (self.noncontributing_provenance_ids, "noncontributing provenance IDs"),
            (self.unknown_provenance_ids, "unknown provenance IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        contributing = set(self.contributing_event_group_ids)
        noncontributing = set(self.noncontributing_event_group_ids)
        unknown = set(self.unknown_event_group_ids)
        if contributing & noncontributing or contributing & unknown or noncontributing & unknown:
            raise ValueError("family event categories must be disjoint")
        if contributing | noncontributing | unknown != set(self.retained_event_group_ids):
            raise ValueError("family event categories must partition retained events")
        if set(self.provenance_ids) != (
            set(self.contributing_provenance_ids)
            | set(self.noncontributing_provenance_ids)
            | set(self.unknown_provenance_ids)
        ):
            raise ValueError("family provenance categories must cover retained provenance")
        if self.status == "computed":
            if self.family_value is None or len(self.contributing_event_group_ids) != 1:
                raise ValueError("computed family maximum requires one numeric contributor")
        elif (
            self.family_value is not None
            or self.contributing_event_group_ids
            or self.noncontributing_event_group_ids
            or set(self.unknown_event_group_ids) != set(self.retained_event_group_ids)
        ):
            raise ValueError("unknown-only family cannot claim a numeric event")
        return self

    def verify(self) -> bool:
        try:
            validated = SignalFamilyMaximum.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(
                mode="json", exclude={"family_maximum_id", "family_maximum_sha256"}
            )
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and self.family_maximum_id == f"sfm_{digest[:24]}"
            and self.family_maximum_sha256 == digest
        )


class CandidateFamilyAggregation(StrictFrozenModel):
    """Candidate evidence strength from distinct family maxima."""

    schema_version: Literal["nurse.v2.candidate-family-aggregation/1"] = (
        CANDIDATE_FAMILY_AGGREGATION_SCHEMA_VERSION
    )
    candidate_aggregation_id: str = Field(pattern=r"^cfa_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    aggregator_version: str = Field(min_length=1, max_length=128)
    aggregator_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_status: Literal["no_retained_event", "unknown_only", "computed"]
    family_maxima: tuple[SignalFamilyMaximum, ...]
    retained_family_keys: tuple[PrimarySignalFamily, ...]
    selected_family_keys: tuple[PrimarySignalFamily, ...]
    noncontributing_family_keys: tuple[PrimarySignalFamily, ...]
    unknown_family_keys: tuple[PrimarySignalFamily, ...]
    evidence_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    is_root_cause_probability: Literal[False] = False
    provenance_ids: tuple[str, ...]
    aggregation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_family_partition(self) -> Self:
        if self.family_maxima != tuple(
            sorted(self.family_maxima, key=lambda item: item.primary_signal_family)
        ):
            raise ValueError("family maxima must be canonically ordered")
        family_keys = tuple(item.primary_signal_family for item in self.family_maxima)
        if len(family_keys) != len(set(family_keys)):
            raise ValueError("family maxima must be unique")
        retained_event_ids = tuple(
            event_group_id
            for item in self.family_maxima
            for event_group_id in item.retained_event_group_ids
        )
        if len(retained_event_ids) != len(set(retained_event_ids)):
            raise ValueError("one candidate event cannot be retained by multiple signal families")
        for values, name in (
            (self.retained_family_keys, "retained family keys"),
            (self.noncontributing_family_keys, "noncontributing family keys"),
            (self.unknown_family_keys, "unknown family keys"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.retained_family_keys != tuple(sorted(family_keys)):
            raise ValueError("retained families must exactly match family maxima")
        selected = set(self.selected_family_keys)
        noncontributing = set(self.noncontributing_family_keys)
        unknown = set(self.unknown_family_keys)
        if selected & noncontributing or selected & unknown or noncontributing & unknown:
            raise ValueError("candidate family categories must be disjoint")
        if selected | noncontributing | unknown != set(self.retained_family_keys):
            raise ValueError("candidate family categories must partition retained families")
        definition = resolve_declared_family_aggregator(self.aggregator_version)
        if self.aggregator_definition_sha256 != definition.definition_sha256:
            raise ValueError("candidate aggregator definition digest differs")
        maximum_by_key = {item.primary_signal_family: item for item in self.family_maxima}
        numeric = tuple(
            sorted(
                (item for item in self.family_maxima if item.family_value is not None),
                key=lambda item: (-cast(float, item.family_value), item.primary_signal_family),
            )
        )
        expected_selected = tuple(
            item.primary_signal_family
            for item in numeric[: definition.maximum_contributing_families]
        )
        if self.selected_family_keys != expected_selected:
            raise ValueError("selected families do not match declared aggregator ordering")
        if self.noncontributing_family_keys != tuple(
            sorted(item.primary_signal_family for item in numeric[len(expected_selected) :])
        ):
            raise ValueError("noncontributing numeric families are incomplete")
        if self.unknown_family_keys != tuple(
            sorted(
                item.primary_signal_family
                for item in self.family_maxima
                if item.family_value is None
            )
        ):
            raise ValueError("unknown family audit is incomplete")
        if not self.family_maxima:
            expected_status = "no_retained_event"
        elif not numeric:
            expected_status = "unknown_only"
        else:
            expected_status = "computed"
        if self.evidence_status != expected_status:
            raise ValueError("candidate evidence status differs from retained evidence")
        implementation = _AGGREGATOR_IMPLEMENTATIONS[self.aggregator_version]
        expected_strength = implementation.aggregate(
            tuple(
                cast(float, maximum_by_key[key].family_value) for key in self.selected_family_keys
            )
        )
        if self.evidence_strength != expected_strength:
            raise ValueError("candidate evidence strength differs from declared aggregator")
        if set(self.provenance_ids) != {
            value for item in self.family_maxima for value in item.provenance_ids
        }:
            raise ValueError("candidate provenance must cover all retained families")
        return self

    def verify(self) -> bool:
        try:
            validated = CandidateFamilyAggregation.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or any(not item.verify() for item in self.family_maxima):
                return False
            unsigned = validated.model_dump(
                mode="json", exclude={"candidate_aggregation_id", "aggregation_sha256"}
            )
            digest = _digest(unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return (
            self.candidate_aggregation_id == f"cfa_{digest[:24]}"
            and self.aggregation_sha256 == digest
        )


class UnboundEventGroupAudit(StrictFrozenModel):
    """Source-complete audit for one retained group with no candidate binding."""

    schema_version: Literal["nurse.v2.unbound-event-group-audit/1"] = (
        UNBOUND_EVENT_GROUP_AUDIT_SCHEMA_VERSION
    )
    audit_id: str = Field(pattern=r"^uega_[0-9a-f]{24}$")
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    primary_signal_family: PrimarySignalFamily
    base_strength_id: str = Field(pattern=r"^egs_[0-9a-f]{24}$")
    source_fingerprints: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_sources(self) -> Self:
        for values, name in (
            (self.source_fingerprints, "unbound source fingerprints"),
            (self.provenance_ids, "unbound provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            validated = UnboundEventGroupAudit.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"audit_id", "audit_sha256"})
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and self.audit_id == f"uega_{digest[:24]}"
            and self.audit_sha256 == digest
        )


class FamilyAggregationResult(StrictFrozenModel):
    """Case-atomic, content-addressed family aggregation result."""

    schema_version: Literal["nurse.v2.family-aggregation-result/2"] = (
        FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION
    )
    family_aggregation_result_id: str = Field(pattern=r"^far_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-family-aggregation/1"] = FAMILY_AGGREGATION_POLICY_VERSION
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregator_version: str = Field(min_length=1, max_length=128)
    aggregator_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_strength_result_id: str = Field(pattern=r"^esr_[0-9a-f]{24}$")
    event_strength_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retained_event_group_ids: tuple[str, ...]
    aggregated_event_group_ids: tuple[str, ...]
    unbound_event_group_ids: tuple[str, ...]
    unbound_event_group_audits: tuple[UnboundEventGroupAudit, ...]
    candidates: tuple[CandidateFamilyAggregation, ...]
    provenance_ids: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_canonical_result(self) -> Self:
        if self.policy_sha256 != FAMILY_AGGREGATION_POLICY_SHA256:
            raise ValueError("family aggregation policy digest differs")
        definition = resolve_declared_family_aggregator(self.aggregator_version)
        if self.aggregator_definition_sha256 != definition.definition_sha256:
            raise ValueError("family aggregator definition digest differs")
        if self.candidates != tuple(
            sorted(self.candidates, key=lambda item: item.candidate_service_key)
        ):
            raise ValueError("candidate aggregations must be canonically ordered")
        if len({item.candidate_service_key for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate aggregations must be unique")
        for values, name in (
            (self.retained_event_group_ids, "retained event group IDs"),
            (self.aggregated_event_group_ids, "aggregated event group IDs"),
            (self.unbound_event_group_ids, "unbound event group IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        aggregated = set(self.aggregated_event_group_ids)
        unbound = set(self.unbound_event_group_ids)
        if aggregated & unbound:
            raise ValueError("aggregated and unbound events must be disjoint")
        if aggregated | unbound != set(self.retained_event_group_ids):
            raise ValueError("aggregated and unbound events must partition retained events")
        if self.unbound_event_group_audits != tuple(
            sorted(self.unbound_event_group_audits, key=lambda item: item.event_group_id)
        ):
            raise ValueError("unbound event audits must be canonically ordered")
        audited_unbound_ids = tuple(item.event_group_id for item in self.unbound_event_group_audits)
        if len(audited_unbound_ids) != len(set(audited_unbound_ids)):
            raise ValueError("unbound event audits must be unique")
        if set(audited_unbound_ids) != unbound:
            raise ValueError("unbound event audits must exactly cover unbound events")
        candidate_events = {
            event_group_id
            for candidate in self.candidates
            for family in candidate.family_maxima
            for event_group_id in family.retained_event_group_ids
        }
        if candidate_events != aggregated:
            raise ValueError("candidate family audits must cover aggregated events")
        if any(
            item.aggregator_version != self.aggregator_version
            or item.aggregator_definition_sha256 != self.aggregator_definition_sha256
            for item in self.candidates
        ):
            raise ValueError("candidate aggregations use a different declaration")
        expected_provenance = {
            value for item in self.candidates for value in item.provenance_ids
        } | {value for item in self.unbound_event_group_audits for value in item.provenance_ids}
        if set(self.provenance_ids) != expected_provenance:
            raise ValueError("result provenance must cover every retained event group")
        return self

    def verify(self) -> bool:
        try:
            validated = FamilyAggregationResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if (
                validated != self
                or any(not item.verify() for item in self.candidates)
                or any(not item.verify() for item in self.unbound_event_group_audits)
            ):
                return False
            unsigned = validated.model_dump(
                mode="json", exclude={"family_aggregation_result_id", "result_sha256"}
            )
            digest = _digest(unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return (
            self.family_aggregation_result_id == f"far_{digest[:24]}"
            and self.result_sha256 == digest
        )

    def verify_inputs(
        self,
        *,
        event_strength: EventStrengthResult,
        aggregator_version: str,
        role_compilation: RoleCompilationResult,
        proposals: Sequence[RoleBindingProposal],
        guard_results: Sequence[HypothesisGuardResult],
        predicates: Sequence[PredicateResult],
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
        query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
    ) -> bool:
        try:
            expected = compile_family_aggregation(
                event_strength=event_strength,
                aggregator_version=aggregator_version,
                role_compilation=role_compilation,
                proposals=proposals,
                guard_results=guard_results,
                predicates=predicates,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
                mechanism_library=mechanism_library,
                scope_interval=scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=semantic_context,
                card_expansion_policy=card_expansion_policy,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return expected == self


@dataclass(frozen=True)
class _EventValue:
    event_group_id: str
    value: float | None
    provenance_ids: tuple[str, ...]


def _build_family_maximum(
    *,
    candidate_service_key: str,
    family: PrimarySignalFamily,
    events: Sequence[_EventValue],
) -> SignalFamilyMaximum:
    # Exact EventStrengthResult replay guarantees canonical (candidate, event)
    # ordering; each per-family subsequence preserves it without re-sorting.
    retained = tuple(events)
    numeric = tuple(item for item in retained if item.value is not None)
    unknown = tuple(item for item in retained if item.value is None)
    family_value: float | None
    contributing: tuple[_EventValue, ...]
    noncontributing: tuple[_EventValue, ...]
    status: Literal["computed", "unknown_only"]
    if numeric:
        family_value = max(cast(float, item.value) for item in numeric)
        contributor = min(
            (item for item in numeric if item.value == family_value),
            key=lambda item: item.event_group_id,
        )
        contributing = (contributor,)
        noncontributing = tuple(
            item for item in numeric if item.event_group_id != contributor.event_group_id
        )
        status = "computed"
    else:
        family_value = None
        contributing = ()
        noncontributing = ()
        status = "unknown_only"

    def provenance(values: Sequence[_EventValue]) -> tuple[str, ...]:
        return tuple(sorted({value for item in values for value in item.provenance_ids}))

    unsigned: dict[str, object] = {
        "schema_version": SIGNAL_FAMILY_MAXIMUM_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "primary_signal_family": family,
        "status": status,
        "family_value": family_value,
        "contributing_event_group_ids": tuple(item.event_group_id for item in contributing),
        "retained_event_group_ids": tuple(item.event_group_id for item in retained),
        "noncontributing_event_group_ids": tuple(item.event_group_id for item in noncontributing),
        "unknown_event_group_ids": tuple(item.event_group_id for item in unknown),
        "contributing_provenance_ids": provenance(contributing),
        "noncontributing_provenance_ids": provenance(noncontributing),
        "unknown_provenance_ids": provenance(unknown),
        "provenance_ids": provenance(retained),
    }
    digest = _digest(unsigned)
    return SignalFamilyMaximum.model_validate(
        {
            **unsigned,
            "family_maximum_id": f"sfm_{digest[:24]}",
            "family_maximum_sha256": digest,
        }
    )


def _build_candidate_aggregation(
    *,
    candidate_service_key: str,
    events: Sequence[tuple[PrimarySignalFamily, _EventValue]],
    implementation: DeclaredFamilyAggregator,
) -> CandidateFamilyAggregation:
    by_family: dict[PrimarySignalFamily, list[_EventValue]] = defaultdict(list)
    for family, event in events:
        by_family[family].append(event)
    maxima = tuple(
        _build_family_maximum(
            candidate_service_key=candidate_service_key,
            family=family,
            events=by_family[family],
        )
        for family in sorted(by_family)
    )
    numeric = tuple(
        sorted(
            (item for item in maxima if item.family_value is not None),
            key=lambda item: (-cast(float, item.family_value), item.primary_signal_family),
        )
    )
    selected = numeric[: implementation.definition.maximum_contributing_families]
    selected_keys = tuple(item.primary_signal_family for item in selected)
    unknown_keys = tuple(
        sorted(item.primary_signal_family for item in maxima if item.family_value is None)
    )
    if not maxima:
        status = "no_retained_event"
    elif not numeric:
        status = "unknown_only"
    else:
        status = "computed"
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_FAMILY_AGGREGATION_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "aggregator_version": implementation.definition.aggregator_version,
        "aggregator_definition_sha256": implementation.definition.definition_sha256,
        "evidence_status": status,
        "family_maxima": tuple(item.model_dump(mode="json") for item in maxima),
        "retained_family_keys": tuple(sorted(by_family)),
        "selected_family_keys": selected_keys,
        "noncontributing_family_keys": tuple(
            sorted(item.primary_signal_family for item in numeric[len(selected) :])
        ),
        "unknown_family_keys": unknown_keys,
        "evidence_strength": implementation.aggregate(
            tuple(cast(float, item.family_value) for item in selected)
        ),
        "is_root_cause_probability": False,
        "provenance_ids": tuple(
            sorted({value for item in maxima for value in item.provenance_ids})
        ),
    }
    digest = _digest(unsigned)
    return CandidateFamilyAggregation.model_validate(
        {
            **unsigned,
            "candidate_aggregation_id": f"cfa_{digest[:24]}",
            "aggregation_sha256": digest,
        }
    )


def _build_unbound_event_group_audit(
    *,
    event_group_id: str,
    primary_signal_family: PrimarySignalFamily,
    base_strength_id: str,
    source_fingerprints: Sequence[str],
    provenance_ids: Sequence[str],
) -> UnboundEventGroupAudit:
    unsigned: dict[str, object] = {
        "schema_version": UNBOUND_EVENT_GROUP_AUDIT_SCHEMA_VERSION,
        "event_group_id": event_group_id,
        "primary_signal_family": primary_signal_family,
        "base_strength_id": base_strength_id,
        "source_fingerprints": tuple(sorted(set(source_fingerprints))),
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = _digest(unsigned)
    return UnboundEventGroupAudit.model_validate(
        {**unsigned, "audit_id": f"uega_{digest[:24]}", "audit_sha256": digest}
    )


def compile_family_aggregation(
    *,
    event_strength: EventStrengthResult,
    aggregator_version: str,
    role_compilation: RoleCompilationResult,
    proposals: Sequence[RoleBindingProposal],
    guard_results: Sequence[HypothesisGuardResult],
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> FamilyAggregationResult:
    """Compile family maxima only after exact grouping and strength replay."""

    definition = resolve_declared_family_aggregator(aggregator_version)
    implementation = _AGGREGATOR_IMPLEMENTATIONS[definition.aggregator_version]
    if type(event_grouping) is not EventGroupingResult:
        raise ValueError("family aggregation requires an exact event grouping result")
    replayed_grouping = replay_event_grouping_result(event_grouping)
    if type(event_strength) is not EventStrengthResult or not event_strength.verify_inputs(
        role_compilation=role_compilation,
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=replayed_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("event strength result does not replay from frozen inputs")
    if (
        event_strength.opaque_case_id != replayed_grouping.opaque_case_id
        or event_strength.event_grouping_result_sha256 != replayed_grouping.result_sha256
    ):
        raise ValueError("event grouping and event strength lineages differ")

    parts = _compile_family_parts(
        event_grouping=replayed_grouping, group_strengths=event_strength.group_strengths,
        candidate_event_adjustments=event_strength.candidate_event_adjustments,
        candidate_universe=candidate_universe, implementation=implementation,
    )
    unsigned: dict[str, object] = {
        "schema_version": FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
        "opaque_case_id": replayed_grouping.opaque_case_id,
        "policy_version": FAMILY_AGGREGATION_POLICY_VERSION,
        "policy_sha256": FAMILY_AGGREGATION_POLICY_SHA256,
        "aggregator_version": definition.aggregator_version,
        "aggregator_definition_sha256": definition.definition_sha256,
        "event_grouping_result_sha256": replayed_grouping.result_sha256,
        "event_strength_result_id": event_strength.event_strength_result_id,
        "event_strength_result_sha256": event_strength.result_sha256,
        **parts,
    }
    digest = _digest(unsigned)
    result = FamilyAggregationResult.model_validate(
        {
            **unsigned,
            "family_aggregation_result_id": f"far_{digest[:24]}",
            "result_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("family aggregation result verification failed")
    return result


def _compile_family_parts(
    *, event_grouping: EventGroupingResult, group_strengths: tuple[EventGroupBaseStrength, ...],
    candidate_event_adjustments: tuple[CandidateEventRoleAdjustment, ...],
    candidate_universe: CandidateUniverse, implementation: DeclaredFamilyAggregator,
) -> dict[str, object]:
    """Same numeric/lineage core; source authority stays with the calling gate."""
    replayed_grouping = event_grouping
    group_by_id = {item.event_group_id: item for item in replayed_grouping.event_groups}
    base_by_event = {item.event_group_id: item for item in group_strengths}
    events_by_candidate: dict[str, list[tuple[PrimarySignalFamily, _EventValue]]] = defaultdict(
        list
    )
    for adjustment in candidate_event_adjustments:
        group = group_by_id.get(adjustment.event_group_id)
        base = base_by_event.get(adjustment.event_group_id)
        if group is None or base is None or adjustment.base_strength_id != base.strength_id:
            raise ValueError("event strength references an event outside grouping lineage")
        provenance_ids = tuple(
            sorted(
                set(group.provenance_ids)
                | set(base.provenance_ids)
                | set(adjustment.provenance_ids)
            )
        )
        events_by_candidate[adjustment.candidate_service_key].append(
            (
                group.primary_signal_family,
                _EventValue(
                    event_group_id=adjustment.event_group_id,
                    value=adjustment.adjusted_strength,
                    provenance_ids=provenance_ids,
                ),
            )
        )

    candidate_keys = tuple(
        sorted(candidate.service_entity_key for candidate in candidate_universe.candidates)
    )
    candidates = tuple(
        _build_candidate_aggregation(
            candidate_service_key=candidate_key,
            events=events_by_candidate.get(candidate_key, ()),
            implementation=implementation,
        )
        for candidate_key in candidate_keys
    )
    retained_event_ids = tuple(sorted(base_by_event))
    aggregated_event_ids = tuple(
        sorted(
            {adjustment.event_group_id for adjustment in candidate_event_adjustments}
        )
    )
    unbound_event_ids = tuple(sorted(set(retained_event_ids) - set(aggregated_event_ids)))
    unbound_event_audits = tuple(
        _build_unbound_event_group_audit(
            event_group_id=event_group_id,
            primary_signal_family=group_by_id[event_group_id].primary_signal_family,
            base_strength_id=base_by_event[event_group_id].strength_id,
            source_fingerprints=tuple(
                sorted(
                    set(group_by_id[event_group_id].source_fingerprints)
                    | set(base_by_event[event_group_id].source_fingerprints)
                )
            ),
            provenance_ids=tuple(
                sorted(
                    set(group_by_id[event_group_id].provenance_ids)
                    | set(base_by_event[event_group_id].provenance_ids)
                )
            ),
        )
        for event_group_id in unbound_event_ids
    )
    return {
        "retained_event_group_ids": retained_event_ids,
        "aggregated_event_group_ids": aggregated_event_ids,
        "unbound_event_group_ids": unbound_event_ids,
        "unbound_event_group_audits": tuple(
            item.model_dump(mode="json") for item in unbound_event_audits
        ),
        "candidates": tuple(item.model_dump(mode="json") for item in candidates),
        "provenance_ids": tuple(
            sorted(
                {value for item in candidates for value in item.provenance_ids}
                | {value for item in unbound_event_audits for value in item.provenance_ids}
            )
        ),
    }
