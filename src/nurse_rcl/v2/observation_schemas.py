"""Versioned V2 contracts from dataset scope through bounded evidence cards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

DATASET_MANIFEST_SCHEMA_VERSION: Literal["nurse.v2.dataset-manifest/1"] = (
    "nurse.v2.dataset-manifest/1"
)
TIME_CONTRACT_SCHEMA_VERSION: Literal["nurse.v2.case-time-contract/1"] = (
    "nurse.v2.case-time-contract/1"
)
LEGACY_OBSERVATION_SCHEMA_VERSION: Literal["nurse.v2.observation/1"] = "nurse.v2.observation/1"
OBSERVATION_SCHEMA_VERSION: Literal["nurse.v2.observation/2"] = "nurse.v2.observation/2"
ENTITY_BINDING_SCHEMA_VERSION: Literal["nurse.v2.entity-binding/1"] = "nurse.v2.entity-binding/1"
LEGACY_EVENT_GROUP_SCHEMA_VERSION: Literal["nurse.v2.event-group/1"] = "nurse.v2.event-group/1"
CANDIDATE_UNIVERSE_SCHEMA_VERSION: Literal["nurse.v2.candidate-universe/1"] = (
    "nurse.v2.candidate-universe/1"
)
FOCUS_SET_SCHEMA_VERSION: Literal["nurse.v2.semantic-focus-set/1"] = "nurse.v2.semantic-focus-set/1"
EVIDENCE_CARD_SCHEMA_VERSION: Literal["nurse.v2.evidence-card/1"] = "nurse.v2.evidence-card/1"

TelemetryModality = Literal["deployment", "metrics", "logs", "traces"]
SignalType = Literal[
    "counter",
    "gauge",
    "latency",
    "error",
    "sparse",
    "histogram",
    "sum",
    "count",
    "unknown",
]
EvidenceState = Literal[
    "abnormal",
    "verified_normal",
    "zero",
    "no_record",
    "unknown",
    "degraded",
]
CandidateEvidenceState = Literal[
    "supported",
    "zero_support",
    "unknown",
    "degraded",
]
FocusSelectionReason = Literal[
    "neutral_evidence",
    "lifecycle",
    "call_failure",
    "log_error",
    "required_one_hop_neighbor",
]


class TimeInterval(StrictFrozenModel):
    """Half-open UTC Unix-nanosecond interval ``[start, end)``."""

    start_unix_ns: int = Field(ge=0)
    end_unix_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_width(self) -> Self:
        if self.end_unix_ns <= self.start_unix_ns:
            raise ValueError("time interval end must be greater than start")
        return self


class CaseTimeContract(StrictFrozenModel):
    """Decision-time boundary shared by every method evaluating one case."""

    schema_version: Literal["nurse.v2.case-time-contract/1"] = TIME_CONTRACT_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    t0_unix_ns: int = Field(ge=0)
    t0_source: Literal["benchmark_fault_time", "shared_detector", "real_alarm"]
    localization_scope: Literal["given_fault_time", "detector_triggered", "alarm_triggered"]
    baseline_window: TimeInterval
    fault_window: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    timezone: str = Field(min_length=1)
    expected_cadence_ns: int = Field(gt=0)
    clock_uncertainty_ns: int = Field(ge=0)
    missing_modalities: tuple[TelemetryModality, ...] = ()
    late_data_observed: bool

    @model_validator(mode="after")
    def require_contiguous_decision_boundary(self) -> Self:
        if self.baseline_window.end_unix_ns != self.t0_unix_ns:
            raise ValueError("baseline window must end at t0")
        if self.fault_window.start_unix_ns != self.t0_unix_ns:
            raise ValueError("fault window must start at t0")
        if self.decision_time_unix_ns != self.fault_window.end_unix_ns:
            raise ValueError("decision time must equal fault window end")
        expected_scope = {
            "benchmark_fault_time": "given_fault_time",
            "shared_detector": "detector_triggered",
            "real_alarm": "alarm_triggered",
        }[self.t0_source]
        if self.localization_scope != expected_scope:
            raise ValueError("localization scope does not match t0 source")
        if self.missing_modalities != tuple(sorted(set(self.missing_modalities))):
            raise ValueError("missing modalities must be sorted and unique")
        return self


class DatasetManifest(StrictFrozenModel):
    """Inference-safe dataset contract over opaque case identities only."""

    schema_version: Literal["nurse.v2.dataset-manifest/1"] = DATASET_MANIFEST_SCHEMA_VERSION
    dataset: DatasetName
    adapter_version: str = Field(min_length=1)
    target_granularity: Literal["service"] = "service"
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=0)
    cases: tuple[CaseTimeContract, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        expected = tuple(sorted(self.cases, key=lambda item: item.opaque_case_id))
        if self.cases != expected or self.case_count != len(self.cases):
            return False
        if len({item.opaque_case_id for item in self.cases}) != len(self.cases):
            return False
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        return self.manifest_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_dataset_manifest(
    *,
    dataset: DatasetName,
    adapter_version: str,
    target_granularity: Literal["service"],
    source_inventory_sha256: str,
    adapter_configuration_sha256: str,
    cases: Sequence[CaseTimeContract],
) -> DatasetManifest:
    """Create a canonically ordered, content-addressed dataset manifest."""

    ordered = tuple(sorted(cases, key=lambda item: item.opaque_case_id))
    if len({item.opaque_case_id for item in ordered}) != len(ordered):
        raise ValueError("dataset case identities must be unique")
    unsigned: dict[str, object] = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "adapter_version": adapter_version,
        "target_granularity": target_granularity,
        "source_inventory_sha256": source_inventory_sha256,
        "adapter_configuration_sha256": adapter_configuration_sha256,
        "case_count": len(ordered),
        "cases": [item.model_dump(mode="json") for item in ordered],
    }
    manifest = DatasetManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not manifest.verify():  # pragma: no cover - construction invariant
        raise AssertionError("dataset manifest verification failed")
    return manifest


class WindowSummary(StrictFrozenModel):
    """Summary that distinguishes no record from observed zero and normal."""

    record_state: Literal["observed", "no_record"]
    valid_point_count: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    median: float | None = Field(default=None, allow_inf_nan=False)
    mad: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    event_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_consistent_record_state(self) -> Self:
        numeric = (self.median, self.mad, self.minimum, self.maximum)
        if self.record_state == "no_record":
            if (
                self.valid_point_count != 0
                or self.coverage_fraction != 0.0
                or any(value is not None for value in numeric)
                or self.event_count is not None
            ):
                raise ValueError("no-record summary cannot contain observations")
            return self
        if self.valid_point_count < 1:
            raise ValueError("observed summary requires at least one valid point")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("summary minimum cannot exceed maximum")
            if self.median is not None and not (self.minimum <= self.median <= self.maximum):
                raise ValueError("summary median must be within minimum and maximum")
        return self


class _ObservationFields(StrictFrozenModel):
    """Fields shared by sibling observation schema versions."""

    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    entity_binding_id: str | None = Field(
        default=None,
        pattern=r"^bind_[0-9a-f]{24}$",
    )
    modality: TelemetryModality
    signal_family: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    signal_type: SignalType
    unit: str = Field(min_length=1)
    interval: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    aggregation: str = Field(min_length=1)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_summary: WindowSummary
    fault_summary: WindowSummary
    evidence_state: EvidenceState
    anomaly_strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    coverage_quality: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_valid_points: int = Field(ge=1)
    transform: str = Field(min_length=1)
    quality_flags: tuple[str, ...]
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    formal_scoring_eligible: bool
    adapter_semantics_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def require_bounded_typed_evidence(self) -> Self:
        if self.interval.end_unix_ns > self.decision_time_unix_ns:
            raise ValueError("observation extends beyond decision time")
        for values, name in (
            (self.quality_flags, "quality flags"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.signal_type == "unknown" and (
            self.formal_scoring_eligible or self.evidence_state != "unknown"
        ):
            raise ValueError("unknown signal type must remain unknown and outside formal scoring")
        if self.formal_scoring_eligible and self.adapter_semantics_sha256 is None:
            raise ValueError("formal scoring requires frozen adapter semantics")
        insufficient = any(
            summary.record_state == "observed"
            and summary.valid_point_count < self.minimum_valid_points
            for summary in (self.baseline_summary, self.fault_summary)
        )
        if insufficient and self.evidence_state not in {"unknown", "degraded"}:
            raise ValueError("insufficient points require unknown or degraded evidence")
        if self.evidence_state == "zero" and self.anomaly_strength not in {0.0, None}:
            raise ValueError("zero evidence cannot have positive anomaly strength")
        if self.evidence_state == "no_record" and not (
            self.baseline_summary.record_state == "no_record"
            and self.fault_summary.record_state == "no_record"
        ):
            raise ValueError("no-record evidence requires two no-record summaries")
        return self


class ObservationV1(_ObservationFields):
    """Recognizable legacy observation contract without association anchoring."""

    schema_version: Literal["nurse.v2.observation/1"] = LEGACY_OBSERVATION_SCHEMA_VERSION


class Observation(_ObservationFields):
    """Current observation contract with an optional association-record anchor."""

    schema_version: Literal["nurse.v2.observation/2"] = OBSERVATION_SCHEMA_VERSION
    association_source_record_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class EntityBinding(StrictFrozenModel):
    """Time-aware entity binding that retains conflict and unbound states."""

    schema_version: Literal["nurse.v2.entity-binding/1"] = ENTITY_BINDING_SCHEMA_VERSION
    binding_id: str = Field(pattern=r"^bind_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observed_entity_key: str = Field(min_length=1)
    entity_kind: Literal["service", "workload", "pod", "container", "untyped"]
    status: Literal["bound", "unbound", "conflicted"]
    service_entity_key: str | None = Field(default=None, min_length=1)
    conflict_service_keys: tuple[str, ...]
    valid_interval: TimeInterval
    source_fingerprints: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_explicit_binding_state(self) -> Self:
        for values, name in (
            (self.conflict_service_keys, "conflict service keys"),
            (self.source_fingerprints, "source fingerprints"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_fingerprints
        ):
            raise ValueError("source fingerprints must contain SHA-256 values")
        if self.status == "bound":
            if self.service_entity_key is None or self.conflict_service_keys:
                raise ValueError("bound binding requires one service and no conflicts")
        elif self.status == "unbound":
            if self.service_entity_key is not None:
                raise ValueError("unbound binding cannot guess a service")
            if self.conflict_service_keys:
                raise ValueError("unbound binding cannot contain conflict candidates")
        elif self.service_entity_key is not None or len(self.conflict_service_keys) < 2:
            raise ValueError(
                "conflicted binding requires at least two candidates and no chosen service"
            )
        return self


class EventGroupV1(StrictFrozenModel):
    """Recognizable legacy event-group contract."""

    schema_version: Literal["nurse.v2.event-group/1"] = LEGACY_EVENT_GROUP_SCHEMA_VERSION
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    association_basis: Literal[
        "shared_export_id",
        "explicit_source_relation",
        "deterministic_association_key",
    ]
    association_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_ids: tuple[str, ...] = Field(min_length=1)
    entity_binding_ids: tuple[str, ...]
    signal_family: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    interval: TimeInterval
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_references(self) -> Self:
        for values, name in (
            (self.observation_ids, "observation IDs"),
            (self.entity_binding_ids, "entity binding IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self


class CandidateEntry(StrictFrozenModel):
    """One valid service in C, including zero-support and unknown candidates."""

    service_entity_key: str = Field(min_length=1)
    binding_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    evidence_state: CandidateEvidenceState

    @model_validator(mode="after")
    def require_canonical_evidence(self) -> Self:
        for values, name in (
            (self.binding_ids, "binding IDs"),
            (self.observation_ids, "observation IDs"),
            (self.event_group_ids, "event group IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        has_evidence = bool(self.observation_ids or self.event_group_ids)
        if self.evidence_state == "supported" and not has_evidence:
            raise ValueError("supported candidate requires evidence references")
        if self.evidence_state == "zero_support" and has_evidence:
            raise ValueError("zero-support candidate cannot cite retained evidence")
        return self


class CandidateUniverse(StrictFrozenModel):
    """Content-addressed complete rank-eligible service universe C."""

    schema_version: Literal["nurse.v2.candidate-universe/1"] = CANDIDATE_UNIVERSE_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1)
    candidates: tuple[CandidateEntry, ...] = Field(min_length=1)
    universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        expected = tuple(sorted(self.candidates, key=lambda item: item.service_entity_key))
        if self.candidates != expected or self.candidate_count != len(self.candidates):
            return False
        if len({item.service_entity_key for item in self.candidates}) != len(self.candidates):
            return False
        unsigned = self.model_dump(mode="json", exclude={"universe_sha256"})
        return self.universe_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_candidate_universe(
    *,
    opaque_case_id: str,
    observation_contract_sha256: str,
    candidates: Sequence[CandidateEntry],
) -> CandidateUniverse:
    """Freeze complete C without filtering zero-support or unknown services."""

    ordered = tuple(sorted(candidates, key=lambda item: item.service_entity_key))
    if not ordered:
        raise ValueError("candidate universe cannot be empty")
    if len({item.service_entity_key for item in ordered}) != len(ordered):
        raise ValueError("candidate service keys must be unique")
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_UNIVERSE_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "observation_contract_sha256": observation_contract_sha256,
        "candidate_count": len(ordered),
        "candidates": [item.model_dump(mode="json") for item in ordered],
    }
    universe = CandidateUniverse.model_validate(
        {
            **unsigned,
            "universe_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not universe.verify():  # pragma: no cover - construction invariant
        raise AssertionError("candidate universe verification failed")
    return universe


class FocusDecision(StrictFrozenModel):
    """Auditable Q inclusion or exclusion decision for one member of C."""

    service_entity_key: str = Field(min_length=1)
    selected: bool
    selection_phase: Literal["base", "expanded", "none"]
    selection_reasons: tuple[FocusSelectionReason, ...]
    exclusion_reason: Literal["not_selected_by_policy", "q_max_truncation"] | None

    @model_validator(mode="after")
    def require_consistent_focus_decision(self) -> Self:
        if self.selection_reasons != tuple(sorted(set(self.selection_reasons))):
            raise ValueError("focus selection reasons must be sorted and unique")
        if self.selected:
            if self.selection_phase == "none" or not self.selection_reasons:
                raise ValueError("selected focus candidate requires phase and reason")
            if self.exclusion_reason is not None:
                raise ValueError("selected focus candidate cannot have exclusion reason")
            if self.selection_phase == "base" and self.selection_reasons != ("neutral_evidence",):
                raise ValueError("base focus selection must use neutral evidence only")
        elif (
            self.selection_phase != "none"
            or self.selection_reasons
            or self.exclusion_reason is None
        ):
            raise ValueError("unselected focus candidate requires one exclusion reason")
        return self


class SemanticFocusSet(StrictFrozenModel):
    """Content-addressed Q allocation that never changes membership in C."""

    schema_version: Literal["nurse.v2.semantic-focus-set/1"] = FOCUS_SET_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    q_base: int = Field(ge=0)
    q_max: int = Field(ge=1)
    selected_count: int = Field(ge=0)
    selected_service_keys: tuple[str, ...]
    decisions: tuple[FocusDecision, ...] = Field(min_length=1)
    focus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_valid_limits(self) -> Self:
        if self.q_base > self.q_max:
            raise ValueError("q_base cannot exceed q_max")
        return self

    def verify(self) -> bool:
        expected = tuple(sorted(self.decisions, key=lambda item: item.service_entity_key))
        if self.decisions != expected:
            return False
        if len({item.service_entity_key for item in self.decisions}) != len(self.decisions):
            return False
        selected = tuple(item.service_entity_key for item in self.decisions if item.selected)
        if self.selected_service_keys != selected or self.selected_count != len(selected):
            return False
        if self.selected_count > self.q_max:
            return False
        if (
            sum(item.selected and item.selection_phase == "base" for item in self.decisions)
            > self.q_base
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"focus_sha256"})
        return self.focus_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_focus_set(
    *,
    candidate_universe: CandidateUniverse,
    q_base: int,
    q_max: int,
    decisions: Sequence[FocusDecision],
) -> SemanticFocusSet:
    """Freeze Q while requiring an audit decision for every member of C."""

    if not candidate_universe.verify():
        raise ValueError("candidate universe verification failed")
    ordered = tuple(sorted(decisions, key=lambda item: item.service_entity_key))
    candidate_keys = {item.service_entity_key for item in candidate_universe.candidates}
    decision_keys = {item.service_entity_key for item in ordered}
    if decision_keys != candidate_keys or len(ordered) != len(candidate_keys):
        raise ValueError("focus decisions must exactly cover candidate universe")
    selected = tuple(item.service_entity_key for item in ordered if item.selected)
    unsigned: dict[str, object] = {
        "schema_version": FOCUS_SET_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "q_base": q_base,
        "q_max": q_max,
        "selected_count": len(selected),
        "selected_service_keys": selected,
        "decisions": [item.model_dump(mode="json") for item in ordered],
    }
    focus = SemanticFocusSet.model_validate(
        {
            **unsigned,
            "focus_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not focus.verify():
        raise ValueError("semantic focus set violates q_base or q_max")
    return focus


class EvidenceCard(StrictFrozenModel):
    """Opaque, label-free references exposed for one candidate in C."""

    schema_version: Literal["nurse.v2.evidence-card/1"] = EVIDENCE_CARD_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    service_entity_key: str = Field(min_length=1)
    access_reason: Literal["focus_q", "approved_expansion"]
    decision_time_unix_ns: int = Field(ge=0)
    observation_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    entity_binding_ids: tuple[str, ...]
    quality_flags: tuple[str, ...]
    card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_references(self) -> Self:
        for values, name in (
            (self.observation_ids, "observation IDs"),
            (self.event_group_ids, "event group IDs"),
            (self.entity_binding_ids, "entity binding IDs"),
            (self.quality_flags, "quality flags"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"card_sha256"})
        return self.card_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_evidence_card(
    *,
    candidate_universe: CandidateUniverse,
    service_entity_key: str,
    access_reason: Literal["focus_q", "approved_expansion"],
    decision_time_unix_ns: int,
    observation_ids: Sequence[str],
    event_group_ids: Sequence[str],
    entity_binding_ids: Sequence[str],
    quality_flags: Sequence[str],
) -> EvidenceCard:
    """Create a card only for a member of frozen C using canonical references."""

    if not candidate_universe.verify():
        raise ValueError("candidate universe verification failed")
    if service_entity_key not in {
        item.service_entity_key for item in candidate_universe.candidates
    }:
        raise ValueError("evidence card candidate is outside candidate universe")
    unsigned: dict[str, object] = {
        "schema_version": EVIDENCE_CARD_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "service_entity_key": service_entity_key,
        "access_reason": access_reason,
        "decision_time_unix_ns": decision_time_unix_ns,
        "observation_ids": tuple(sorted(set(observation_ids))),
        "event_group_ids": tuple(sorted(set(event_group_ids))),
        "entity_binding_ids": tuple(sorted(set(entity_binding_ids))),
        "quality_flags": tuple(sorted(set(quality_flags))),
    }
    card = EvidenceCard.model_validate(
        {**unsigned, "card_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not card.verify():  # pragma: no cover - construction invariant
        raise AssertionError("evidence card verification failed")
    return card
