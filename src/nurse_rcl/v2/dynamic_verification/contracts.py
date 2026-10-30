"""Private source contracts; nested source rows are not independent outbound roots."""

from __future__ import annotations

from typing import Annotated, Literal, cast, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis, PredicateResult
from nurse_rcl.v2.event_grouping import EventGroup
from nurse_rcl.v2.graph_snapshots import GraphSnapshot, GraphSnapshotEdge
from nurse_rcl.v2.observation_quality import ObservationQualityResult
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.signal_transforms import SignalTransformResult, SourceWindowInput
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution
from nurse_rcl.v2.temporal_entity_registry import EntityKind

from .safety import canonical, exact_field, plain, preflight, snapshot

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ObservationId = Annotated[str, Field(pattern=r"^obs_[0-9a-f]{24}$")]
SeriesKey = Annotated[str, Field(pattern=r"^series_[0-9a-f]{64}$")]
SourceWindowKind = Literal["ScalarWindowInput", "SparseEventWindowInput", "HistogramWindowInput"]


class _SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def exact_python_fields(cls, value: object, info: ValidationInfo) -> object:
        preflight(value, (cls,))
        if info.mode == "json" and not exact_field(value, cls, json_mode=True):
            raise ValueError("dynamic source JSON requires exact declared field shapes")
        if info.mode == "python" and type(value) is dict:
            for name, item in cast(dict[str, object], value).items():
                if name not in cls.model_fields or not exact_field(
                    item, cls.model_fields[name].annotation
                ):
                    raise ValueError("dynamic source constructor requires exact typed fields")
        return value

    @field_validator("*", mode="before")
    @classmethod
    def json_tuple_fields(cls, value: object, info: ValidationInfo) -> object:
        # Before validators expose JSON arrays as Python lists in Pydantic 2.11.
        # Only JSON parsing may restore declared tuples; Python lists stay invalid.
        if (
            info.mode == "json"
            and info.field_name is not None
            and get_origin(cls.model_fields[info.field_name].annotation) is tuple
            and type(value) is list
        ):
            return tuple(value)
        return value


class NamedSourceWindow(_SourceModel):
    name: str = Field(min_length=1, max_length=128)
    window: SourceWindowInput


class SourceIdentity(_SourceModel):
    observed_entity_key: str = Field(min_length=1)
    entity_kind: EntityKind
    source_fingerprint: SHA256
    adapter_semantics_sha256: SHA256


def series_identity_key(
    *,
    case: str,
    freeze: str,
    candidate: str | None,
    identity: SourceIdentity,
    quality: ObservationQualityResult,
) -> str:
    """Canonical series semantics exclude observation aliases and raw point order."""
    return "series_" + sha256_hex(
        canonical_json_bytes(
            {
                "case": case,
                "freeze": freeze,
                "candidate": candidate,
                "unbound_entity": (
                    (identity.entity_kind, identity.observed_entity_key)
                    if candidate is None
                    else None
                ),
                "source": identity.source_fingerprint,
                "adapter": identity.adapter_semantics_sha256,
                "component": quality.component_name,
                "unit": quality.unit,
                "aggregation": quality.aggregation,
            }
        )
    )


class VerificationSourceRecord(SourceIdentity):
    observation_id: ObservationId
    source_inputs: tuple[NamedSourceWindow, ...] = Field(min_length=1)
    transform_result: SignalTransformResult
    quality_result: ObservationQualityResult


class SourceBlockReference(_SourceModel):
    name: str = Field(min_length=1, max_length=128)
    schema_version: Literal["nurse.v2.signal-transform-source-inputs/2"] = (
        "nurse.v2.signal-transform-source-inputs/2"
    )
    input_kind: SourceWindowKind
    sha256: SHA256
    size_bytes: int = Field(gt=0, le=64 * 1024 * 1024)


class VerificationSeries(_SourceModel):
    canonical_series_key: SeriesKey
    candidate_service_key: str | None
    source_fingerprint: SHA256
    adapter_semantics_sha256: SHA256
    transform_component: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    aggregation: str = Field(min_length=1)
    source_input_sha256: SHA256
    transform_result: SignalTransformResult
    quality_result: ObservationQualityResult
    source_blocks: tuple[SourceBlockReference, ...] = Field(min_length=1)
    expected_cadence_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def closed_semantics(self) -> Self:
        transform, quality = self.transform_result, self.quality_result
        if (
            not transform.verify()
            or not quality.verify_transform(transform_result=transform)
            or self.adapter_semantics_sha256 != transform.adapter_semantics_sha256
            or self.source_input_sha256 != transform.source_input_sha256
            or self.transform_component != quality.component_name
            or self.unit != quality.unit
            or self.aggregation != quality.aggregation
            or self.expected_cadence_ns != quality.context.expected_cadence_ns
            or (self.candidate_service_key is not None)
            != (quality.context.binding_status == "bound")
        ):
            raise ValueError("dynamic series transform/quality semantics are inconsistent")
        names = tuple(block.name for block in self.source_blocks)
        expected_names = {
            name
            for window in transform.windows
            for attribution in window.component_raw_attributions
            for name in attribution.direct_source_names
        }
        expected_kind = {
            "event_exposure_rate": "SparseEventWindowInput",
            "cumulative_histogram_percentiles": "HistogramWindowInput",
        }.get(transform.transform, "ScalarWindowInput")
        if names != tuple(sorted(expected_names)) or any(
            block.input_kind != expected_kind for block in self.source_blocks
        ):
            raise ValueError("dynamic series source block domain is inconsistent")
        return self


class VerificationSourceRow(_SourceModel):
    observation_id: ObservationId
    candidate_service_key: str | None
    event_group_ids: tuple[str, ...]
    canonical_series_key: SeriesKey | None
    availability: Literal["available", "unavailable"]
    reason_codes: tuple[str, ...]
    source_identity: SourceIdentity | None

    @model_validator(mode="after")
    def closed_availability(self) -> Self:
        if self.event_group_ids != tuple(sorted(set(self.event_group_ids))):
            raise ValueError("dynamic source event references must be sorted and unique")
        if (self.canonical_series_key is None) != (self.source_identity is None):
            raise ValueError("dynamic source row lacks its corresponding source identity")
        expected = (
            ("source_unavailable",)
            if self.canonical_series_key is None
            else ("entity_binding_unavailable",)
            if self.candidate_service_key is None
            else ()
        )
        if self.reason_codes != expected or self.availability != (
            "unavailable" if expected else "available"
        ):
            raise ValueError("dynamic source row availability is inconsistent")
        return self


class VerificationSourceCatalog(_SourceModel):
    schema_version: Literal["nurse.v2.verification-source-catalog/1"] = (
        "nurse.v2.verification-source-catalog/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    time_contract_sha256: SHA256
    source_records_sha256: SHA256
    rows: tuple[VerificationSourceRow, ...]
    series: tuple[VerificationSeries, ...]
    catalog_sha256: SHA256

    @model_validator(mode="after")
    def closed_rows(self) -> Self:
        ids = tuple(row.observation_id for row in self.rows)
        keys = tuple(row.canonical_series_key for row in self.series)
        if ids != tuple(sorted(set(ids))) or keys != tuple(sorted(set(keys))):
            raise ValueError("dynamic source catalog rows and series must be unique and ordered")
        if {
            row.canonical_series_key for row in self.rows if row.canonical_series_key is not None
        } != set(keys):
            raise ValueError("dynamic source catalog series references are not closed")
        by_key = {item.canonical_series_key: item for item in self.series}
        for row in self.rows:
            if row.canonical_series_key is None:
                continue
            item = by_key[row.canonical_series_key]
            identity = row.source_identity
            if (
                identity is None
                or row.candidate_service_key != item.candidate_service_key
                or identity.source_fingerprint != item.source_fingerprint
                or identity.adapter_semantics_sha256 != item.adapter_semantics_sha256
                or row.canonical_series_key
                != series_identity_key(
                    case=self.opaque_case_id,
                    freeze=self.candidate_freeze_sha256,
                    candidate=row.candidate_service_key,
                    identity=identity,
                    quality=item.quality_result,
                )
            ):
                raise ValueError("dynamic source row identity differs from its series")
        return self

    def verify(self) -> bool:
        """Internal integrity only; independent inputs remain mandatory for replay."""
        try:
            if type(self) is not VerificationSourceCatalog:
                return False
            before = snapshot(self, (VerificationSourceCatalog,))
            raw = before[0]
            rebuilt = VerificationSourceCatalog.model_validate_json(raw)
            unsigned = cast(dict[str, object], plain(rebuilt))
            unsigned.pop("catalog_sha256")
            return (
                raw == canonical(rebuilt, (VerificationSourceCatalog,))
                and self.catalog_sha256 == sha256_hex(canonical_json_bytes(unsigned))
                and snapshot(self, (VerificationSourceCatalog,)) == before
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


class DynamicVerificationPolicy(_SourceModel):
    schema_version: Literal["nurse.v2.dynamic-verification-policy/1"] = (
        "nurse.v2.dynamic-verification-policy/1"
    )
    operator: Literal["measure_signal_episodes"] = "measure_signal_episodes"
    quality_configuration_rule: Literal["inherit_frozen_source"] = "inherit_frozen_source"
    minimum_consecutive_samples: Literal[3] = 3
    minimum_span_cadences: Literal[2] = 2
    maximum_gap_numerator: Literal[3] = 3
    maximum_gap_denominator: Literal[2] = 2
    case_query_budget: Literal[8] = 8
    tie_seed: Literal["nurse-v2-b0-dynamic-verification/1"] = "nurse-v2-b0-dynamic-verification/1"
    consumption_version: Literal["dynamic-rule-consumption/1"] = "dynamic-rule-consumption/1"
    priority_version: Literal["feedback-lexicographic/1"] = "feedback-lexicographic/1"
    counting_rule: Literal["submitted_once_completed_unknown_counts_reuse_free/1"] = (
        "submitted_once_completed_unknown_counts_reuse_free/1"
    )
    termination_rule: Literal["budget_exhausted_or_no_legal_progress/1"] = (
        "budget_exhausted_or_no_legal_progress/1"
    )
    policy_sha256: SHA256

    def verify(self) -> bool:
        return _verify_request_root(self, DynamicVerificationPolicy, "policy_sha256")


EPISODE_OUTPUT_FIELDS = (
    "baseline_scale",
    "episodes",
    "persistent_deviation_present",
    "persistent_down",
    "persistent_up",
    "points",
    "quality",
    "reason_codes",
)


class DynamicVerificationRequest(_SourceModel):
    schema_version: Literal["nurse.v2.dynamic-verification-request/1"] = (
        "nurse.v2.dynamic-verification-request/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    time_contract_sha256: SHA256
    canonical_series_key: SeriesKey
    candidate_service_key: str | None
    transform_component: str = Field(min_length=1)
    baseline_window: TimeInterval
    fault_window: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    permission_scope: TimeInterval
    allowed_candidate_service_keys: tuple[str, ...]
    availability: Literal["available", "unavailable"]
    reason_codes: tuple[str, ...]
    operator: Literal["measure_signal_episodes"] = "measure_signal_episodes"
    policy_sha256: SHA256
    quality_configuration_sha256: SHA256
    output_fields: tuple[str, ...] = EPISODE_OUTPUT_FIELDS
    request_sha256: SHA256

    @model_validator(mode="after")
    def closed_scope(self) -> Self:
        if self.output_fields != EPISODE_OUTPUT_FIELDS:
            raise ValueError("dynamic request must return all fixed measurement fields")
        keys = self.allowed_candidate_service_keys
        if keys != tuple(sorted(set(keys))) or (
            self.candidate_service_key is not None and self.candidate_service_key not in keys
        ):
            raise ValueError("dynamic request is outside frozen candidate permissions")
        if not (
            self.permission_scope.start_unix_ns
            <= self.baseline_window.start_unix_ns
            < self.baseline_window.end_unix_ns
            == self.fault_window.start_unix_ns
            < self.fault_window.end_unix_ns
            == self.decision_time_unix_ns
            <= self.permission_scope.end_unix_ns
        ):
            raise ValueError("dynamic request must retain full contiguous frozen B/F")
        expected = () if self.candidate_service_key is not None else ("entity_binding_unavailable",)
        if self.reason_codes != expected or self.availability != (
            "unavailable" if expected else "available"
        ):
            raise ValueError("dynamic request availability differs from candidate permission")
        return self

    @property
    def canonical_request_key(self) -> str:
        return "request_" + self.request_sha256

    def verify(self) -> bool:
        """Internal shape/digest only; execution must reconstruct from independent inputs."""
        return _verify_request_root(self, DynamicVerificationRequest, "request_sha256")


def _verify_request_root(
    value: _SourceModel, exact_type: type[_SourceModel], hash_field: str
) -> bool:
    try:
        if type(value) is not exact_type:
            return False
        before = snapshot(value, (exact_type,))
        rebuilt = exact_type.model_validate_json(before[0])
        unsigned = cast(dict[str, object], plain(rebuilt))
        digest = unsigned.pop(hash_field)
        return (
            digest == sha256_hex(canonical_json_bytes(unsigned))
            and canonical(rebuilt, (exact_type,)) == before[0]
            and snapshot(value, (exact_type,)) == before
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


PointState = Literal["up", "down", "within_band", "unknown"]
Direction = Literal["up", "down"]
TruthValue = Literal["true", "false", "unknown"]
OnsetReason = Literal["observed_bracket", "window_start", "gap", "no_normal_bracket"]
EpisodeBoundary = Literal[
    "window_start",
    "window_end",
    "within_band",
    "unknown_point",
    "direction_reversal",
    "sampling_gap",
    "source_interval_gap",
]


class SignalEpisodePoint(_SourceModel):
    sample_index: int = Field(ge=0)
    timestamp_unix_ns: int = Field(ge=0)
    source_interval: TimeInterval | None
    value: float
    signed_z: float | None
    state: PointState
    reason_codes: tuple[str, ...]


class SignalEpisode(_SourceModel):
    episode_id: str = Field(pattern=r"^episode_[0-9a-f]{64}$")
    canonical_series_key: SeriesKey
    request_sha256: SHA256
    policy_sha256: SHA256
    sample_indices: tuple[int, ...] = Field(min_length=1)
    direction: Direction
    support_span_ns: int = Field(ge=0)
    qualifying: bool
    start_reason: EpisodeBoundary
    end_reason: EpisodeBoundary
    onset_lower_unix_ns: int | None = Field(ge=0)
    onset_upper_unix_ns: int = Field(ge=0)
    onset_reason: OnsetReason
    left_censored: bool
    right_censored: bool


class SignalEpisodeMeasurement(_SourceModel):
    schema_version: Literal["nurse.v2.signal-episode-measurement/1"] = (
        "nurse.v2.signal-episode-measurement/1"
    )
    request_sha256: SHA256
    catalog_sha256: SHA256
    policy_sha256: SHA256
    canonical_series_key: SeriesKey
    source_input_sha256: SHA256
    baseline_median: float | None
    baseline_scale: float | None = Field(gt=0.0)
    abnormal_z_threshold: float = Field(gt=0.0)
    points: tuple[SignalEpisodePoint, ...]
    episodes: tuple[SignalEpisode, ...]
    persistent_up: TruthValue
    persistent_down: TruthValue
    persistent_deviation_present: TruthValue
    quality: ObservationQualityResult
    reason_codes: tuple[str, ...]
    measurement_sha256: SHA256

    def verify(self) -> bool:
        """Internal integrity only; not proof that these points came from sources."""
        return _verify_request_root(self, SignalEpisodeMeasurement, "measurement_sha256")


ExpressionId = Annotated[str, Field(pattern=r"^expr_[0-9a-f]{64}$")]
ExpressionLeafKind = Literal[
    "original_window",
    "directional_persistence",
    "directional_time",
    "summary_not_persistent",
    "pending",
]
ExpressionKind = Literal[
    "original_window",
    "directional_persistence",
    "directional_time",
    "summary_not_persistent",
    "pending",
    "and",
    "or",
]


class DynamicFieldReference(_SourceModel):
    """A precise retained input reference, not a source-authentication certificate."""

    source_kind: Literal[
        "original_predicate",
        "observation",
        "measurement",
        "quality",
        "request",
        "time_pair",
        "clock_resolution",
        "summary",
        "transform",
        "policy",
    ]
    source_sha256: SHA256
    field_path: str = Field(min_length=1, max_length=1024)


def field_reference_key(value: DynamicFieldReference) -> tuple[str, str, str]:
    return value.source_kind, value.source_sha256, value.field_path


class DynamicSupportSource(_SourceModel):
    """One actual source; aliases and all consumed fields remain visible.

    Identity excludes aliases, evidence type and quality. A window summary and an
    episode from the same physical source cannot become two independent votes.
    Consumers still have to replay candidate/source/adapter and quality bindings.
    """

    candidate_service_key: str = Field(min_length=1)
    source_fingerprint: SHA256
    adapter_semantics_sha256: SHA256
    observation_ids: tuple[ObservationId, ...] = Field(min_length=1)
    field_references: tuple[DynamicFieldReference, ...] = Field(min_length=1)
    coverage_quality: float | None = Field(ge=0.0, le=1.0)

    @property
    def source_key(self) -> str:
        return "support_" + sha256_hex(
            canonical_json_bytes(
                {
                    "candidate_service_key": self.candidate_service_key,
                    "source_fingerprint": self.source_fingerprint,
                    "adapter_semantics_sha256": self.adapter_semantics_sha256,
                }
            )
        )

    @model_validator(mode="after")
    def canonical_source_references(self) -> Self:
        if self.observation_ids != tuple(sorted(set(self.observation_ids))):
            raise ValueError("dynamic support observation aliases must be sorted and unique")
        keys = tuple(field_reference_key(item) for item in self.field_references)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("dynamic support field references must be sorted and unique")
        return self


class DynamicExpressionNode(_SourceModel):
    """Private DAG row; content identity does not prove a leaf's telemetry origin."""

    expression_id: ExpressionId
    kind: ExpressionKind
    truth_value: TruthValue
    child_ids: tuple[ExpressionId, ...]
    field_references: tuple[DynamicFieldReference, ...]
    support_sources: tuple[DynamicSupportSource, ...]
    pending_demand_keys: tuple[str, ...]
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @property
    def minimum_support_quality(self) -> float | None:
        if not self.support_sources or any(
            item.coverage_quality is None for item in self.support_sources
        ):
            return None
        return min(cast(float, item.coverage_quality) for item in self.support_sources)

    @model_validator(mode="after")
    def canonical_node_shape(self) -> Self:
        for values in (self.child_ids, self.pending_demand_keys, self.reason_codes):
            if values != tuple(sorted(set(values))) or any(not item for item in values):
                raise ValueError("dynamic expression collections must be sorted and unique")
        if bool(self.child_ids) != (self.kind in {"and", "or"}):
            raise ValueError(
                "dynamic expression operators require nonempty children; leaves do not"
            )
        references = tuple(field_reference_key(item) for item in self.field_references)
        if references != tuple(sorted(set(references))):
            raise ValueError("dynamic expression field references must be sorted and unique")
        keys = tuple(item.source_key for item in self.support_sources)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("dynamic expression support sources must be sorted and unique")
        if self.truth_value != "true" and self.support_sources:
            raise ValueError("nontrue dynamic expressions cannot provide positive support")
        if self.kind == "pending" and self.truth_value != "unknown":
            raise ValueError("pending dynamic expressions must remain unknown")
        if any(
            field_reference_key(ref) not in set(references)
            for item in self.support_sources
            for ref in item.field_references
        ):
            raise ValueError("dynamic expression support references are not closed")
        return self


class DynamicTimeSide(_SourceModel):
    """Complete frozen source domain; multiple series must not be cherry-picked."""

    candidate_service_key: str = Field(min_length=1)
    predicate_kinds: tuple[str, ...] = Field(min_length=1)
    observation_ids: tuple[ObservationId, ...]
    event_group_ids: tuple[str, ...]
    canonical_series_keys: tuple[SeriesKey, ...]
    request_keys: tuple[str, ...]
    direction: Direction

    @model_validator(mode="after")
    def canonical_domain(self) -> Self:
        for values in (
            self.predicate_kinds,
            self.observation_ids,
            self.event_group_ids,
            self.canonical_series_keys,
            self.request_keys,
        ):
            if values != tuple(sorted(set(values))) or any(not value for value in values):
                raise ValueError("dynamic time side requires complete canonical domains")
        return self


class DynamicLocalTimePair(_SourceModel):
    """Nested fixed-recipe row, not a stand-alone time or causal certificate."""

    time_pair_key: str = Field(pattern=r"^timepair_[0-9a-f]{64}$")
    condition_key: str = Field(pattern=r"^condition_[0-9a-f]{64}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    time_contract_sha256: SHA256
    catalog_sha256: SHA256
    source_rule_set_sha256: SHA256
    source_kind: Literal["mechanism_seed"] = "mechanism_seed"
    seed_hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    seed_predicate_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    predicate_kind: Literal[
        "incoming_load_precedes_local_pressure", "dependency_failure_precedes_local_symptom"
    ]
    requirement: Literal["optional"] = "optional"
    source: DynamicTimeSide
    target: DynamicTimeSide
    relation_window: TimeInterval

    @property
    def time_pair_sha256(self) -> str:
        return self.time_pair_key.removeprefix("timepair_")

    @model_validator(mode="after")
    def closed_local_pair(self) -> Self:
        if self.source.candidate_service_key != self.target.candidate_service_key:
            raise ValueError("local mechanism time pair cannot authorize a cross-candidate edge")
        document = cast(dict[str, object], plain(self))
        document.pop("time_pair_key")
        if self.time_pair_key != "timepair_" + sha256_hex(canonical_json_bytes(document)):
            raise ValueError("dynamic time pair content identity mismatch")
        return self


class DynamicGraphTimeBinding(_SourceModel):
    """One pre-query candidate/event/mechanism/role domain, not an accepted role."""

    candidate_service_key: str = Field(min_length=1)
    event_group_id: str = Field(min_length=1)
    mechanism_key: str = Field(min_length=1)
    role: Literal["source_support", "symptom_support"]
    seed: DiagnosticHypothesis
    event: EventGroup
    necessary_predicates: tuple[PredicateResult, ...]
    domain: DynamicTimeSide | None
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closed_binding(self) -> Self:
        if (
            self.candidate_service_key != self.seed.candidate_service_key
            or self.mechanism_key != self.seed.mechanism_key
            or self.event_group_id != self.event.event_group_id
            or self.event_group_id not in self.seed.event_group_ids
            or tuple(row.predicate_result_id for row in self.necessary_predicates)
            != tuple(sorted(self.seed.necessary_predicate_result_ids))
            or self.reason_codes != tuple(sorted(set(self.reason_codes)))
        ):
            raise ValueError("graph time binding is not closed over its seed/event")
        if self.domain is not None and (
            self.domain.candidate_service_key != self.candidate_service_key
            or self.domain.event_group_ids != (self.event_group_id,)
            or not set(self.domain.observation_ids) <= set(self.event.observation_ids)
        ):
            raise ValueError("graph time domain differs from its event binding")
        return self


class DynamicGraphTimePair(_SourceModel):
    """Nested graph-derived comparison; complete inputs must independently replay it."""

    time_pair_key: str = Field(pattern=r"^timepair_[0-9a-f]{64}$")
    condition_key: str = Field(pattern=r"^condition_[0-9a-f]{64}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    time_contract_sha256: SHA256
    catalog_sha256: SHA256
    source_rule_set_sha256: SHA256
    source_kind: Literal["graph_edge"] = "graph_edge"
    graph_snapshot_sha256: SHA256
    relation_policy_sha256: SHA256
    edge: GraphSnapshotEdge
    source: DynamicGraphTimeBinding
    target: DynamicGraphTimeBinding
    relation_window: TimeInterval

    @property
    def time_pair_sha256(self) -> str:
        return self.time_pair_key.removeprefix("timepair_")

    @model_validator(mode="after")
    def closed_graph_pair(self) -> Self:
        if self.source.role != "source_support" or self.target.role != "symptom_support":
            raise ValueError("graph time pair requires source and symptom binding roles")
        document = cast(dict[str, object], plain(self))
        document.pop("time_pair_key")
        if self.time_pair_key != "timepair_" + sha256_hex(canonical_json_bytes(document)):
            raise ValueError("graph time pair content identity mismatch")
        return self


class DynamicDemand(_SourceModel):
    """A semantic need, separate from the paid measurement identity and seed count."""

    demand_key: str = Field(pattern=r"^demand_[0-9a-f]{64}$")
    candidate_service_key: str = Field(min_length=1)
    seed_ids: tuple[str, ...] = Field(min_length=1)
    source_predicate_ids: tuple[str, ...]
    condition_key: str = Field(pattern=r"^condition_[0-9a-f]{64}$")
    predicate_kind: str = Field(min_length=1)
    requirement: Literal["necessary", "optional", "refuting"]
    observation_ids: tuple[ObservationId, ...]
    request_keys: tuple[str, ...]
    time_pair_key: str | None
    time_side: Literal["source", "target"] | None
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_demand(self) -> Self:
        for values in (
            self.seed_ids,
            self.source_predicate_ids,
            self.observation_ids,
            self.request_keys,
            self.reason_codes,
        ):
            if values != tuple(sorted(set(values))) or any(not item for item in values):
                raise ValueError("dynamic demand references must be sorted and unique")
        if (self.time_pair_key is None) != (self.time_side is None):
            raise ValueError("dynamic time demand requires a pair and side together")
        if self.time_pair_key is not None and self.requirement != "optional":
            raise ValueError("dynamic time cannot become a necessary or refuting demand")
        if self.demand_key != dynamic_demand_key(cast(dict[str, object], plain(self))):
            raise ValueError("dynamic demand semantic identity mismatch")
        return self


def dynamic_demand_key(value: dict[str, object]) -> str:
    document = dict(value)
    # Provenance multiplicity must not create extra priority votes.
    for name in (
        "demand_key",
        "seed_ids",
        "source_predicate_ids",
        "observation_ids",
        "reason_codes",
    ):
        document.pop(name, None)
    return "demand_" + sha256_hex(canonical_json_bytes(document))


class DynamicRequestCatalog(_SourceModel):
    """Nested pre-query directory. Internal verify is not independent source authority."""

    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    candidate_service_keys: tuple[str, ...] = Field(min_length=1)
    time_contract_sha256: SHA256
    source_rule_set_sha256: SHA256
    catalog_sha256: SHA256
    policy_sha256: SHA256
    requests: tuple[DynamicVerificationRequest, ...]
    demands: tuple[DynamicDemand, ...]
    time_pairs: tuple[DynamicLocalTimePair | DynamicGraphTimePair, ...]
    graph_snapshots: tuple[GraphSnapshot, ...]
    request_catalog_sha256: SHA256

    @model_validator(mode="after")
    def closed_directory(self) -> Self:
        request_keys = tuple(row.canonical_request_key for row in self.requests)
        pair_keys = tuple(row.time_pair_key for row in self.time_pairs)
        for keys in (
            self.candidate_service_keys,
            request_keys,
            pair_keys,
            tuple(row.demand_key for row in self.demands),
            tuple(row.snapshot_sha256 for row in self.graph_snapshots),
        ):
            if keys != tuple(sorted(set(keys))):
                raise ValueError("dynamic request catalog domains must be ordered and unique")
        if any(not row.verify() for row in self.requests + self.graph_snapshots):
            raise ValueError("dynamic request catalog contains invalid frozen sources")
        for request in self.requests:
            if (
                request.opaque_case_id != self.opaque_case_id
                or request.candidate_freeze_sha256 != self.candidate_freeze_sha256
                or request.time_contract_sha256 != self.time_contract_sha256
                or request.catalog_sha256 != self.catalog_sha256
                or request.policy_sha256 != self.policy_sha256
                or request.allowed_candidate_service_keys != self.candidate_service_keys
            ):
                raise ValueError("dynamic request catalog request headers differ")
        for pair in self.time_pairs:
            if (
                pair.opaque_case_id != self.opaque_case_id
                or pair.candidate_freeze_sha256 != self.candidate_freeze_sha256
                or pair.time_contract_sha256 != self.time_contract_sha256
                or pair.catalog_sha256 != self.catalog_sha256
                or pair.source_rule_set_sha256 != self.source_rule_set_sha256
            ):
                raise ValueError("dynamic request catalog time pair headers differ")
        for graph in self.graph_snapshots:
            if (
                graph.opaque_case_id != self.opaque_case_id
                or graph.candidate_freeze_sha256 != self.candidate_freeze_sha256
                or graph.node_service_keys != self.candidate_service_keys
            ):
                raise ValueError("dynamic request catalog graph headers differ")
        requests_by_key = {row.canonical_request_key: row for row in self.requests}
        if any(
            row.candidate_service_key not in self.candidate_service_keys
            or not set(row.request_keys) <= set(request_keys)
            or (row.time_pair_key is not None and row.time_pair_key not in pair_keys)
            for row in self.demands
        ):
            raise ValueError("dynamic request catalog demand references are not closed")
        if any(
            requests_by_key[key].candidate_service_key != row.candidate_service_key
            for row in self.demands
            for key in row.request_keys
        ):
            raise ValueError("dynamic demand cannot borrow another candidate's request")
        graphs_by_key = {row.snapshot_sha256: row for row in self.graph_snapshots}
        time_demands = {
            (row.time_pair_key, row.time_side): row
            for row in self.demands
            if row.time_pair_key is not None
        }
        if len(time_demands) != sum(row.time_pair_key is not None for row in self.demands):
            raise ValueError("dynamic time pair side has duplicate demands")
        for pair in self.time_pairs:
            if type(pair) is DynamicGraphTimePair:
                graph_source = graphs_by_key.get(pair.graph_snapshot_sha256)
                if graph_source is None or pair.edge not in graph_source.edges:
                    raise ValueError("dynamic time pair lacks its full graph source")
            for side_name in ("source", "target"):
                domain: DynamicTimeSide | None
                predicate_ids: tuple[str, ...]
                kind: str
                if type(pair) is DynamicLocalTimePair:
                    domain = pair.source if side_name == "source" else pair.target
                    candidate = domain.candidate_service_key
                    seed_ids = (pair.seed_hypothesis_id,)
                    predicate_ids = (pair.seed_predicate_id,)
                    kind = pair.predicate_kind
                elif type(pair) is DynamicGraphTimePair:
                    binding = pair.source if side_name == "source" else pair.target
                    domain = binding.domain
                    candidate = binding.candidate_service_key
                    seed_ids = (binding.seed.hypothesis_id,)
                    predicate_ids = tuple(
                        row.predicate_result_id for row in binding.necessary_predicates
                    )
                    kind = "source_before_target"
                else:
                    raise ValueError("dynamic directory requires an exact declared time pair")
                demand = time_demands.get((pair.time_pair_key, side_name))
                if demand is None or (
                    demand.condition_key != pair.condition_key
                    or demand.candidate_service_key != candidate
                    or demand.seed_ids != seed_ids
                    or demand.source_predicate_ids != predicate_ids
                    or demand.predicate_kind != kind
                    or demand.observation_ids != (domain.observation_ids if domain else ())
                    or demand.request_keys != (domain.request_keys if domain else ())
                ):
                    raise ValueError("dynamic time demand differs from its frozen pair side")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not DynamicRequestCatalog:
                return False
            before = snapshot(self, (DynamicRequestCatalog,))
            fresh = DynamicRequestCatalog.model_validate_json(before[0])
            document = cast(dict[str, object], plain(fresh))
            document.pop("request_catalog_sha256")
            return (
                canonical(fresh, (DynamicRequestCatalog,)) == before[0]
                and self.request_catalog_sha256 == sha256_hex(canonical_json_bytes(document))
                and snapshot(self, (DynamicRequestCatalog,)) == before
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


class DynamicConditionEvaluation(_SourceModel):
    seed_hypothesis_id: str
    seed_predicate_id: str
    requirement: Literal["necessary", "optional", "refuting"]
    demand_key: str
    original_expression_id: ExpressionId
    expression_id: ExpressionId


class DynamicSeedEvaluation(_SourceModel):
    """Condition mapping only: this is neither a final hypothesis nor a role grant."""

    seed_hypothesis_id: str
    candidate_service_key: str
    mechanism_key: str
    necessary_expression_ids: tuple[ExpressionId, ...] = Field(min_length=1)
    optional_expression_ids: tuple[ExpressionId, ...]
    refuting_expression_ids: tuple[ExpressionId, ...]
    necessary_expression_id: ExpressionId


class DynamicTimeSideEvaluation(_SourceModel):
    """Observed-prefix diagnostics for selection; never contains unqueried samples."""

    demand_key: str
    onset_bounds: tuple[int, int] | None
    field_references: tuple[DynamicFieldReference, ...]
    reason_codes: tuple[str, ...]
    measurement_pending: bool

    @field_validator("onset_bounds", mode="before")
    @classmethod
    def json_optional_bounds(cls, value: object, info: ValidationInfo) -> object:
        # The generic tuple restorer does not cover Optional[tuple[int, int]].
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        return value


class DynamicTimeEvaluation(_SourceModel):
    time_pair_key: str
    expression_id: ExpressionId
    source: DynamicTimeSideEvaluation
    target: DynamicTimeSideEvaluation


class DynamicRequestConsumption(_SourceModel):
    """An audit read is not positive support and never implies score consumption."""

    request_key: str
    measurement_sha256: SHA256 | None
    expression_ids: tuple[ExpressionId, ...]
    reason_codes: tuple[str, ...] = Field(min_length=1)


class DynamicPredicateEvaluation(_SourceModel):
    """Pure complete evaluation; session/guard must still replay actual A4 blocks."""

    schema_version: Literal["nurse.v2.dynamic-predicate-evaluation/1"] = (
        "nurse.v2.dynamic-predicate-evaluation/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    source_rule_set_sha256: SHA256
    original_binding_sha256: SHA256
    time_contract_sha256: SHA256
    catalog_sha256: SHA256
    policy_sha256: SHA256
    request_catalog_sha256: SHA256
    clock_resolution: FrozenClockResolution
    original_predicates: tuple[PredicateResult, ...]
    expression_nodes: tuple[DynamicExpressionNode, ...] = Field(min_length=1)
    conditions: tuple[DynamicConditionEvaluation, ...]
    seed_evaluations: tuple[DynamicSeedEvaluation, ...] = Field(min_length=1)
    time_evaluations: tuple[DynamicTimeEvaluation, ...]
    request_consumption: tuple[DynamicRequestConsumption, ...]
    pending_demand_keys: tuple[str, ...]
    evaluation_sha256: SHA256

    @property
    def root_expression_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    key
                    for row in self.conditions
                    for key in (row.expression_id, row.original_expression_id)
                }
                | {row.necessary_expression_id for row in self.seed_evaluations}
                | {row.expression_id for row in self.time_evaluations}
            )
        )

    @model_validator(mode="after")
    def closed_evaluation(self) -> Self:
        for keys in (
            tuple(row.predicate_result_id for row in self.original_predicates),
            tuple(row.seed_hypothesis_id for row in self.seed_evaluations),
            tuple(row.time_pair_key for row in self.time_evaluations),
            tuple(row.request_key for row in self.request_consumption),
            self.pending_demand_keys,
        ):
            if keys != tuple(sorted(set(keys))):
                raise ValueError("dynamic evaluation mappings must be ordered and unique")
        condition_keys = tuple(
            (row.seed_hypothesis_id, row.seed_predicate_id) for row in self.conditions
        )
        if condition_keys != tuple(sorted(set(condition_keys))):
            raise ValueError("dynamic evaluation condition mappings must be ordered and unique")
        nodes = {row.expression_id: row for row in self.expression_nodes}
        if len(nodes) != len(self.expression_nodes) or not set(self.root_expression_ids) <= set(
            nodes
        ):
            raise ValueError("dynamic evaluation expression references are not closed")
        if self.pending_demand_keys != tuple(
            sorted({key for node in nodes.values() for key in node.pending_demand_keys})
        ):
            raise ValueError("dynamic evaluation pending demands differ from its complete DAG")
        for seed in self.seed_evaluations:
            for requirement, ids in (
                ("necessary", seed.necessary_expression_ids),
                ("optional", seed.optional_expression_ids),
                ("refuting", seed.refuting_expression_ids),
            ):
                expected = tuple(
                    sorted(
                        {
                            row.expression_id
                            for row in self.conditions
                            if row.seed_hypothesis_id == seed.seed_hypothesis_id
                            and row.requirement == requirement
                        }
                    )
                )
                if ids != expected:
                    raise ValueError("dynamic evaluation seed condition mapping differs")
            necessary = nodes[seed.necessary_expression_id]
            if necessary.kind != "and" or necessary.child_ids != seed.necessary_expression_ids:
                raise ValueError("dynamic evaluation necessary condition conjunction differs")
        if {row.seed_hypothesis_id for row in self.conditions} != {
            row.seed_hypothesis_id for row in self.seed_evaluations
        }:
            raise ValueError("dynamic evaluation contains unmapped seed conditions")
        for row in self.request_consumption:
            expected_consumers = tuple(
                sorted(
                    node.expression_id
                    for node in self.expression_nodes
                    if any(
                        ref.source_kind == "measurement"
                        and ref.source_sha256 == row.measurement_sha256
                        for ref in node.field_references
                    )
                )
            )
            if row.expression_ids != expected_consumers:
                raise ValueError(
                    "dynamic evaluation request consumption differs from actual fields"
                )
        if (
            self.clock_resolution.opaque_case_id != self.opaque_case_id
            or self.clock_resolution.time_contract_sha256 != self.time_contract_sha256
            or not self.clock_resolution.verify()
            or any(not row.verify() for row in self.original_predicates)
        ):
            raise ValueError("dynamic evaluation original predicate/clock integrity failed")
        return self

    def verify(self) -> bool:
        """Internal DAG/closure only. Independent input pins and source replay are mandatory."""
        from .predicate_evaluation import replay_dynamic_expression_dag

        try:
            if type(self) is not DynamicPredicateEvaluation:
                return False
            before = snapshot(self, (DynamicPredicateEvaluation,))
            fresh = DynamicPredicateEvaluation.model_validate_json(before[0])
            replay_dynamic_expression_dag(
                nodes=fresh.expression_nodes, root_ids=fresh.root_expression_ids
            )
            document = cast(dict[str, object], plain(fresh))
            document.pop("evaluation_sha256")
            return (
                canonical(fresh, (DynamicPredicateEvaluation,)) == before[0]
                and self.evaluation_sha256 == sha256_hex(canonical_json_bytes(document))
                and snapshot(self, (DynamicPredicateEvaluation,)) == before
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
