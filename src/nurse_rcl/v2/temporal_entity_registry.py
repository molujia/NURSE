"""Source- and time-aware entity binding for inference-safe V2 observations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import EntityBinding, Observation, TimeInterval

ENTITY_METADATA_RECORD_SCHEMA_VERSION: Literal["nurse.v2.entity-metadata-record/1"] = (
    "nurse.v2.entity-metadata-record/1"
)
OBSERVATION_BINDING_RESOLUTION_SCHEMA_VERSION: Literal[
    "nurse.v2.observation-binding-resolution/1"
] = "nurse.v2.observation-binding-resolution/1"
TEMPORAL_ENTITY_REGISTRY_SCHEMA_VERSION: Literal["nurse.v2.temporal-entity-registry/1"] = (
    "nurse.v2.temporal-entity-registry/1"
)

MetadataSource = Literal["deployment", "metrics", "logs", "traces"]
EntityKind = Literal["service", "workload", "pod", "container", "untyped"]
MetadataRelation = Literal["explicit_service_binding", "collector_source"]
ResolutionReason = Literal[
    "complete_explicit_mapping",
    "conflicting_service_claims",
    "partial_metadata_coverage",
    "service_changed_during_observation",
    "collector_only_metadata",
    "no_explicit_service_mapping",
]


def _stable_id(prefix: str, payload: object) -> str:
    return f"{prefix}_{sha256_hex(canonical_json_bytes(payload))[:24]}"


class EntityMetadataRecord(StrictFrozenModel):
    """One inference-safe metadata assertion with an explicit validity interval."""

    schema_version: Literal["nurse.v2.entity-metadata-record/1"] = (
        ENTITY_METADATA_RECORD_SCHEMA_VERSION
    )
    metadata_record_id: str = Field(pattern=r"^meta_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    metadata_source: MetadataSource
    entity_kind: EntityKind
    observed_entity_key: str = Field(min_length=1)
    relation: MetadataRelation
    service_entity_key: str | None = Field(default=None, min_length=1)
    valid_interval: TimeInterval
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_explicit_relation(self) -> Self:
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("metadata provenance IDs must be sorted and unique")
        if self.relation == "explicit_service_binding":
            if self.service_entity_key is None:
                raise ValueError("explicit service binding requires a service key")
        elif self.service_entity_key is not None:
            raise ValueError("collector source cannot claim a service")
        if self.metadata_record_id != _metadata_record_id(self):
            raise ValueError("metadata record ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = EntityMetadataRecord.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


class ObservationEntityInput(StrictFrozenModel):
    """An observation paired with its explicit observed entity identity."""

    observation: Observation
    entity_kind: EntityKind
    observed_entity_key: str = Field(min_length=1)


class ObservationBindingResolution(StrictFrozenModel):
    """Auditable per-observation result, separate from the registry timeline."""

    schema_version: Literal["nurse.v2.observation-binding-resolution/1"] = (
        OBSERVATION_BINDING_RESOLUTION_SCHEMA_VERSION
    )
    resolution_id: str = Field(pattern=r"^res_[0-9a-f]{24}$")
    observation: Observation
    entity_kind: EntityKind
    observed_entity_key: str = Field(min_length=1)
    binding: EntityBinding
    reason: ResolutionReason

    @model_validator(mode="after")
    def require_linked_observation(self) -> Self:
        if self.observation.entity_binding_id != self.binding.binding_id:
            raise ValueError("resolved observation must reference its binding")
        if self.observation.opaque_case_id != self.binding.opaque_case_id:
            raise ValueError("observation and binding must share one opaque case")
        if self.entity_kind != self.binding.entity_kind:
            raise ValueError("resolution entity kind must match binding")
        if self.observed_entity_key != self.binding.observed_entity_key:
            raise ValueError("resolution entity key must match binding")
        valid_reasons = {
            "bound": {"complete_explicit_mapping"},
            "conflicted": {"conflicting_service_claims"},
            "unbound": {
                "partial_metadata_coverage",
                "service_changed_during_observation",
                "collector_only_metadata",
                "no_explicit_service_mapping",
            },
        }
        if self.reason not in valid_reasons[self.binding.status]:
            raise ValueError("resolution reason does not match binding status")
        if self.resolution_id != _resolution_id(self):
            raise ValueError("resolution ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = ObservationBindingResolution.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            binding_verified = self.binding.binding_id == _binding_id(self.binding)
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self and binding_verified


class TemporalEntityRegistry(StrictFrozenModel):
    """Content-addressed registry timeline plus atomic observation resolutions."""

    schema_version: Literal["nurse.v2.temporal-entity-registry/1"] = (
        TEMPORAL_ENTITY_REGISTRY_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    decision_time_unix_ns: int = Field(ge=0)
    metadata_records: tuple[EntityMetadataRecord, ...]
    timeline_bindings: tuple[EntityBinding, ...]
    resolutions: tuple[ObservationBindingResolution, ...]
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = TemporalEntityRegistry.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated._verify_hash_and_structure():
                return False
            expected = _rebuild_temporal_entity_registry(validated)
            return expected == validated
        except (AttributeError, TypeError, ValueError):
            return False

    def _verify_hash_and_structure(self) -> bool:
        if self.metadata_records != tuple(
            sorted(self.metadata_records, key=lambda item: item.metadata_record_id)
        ):
            return False
        if self.resolutions != tuple(
            sorted(self.resolutions, key=lambda item: item.observation.observation_id)
        ):
            return False
        if self.timeline_bindings != tuple(sorted(self.timeline_bindings, key=_binding_sort_key)):
            return False
        if any(not item.verify() for item in self.metadata_records):
            return False
        if len({item.metadata_record_id for item in self.metadata_records}) != len(
            self.metadata_records
        ):
            return False
        if any(not item.verify() for item in self.resolutions):
            return False
        if any(item.binding_id != _binding_id(item) for item in self.timeline_bindings):
            return False
        unsigned = self.model_dump(mode="json", exclude={"registry_sha256"})
        return self.registry_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _metadata_record_id(record: EntityMetadataRecord) -> str:
    return _stable_id(
        "meta",
        record.model_dump(mode="json", exclude={"metadata_record_id"}),
    )


def build_entity_metadata_record(
    *,
    opaque_case_id: str,
    metadata_source: MetadataSource,
    entity_kind: EntityKind,
    observed_entity_key: str,
    relation: MetadataRelation,
    service_entity_key: str | None,
    valid_interval: TimeInterval,
    source_fingerprint: str,
    provenance_ids: Sequence[str],
) -> EntityMetadataRecord:
    """Build one stable metadata record without label or source-path fields."""

    unsigned: dict[str, object] = {
        "schema_version": ENTITY_METADATA_RECORD_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "metadata_source": metadata_source,
        "entity_kind": entity_kind,
        "observed_entity_key": observed_entity_key,
        "relation": relation,
        "service_entity_key": service_entity_key,
        "valid_interval": valid_interval.model_dump(mode="json"),
        "source_fingerprint": source_fingerprint,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    return EntityMetadataRecord.model_validate(
        {
            **unsigned,
            "metadata_record_id": _stable_id("meta", unsigned),
        }
    )


def _binding_id(binding: EntityBinding) -> str:
    return _stable_id(
        "bind",
        binding.model_dump(mode="json", exclude={"binding_id"}),
    )


def _build_binding(
    *,
    opaque_case_id: str,
    entity_kind: EntityKind,
    observed_entity_key: str,
    status: Literal["bound", "unbound", "conflicted"],
    service_entity_key: str | None,
    conflict_service_keys: tuple[str, ...],
    valid_interval: TimeInterval,
    source_fingerprints: tuple[str, ...],
    provenance_ids: tuple[str, ...],
) -> EntityBinding:
    unsigned: dict[str, object] = {
        "schema_version": "nurse.v2.entity-binding/1",
        "opaque_case_id": opaque_case_id,
        "observed_entity_key": observed_entity_key,
        "entity_kind": entity_kind,
        "status": status,
        "service_entity_key": service_entity_key,
        "conflict_service_keys": conflict_service_keys,
        "valid_interval": valid_interval.model_dump(mode="json"),
        "source_fingerprints": source_fingerprints,
        "provenance_ids": provenance_ids,
    }
    return EntityBinding.model_validate(
        {
            **unsigned,
            "binding_id": _stable_id("bind", unsigned),
        }
    )


def _resolution_id(resolution: ObservationBindingResolution) -> str:
    return _stable_id(
        "res",
        resolution.model_dump(mode="json", exclude={"resolution_id"}),
    )


def _resolve_observation(
    *,
    item: ObservationEntityInput,
    records: tuple[EntityMetadataRecord, ...],
) -> ObservationBindingResolution:
    relevant = tuple(
        record
        for record in records
        if record.entity_kind == item.entity_kind
        and record.observed_entity_key == item.observed_entity_key
        and record.valid_interval.start_unix_ns < item.observation.interval.end_unix_ns
        and record.valid_interval.end_unix_ns > item.observation.interval.start_unix_ns
    )
    boundaries = {
        item.observation.interval.start_unix_ns,
        item.observation.interval.end_unix_ns,
    }
    for record in relevant:
        boundaries.add(
            max(
                item.observation.interval.start_unix_ns,
                record.valid_interval.start_unix_ns,
            )
        )
        boundaries.add(
            min(
                item.observation.interval.end_unix_ns,
                record.valid_interval.end_unix_ns,
            )
        )
    points = sorted(boundaries)
    segment_services = tuple(
        {
            record.service_entity_key
            for record in relevant
            if record.relation == "explicit_service_binding"
            and record.service_entity_key is not None
            and record.valid_interval.start_unix_ns <= start
            and record.valid_interval.end_unix_ns >= end
        }
        for start, end in pairwise(points)
        if start < end
    )
    services = tuple(sorted(set().union(*segment_services))) if segment_services else ()
    has_conflict = any(len(values) > 1 for values in segment_services)
    has_gap = any(not values for values in segment_services)
    status: Literal["bound", "unbound", "conflicted"]
    reason: ResolutionReason
    if has_conflict:
        status = "conflicted"
        service = None
        conflicts = services
        reason = "conflicting_service_claims"
    elif not has_gap and len(services) == 1:
        status = "bound"
        service = services[0]
        conflicts = ()
        reason = "complete_explicit_mapping"
    else:
        status = "unbound"
        service = None
        conflicts = ()
        if has_gap and services:
            reason = "partial_metadata_coverage"
        elif len(services) > 1:
            reason = "service_changed_during_observation"
        elif relevant:
            reason = "collector_only_metadata"
        else:
            reason = "no_explicit_service_mapping"
    binding = _build_binding(
        opaque_case_id=item.observation.opaque_case_id,
        entity_kind=item.entity_kind,
        observed_entity_key=item.observed_entity_key,
        status=status,
        service_entity_key=service,
        conflict_service_keys=conflicts,
        valid_interval=item.observation.interval,
        source_fingerprints=tuple(
            sorted(
                {record.source_fingerprint for record in relevant}
                or {item.observation.source_fingerprint}
            )
        ),
        provenance_ids=tuple(
            sorted(
                {provenance_id for record in relevant for provenance_id in record.provenance_ids}
                | set(item.observation.provenance_ids)
            )
        ),
    )
    resolved_observation = Observation.model_validate(
        {
            **item.observation.model_dump(mode="python"),
            "entity_binding_id": binding.binding_id,
        }
    )
    unsigned: dict[str, object] = {
        "schema_version": OBSERVATION_BINDING_RESOLUTION_SCHEMA_VERSION,
        "observation": resolved_observation.model_dump(mode="json"),
        "entity_kind": item.entity_kind,
        "observed_entity_key": item.observed_entity_key,
        "binding": binding.model_dump(mode="json"),
        "reason": reason,
    }
    return ObservationBindingResolution.model_validate(
        {
            **unsigned,
            "resolution_id": _stable_id("res", unsigned),
        }
    )


def _build_timeline(
    records: tuple[EntityMetadataRecord, ...],
) -> tuple[EntityBinding, ...]:
    grouped: dict[tuple[EntityKind, str], list[EntityMetadataRecord]] = {}
    for record in records:
        key = (record.entity_kind, record.observed_entity_key)
        grouped.setdefault(key, []).append(record)
    bindings = []
    for (entity_kind, observed_entity_key), entity_records in grouped.items():
        boundaries = sorted(
            {
                boundary
                for record in entity_records
                for boundary in (
                    record.valid_interval.start_unix_ns,
                    record.valid_interval.end_unix_ns,
                )
            }
        )
        for start, end in pairwise(boundaries):
            active = tuple(
                record
                for record in entity_records
                if record.valid_interval.start_unix_ns <= start
                and record.valid_interval.end_unix_ns >= end
            )
            if not active:  # pragma: no cover - endpoints come from records
                continue
            explicit = tuple(
                record
                for record in active
                if record.relation == "explicit_service_binding"
                and record.service_entity_key is not None
            )
            service_keys: set[str] = set()
            for record in explicit:
                if record.service_entity_key is not None:
                    service_keys.add(record.service_entity_key)
            services = tuple(sorted(service_keys))
            if len(services) == 1:
                status: Literal["bound", "unbound", "conflicted"] = "bound"
                service = services[0]
                conflicts: tuple[str, ...] = ()
                supporting = explicit
            elif len(services) > 1:
                status = "conflicted"
                service = None
                conflicts = services
                supporting = explicit
            else:
                status = "unbound"
                service = None
                conflicts = ()
                supporting = active
            bindings.append(
                _build_binding(
                    opaque_case_id=active[0].opaque_case_id,
                    entity_kind=entity_kind,
                    observed_entity_key=observed_entity_key,
                    status=status,
                    service_entity_key=service,
                    conflict_service_keys=conflicts,
                    valid_interval=TimeInterval(start_unix_ns=start, end_unix_ns=end),
                    source_fingerprints=tuple(
                        sorted({record.source_fingerprint for record in supporting})
                    ),
                    provenance_ids=tuple(
                        sorted(
                            {
                                provenance_id
                                for record in supporting
                                for provenance_id in record.provenance_ids
                            }
                        )
                    ),
                )
            )
    return tuple(sorted(bindings, key=_binding_sort_key))


def _binding_sort_key(binding: EntityBinding) -> tuple[object, ...]:
    return (
        binding.entity_kind,
        binding.observed_entity_key,
        binding.valid_interval.start_unix_ns,
        binding.valid_interval.end_unix_ns,
        binding.binding_id,
    )


def build_temporal_entity_registry(
    *,
    opaque_case_id: str,
    decision_time_unix_ns: int,
    metadata_records: Sequence[EntityMetadataRecord],
    observations: Sequence[ObservationEntityInput],
) -> TemporalEntityRegistry:
    """Build a deterministic registry and retain every observation resolution."""

    validated_records = tuple(
        EntityMetadataRecord.model_validate(item.model_dump(mode="python"))
        for item in metadata_records
    )
    metadata_record_ids = tuple(item.metadata_record_id for item in validated_records)
    if len(metadata_record_ids) != len(set(metadata_record_ids)):
        raise ValueError("metadata record IDs must be unique")
    ordered_records = tuple(sorted(validated_records, key=lambda item: item.metadata_record_id))
    for record in ordered_records:
        if record.opaque_case_id != opaque_case_id:
            raise ValueError("metadata record belongs to a different opaque case")
        if record.valid_interval.end_unix_ns > decision_time_unix_ns:
            raise ValueError("metadata record extends beyond decision time")
    validated_observations = tuple(
        ObservationEntityInput.model_validate(item.model_dump(mode="python"))
        for item in observations
    )
    observation_ids = tuple(item.observation.observation_id for item in validated_observations)
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("observation IDs must be unique")
    for item in validated_observations:
        if item.observation.opaque_case_id != opaque_case_id:
            raise ValueError("observation belongs to a different opaque case")
        if item.observation.decision_time_unix_ns != decision_time_unix_ns:
            raise ValueError("observation decision time does not match registry")
    ordered_observations = tuple(
        sorted(
            validated_observations,
            key=lambda item: item.observation.observation_id,
        )
    )
    resolutions = tuple(
        _resolve_observation(item=item, records=ordered_records) for item in ordered_observations
    )
    timeline = _build_timeline(ordered_records)
    unsigned: dict[str, object] = {
        "schema_version": TEMPORAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "decision_time_unix_ns": decision_time_unix_ns,
        "metadata_records": [item.model_dump(mode="json") for item in ordered_records],
        "timeline_bindings": [item.model_dump(mode="json") for item in timeline],
        "resolutions": [item.model_dump(mode="json") for item in resolutions],
    }
    result = TemporalEntityRegistry.model_validate(
        {
            **unsigned,
            "registry_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result._verify_hash_and_structure():  # pragma: no cover - construction invariant
        raise AssertionError("temporal entity registry verification failed")
    return result


def _rebuild_temporal_entity_registry(
    registry: TemporalEntityRegistry,
) -> TemporalEntityRegistry:
    observations = tuple(
        ObservationEntityInput(
            observation=Observation.model_validate(
                {
                    **resolution.observation.model_dump(mode="python"),
                    "entity_binding_id": None,
                }
            ),
            entity_kind=resolution.entity_kind,
            observed_entity_key=resolution.observed_entity_key,
        )
        for resolution in registry.resolutions
    )
    return build_temporal_entity_registry(
        opaque_case_id=registry.opaque_case_id,
        decision_time_unix_ns=registry.decision_time_unix_ns,
        metadata_records=registry.metadata_records,
        observations=observations,
    )


def replay_temporal_entity_registry(
    registry: TemporalEntityRegistry,
) -> TemporalEntityRegistry:
    """Rebuild the registry from retained inputs and reject any derived-state drift."""

    try:
        validated = TemporalEntityRegistry.model_validate(
            registry.model_dump(mode="json", warnings="error")
        )
        if validated != registry or not validated._verify_hash_and_structure():
            raise ValueError("registry structure failed verification")
        expected = _rebuild_temporal_entity_registry(validated)
        if expected != validated:
            raise ValueError("derived registry state drifted")
        return expected
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "temporal entity registry cannot be replayed because it does not replay exactly"
        ) from error
