"""Bind all original observations to independently pinned, replayable source windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.observation_quality import summarize_transformed_component
from nurse_rcl.v2.observation_schemas import CaseTimeContract, Observation
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.temporal_entity_registry import ObservationBindingResolution

from .contracts import (
    NamedSourceWindow,
    SourceBlockReference,
    SourceIdentity,
    SourceWindowKind,
    VerificationSeries,
    VerificationSourceCatalog,
    VerificationSourceRecord,
    VerificationSourceRow,
    series_identity_key,
)
from .safety import canonical, plain, preflight, require_pin, require_unchanged, snapshot
from .source_store import (
    SourceBlockStore,
    read_source_window,
    require_store_unchanged,
    store_snapshot,
)

__all__ = [
    "NamedSourceWindow",
    "VerificationSourceRecord",
    "build_verification_source_catalog",
    "require_verification_source_catalog",
    "source_records_sha256",
]
_ROOTS = (RuleHypothesisEnvelope, CaseTimeContract, VerificationSourceRecord)


def source_records_sha256(records: tuple[VerificationSourceRecord, ...]) -> str:
    if type(records) is not tuple:
        raise ValueError("dynamic source records require an exact tuple")
    preflight(records, (VerificationSourceRecord,))
    if any(type(row) is not VerificationSourceRecord for row in records):
        raise ValueError("dynamic source record requires its exact type")
    ids = [row.observation_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dynamic source observation record")
    documents = []
    for row in sorted(records, key=lambda item: item.observation_id):
        names = tuple(item.name for item in row.source_inputs)
        if len(names) != len(set(names)):
            raise ValueError("duplicate dynamic source window name")
        document = cast(dict[str, object], plain(row))
        document["source_inputs"] = [
            plain(item) for item in sorted(row.source_inputs, key=lambda item: item.name)
        ]
        documents.append(document)
    return sha256_hex(canonical_json_bytes(documents))


def _verify_record(
    record: VerificationSourceRecord,
    observation: Observation,
    resolution: ObservationBindingResolution,
    contract: CaseTimeContract,
) -> None:
    transform, quality = record.transform_result, record.quality_result
    windows = {row.name: row.window for row in record.source_inputs}
    if len(windows) != len(record.source_inputs) or not transform.verify_source_inputs(windows):
        raise ValueError("dynamic source transform differs from raw windows")
    rebuilt = summarize_transformed_component(
        transform_result=transform,
        component_name=quality.component_name,
        context=quality.context,
        configuration=quality.configuration,
    )
    if canonical(rebuilt, (type(rebuilt),)) != canonical(quality, (type(quality),)):
        raise ValueError("dynamic quality source mismatch")
    if (
        record.observed_entity_key != resolution.observed_entity_key
        or record.entity_kind != resolution.entity_kind
        or record.source_fingerprint != observation.source_fingerprint
        or record.adapter_semantics_sha256 != observation.adapter_semantics_sha256
        or record.adapter_semantics_sha256 != transform.adapter_semantics_sha256
    ):
        raise ValueError("dynamic source identity differs from original observation")
    if (
        transform.windows[0].interval != contract.baseline_window
        or transform.windows[1].interval != contract.fault_window
        or observation.interval != contract.fault_window
        or observation.decision_time_unix_ns != contract.decision_time_unix_ns
        or quality.context.binding_status != resolution.binding.status
        or quality.context.clock_uncertainty_ns != contract.clock_uncertainty_ns
        or quality.context.late_data_observed != contract.late_data_observed
    ):
        raise ValueError("dynamic source differs from frozen time or binding")
    for name in (
        "signal_type",
        "unit",
        "aggregation",
        "baseline_summary",
        "fault_summary",
        "evidence_state",
        "anomaly_strength",
        "coverage_quality",
        "minimum_valid_points",
        "formal_scoring_eligible",
        "quality_flags",
    ):
        if getattr(observation, name) != getattr(quality, name):
            raise ValueError(f"dynamic source observation mismatch: {name}")
    if observation.transform != transform.transform:
        raise ValueError("dynamic source observation transform mismatch")


def _series(
    record: VerificationSourceRecord, candidate: str | None, envelope: RuleHypothesisEnvelope
) -> VerificationSeries:
    transform, quality = record.transform_result, record.quality_result
    identity = SourceIdentity(
        observed_entity_key=record.observed_entity_key,
        entity_kind=record.entity_kind,
        source_fingerprint=record.source_fingerprint,
        adapter_semantics_sha256=record.adapter_semantics_sha256,
    )
    blocks = []
    for item in sorted(record.source_inputs, key=lambda row: row.name):
        input_kind = cast(SourceWindowKind, type(item.window).__name__)
        raw = canonical_json_bytes(
            {
                "schema_version": "nurse.v2.signal-transform-source-inputs/2",
                "inputs": {item.name: {"input_kind": input_kind, "document": plain(item.window)}},
            }
        )
        blocks.append(
            SourceBlockReference(
                name=item.name, input_kind=input_kind, sha256=sha256_hex(raw), size_bytes=len(raw)
            )
        )
    return VerificationSeries(
        canonical_series_key=series_identity_key(
            case=envelope.candidate_universe.opaque_case_id,
            freeze=envelope.frozen_candidates.freeze_sha256,
            candidate=candidate,
            identity=identity,
            quality=quality,
        ),
        candidate_service_key=candidate,
        source_fingerprint=record.source_fingerprint,
        adapter_semantics_sha256=record.adapter_semantics_sha256,
        transform_component=quality.component_name,
        unit=quality.unit,
        aggregation=quality.aggregation,
        source_input_sha256=transform.source_input_sha256,
        transform_result=transform,
        quality_result=quality,
        source_blocks=tuple(blocks),
        expected_cadence_ns=quality.context.expected_cadence_ns,
    )


def build_verification_source_catalog(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    source_records: tuple[VerificationSourceRecord, ...],
    expected_source_records_sha256: str,
) -> VerificationSourceCatalog:
    if (
        type(envelope) is not RuleHypothesisEnvelope
        or type(time_contract) is not CaseTimeContract
        or type(source_records) is not tuple
    ):
        raise ValueError("dynamic catalog requires exact source roots")
    require_pin(expected_source_records_sha256)
    if any(type(row) is not VerificationSourceRecord for row in source_records):
        raise ValueError("dynamic source record requires its exact type")
    inputs = (envelope, time_contract, source_records)
    before = snapshot(inputs, _ROOTS)
    try:
        if source_records_sha256(source_records) != expected_source_records_sha256:
            raise ValueError("dynamic source independent pin mismatch")
        documents = json.loads(before[0])
        source = RuleHypothesisEnvelope.model_validate_json(canonical_json_bytes(documents[0]))
        contract = CaseTimeContract.model_validate_json(canonical_json_bytes(documents[1]))
        records = tuple(
            VerificationSourceRecord.model_validate_json(canonical_json_bytes(row))
            for row in documents[2]
        )
        if canonical((source, contract, records), _ROOTS) != before[0] or not source.verify():
            raise ValueError("dynamic source canonical frozen-input replay failed")
        if (
            contract.opaque_case_id != source.candidate_universe.opaque_case_id
            or contract.decision_time_unix_ns != source.entity_registry.decision_time_unix_ns
            or source.scope_interval.start_unix_ns > contract.baseline_window.start_unix_ns
            or source.scope_interval.end_unix_ns < contract.fault_window.end_unix_ns
        ):
            raise ValueError("dynamic source case time scope mismatch")
        observations = {row.observation_id: row for row in source.event_grouping.observations}
        by_id = {row.observation_id: row for row in records}
        if not set(by_id) <= set(observations):
            raise ValueError("dynamic source record is outside the original observations")
        resolutions = {
            row.observation.observation_id: row for row in source.entity_registry.resolutions
        }
        rows: list[VerificationSourceRow] = []
        series_by_key: dict[str, VerificationSeries] = {}
        for observation_id, observation in sorted(observations.items()):
            resolution = resolutions[observation_id]
            candidate = (
                resolution.binding.service_entity_key
                if resolution.binding.status == "bound"
                else None
            )
            event_ids = tuple(
                sorted(
                    row.event_group_id
                    for row in source.event_grouping.event_groups
                    if observation_id in row.observation_ids
                )
            )
            record = by_id.get(observation_id)
            key, identity = None, None
            reasons: tuple[str, ...] = ("source_unavailable",)
            if record is not None:
                _verify_record(record, observation, resolution, contract)
                item = _series(record, candidate, source)
                key = item.canonical_series_key
                if key in series_by_key and series_by_key[key] != item:
                    raise ValueError("conflicting dynamic sources claim one canonical series")
                series_by_key[key] = item
                identity = SourceIdentity(
                    observed_entity_key=record.observed_entity_key,
                    entity_kind=record.entity_kind,
                    source_fingerprint=record.source_fingerprint,
                    adapter_semantics_sha256=record.adapter_semantics_sha256,
                )
                reasons = () if candidate is not None else ("entity_binding_unavailable",)
            rows.append(
                VerificationSourceRow(
                    observation_id=observation_id,
                    candidate_service_key=candidate,
                    event_group_ids=event_ids,
                    canonical_series_key=key,
                    availability="unavailable" if reasons else "available",
                    reason_codes=reasons,
                    source_identity=identity,
                )
            )
        result = VerificationSourceCatalog(
            opaque_case_id=contract.opaque_case_id,
            candidate_freeze_sha256=source.frozen_candidates.freeze_sha256,
            time_contract_sha256=sha256_hex(canonical(contract, (CaseTimeContract,))),
            source_records_sha256=expected_source_records_sha256,
            rows=tuple(rows),
            series=tuple(series_by_key[key] for key in sorted(series_by_key)),
            catalog_sha256="0" * 64,
        )
        unsigned = cast(dict[str, object], plain(result))
        unsigned.pop("catalog_sha256")
        return result.model_copy(
            update={"catalog_sha256": sha256_hex(canonical_json_bytes(unsigned))}
        )
    finally:
        require_unchanged(inputs, _ROOTS, before)


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedMeasurementSources:
    """Fresh replay values, never an authority token accepted by downstream APIs."""

    catalog: VerificationSourceCatalog
    source_records: tuple[VerificationSourceRecord, ...]


def require_verification_source_catalog(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    source_records: tuple[VerificationSourceRecord, ...],
    expected_source_records_sha256: str,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    store: SourceBlockStore,
) -> VerifiedMeasurementSources:
    if (
        type(catalog) is not VerificationSourceCatalog
        or type(envelope) is not RuleHypothesisEnvelope
        or type(time_contract) is not CaseTimeContract
        or type(source_records) is not tuple
    ):
        raise ValueError("dynamic catalog replay requires exact source roots")
    if any(type(row) is not VerificationSourceRecord for row in source_records):
        raise ValueError("dynamic catalog replay requires exact source records")
    require_pin(expected_catalog_sha256)
    require_pin(expected_source_records_sha256)
    state = store_snapshot(store)
    roots = (*_ROOTS, VerificationSourceCatalog)
    originals = (envelope, time_contract, source_records, catalog)
    before = snapshot(originals, roots)
    try:
        rebuilt = build_verification_source_catalog(
            envelope=envelope,
            time_contract=time_contract,
            source_records=source_records,
            expected_source_records_sha256=expected_source_records_sha256,
        )
        if rebuilt.catalog_sha256 != expected_catalog_sha256 or canonical(
            rebuilt, (VerificationSourceCatalog,)
        ) != canonical(catalog, (VerificationSourceCatalog,)):
            raise ValueError("dynamic catalog differs from independently pinned sources")
        windows = {
            series.canonical_series_key: tuple(
                NamedSourceWindow(
                    name=block.name, window=read_source_window(store=store, block=block)
                )
                for block in series.source_blocks
            )
            for series in rebuilt.series
        }
        series_by_key = {row.canonical_series_key: row for row in rebuilt.series}
        records = []
        for row in rebuilt.rows:
            if row.canonical_series_key is None or row.source_identity is None:
                continue
            series = series_by_key[row.canonical_series_key]
            identity = row.source_identity
            records.append(
                VerificationSourceRecord(
                    observation_id=row.observation_id,
                    observed_entity_key=identity.observed_entity_key,
                    entity_kind=identity.entity_kind,
                    source_fingerprint=identity.source_fingerprint,
                    adapter_semantics_sha256=identity.adapter_semantics_sha256,
                    source_inputs=windows[row.canonical_series_key],
                    transform_result=series.transform_result,
                    quality_result=series.quality_result,
                )
            )
        replayed_records = tuple(records)
        replayed = build_verification_source_catalog(
            envelope=envelope,
            time_contract=time_contract,
            source_records=replayed_records,
            expected_source_records_sha256=expected_source_records_sha256,
        )
        if canonical(replayed, (VerificationSourceCatalog,)) != canonical(
            rebuilt, (VerificationSourceCatalog,)
        ):
            raise ValueError("source blocks cannot reconstruct the complete pinned catalog")
        return VerifiedMeasurementSources(catalog=replayed, source_records=replayed_records)
    finally:
        require_unchanged(originals, roots, before)
        require_store_unchanged(store, state)
