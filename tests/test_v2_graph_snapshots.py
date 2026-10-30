from __future__ import annotations

import importlib
import uuid
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import freeze_candidate_universe
from nurse_rcl.v2.event_grouping import AdapterAssociationSourceField
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    CaseTimeContract,
    TimeInterval,
    build_candidate_universe,
)
from nurse_rcl.v2.opaque_ids import EvaluatorIdentitySource, build_identity_artifacts
from nurse_rcl.v2.temporal_entity_registry import (
    build_entity_metadata_record,
    build_temporal_entity_registry,
)

CASE_ID = "opq_" + "a" * 32
CONTRACT_SHA256 = "b" * 64


def graphs() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.graph_snapshots")


def interval(start: int, end: int) -> TimeInterval:
    return TimeInterval(start_unix_ns=start, end_unix_ns=end)


def graph_context(opaque_case_id: str = CASE_ID) -> tuple[Any, Any, Any, Any]:
    time_contract = CaseTimeContract(
        opaque_case_id=opaque_case_id,
        t0_unix_ns=200,
        t0_source="shared_detector",
        localization_scope="detector_triggered",
        baseline_window=interval(100, 200),
        fault_window=interval(200, 300),
        decision_time_unix_ns=300,
        timezone="UTC",
        expected_cadence_ns=10,
        clock_uncertainty_ns=0,
        missing_modalities=(),
        late_data_observed=False,
    )
    records = tuple(
        build_entity_metadata_record(
            opaque_case_id=opaque_case_id,
            metadata_source="deployment",
            entity_kind="service",
            observed_entity_key=service,
            relation="explicit_service_binding",
            service_entity_key=service,
            valid_interval=interval(100, 300),
            source_fingerprint=character * 64,
            provenance_ids=(f"deployment:{service}",),
        )
        for service, character in (("checkout", "1"), ("payment", "2"))
    )
    registry = build_temporal_entity_registry(
        opaque_case_id=opaque_case_id,
        decision_time_unix_ns=300,
        metadata_records=records,
        observations=(),
    )
    universe = build_candidate_universe(
        opaque_case_id=opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        candidates=tuple(
            CandidateEntry(
                service_entity_key=service,
                binding_ids=(),
                observation_ids=(),
                event_group_ids=(),
                evidence_state="zero_support",
            )
            for service in ("checkout", "payment")
        ),
    )
    return time_contract, universe, freeze_candidate_universe(universe), registry


def coverages(
    module: ModuleType,
    time_contract: Any,
    *,
    fault: float = 1.0,
    opaque_case_id: str = CASE_ID,
) -> tuple[Any, ...]:
    return tuple(
        module.build_graph_adapter_parent_record(
            opaque_case_id=opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind=kind,
            interval=window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=round(fraction * 10),
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value=f"series:coverage-{kind}",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value=f"export:coverage-{kind}",
                ),
            ),
        )
        for kind, window, fraction in (
            ("normal", time_contract.baseline_window, 1.0),
            ("fault", time_contract.fault_window, fault),
        )
    )


def source(
    module: ModuleType,
    time_contract: Any,
    *,
    kind: str = "normal",
    caller: str = "checkout",
    callee: str = "payment",
    relation: str = "service_call",
    direction: str = "caller_to_callee",
    state: str = "observed",
    requests: int | None = 10,
    errors: int | None = 0,
    latency: float | None = 0.1,
    fingerprint: str = "5",
    provenance: str = "trace:one",
    opaque_case_id: str = CASE_ID,
    observation_contract_sha256: str = CONTRACT_SHA256,
    source_interval: TimeInterval | None = None,
) -> Any:
    window = time_contract.baseline_window if kind == "normal" else time_contract.fault_window
    return module.build_graph_adapter_parent_record(
        opaque_case_id=opaque_case_id,
        observation_contract_sha256=observation_contract_sha256,
        window_kind=kind,
        interval=source_interval or window,
        parent_artifact_kind="call_measurement",
        caller_service_key=caller,
        callee_service_key=callee,
        relation_type=relation,
        direction=direction,
        observation_state=state,
        request_count=requests,
        error_count=errors,
        latency_p95_seconds=latency,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:{fingerprint}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:{sha256_hex(provenance.encode('utf-8'))[:16]}",
            ),
        ),
    )


def canonical_source_input(
    module: ModuleType,
    time_contract: Any,
    *,
    kind: str = "normal",
    caller: str = "checkout",
    callee: str = "payment",
    relation: str = "service_call",
    direction: str = "caller_to_callee",
    state: str = "observed",
    requests: int | None = 10,
    errors: int | None = 0,
    latency: float | None = 0.1,
    source_token: str = "one",
    provenance_token: str = "one",
    opaque_case_id: str = CASE_ID,
) -> Any:
    window = time_contract.baseline_window if kind == "normal" else time_contract.fault_window
    return module.build_graph_adapter_parent_record(
        opaque_case_id=opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        window_kind=kind,
        interval=window,
        parent_artifact_kind="call_measurement",
        caller_service_key=caller,
        callee_service_key=callee,
        relation_type=relation,
        direction=direction,
        observation_state=state,
        request_count=requests,
        error_count=errors,
        latency_p95_seconds=latency,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:{source_token}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:{provenance_token}",
            ),
        ),
    )


def canonical_coverage_inputs(
    module: ModuleType,
    time_contract: Any,
    *,
    fault_valid_points: int = 10,
    opaque_case_id: str = CASE_ID,
) -> tuple[Any, ...]:
    return tuple(
        module.build_graph_adapter_parent_record(
            opaque_case_id=opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind=kind,
            interval=window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=valid_points,
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value=f"series:coverage-{kind}",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value=f"export:coverage-{kind}",
                ),
            ),
        )
        for kind, window, valid_points in (
            ("normal", time_contract.baseline_window, 10),
            ("fault", time_contract.fault_window, fault_valid_points),
        )
    )


def coverage_parent(
    module: ModuleType,
    time_contract: Any,
    *,
    kind: str = "normal",
    state: str = "observed",
    valid_sample_count: int | None,
    expected_sample_count: int | None,
    token: str,
) -> Any:
    window = time_contract.baseline_window if kind == "normal" else time_contract.fault_window
    return module.build_graph_adapter_parent_record(
        opaque_case_id=time_contract.opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        window_kind=kind,
        interval=window,
        parent_artifact_kind="coverage_samples",
        collection_state=state,
        valid_sample_count=valid_sample_count,
        expected_sample_count=expected_sample_count,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:coverage-{token}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:coverage-{token}",
            ),
        ),
    )


def inference_identity_context() -> tuple[Any, Any, Any, Any, Any]:
    _, identity_manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="private-graph-replay-root-case",
                incident_group_id="grp_" + "9" * 24,
                answer_services=("private-answer",),
            ),
        ),
        method="random_uuid_v4",
        uuid_factory=lambda: uuid.UUID("92345678-1234-4234-8234-1234567890ab"),
    )
    opaque_case_id = identity_manifest.records[0].opaque_case_id
    return (*graph_context(opaque_case_id), identity_manifest)


def adapter_parent_source(
    module: ModuleType,
    time_contract: Any,
    *,
    requests: int = 10,
    source_token: str = "trusted",
    provenance_token: str = "trusted",
) -> Any:
    return module.build_graph_adapter_parent_record(
        opaque_case_id=time_contract.opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        window_kind="normal",
        interval=time_contract.baseline_window,
        parent_artifact_kind="call_measurement",
        caller_service_key="checkout",
        callee_service_key="payment",
        relation_type="service_call",
        direction="caller_to_callee",
        observation_state="observed",
        request_count=requests,
        error_count=0,
        latency_p95_seconds=0.1,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:{source_token}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:{provenance_token}",
            ),
        ),
    )


def adapter_parent_coverages(
    module: ModuleType,
    time_contract: Any,
    *,
    fault_valid_points: int = 2,
    fault_provenance_token: str = "trusted-fault",
) -> tuple[Any, Any]:
    return tuple(
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind=kind,
            interval=window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=valid_points,
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value=f"series:coverage-{kind}",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value=(
                        f"export:{fault_provenance_token}"
                        if kind == "fault"
                        else "export:trusted-normal"
                    ),
                ),
            ),
        )
        for kind, window, valid_points in (
            ("normal", time_contract.baseline_window, 10),
            ("fault", time_contract.fault_window, fault_valid_points),
        )
    )


def freeze_graph_inputs(
    module: ModuleType,
    time_contract: Any,
    *,
    source_parents: tuple[Any, ...],
    coverage_parents: tuple[Any, ...],
) -> Any:
    return module.build_frozen_graph_input_manifest(
        opaque_case_id=time_contract.opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        parent_records=coverage_parents + source_parents,
    )


def unchecked_graph_manifest_with_counts(
    module: ModuleType,
    time_contract: Any,
    *,
    request_count: int,
    error_count: int,
) -> Any:
    """Build a correctly rehashed hostile manifest without schema validation."""

    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    base_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    parent_document = source_parent.model_dump(mode="json")
    parent_document.update(
        {
            "request_count": request_count,
            "error_count": error_count,
        }
    )
    parent_payload_fields = (
        "parent_artifact_kind",
        "opaque_case_id",
        "observation_contract_sha256",
        "window_kind",
        "interval",
        "caller_service_key",
        "callee_service_key",
        "relation_type",
        "direction",
        "observation_state",
        "request_count",
        "error_count",
        "latency_p95_seconds",
        "collection_state",
        "valid_sample_count",
        "expected_sample_count",
        "source_identity_fields",
        "provenance_fields",
    )
    parent_payload = {key: parent_document[key] for key in parent_payload_fields}
    parent_artifact_bytes_sha256 = sha256_hex(canonical_json_bytes(parent_payload))
    parent_selector_payload = {
        "parent_artifact_kind": parent_document["parent_artifact_kind"],
        "parent_artifact_bytes_sha256": parent_artifact_bytes_sha256,
    }
    parent_selector = "gsel_" + sha256_hex(canonical_json_bytes(parent_selector_payload))[:24]
    parent_document.update(
        {
            "parent_selector": parent_selector,
            "parent_artifact_bytes_sha256": parent_artifact_bytes_sha256,
        }
    )
    parent_unsigned = {
        key: value
        for key, value in parent_document.items()
        if key not in {"parent_record_id", "parent_record_sha256"}
    }
    parent_record_sha256 = sha256_hex(canonical_json_bytes(parent_unsigned))
    hostile_parent = source_parent.model_copy(
        update={
            "request_count": request_count,
            "error_count": error_count,
            "parent_selector": parent_selector,
            "parent_artifact_bytes_sha256": parent_artifact_bytes_sha256,
            "parent_record_id": f"gpar_{parent_record_sha256[:24]}",
            "parent_record_sha256": parent_record_sha256,
        }
    )
    manifest_parents = tuple(
        sorted(
            (*coverage_parents, hostile_parent),
            key=lambda item: item.parent_record_sha256,
        )
    )
    manifest_document = base_manifest.model_dump(mode="json")
    manifest_document.update(
        {
            "parent_records": tuple(item.model_dump(mode="json") for item in manifest_parents),
            "source_parent_selectors": (parent_selector,),
        }
    )
    manifest_unsigned = {
        key: value for key, value in manifest_document.items() if key != "manifest_sha256"
    }
    manifest_sha256 = sha256_hex(canonical_json_bytes(manifest_unsigned))
    return base_manifest.model_copy(
        update={
            "parent_records": manifest_parents,
            "source_parent_selectors": (parent_selector,),
            "manifest_sha256": manifest_sha256,
        }
    )


def unchecked_graph_manifest_with_boolean_coverage_count(
    module: ModuleType,
    time_contract: Any,
    *,
    count_field: str,
    boolean_value: bool,
) -> tuple[Any, Any]:
    """Inject a bool that Pydantic v2 would otherwise coerce to an integer."""

    boolean_token = str(boolean_value).lower()
    base_valid = int(boolean_value) if count_field == "valid_sample_count" else 0
    target_coverage = module.build_graph_adapter_parent_record(
        opaque_case_id=time_contract.opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        window_kind="normal",
        interval=time_contract.baseline_window,
        parent_artifact_kind="coverage_samples",
        collection_state="observed",
        valid_sample_count=base_valid,
        expected_sample_count=1,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:boolean-coverage-{count_field}-{boolean_token}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:boolean-coverage-{count_field}-{boolean_token}",
            ),
        ),
    )
    fault_coverage = adapter_parent_coverages(module, time_contract)[1]
    source_parent = adapter_parent_source(module, time_contract)
    base_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=(target_coverage, fault_coverage),
    )
    poisoned_coverage = target_coverage.model_copy(update={count_field: boolean_value})
    poisoned_parents = tuple(
        poisoned_coverage
        if item.parent_record_sha256 == target_coverage.parent_record_sha256
        else item
        for item in base_manifest.parent_records
    )
    return base_manifest, base_manifest.model_copy(update={"parent_records": poisoned_parents})


def canonical_source_from_parent(module: ModuleType, manifest: Any, parent: Any) -> Any:
    return module.build_canonical_graph_source_input(
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        parent_selector=parent.parent_selector,
    )


def canonical_coverages_from_manifest(module: ModuleType, manifest: Any) -> tuple[Any, ...]:
    return tuple(
        module.build_canonical_graph_coverage_input(
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            parent_selector=selector,
        )
        for selector in manifest.coverage_parent_selectors
    )


def derived_source_from_parent(module: ModuleType, manifest: Any, parent: Any) -> Any:
    return module.build_graph_source_record(
        canonical_input=canonical_source_from_parent(module, manifest, parent),
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )


def derived_coverages_from_manifest(module: ModuleType, manifest: Any) -> tuple[Any, ...]:
    return tuple(
        module.build_graph_window_coverage(
            canonical_input=item,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        )
        for item in canonical_coverages_from_manifest(module, manifest)
    )


def pair_from_parents(
    module: ModuleType,
    *,
    time_contract: Any,
    candidate_universe: Any,
    frozen_candidates: Any,
    entity_registry: Any,
    coverage_parents: tuple[Any, ...],
    source_parents: tuple[Any, ...],
    minimum_coverage_fraction: float = 0.8,
) -> tuple[Any, Any]:
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=source_parents,
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        minimum_coverage_fraction=minimum_coverage_fraction,
    )
    return pair, manifest


@pytest.mark.parametrize("entrypoint", ("verify_inputs", "boundary"))
@pytest.mark.parametrize("callback_method", ("__len__", "__iter__"))
def test_manifest_digest_str_subclasses_are_rejected_without_callbacks(
    entrypoint: str,
    callback_method: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=adapter_parent_coverages(module, time_contract),
        source_parents=(adapter_parent_source(module, time_contract),),
    )

    class CallbackTrackingDigest(str):
        callback_count: int

        def __new__(cls, value: str) -> CallbackTrackingDigest:
            instance = str.__new__(cls, value)
            instance.callback_count = 0
            return instance

        def __len__(self) -> int:
            if callback_method == "__len__":
                self.callback_count += 1
            return str.__len__(self)

        def __iter__(self) -> Iterator[str]:
            if callback_method == "__iter__":
                self.callback_count += 1
            return str.__iter__(self)

    hostile_digest = CallbackTrackingDigest(manifest.manifest_sha256)
    replay_arguments = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": manifest,
        "expected_graph_input_manifest_sha256": hostile_digest,
        "expected_minimum_coverage_fraction": 0.8,
    }

    if entrypoint == "verify_inputs":
        assert not pair.verify_inputs(**replay_arguments)
    else:
        with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
            boundary.audit_replayed_graph_snapshot_boundary(
                pair,
                **replay_arguments,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            )
    assert hostile_digest.callback_count == 0


def test_externally_pinned_manifest_rejects_fully_rehashed_source_root() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    coverage_parents = adapter_parent_coverages(module, time_contract)
    original_source = adapter_parent_source(module, time_contract)
    original_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(original_source,),
        coverage_parents=coverage_parents,
    )
    forged_source = adapter_parent_source(
        module,
        time_contract,
        requests=777,
        source_token="forged",
        provenance_token="forged",
    )
    forged_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(forged_source,),
        coverage_parents=coverage_parents,
    )
    forged_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=forged_manifest,
        expected_graph_input_manifest_sha256=forged_manifest.manifest_sha256,
    )

    assert forged_source.verify() and forged_manifest.verify() and forged_pair.verify()
    assert forged_pair.normal_snapshot.edges[0].traffic.value == 777.0
    assert not forged_pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=forged_manifest,
        expected_graph_input_manifest_sha256=original_manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            forged_pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=forged_manifest,
            expected_graph_input_manifest_sha256=original_manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


def test_externally_pinned_manifest_rejects_fully_rehashed_coverage_root() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    original_coverages = adapter_parent_coverages(module, time_contract, fault_valid_points=2)
    original_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=original_coverages,
    )
    forged_coverages = adapter_parent_coverages(
        module,
        time_contract,
        fault_valid_points=10,
        fault_provenance_token="forged-fault",
    )
    forged_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=forged_coverages,
    )
    forged_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=forged_manifest,
        expected_graph_input_manifest_sha256=forged_manifest.manifest_sha256,
    )

    assert forged_manifest.verify() and forged_pair.verify()
    assert forged_pair.fault_snapshot.coverage.coverage_fraction == 1.0
    assert not forged_pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=forged_manifest,
        expected_graph_input_manifest_sha256=original_manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            forged_pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=forged_manifest,
            expected_graph_input_manifest_sha256=original_manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


def test_rehashed_parent_bytes_and_manifest_cannot_replace_pinned_manifest() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    coverage_parents = adapter_parent_coverages(module, time_contract)
    original_parent = adapter_parent_source(module, time_contract)
    original_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(original_parent,),
        coverage_parents=coverage_parents,
    )
    replacement_parent = adapter_parent_source(
        module,
        time_contract,
        provenance_token="replacement-parent-bytes",
    )
    replacement_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(replacement_parent,),
        coverage_parents=coverage_parents,
    )

    assert replacement_parent.verify() and replacement_manifest.verify()
    assert replacement_parent.parent_artifact_bytes_sha256 != (
        original_parent.parent_artifact_bytes_sha256
    )
    assert replacement_manifest.manifest_sha256 != original_manifest.manifest_sha256
    with pytest.raises(ValueError, match="pinned graph input manifest digest mismatch"):
        module.build_graph_snapshot_pair(
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=replacement_manifest,
            expected_graph_input_manifest_sha256=original_manifest.manifest_sha256,
        )


def test_signed_zero_latency_has_one_end_to_end_content_address() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    coverage_parents = adapter_parent_coverages(module, time_contract)
    positive_parent = canonical_source_input(
        module,
        time_contract,
        latency=0.0,
        source_token="signed-zero",
        provenance_token="signed-zero",
    )
    negative_parent = canonical_source_input(
        module,
        time_contract,
        latency=-0.0,
        source_token="signed-zero",
        provenance_token="signed-zero",
    )
    positive_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(positive_parent,),
        coverage_parents=coverage_parents,
    )
    negative_manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(negative_parent,),
        coverage_parents=coverage_parents,
    )
    positive_input = canonical_source_from_parent(module, positive_manifest, positive_parent)
    negative_input = canonical_source_from_parent(module, negative_manifest, negative_parent)
    positive_record = derived_source_from_parent(module, positive_manifest, positive_parent)
    negative_record = derived_source_from_parent(module, negative_manifest, negative_parent)
    positive_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=positive_manifest,
        expected_graph_input_manifest_sha256=positive_manifest.manifest_sha256,
    )
    negative_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=negative_manifest,
        expected_graph_input_manifest_sha256=negative_manifest.manifest_sha256,
    )

    assert positive_parent == negative_parent
    assert positive_manifest == negative_manifest
    assert positive_input == negative_input
    assert positive_record == negative_record
    assert positive_pair == negative_pair
    assert positive_pair.normal_snapshot.snapshot_sha256 == (
        negative_pair.normal_snapshot.snapshot_sha256
    )


def test_signed_zero_coverage_metrics_and_thresholds_use_canonical_positive_zero() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    coverage_parents = canonical_coverage_inputs(
        module,
        time_contract,
        fault_valid_points=0,
    )
    source_parent = canonical_source_input(module, time_contract, latency=0.0)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    zero_coverage = next(
        item
        for item in derived_coverages_from_manifest(module, manifest)
        if item.window_kind == "fault"
    )
    signed_coverage_document = zero_coverage.model_dump(mode="json")
    signed_coverage_document["coverage_fraction"] = -0.0
    normalized_coverage = module.GraphWindowCoverage.model_validate(signed_coverage_document)

    positive_metric = module.GraphMetricSummary(
        state="zero",
        value=0.0,
        aggregation="sum_unique_sources",
        contributing_source_record_ids=(),
    )
    negative_metric = module.GraphMetricSummary(
        state="zero",
        value=-0.0,
        aggregation="sum_unique_sources",
        contributing_source_record_ids=(),
    )
    positive_summary = module.GraphCoverageSummary(
        observation_state="no_record",
        coverage_state="insufficient",
        valid_sample_count=0,
        expected_sample_count=10,
        coverage_fraction=0.0,
        minimum_coverage_fraction=0.0,
        provenance_ids=("coverage:signed-zero",),
    )
    negative_summary = module.GraphCoverageSummary(
        observation_state="no_record",
        coverage_state="insufficient",
        valid_sample_count=0,
        expected_sample_count=10,
        coverage_fraction=-0.0,
        minimum_coverage_fraction=-0.0,
        provenance_ids=("coverage:signed-zero",),
    )
    positive_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        minimum_coverage_fraction=0.0,
    )
    negative_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        minimum_coverage_fraction=-0.0,
    )

    assert normalized_coverage == zero_coverage
    assert canonical_json_bytes(negative_metric.model_dump(mode="json")) == (
        canonical_json_bytes(positive_metric.model_dump(mode="json"))
    )
    assert canonical_json_bytes(negative_summary.model_dump(mode="json")) == (
        canonical_json_bytes(positive_summary.model_dump(mode="json"))
    )
    assert negative_pair == positive_pair
    assert negative_pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=-0.0,
    )


@pytest.mark.parametrize(
    ("valid_sample_count", "expected_sample_count", "expected_state"),
    (
        (8_000_000_000_000_000_000, 10_000_000_000_000_000_001, "insufficient"),
        (8, 10, "sufficient"),
        (81, 100, "sufficient"),
    ),
)
def test_coverage_sufficiency_uses_exact_counts_across_build_and_replay(
    valid_sample_count: int,
    expected_sample_count: int,
    expected_state: str,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    normal_coverage = coverage_parent(
        module,
        time_contract,
        valid_sample_count=valid_sample_count,
        expected_sample_count=expected_sample_count,
        token=f"exact-{valid_sample_count}-{expected_sample_count}",
    )
    fault_coverage = coverage_parent(
        module,
        time_contract,
        kind="fault",
        valid_sample_count=10,
        expected_sample_count=10,
        token="exact-fault",
    )
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=(normal_coverage, fault_coverage),
        source_parents=(source(module, time_contract),),
    )

    derived = next(
        item
        for item in derived_coverages_from_manifest(module, manifest)
        if item.window_kind == "normal"
    )
    summary = pair.normal_snapshot.coverage
    assert summary.coverage_state == expected_state
    assert derived.valid_sample_count == valid_sample_count
    assert derived.expected_sample_count == expected_sample_count
    assert summary.valid_sample_count == valid_sample_count
    assert summary.expected_sample_count == expected_sample_count
    assert summary.coverage_fraction == valid_sample_count / expected_sample_count
    assert pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )


def test_exact_coverage_boundary_replays_rounded_below_threshold_as_insufficient() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=(
            coverage_parent(
                module,
                time_contract,
                valid_sample_count=8_000_000_000_000_000_000,
                expected_sample_count=10_000_000_000_000_000_001,
                token="boundary-rounded-below",
            ),
            coverage_parent(
                module,
                time_contract,
                kind="fault",
                valid_sample_count=10,
                expected_sample_count=10,
                token="boundary-fault",
            ),
        ),
        source_parents=(
            source(module, time_contract, opaque_case_id=time_contract.opaque_case_id),
        ),
    )

    assert pair.normal_snapshot.coverage.coverage_fraction == 0.8
    assert pair.normal_snapshot.coverage.coverage_state == "insufficient"
    audit = boundary.audit_replayed_graph_snapshot_boundary(
        pair,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
    )
    assert audit.artifact_count == 1


@pytest.mark.parametrize(
    (
        "state",
        "valid_sample_count",
        "expected_sample_count",
        "threshold",
        "expected_coverage_state",
    ),
    (
        ("observed", 0, 10, -0.0, "sufficient"),
        ("no_record", 0, 10, 0.0, "insufficient"),
        ("unknown", None, None, 0.0, "unknown"),
    ),
)
def test_exact_coverage_counts_preserve_observation_state_and_zero_threshold(
    state: str,
    valid_sample_count: int | None,
    expected_sample_count: int | None,
    threshold: float,
    expected_coverage_state: str,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    target_coverage = coverage_parent(
        module,
        time_contract,
        state=state,
        valid_sample_count=valid_sample_count,
        expected_sample_count=expected_sample_count,
        token=f"state-{state}",
    )
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=(
            target_coverage,
            coverage_parent(
                module,
                time_contract,
                kind="fault",
                valid_sample_count=10,
                expected_sample_count=10,
                token=f"state-{state}-fault",
            ),
        ),
        source_parents=(source(module, time_contract),),
        minimum_coverage_fraction=threshold,
    )

    derived = next(
        item
        for item in derived_coverages_from_manifest(module, manifest)
        if item.window_kind == "normal"
    )
    summary = pair.normal_snapshot.coverage
    assert pair.minimum_coverage_fraction == 0.0
    assert derived.valid_sample_count == valid_sample_count
    assert derived.expected_sample_count == expected_sample_count
    assert summary.valid_sample_count == valid_sample_count
    assert summary.expected_sample_count == expected_sample_count
    assert summary.coverage_state == expected_coverage_state


@pytest.mark.parametrize(
    ("count_field", "valid_sample_count", "expected_sample_count"),
    (
        ("valid_sample_count", (1 << 64), (1 << 64)),
        ("expected_sample_count", 0, (1 << 64)),
    ),
)
def test_coverage_sample_counts_reject_values_above_uint64_resource_cap(
    count_field: str,
    valid_sample_count: int,
    expected_sample_count: int,
) -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    maximum = (1 << 64) - 1
    accepted = coverage_parent(
        module,
        time_contract,
        valid_sample_count=maximum,
        expected_sample_count=maximum,
        token=f"uint64-maximum-{count_field}",
    )
    assert getattr(accepted, count_field) == maximum

    with pytest.raises(ValueError, match="uint64 coverage count limit"):
        coverage_parent(
            module,
            time_contract,
            valid_sample_count=valid_sample_count,
            expected_sample_count=expected_sample_count,
            token=f"uint64-overflow-{count_field}",
        )


def test_coverage_models_reject_rehashed_fraction_inconsistent_with_exact_counts() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=(
            coverage_parent(
                module,
                time_contract,
                valid_sample_count=8,
                expected_sample_count=10,
                token="inconsistent-normal",
            ),
            coverage_parent(
                module,
                time_contract,
                kind="fault",
                valid_sample_count=10,
                expected_sample_count=10,
                token="inconsistent-fault",
            ),
        ),
        source_parents=(source(module, time_contract),),
    )
    derived = next(
        item
        for item in derived_coverages_from_manifest(module, manifest)
        if item.window_kind == "normal"
    )
    coverage_document = derived.model_dump(mode="json")
    coverage_document["coverage_fraction"] = 0.9
    coverage_unsigned = {
        key: value
        for key, value in coverage_document.items()
        if key not in {"coverage_id", "coverage_sha256"}
    }
    coverage_digest = sha256_hex(canonical_json_bytes(coverage_unsigned))
    coverage_document["coverage_id"] = f"gwc_{coverage_digest[:24]}"
    coverage_document["coverage_sha256"] = coverage_digest

    summary_document = pair.normal_snapshot.coverage.model_dump(mode="json")
    summary_document["coverage_fraction"] = 0.9

    with pytest.raises(ValidationError, match="fraction does not match exact counts"):
        module.GraphWindowCoverage.model_validate(coverage_document)
    with pytest.raises(ValidationError, match="fraction does not match exact counts"):
        module.GraphCoverageSummary.model_validate(summary_document)


def test_verify_inputs_rejects_externally_mismatched_coverage_threshold_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract, fault=0.2),
        source_parents=(source(module, time_contract),),
        minimum_coverage_fraction=0.0,
    )
    replay_calls = 0
    original_replay = module._build_graph_snapshot_pair_from_context

    def counting_replay(*args: object, **kwargs: object) -> Any:
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(module, "_build_graph_snapshot_pair_from_context", counting_replay)

    assert pair.edge_changes[0].change_state == "removed"
    assert not pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    assert replay_calls == 0


def test_boundary_rejects_externally_mismatched_coverage_threshold_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(
            module,
            time_contract,
            fault=0.2,
            opaque_case_id=time_contract.opaque_case_id,
        ),
        source_parents=(
            source(
                module,
                time_contract,
                opaque_case_id=time_contract.opaque_case_id,
            ),
        ),
        minimum_coverage_fraction=0.0,
    )
    replay_calls = 0
    original_replay = module._build_graph_snapshot_pair_from_context

    def counting_replay(*args: object, **kwargs: object) -> Any:
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(module, "_build_graph_snapshot_pair_from_context", counting_replay)

    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )
    assert replay_calls == 0


def test_no_record_coverage_is_never_sufficient_at_zero_threshold() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    coverage_parents = (
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="normal",
            interval=time_contract.baseline_window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=10,
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value="series:no-record-normal",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value="export:no-record-normal",
                ),
            ),
        ),
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="fault",
            interval=time_contract.fault_window,
            parent_artifact_kind="coverage_samples",
            collection_state="no_record",
            valid_sample_count=0,
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value="series:no-record-fault",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value="export:no-record-fault",
                ),
            ),
        ),
    )
    pair, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverage_parents,
        source_parents=(source(module, time_contract),),
        minimum_coverage_fraction=-0.0,
    )

    assert pair.minimum_coverage_fraction == 0.0
    assert pair.fault_snapshot.coverage.observation_state == "no_record"
    assert pair.fault_snapshot.coverage.coverage_state == "insufficient"
    assert pair.edge_changes[0].fault_presence == "not_observed_insufficient_coverage"
    assert pair.edge_changes[0].change_state == "unknown"
    with pytest.raises(ValidationError, match="no-record coverage cannot be sufficient"):
        module.GraphCoverageSummary(
            observation_state="no_record",
            coverage_state="sufficient",
            valid_sample_count=0,
            expected_sample_count=10,
            coverage_fraction=0.0,
            minimum_coverage_fraction=0.0,
            provenance_ids=("coverage:no-record",),
        )


@pytest.mark.parametrize("count_field", ("request_count", "error_count"))
def test_graph_parent_rejects_individual_counts_above_float64_exact_integer_limit(
    count_field: str,
) -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    huge = 10**400
    maximum_exact_count = (1 << 53) - 1
    count_values = {
        "request_count": huge if count_field == "request_count" else maximum_exact_count,
        "error_count": huge if count_field == "error_count" else 0,
    }

    with pytest.raises(ValueError, match="float64 exact integer"):
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="normal",
            interval=time_contract.baseline_window,
            parent_artifact_kind="call_measurement",
            caller_service_key="checkout",
            callee_service_key="payment",
            relation_type="service_call",
            direction="caller_to_callee",
            observation_state="observed",
            request_count=count_values["request_count"],
            error_count=count_values["error_count"],
            latency_p95_seconds=0.1,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value=f"series:huge-{count_field}",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value=f"export:huge-{count_field}",
                ),
            ),
        )


def test_graph_pair_rejects_aggregate_counts_above_float64_exact_integer_limit() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    maximum_exact_count = (1 << 53) - 1
    large = canonical_source_input(
        module,
        time_contract,
        requests=maximum_exact_count,
        errors=maximum_exact_count,
        source_token="aggregate-large",
        provenance_token="aggregate-large",
    )
    extra = canonical_source_input(
        module,
        time_contract,
        requests=1,
        errors=1,
        source_token="aggregate-extra",
        provenance_token="aggregate-extra",
    )
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(large, extra),
        coverage_parents=adapter_parent_coverages(module, time_contract),
    )

    with pytest.raises(ValueError, match="aggregate request_count.*float64 exact integer"):
        module.build_graph_snapshot_pair(
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("entrypoint", ("replay", "boundary"))
@pytest.mark.parametrize("count_field", ("request_count", "error_count"))
def test_huge_individual_counts_are_controlled_at_replay_and_boundary(
    entrypoint: str,
    count_field: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    safe_pair, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=adapter_parent_coverages(module, time_contract),
        source_parents=(adapter_parent_source(module, time_contract),),
    )
    huge = 10**400
    hostile_manifest = unchecked_graph_manifest_with_counts(
        module,
        time_contract,
        request_count=huge,
        error_count=huge if count_field == "error_count" else 0,
    )
    replay_kwargs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": hostile_manifest,
        "expected_graph_input_manifest_sha256": hostile_manifest.manifest_sha256,
        "expected_minimum_coverage_fraction": 0.8,
    }

    if entrypoint == "replay":
        assert not safe_pair.verify_inputs(**replay_kwargs)
        return

    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            safe_pair,
            **replay_kwargs,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


@pytest.mark.parametrize("entrypoint", ("direct_model", "builder"))
@pytest.mark.parametrize("count_field", ("valid_sample_count", "expected_sample_count"))
@pytest.mark.parametrize("boolean_value", (True, False))
def test_graph_coverage_counts_reject_booleans_before_integer_coercion(
    entrypoint: str,
    count_field: str,
    boolean_value: bool,
) -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    boolean_token = str(boolean_value).lower()
    base_valid = int(boolean_value) if count_field == "valid_sample_count" else 0
    base_parent = module.build_graph_adapter_parent_record(
        opaque_case_id=time_contract.opaque_case_id,
        observation_contract_sha256=CONTRACT_SHA256,
        window_kind="normal",
        interval=time_contract.baseline_window,
        parent_artifact_kind="coverage_samples",
        collection_state="observed",
        valid_sample_count=base_valid,
        expected_sample_count=1,
        source_identity_fields=(
            AdapterAssociationSourceField(
                field_kind="counter_series_id",
                raw_value=f"series:boolean-{entrypoint}-{count_field}-{boolean_token}",
            ),
        ),
        provenance_fields=(
            AdapterAssociationSourceField(
                field_kind="export_id",
                raw_value=f"export:boolean-{entrypoint}-{count_field}-{boolean_token}",
            ),
        ),
    )
    if entrypoint == "direct_model":
        poisoned_document = base_parent.model_dump(mode="json")
        poisoned_document[count_field] = boolean_value
        with pytest.raises(ValidationError, match="exact integer coverage count"):
            module.GraphAdapterParentRecord.model_validate(poisoned_document)
        return

    builder_counts = {
        "valid_sample_count": base_valid,
        "expected_sample_count": 1,
    }
    builder_counts[count_field] = boolean_value
    with pytest.raises(ValueError, match="exact integer coverage count"):
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="normal",
            interval=time_contract.baseline_window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=builder_counts["valid_sample_count"],
            expected_sample_count=builder_counts["expected_sample_count"],
            source_identity_fields=base_parent.source_identity_fields,
            provenance_fields=base_parent.provenance_fields,
        )


@pytest.mark.parametrize("entrypoint", ("replay", "boundary"))
@pytest.mark.parametrize("count_field", ("valid_sample_count", "expected_sample_count"))
@pytest.mark.parametrize("boolean_value", (True, False))
def test_boolean_coverage_counts_are_controlled_at_replay_and_boundary(
    entrypoint: str,
    count_field: str,
    boolean_value: bool,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    base_manifest, poisoned_manifest = unchecked_graph_manifest_with_boolean_coverage_count(
        module,
        time_contract,
        count_field=count_field,
        boolean_value=boolean_value,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=base_manifest,
        expected_graph_input_manifest_sha256=base_manifest.manifest_sha256,
    )
    expected_normal_coverage_state = (
        "sufficient" if count_field == "valid_sample_count" and boolean_value else "insufficient"
    )
    assert pair.normal_snapshot.coverage.coverage_state == expected_normal_coverage_state
    replay_kwargs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": poisoned_manifest,
        "expected_graph_input_manifest_sha256": poisoned_manifest.manifest_sha256,
        "expected_minimum_coverage_fraction": 0.8,
    }
    if entrypoint == "replay":
        assert not pair.verify_inputs(**replay_kwargs)
        return

    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            **replay_kwargs,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


@pytest.mark.parametrize("integer_value", (0, 1))
def test_integer_latency_is_converted_to_float_before_content_hashing(
    integer_value: int,
) -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    integer_parent = canonical_source_input(
        module,
        time_contract,
        latency=integer_value,
        source_token=f"integer-latency-{integer_value}",
        provenance_token=f"integer-latency-{integer_value}",
    )
    float_parent = canonical_source_input(
        module,
        time_contract,
        latency=float(integer_value),
        source_token=f"integer-latency-{integer_value}",
        provenance_token=f"integer-latency-{integer_value}",
    )

    assert integer_parent == float_parent
    assert type(integer_parent.latency_p95_seconds) is float


@pytest.mark.parametrize("integer_value", (0, 1))
def test_integer_coverage_threshold_is_converted_to_float_before_pair_hashing(
    integer_value: int,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    source_parent = adapter_parent_source(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=adapter_parent_coverages(module, time_contract),
    )
    integer_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        minimum_coverage_fraction=integer_value,
    )
    float_pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        minimum_coverage_fraction=float(integer_value),
    )

    assert integer_pair == float_pair
    assert type(integer_pair.minimum_coverage_fraction) is float


@pytest.mark.parametrize("entrypoint", ("latency", "threshold"))
@pytest.mark.parametrize(
    "invalid_value",
    (True, False, 10**1000, "1"),
    ids=("true", "false", "overflowing-int", "numeric-string"),
)
def test_graph_float_builders_reject_noncanonical_runtime_numbers(
    entrypoint: str,
    invalid_value: Any,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()

    if entrypoint == "latency":
        with pytest.raises(ValueError, match="graph float"):
            canonical_source_input(
                module,
                time_contract,
                latency=invalid_value,
                source_token=f"invalid-latency-{type(invalid_value).__name__}",
                provenance_token=f"invalid-latency-{type(invalid_value).__name__}",
            )
        return

    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(adapter_parent_source(module, time_contract),),
        coverage_parents=adapter_parent_coverages(module, time_contract),
    )
    with pytest.raises(ValueError, match="graph float"):
        module.build_graph_snapshot_pair(
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            minimum_coverage_fraction=invalid_value,
        )


@pytest.mark.parametrize("model_kind", ("metric", "coverage"))
def test_float_bearing_models_reject_boolean_values_before_pydantic_coercion(
    model_kind: str,
) -> None:
    module = graphs()

    with pytest.raises(ValidationError, match="graph float"):
        if model_kind == "metric":
            module.GraphMetricSummary(
                state="observed",
                value=True,
                aggregation="sum_unique_sources",
                contributing_source_record_ids=(),
            )
        else:
            module.GraphCoverageSummary(
                observation_state="observed",
                coverage_state="sufficient",
                valid_sample_count=1,
                expected_sample_count=1,
                coverage_fraction=True,
                minimum_coverage_fraction=False,
                provenance_ids=("coverage:boolean",),
            )


def test_bulk_pair_build_and_replay_validate_the_manifest_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    source_parents = tuple(
        canonical_source_input(
            module,
            time_contract,
            source_token=f"bulk-{index}",
            provenance_token=f"bulk-{index}",
        )
        for index in range(8)
    )
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=source_parents,
        coverage_parents=coverage_parents,
    )
    original_require = module._require_pinned_graph_input_manifest
    validation_calls = 0

    def counting_require(*args: object, **kwargs: object) -> Any:
        nonlocal validation_calls
        validation_calls += 1
        return original_require(*args, **kwargs)

    monkeypatch.setattr(module, "_require_pinned_graph_input_manifest", counting_require)
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )

    assert validation_calls == 1
    validation_calls = 0
    assert pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    assert validation_calls == 1


def test_bulk_pair_uses_a_prebuilt_parent_selector_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    source_parents = tuple(
        canonical_source_input(
            module,
            time_contract,
            source_token=f"indexed-{index}",
            provenance_token=f"indexed-{index}",
        )
        for index in range(8)
    )
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=source_parents,
        coverage_parents=adapter_parent_coverages(module, time_contract),
    )
    linear_lookup_calls = 0
    original_lookup = module._parent_by_selector

    def counting_linear_lookup(*args: object, **kwargs: object) -> Any:
        nonlocal linear_lookup_calls
        linear_lookup_calls += 1
        return original_lookup(*args, **kwargs)

    monkeypatch.setattr(module, "_parent_by_selector", counting_linear_lookup)
    module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )

    assert linear_lookup_calls == 0


def test_verified_parent_index_rejects_duplicate_selectors() -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    parent = adapter_parent_source(module, time_contract)

    with pytest.raises(ValueError, match="parent index selectors must be unique"):
        module._build_parent_selector_index((parent, parent))


def test_canonical_graph_inputs_only_reference_pinned_parent_selectors() -> None:
    module = graphs()
    forbidden_fields = {
        "caller_service_key",
        "callee_service_key",
        "relation_type",
        "direction",
        "observation_state",
        "request_count",
        "error_count",
        "latency_p95_seconds",
        "collection_state",
        "valid_sample_count",
        "expected_sample_count",
        "source_identity_fields",
        "provenance_fields",
    }
    required_reference_fields = {
        "graph_input_manifest_sha256",
        "parent_selector",
        "parent_record_sha256",
    }

    assert forbidden_fields.isdisjoint(module.CanonicalGraphSourceInput.model_fields)
    assert forbidden_fields.isdisjoint(module.CanonicalGraphCoverageInput.model_fields)
    assert required_reference_fields <= set(module.CanonicalGraphSourceInput.model_fields)
    assert required_reference_fields <= set(module.CanonicalGraphCoverageInput.model_fields)


def test_replay_rejects_exact_upstream_instance_method_shadow_without_executing_it() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    supplied_source = derived_source_from_parent(module, manifest, source_parent)
    callback_calls = 0

    def malicious_model_dump(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("caller-controlled model_dump executed")

    assert type(supplied_source) is module.GraphSourceRecord
    object.__setattr__(supplied_source, "model_dump", malicious_model_dump)

    assert not pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        source_records=(supplied_source,),
    )
    assert callback_calls == 0


def test_boundary_rejects_exact_pair_method_shadow_without_executing_it() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    callback_calls = 0

    def malicious_verify_inputs(*args: object, **kwargs: object) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("caller-controlled verify_inputs executed")

    assert type(pair) is module.GraphSnapshotPair
    object.__setattr__(pair, "verify_inputs", malicious_verify_inputs)

    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )
    assert callback_calls == 0


@pytest.mark.parametrize("entrypoint", ("verify_inputs", "boundary"))
@pytest.mark.parametrize(
    "poison_kind",
    ("pair_subclass", "top_level_equality", "nested_equality", "threshold_comparison"),
)
def test_poisoned_graph_pair_receiver_never_executes_comparison_callbacks(
    entrypoint: str,
    poison_kind: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    callback_calls = 0

    class UniversalEquality:
        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return True

        def __ne__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return False

    class ExplodingComparisonFloat(float):
        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            raise RuntimeError("caller-controlled threshold equality executed")

        def __ne__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            raise RuntimeError("caller-controlled threshold inequality executed")

    if poison_kind == "pair_subclass":

        class EqualitySpoofPair(module.GraphSnapshotPair):
            def __eq__(self, other: object) -> bool:
                nonlocal callback_calls
                callback_calls += 1
                return True

        poisoned_pair = EqualitySpoofPair.model_validate(pair.model_dump(mode="json"))
    elif poison_kind == "top_level_equality":
        poisoned_pair = pair.model_copy(update={"pair_sha256": UniversalEquality()})
    elif poison_kind == "nested_equality":
        poisoned_snapshot = pair.normal_snapshot.model_copy(
            update={"snapshot_sha256": UniversalEquality()}
        )
        poisoned_pair = pair.model_copy(update={"normal_snapshot": poisoned_snapshot})
    else:
        poisoned_pair = pair.model_copy(
            update={"minimum_coverage_fraction": ExplodingComparisonFloat(0.8)}
        )

    replay_kwargs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": manifest,
        "expected_graph_input_manifest_sha256": manifest.manifest_sha256,
        "expected_minimum_coverage_fraction": 0.8,
    }
    if entrypoint == "verify_inputs":
        assert not module.GraphSnapshotPair.verify_inputs(poisoned_pair, **replay_kwargs)
    else:
        with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
            boundary.audit_replayed_graph_snapshot_boundary(
                poisoned_pair,
                **replay_kwargs,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            )
    assert callback_calls == 0


def test_adapter_parent_rejects_exact_interval_method_shadow_without_executing_it() -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    supplied_interval = time_contract.baseline_window.model_copy()
    callback_calls = 0

    def malicious_model_dump(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("caller-controlled interval model_dump executed")

    assert type(supplied_interval) is TimeInterval
    object.__setattr__(supplied_interval, "model_dump", malicious_model_dump)

    with pytest.raises(ValueError, match="time interval"):
        module.build_graph_adapter_parent_record(
            opaque_case_id=time_contract.opaque_case_id,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="normal",
            interval=supplied_interval,
            parent_artifact_kind="call_measurement",
            caller_service_key="checkout",
            callee_service_key="payment",
            relation_type="service_call",
            direction="caller_to_callee",
            observation_state="observed",
            request_count=10,
            error_count=0,
            latency_p95_seconds=0.1,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value="series:shadowed-interval",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value="export:shadowed-interval",
                ),
            ),
        )
    assert callback_calls == 0


@pytest.mark.parametrize("entrypoint", ("build", "verify", "boundary"))
@pytest.mark.parametrize(
    ("input_kind", "method_name"),
    (
        ("time_contract", "model_dump"),
        ("candidate_universe", "verify"),
        ("frozen_candidates", "verify"),
        ("entity_registry", "verify"),
    ),
)
def test_pair_entrypoints_reject_exact_base_input_method_shadows_without_execution(
    entrypoint: str,
    input_kind: str,
    method_name: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    public_inputs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
    }
    supplied = public_inputs[input_kind]
    expected_types = {
        "time_contract": CaseTimeContract,
        "candidate_universe": type(universe),
        "frozen_candidates": type(frozen),
        "entity_registry": type(registry),
    }
    callback_calls = 0

    def malicious_method(*args: object, **kwargs: object) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError(f"caller-controlled {input_kind}.{method_name} executed")

    assert type(supplied) is expected_types[input_kind]
    object.__setattr__(supplied, method_name, malicious_method)
    replay_kwargs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": manifest,
        "expected_graph_input_manifest_sha256": manifest.manifest_sha256,
    }

    if entrypoint == "build":
        with pytest.raises(ValueError, match="exact|method shadow"):
            module.build_graph_snapshot_pair(**replay_kwargs)
    elif entrypoint == "verify":
        assert not pair.verify_inputs(
            **replay_kwargs,
            expected_minimum_coverage_fraction=0.8,
        )
    else:
        with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
            boundary.audit_replayed_graph_snapshot_boundary(
                pair,
                **replay_kwargs,
                expected_minimum_coverage_fraction=0.8,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            )
    assert callback_calls == 0


@pytest.mark.parametrize("entrypoint", ("build", "verify", "boundary"))
def test_pair_entrypoints_reject_cross_case_base_inputs(entrypoint: str) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_a, universe_a, frozen_a, registry_a, identity_manifest = inference_identity_context()
    _, universe_b, frozen_b, _ = graph_context("opq_" + "c" * 32)
    source_parent = adapter_parent_source(module, time_a)
    coverage_parents = adapter_parent_coverages(module, time_a)
    manifest_a = freeze_graph_inputs(
        module,
        time_a,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    valid_pair = module.build_graph_snapshot_pair(
        time_contract=time_a,
        candidate_universe=universe_a,
        frozen_candidates=frozen_a,
        entity_registry=registry_a,
        graph_input_manifest=manifest_a,
        expected_graph_input_manifest_sha256=manifest_a.manifest_sha256,
    )
    cross_case_document = valid_pair.model_dump(mode="json")
    cross_case_document["candidate_universe_sha256"] = universe_b.universe_sha256
    cross_case_document["candidate_freeze_sha256"] = frozen_b.freeze_sha256
    for snapshot_name in ("normal_snapshot", "fault_snapshot"):
        snapshot_document = cross_case_document[snapshot_name]
        snapshot_document["candidate_universe_sha256"] = universe_b.universe_sha256
        snapshot_document["candidate_freeze_sha256"] = frozen_b.freeze_sha256
        snapshot_unsigned = {
            key: value for key, value in snapshot_document.items() if key != "snapshot_sha256"
        }
        snapshot_document["snapshot_sha256"] = sha256_hex(canonical_json_bytes(snapshot_unsigned))
    pair_unsigned = {
        key: value for key, value in cross_case_document.items() if key != "pair_sha256"
    }
    cross_case_document["pair_sha256"] = sha256_hex(canonical_json_bytes(pair_unsigned))
    cross_case_pair = module.GraphSnapshotPair.model_validate(cross_case_document)
    mixed_inputs = {
        "time_contract": time_a,
        "candidate_universe": universe_b,
        "frozen_candidates": frozen_b,
        "entity_registry": registry_a,
        "graph_input_manifest": manifest_a,
        "expected_graph_input_manifest_sha256": manifest_a.manifest_sha256,
    }

    assert cross_case_pair.verify()
    if entrypoint == "build":
        with pytest.raises(ValueError, match="case scope"):
            module.build_graph_snapshot_pair(**mixed_inputs)
    elif entrypoint == "verify":
        assert not cross_case_pair.verify_inputs(
            **mixed_inputs,
            expected_minimum_coverage_fraction=0.8,
        )
    else:
        with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
            boundary.audit_replayed_graph_snapshot_boundary(
                cross_case_pair,
                **mixed_inputs,
                expected_minimum_coverage_fraction=0.8,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            )


@pytest.mark.parametrize("entrypoint", ("verify_inputs", "boundary"))
@pytest.mark.parametrize(
    ("upstream_kind", "parameter_name"),
    (
        ("canonical_source", "canonical_source_inputs"),
        ("canonical_coverage", "canonical_coverage_inputs"),
        ("source_record", "source_records"),
        ("window_coverage", "window_coverages"),
    ),
)
def test_optional_replay_container_subclass_never_executes_iteration_callback(
    entrypoint: str,
    upstream_kind: str,
    parameter_name: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    supplied_cases: dict[str, tuple[Any, ...]] = {
        "canonical_source": (canonical_source_from_parent(module, manifest, source_parent),),
        "canonical_coverage": canonical_coverages_from_manifest(module, manifest),
        "source_record": (derived_source_from_parent(module, manifest, source_parent),),
        "window_coverage": derived_coverages_from_manifest(module, manifest),
    }
    supplied_values = supplied_cases[upstream_kind]
    replay_kwargs = {
        "time_contract": time_contract,
        "candidate_universe": universe,
        "frozen_candidates": frozen,
        "entity_registry": registry,
        "graph_input_manifest": manifest,
        "expected_graph_input_manifest_sha256": manifest.manifest_sha256,
        "expected_minimum_coverage_fraction": 0.8,
    }
    if entrypoint == "verify_inputs":
        assert pair.verify_inputs(
            **replay_kwargs,
            **{parameter_name: supplied_values},
        )
    else:
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            **replay_kwargs,
            **{parameter_name: supplied_values},
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )

    callback_calls = 0

    class IterationCallbackTuple(tuple[Any, ...]):
        def __iter__(self) -> Iterator[Any]:
            nonlocal callback_calls
            callback_calls += 1
            raise RuntimeError("caller-controlled replay container iteration executed")

    malicious_values = IterationCallbackTuple(supplied_values)
    if entrypoint == "verify_inputs":
        assert not pair.verify_inputs(
            **replay_kwargs,
            **{parameter_name: malicious_values},
        )
    else:
        with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
            boundary.audit_replayed_graph_snapshot_boundary(
                pair,
                **replay_kwargs,
                **{parameter_name: malicious_values},
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            )
    assert callback_calls == 0


@pytest.mark.parametrize(
    "upstream_kind",
    ("canonical_source", "canonical_coverage", "source_record", "window_coverage"),
)
def test_replay_rejects_equality_spoofing_subclasses_for_all_supplied_upstreams(
    upstream_kind: str,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    source_parent = adapter_parent_source(module, time_contract)
    coverage_parents = adapter_parent_coverages(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    canonical_sources = (canonical_source_from_parent(module, manifest, source_parent),)
    canonical_coverages = canonical_coverages_from_manifest(module, manifest)
    derived_sources = (derived_source_from_parent(module, manifest, source_parent),)
    derived_coverages = derived_coverages_from_manifest(module, manifest)

    class EqualitySpoofCanonicalSource(module.CanonicalGraphSourceInput):
        def __eq__(self, other: object) -> bool:
            return True

    class EqualitySpoofCanonicalCoverage(module.CanonicalGraphCoverageInput):
        def __eq__(self, other: object) -> bool:
            return True

    class EqualitySpoofSourceRecord(module.GraphSourceRecord):
        def __eq__(self, other: object) -> bool:
            return True

    class EqualitySpoofWindowCoverage(module.GraphWindowCoverage):
        def __eq__(self, other: object) -> bool:
            return True

    evil_canonical_source = EqualitySpoofCanonicalSource.model_validate(
        canonical_sources[0].model_dump(mode="json")
    ).model_copy(update={"parent_record_sha256": "f" * 64})
    evil_canonical_coverage = EqualitySpoofCanonicalCoverage.model_validate(
        canonical_coverages[0].model_dump(mode="json")
    ).model_copy(update={"parent_record_sha256": "f" * 64})
    evil_source_record = EqualitySpoofSourceRecord.model_validate(
        derived_sources[0].model_dump(mode="json")
    ).model_copy(update={"request_count": 999})
    evil_window_coverage = EqualitySpoofWindowCoverage.model_validate(
        derived_coverages[0].model_dump(mode="json")
    ).model_copy(update={"coverage_fraction": 0.99})
    supplied_cases = {
        "canonical_source": {"canonical_source_inputs": (evil_canonical_source,)},
        "canonical_coverage": {
            "canonical_coverage_inputs": (
                evil_canonical_coverage,
                canonical_coverages[1],
            )
        },
        "source_record": {"source_records": (evil_source_record,)},
        "window_coverage": {
            "window_coverages": (
                evil_window_coverage,
                derived_coverages[1],
            )
        },
    }
    supplied = supplied_cases[upstream_kind]
    supplied_value = next(iter(supplied.values()))[0]
    verify_accepted = pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        **supplied,
    )
    try:
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
            **supplied,
        )
        boundary_accepted = True
    except ValueError as error:
        boundary_accepted = False
        assert "does not replay from pinned graph inputs" in str(error)

    assert not supplied_value.verify()
    assert not verify_accepted
    assert not boundary_accepted


def test_self_rehashed_graph_source_cannot_replace_unchanged_canonical_input() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    coverage_parents = canonical_coverage_inputs(
        module,
        time_contract,
        opaque_case_id=time_contract.opaque_case_id,
    )
    source_parent = canonical_source_input(
        module,
        time_contract,
        opaque_case_id=time_contract.opaque_case_id,
    )
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    canonical_coverages = canonical_coverages_from_manifest(module, manifest)
    canonical_source = canonical_source_from_parent(module, manifest, source_parent)
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    derived_coverages = derived_coverages_from_manifest(module, manifest)
    derived_source = derived_source_from_parent(module, manifest, source_parent)
    forged_document = derived_source.model_dump(mode="json")
    forged_document.update(
        {
            "request_count": 999,
            "error_count": 999,
            "latency_p95_seconds": 999.0,
            "source_fingerprint": "f" * 64,
            "provenance_ids": ("caller:invented",),
        }
    )
    forged_identity = {
        "opaque_case_id": forged_document["opaque_case_id"],
        "observation_contract_sha256": forged_document["observation_contract_sha256"],
        "window_kind": forged_document["window_kind"],
        "source_fingerprint": forged_document["source_fingerprint"],
        "caller_service_key": forged_document["caller_service_key"],
        "callee_service_key": forged_document["callee_service_key"],
        "relation_type": forged_document["relation_type"],
        "direction": forged_document["direction"],
    }
    forged_document["source_record_id"] = (
        f"gsrc_{sha256_hex(canonical_json_bytes(forged_identity))[:24]}"
    )
    forged_unsigned = {
        key: value
        for key, value in forged_document.items()
        if key not in {"source_record_id", "source_record_sha256"}
    }
    forged_document["source_record_sha256"] = sha256_hex(canonical_json_bytes(forged_unsigned))
    forged_source = module.GraphSourceRecord.model_validate(forged_document)

    assert forged_source.verify()
    assert not pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        canonical_coverage_inputs=canonical_coverages,
        canonical_source_inputs=(canonical_source,),
        window_coverages=derived_coverages,
        source_records=(forged_source,),
    )
    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            canonical_coverage_inputs=canonical_coverages,
            canonical_source_inputs=(canonical_source,),
            window_coverages=derived_coverages,
            source_records=(forged_source,),
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


def test_self_rehashed_coverage_cannot_replace_unchanged_canonical_input() -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    coverage_parents = canonical_coverage_inputs(
        module,
        time_contract,
        opaque_case_id=time_contract.opaque_case_id,
    )
    source_parent = canonical_source_input(
        module,
        time_contract,
        opaque_case_id=time_contract.opaque_case_id,
    )
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    canonical_coverages = canonical_coverages_from_manifest(module, manifest)
    canonical_source = canonical_source_from_parent(module, manifest, source_parent)
    pair = module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    derived_source = derived_source_from_parent(module, manifest, source_parent)
    derived_coverages = derived_coverages_from_manifest(module, manifest)
    forged_document = derived_coverages[1].model_dump(mode="json")
    forged_document.update(
        {
            "state": "observed",
            "valid_sample_count": 99,
            "expected_sample_count": 100,
            "coverage_fraction": 0.99,
            "provenance_ids": ("caller:invented",),
        }
    )
    forged_unsigned = {
        key: value
        for key, value in forged_document.items()
        if key not in {"coverage_id", "coverage_sha256"}
    }
    forged_digest = sha256_hex(canonical_json_bytes(forged_unsigned))
    forged_document["coverage_id"] = f"gwc_{forged_digest[:24]}"
    forged_document["coverage_sha256"] = forged_digest
    forged_coverage = module.GraphWindowCoverage.model_validate(forged_document)
    supplied_coverages = (derived_coverages[0], forged_coverage)

    assert forged_coverage.verify()
    assert not pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        canonical_coverage_inputs=canonical_coverages,
        canonical_source_inputs=(canonical_source,),
        window_coverages=supplied_coverages,
        source_records=(derived_source,),
    )
    with pytest.raises(ValueError, match="does not replay from pinned graph inputs"):
        boundary.audit_replayed_graph_snapshot_boundary(
            pair,
            time_contract=time_contract,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            expected_minimum_coverage_fraction=0.8,
            canonical_coverage_inputs=canonical_coverages,
            canonical_source_inputs=(canonical_source,),
            window_coverages=supplied_coverages,
            source_records=(derived_source,),
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


def test_canonical_graph_input_recomputes_identity_parent_and_frozen_semantics() -> None:
    module = graphs()
    time_contract, _, _, _ = graph_context()
    source_parent = canonical_source_input(module, time_contract)
    coverage_parents = canonical_coverage_inputs(module, time_contract)
    manifest = freeze_graph_inputs(
        module,
        time_contract,
        source_parents=(source_parent,),
        coverage_parents=coverage_parents,
    )
    source_input = canonical_source_from_parent(module, manifest, source_parent)
    derived = derived_source_from_parent(module, manifest, source_parent)

    assert derived.source_fingerprint == source_parent.derived_source_fingerprint
    assert derived.provenance_ids == source_parent.derived_provenance_ids
    assert derived.parent_source_sha256s == (source_parent.parent_artifact_bytes_sha256,)
    assert derived.canonical_input_sha256s == (source_input.canonical_input_sha256,)

    forged_parent = source_input.model_dump(mode="json")
    forged_parent["parent_record_sha256"] = "f" * 64
    parent_unsigned = {
        key: value
        for key, value in forged_parent.items()
        if key not in {"canonical_input_id", "canonical_input_sha256"}
    }
    parent_digest = sha256_hex(canonical_json_bytes(parent_unsigned))
    forged_parent["canonical_input_id"] = f"cgsi_{parent_digest[:24]}"
    forged_parent["canonical_input_sha256"] = parent_digest
    forged_reference = module.CanonicalGraphSourceInput.model_validate(forged_parent)
    with pytest.raises(ValueError, match="parent digest mismatch"):
        module.build_graph_source_record(
            canonical_input=forged_reference,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        )

    forged_semantics = source_parent.model_dump(mode="json")
    forged_semantics["derivation_semantics_sha256"] = "e" * 64
    semantics_unsigned = {
        key: value
        for key, value in forged_semantics.items()
        if key not in {"parent_record_id", "parent_record_sha256"}
    }
    semantics_digest = sha256_hex(canonical_json_bytes(semantics_unsigned))
    forged_semantics["parent_record_id"] = f"gpar_{semantics_digest[:24]}"
    forged_semantics["parent_record_sha256"] = semantics_digest
    with pytest.raises(ValidationError, match="frozen graph derivation semantics"):
        module.GraphAdapterParentRecord.model_validate(forged_semantics)

    with pytest.raises(ValidationError, match="canonical opaque ID grammar"):
        AdapterAssociationSourceField(
            field_kind="source_relation_id",
            raw_value="caller-controlled-identity",
        )
    builder: Any = module.build_canonical_graph_source_input
    with pytest.raises(TypeError):
        builder(
            opaque_case_id=CASE_ID,
            observation_contract_sha256=CONTRACT_SHA256,
            window_kind="normal",
            interval=time_contract.baseline_window,
            caller_service_key="checkout",
            callee_service_key="payment",
            relation_type="service_call",
            direction="caller_to_callee",
            observation_state="observed",
            request_count=10,
            error_count=0,
            latency_p95_seconds=0.1,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id",
                    raw_value="series:one",
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value="export:one",
                ),
            ),
            source_fingerprint="f" * 64,
            provenance_ids=("caller:invented",),
            parent_source_sha256="f" * 64,
            derivation_semantics_sha256="e" * 64,
        )


def test_builds_normal_and_fault_snapshots_with_original_direction_and_change() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    coverage_inputs = coverages(module, time_contract)
    sources = tuple(
        source(
            module,
            time_contract,
            kind=kind,
            requests=requests,
            errors=errors,
            latency=latency,
            fingerprint=character,
            provenance=f"trace:{kind}",
        )
        for kind, requests, errors, latency, character in (
            ("normal", 10, 0, 0.1, "3"),
            ("fault", 20, 2, 0.2, "4"),
        )
    )

    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverage_inputs,
        source_parents=sources,
        minimum_coverage_fraction=0.8,
    )

    assert result.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    assert result.normal_snapshot.interval == time_contract.baseline_window
    assert result.fault_snapshot.interval == time_contract.fault_window
    normal_edge = result.normal_snapshot.edges[0]
    fault_edge = result.fault_snapshot.edges[0]
    assert (
        normal_edge.caller_service_key,
        normal_edge.callee_service_key,
        normal_edge.direction,
    ) == ("checkout", "payment", "caller_to_callee")
    assert normal_edge.traffic.value == 10.0
    assert normal_edge.error_count.state == "verified_normal"
    assert fault_edge.error_rate.value == 0.1
    assert fault_edge.latency_p95_seconds.value == 0.2
    assert result.edge_changes[0].change_state == "increased"
    assert result.normal_snapshot.verify() and result.fault_snapshot.verify()
    assert result.verify()


def test_partitions_cross_case_window_contract_candidate_and_unknown_inputs() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    valid = source(module, time_contract, fingerprint="5")
    foreign_case = source(
        module,
        time_contract,
        opaque_case_id="opq_" + "c" * 32,
        fingerprint="6",
    )
    wrong_contract = source(
        module,
        time_contract,
        observation_contract_sha256="d" * 64,
        fingerprint="7",
    )
    post_decision = source(
        module,
        time_contract,
        kind="fault",
        source_interval=interval(200, 301),
        fingerprint="8",
    )
    outside_candidate = source(
        module,
        time_contract,
        callee="inventory",
        fingerprint="9",
    )
    unknown = source(
        module,
        time_contract,
        kind="fault",
        state="unknown",
        requests=None,
        errors=None,
        latency=None,
        fingerprint="a",
    )
    inputs = (
        unknown,
        outside_candidate,
        post_decision,
        valid,
        wrong_contract,
        foreign_case,
    )

    coverage_parents = coverages(module, time_contract)
    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverage_parents,
        source_parents=inputs,
    )

    valid_record = derived_source_from_parent(module, manifest, valid)
    unknown_record = derived_source_from_parent(module, manifest, unknown)
    foreign_case_record = derived_source_from_parent(module, manifest, foreign_case)
    wrong_contract_record = derived_source_from_parent(module, manifest, wrong_contract)
    post_decision_record = derived_source_from_parent(module, manifest, post_decision)
    outside_candidate_record = derived_source_from_parent(module, manifest, outside_candidate)
    by_sha = {item.source_record_sha256: item for item in result.input_decisions}
    assert by_sha[valid_record.source_record_sha256].disposition == "accepted"
    assert by_sha[unknown_record.source_record_sha256].disposition == "unknown"
    assert by_sha[foreign_case_record.source_record_sha256].reason_codes == ("case_mismatch",)
    assert by_sha[foreign_case_record.source_record_sha256].provenance_ids == (
        foreign_case.derived_provenance_ids
    )
    assert by_sha[wrong_contract_record.source_record_sha256].reason_codes == (
        "observation_contract_mismatch",
    )
    assert set(by_sha[post_decision_record.source_record_sha256].reason_codes) == {
        "post_decision",
        "window_interval_mismatch",
    }
    assert by_sha[outside_candidate_record.source_record_sha256].reason_codes == (
        "endpoint_outside_frozen_candidates",
    )
    assert result.input_occurrence_count == len(inputs)
    assert sum(item.occurrence_count for item in result.input_decisions) == len(inputs)
    assert set(result.accepted_input_sha256s) == {valid_record.source_record_sha256}
    assert set(result.unknown_input_sha256s) == {unknown_record.source_record_sha256}
    assert set(result.rejected_input_sha256s) == {
        foreign_case_record.source_record_sha256,
        wrong_contract_record.source_record_sha256,
        post_decision_record.source_record_sha256,
        outside_candidate_record.source_record_sha256,
    }
    assert result.fault_snapshot.edges[0].observation_state == "unknown"


def test_relation_types_have_explicit_direction_and_shared_resource_is_not_causal() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    with pytest.raises(ValidationError, match="direction does not match"):
        source(
            module,
            time_contract,
            relation="shared_resource_association",
            direction="caller_to_callee",
            requests=None,
            errors=None,
            latency=None,
        )
    with pytest.raises(ValidationError, match="canonical endpoint order"):
        source(
            module,
            time_contract,
            caller="payment",
            callee="checkout",
            relation="shared_resource_association",
            direction="undirected_association",
            requests=None,
            errors=None,
            latency=None,
        )

    relations = (
        source(
            module,
            time_contract,
            relation="received_dependency_failure",
            direction="callee_to_caller",
            requests=None,
            errors=None,
            latency=None,
            fingerprint="b",
        ),
        source(
            module,
            time_contract,
            relation="caller_load_transmission",
            direction="caller_to_callee",
            fingerprint="c",
        ),
        source(
            module,
            time_contract,
            relation="shared_resource_association",
            direction="undirected_association",
            requests=None,
            errors=None,
            latency=None,
            fingerprint="d",
        ),
    )
    result, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=relations,
    )
    by_relation = {item.relation_type: item for item in result.normal_snapshot.edges}

    assert by_relation["received_dependency_failure"].direction == "callee_to_caller"
    assert by_relation["received_dependency_failure"].allows_directed_propagation is True
    assert by_relation["caller_load_transmission"].direction == "caller_to_callee"
    assert by_relation["caller_load_transmission"].allows_directed_propagation is True
    assert by_relation["shared_resource_association"].direction == "undirected_association"
    assert by_relation["shared_resource_association"].allows_directed_propagation is False


@pytest.mark.parametrize(
    ("relation", "direction", "allows_directed_propagation"),
    (
        ("received_dependency_failure", "callee_to_caller", True),
        ("caller_load_transmission", "caller_to_callee", True),
        ("shared_resource_association", "undirected_association", False),
    ),
)
def test_one_source_can_support_service_call_and_diagnostic_relation(
    relation: str,
    direction: str,
    allows_directed_propagation: bool,
) -> None:
    module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    time_contract, universe, frozen, registry, identity_manifest = inference_identity_context()
    shared_source_fingerprint = "service-and-diagnostic"
    service_call = source(
        module,
        time_contract,
        fingerprint=shared_source_fingerprint,
        opaque_case_id=time_contract.opaque_case_id,
    )
    diagnostic_relation = source(
        module,
        time_contract,
        relation=relation,
        direction=direction,
        requests=None if relation == "shared_resource_association" else 10,
        errors=None if relation == "shared_resource_association" else 0,
        latency=None if relation == "shared_resource_association" else 0.1,
        fingerprint=shared_source_fingerprint,
        provenance="trace:diagnostic",
        opaque_case_id=time_contract.opaque_case_id,
    )

    pair, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(
            module,
            time_contract,
            opaque_case_id=time_contract.opaque_case_id,
        ),
        source_parents=(service_call, diagnostic_relation),
    )
    service_record = derived_source_from_parent(module, manifest, service_call)
    diagnostic_record = derived_source_from_parent(module, manifest, diagnostic_relation)
    edges_by_relation = {item.relation_type: item for item in pair.normal_snapshot.edges}

    assert service_record.source_record_id != diagnostic_record.source_record_id
    assert set(edges_by_relation) == {"service_call", relation}
    assert edges_by_relation["service_call"].direction == "caller_to_callee"
    assert edges_by_relation[relation].direction == direction
    assert edges_by_relation[relation].allows_directed_propagation is allows_directed_propagation
    if relation == "shared_resource_association":
        assert edges_by_relation[relation].traffic.value is None
        assert edges_by_relation[relation].error_count.value is None
        assert edges_by_relation[relation].error_rate.value is None
        assert edges_by_relation[relation].latency_p95_seconds.value is None
    assert pair.duplicate_source_record_ids == ()
    assert pair.conflicted_source_record_ids == ()
    assert pair.rejected_input_sha256s == ()
    assert set(pair.accepted_input_sha256s) == {
        service_record.source_record_sha256,
        diagnostic_record.source_record_sha256,
    }
    assert pair.verify()
    assert pair.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )
    receipt = boundary.audit_replayed_graph_snapshot_boundary(
        pair,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
    )
    assert receipt.verify()


def test_low_fault_coverage_keeps_missing_edge_unknown_instead_of_removed() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    normal = source(module, time_contract)

    low_coverage, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract, fault=0.2),
        source_parents=(normal,),
    )
    adequate_coverage, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract, fault=1.0),
        source_parents=(normal,),
    )

    assert low_coverage.edge_changes[0].change_state == "unknown"
    assert low_coverage.edge_changes[0].normal_presence == "observed"
    assert low_coverage.edge_changes[0].fault_presence == ("not_observed_insufficient_coverage")
    assert adequate_coverage.edge_changes[0].change_state == "removed"
    assert adequate_coverage.edge_changes[0].fault_presence == ("not_observed_sufficient_coverage")


def test_duplicate_source_is_counted_once_and_all_provenance_is_retained() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    first = source(
        module,
        time_contract,
        requests=10,
        errors=1,
        fingerprint="e",
        provenance="trace:export-a",
    )
    second_export = source(
        module,
        time_contract,
        requests=10,
        errors=1,
        fingerprint="e",
        provenance="trace:export-b",
    )

    coverage_parents = coverages(module, time_contract)
    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverage_parents,
        source_parents=(second_export, first, first),
    )

    edge = result.normal_snapshot.edges[0]
    first_record = derived_source_from_parent(module, manifest, first)
    assert edge.traffic.value == 10.0
    assert edge.error_count.value == 1.0
    assert edge.source_record_ids == (first_record.source_record_id,)
    assert edge.provenance_ids == tuple(
        sorted(first.derived_provenance_ids + second_export.derived_provenance_ids)
    )
    assert result.duplicate_source_record_ids == (first_record.source_record_id,)
    assert result.input_occurrence_count == 3
    assert sum(item.occurrence_count for item in result.input_decisions) == 3
    assert result.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )


def test_edge_change_keeps_separate_traffic_error_rate_and_latency_changes() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    normal = source(module, time_contract, requests=10, errors=1, latency=0.1)
    fault = source(
        module,
        time_contract,
        kind="fault",
        requests=10,
        errors=2,
        latency=0.2,
        fingerprint="f",
    )

    result, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=(fault, normal),
    )
    change = result.edge_changes[0]

    assert change.change_state == "stable"
    assert change.traffic_change == "stable"
    assert change.error_count_change == "increased"
    assert change.error_rate_change == "increased"
    assert change.latency_change == "increased"


def test_conflicting_records_with_one_source_identity_are_rejected_without_counting() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    first = source(module, time_contract, requests=10, fingerprint="1")
    conflict = source(
        module,
        time_contract,
        requests=11,
        fingerprint="1",
        provenance="trace:conflict",
    )

    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=(conflict, first),
    )

    first_record = derived_source_from_parent(module, manifest, first)
    conflict_record = derived_source_from_parent(module, manifest, conflict)
    assert result.normal_snapshot.edges == ()
    assert result.conflicted_source_record_ids == (first_record.source_record_id,)
    assert set(result.rejected_input_sha256s) == {
        first_record.source_record_sha256,
        conflict_record.source_record_sha256,
    }
    assert {item.reason_codes for item in result.input_decisions} == {
        ("conflicting_duplicate_source",)
    }


def test_metric_states_distinguish_no_record_zero_verified_normal_and_unknown() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    no_record = source(
        module,
        time_contract,
        state="no_record",
        requests=None,
        errors=None,
        latency=None,
    )
    observed_zero = source(
        module,
        time_contract,
        kind="fault",
        requests=0,
        errors=0,
        latency=0.0,
        fingerprint="2",
    )
    result, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=(observed_zero, no_record),
    )

    normal = result.normal_snapshot.edges[0]
    fault = result.fault_snapshot.edges[0]
    assert normal.observation_state == "not_observed"
    assert normal.traffic.state == "no_record"
    assert fault.traffic.state == "zero"
    assert fault.error_count.state == "zero"
    assert fault.latency_p95_seconds.state == "zero"
    assert fault.error_rate.state == "unknown"


def test_mixed_observed_and_unknown_sources_keep_edge_metrics_unknown() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    source_parents = tuple(
        parent
        for kind, observed_fingerprint, unknown_fingerprint in (
            ("normal", "mixed-normal-observed", "mixed-normal-unknown"),
            ("fault", "mixed-fault-observed", "mixed-fault-unknown"),
        )
        for parent in (
            source(
                module,
                time_contract,
                kind=kind,
                requests=10,
                errors=0,
                latency=0.1,
                fingerprint=observed_fingerprint,
            ),
            source(
                module,
                time_contract,
                kind=kind,
                state="unknown",
                requests=None,
                errors=None,
                latency=None,
                fingerprint=unknown_fingerprint,
            ),
        )
    )
    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=source_parents,
    )

    for snapshot, kind in (
        (result.normal_snapshot, "normal"),
        (result.fault_snapshot, "fault"),
    ):
        edge = snapshot.edges[0]
        expected_source_ids = tuple(
            sorted(
                derived_source_from_parent(module, manifest, parent).source_record_id
                for parent in source_parents
                if parent.window_kind == kind
            )
        )
        assert edge.observation_state == "observed"
        for metric in (
            edge.traffic,
            edge.error_count,
            edge.error_rate,
            edge.latency_p95_seconds,
        ):
            assert metric.state == "unknown"
            assert metric.value is None
            assert metric.aggregation == "not_computable"
            assert metric.contributing_source_record_ids == expected_source_ids

    change = result.edge_changes[0]
    assert change.traffic_change == "unknown"
    assert change.error_count_change == "unknown"
    assert change.error_rate_change == "unknown"
    assert change.latency_change == "unknown"


@pytest.mark.parametrize(
    ("metric_name", "missing_fields"),
    (
        ("traffic", {"requests": None}),
        ("error_count", {"errors": None}),
        ("error_rate", {"requests": None}),
        ("latency_p95_seconds", {"latency": None}),
    ),
)
def test_observed_source_missing_required_field_keeps_metric_unknown(
    metric_name: str,
    missing_fields: dict[str, object],
) -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    complete = source(module, time_contract, fingerprint=f"complete-{metric_name}")
    incomplete = source(
        module,
        time_contract,
        fingerprint=f"incomplete-{metric_name}",
        **missing_fields,
    )
    result, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=coverages(module, time_contract),
        source_parents=(complete, incomplete),
    )

    metric = getattr(result.normal_snapshot.edges[0], metric_name)
    expected_source_ids = tuple(
        sorted(
            derived_source_from_parent(module, manifest, parent).source_record_id
            for parent in (complete, incomplete)
        )
    )
    assert metric.state == "unknown"
    assert metric.value is None
    assert metric.aggregation == "not_computable"
    assert metric.contributing_source_record_ids == expected_source_ids


def test_order_invariance_and_exact_replay_reject_a_self_rehashed_child_forgery() -> None:
    module = graphs()
    time_contract, universe, frozen, registry = graph_context()
    window_coverages = coverages(module, time_contract)
    normal = source(module, time_contract, requests=10, errors=1)
    fault = source(
        module,
        time_contract,
        kind="fault",
        requests=20,
        errors=2,
        fingerprint="2",
    )
    original, manifest = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=window_coverages,
        source_parents=(normal, fault),
    )
    reordered, _ = pair_from_parents(
        module,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        coverage_parents=tuple(reversed(window_coverages)),
        source_parents=(fault, normal),
    )
    assert reordered == original

    edge_document = original.normal_snapshot.edges[0].model_dump(mode="json")
    edge_document["traffic"]["value"] = 999.0
    edge_unsigned = {
        key: value for key, value in edge_document.items() if key not in {"edge_id", "edge_sha256"}
    }
    edge_digest = sha256_hex(canonical_json_bytes(edge_unsigned))
    edge_document["edge_id"] = f"gedge_{edge_digest[:24]}"
    edge_document["edge_sha256"] = edge_digest

    normal_document = original.normal_snapshot.model_dump(mode="json")
    normal_document["edges"] = [edge_document]
    normal_unsigned = {
        key: value for key, value in normal_document.items() if key != "snapshot_sha256"
    }
    normal_document["snapshot_sha256"] = sha256_hex(canonical_json_bytes(normal_unsigned))
    pair_document = original.model_dump(mode="json")
    pair_document["normal_snapshot"] = normal_document
    pair_unsigned = {key: value for key, value in pair_document.items() if key != "pair_sha256"}
    pair_document["pair_sha256"] = sha256_hex(canonical_json_bytes(pair_unsigned))
    forged = module.GraphSnapshotPair.model_validate(pair_document)

    assert forged.verify()
    assert not forged.verify_inputs(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    )


def test_graph_snapshot_schemas_are_explicitly_versioned_in_registry() -> None:
    module = graphs()
    compatibility = importlib.import_module("nurse_rcl.v2.schema_compatibility")

    assert module.GRAPH_SNAPSHOT_SCHEMA_VERSION == "nurse.v2.graph-snapshot/4"
    assert compatibility.CURRENT_SCHEMA_VERSIONS["graph_snapshot"] == ("nurse.v2.graph-snapshot/4")
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["graph_snapshot"] == (
        "nurse.v2.graph-snapshot/1",
        "nurse.v2.graph-snapshot/2",
        "nurse.v2.graph-snapshot/3",
    )
    expected_versions = {
        "canonical_graph_coverage_input": "nurse.v2.canonical-graph-coverage-input/2",
        "canonical_graph_source_input": "nurse.v2.canonical-graph-source-input/2",
        "frozen_graph_input_manifest": "nurse.v2.frozen-graph-input-manifest/1",
        "graph_adapter_parent_record": "nurse.v2.graph-adapter-parent-record/1",
        "graph_input_decision": "nurse.v2.graph-input-decision/2",
        "graph_snapshot_pair": "nurse.v2.graph-snapshot-pair/5",
        "graph_source_record": "nurse.v2.graph-source-record/4",
        "graph_window_coverage": "nurse.v2.graph-window-coverage/4",
    }
    assert {
        "graph_source_record",
        "graph_window_coverage",
        "graph_input_decision",
        "graph_snapshot_pair",
    } <= set(compatibility.CURRENT_SCHEMA_VERSIONS)
    assert {
        name: compatibility.CURRENT_SCHEMA_VERSIONS[name] for name in expected_versions
    } == expected_versions
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["graph_source_record"] == (
        "nurse.v2.graph-source-record/1",
        "nurse.v2.graph-source-record/2",
        "nurse.v2.graph-source-record/3",
    )
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["graph_input_decision"] == (
        "nurse.v2.graph-input-decision/1",
    )
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["graph_window_coverage"] == (
        "nurse.v2.graph-window-coverage/1",
        "nurse.v2.graph-window-coverage/2",
        "nurse.v2.graph-window-coverage/3",
    )
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["graph_snapshot_pair"] == (
        "nurse.v2.graph-snapshot-pair/1",
        "nurse.v2.graph-snapshot-pair/2",
        "nurse.v2.graph-snapshot-pair/3",
        "nurse.v2.graph-snapshot-pair/4",
    )
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["canonical_graph_source_input"] == (
        "nurse.v2.canonical-graph-source-input/1",
    )
    assert compatibility.RECOGNIZED_LEGACY_SCHEMA_VERSIONS["canonical_graph_coverage_input"] == (
        "nurse.v2.canonical-graph-coverage-input/1",
    )


def test_inference_boundary_requires_the_complete_exactly_replayed_pair() -> None:
    graph_module = graphs()
    boundary = importlib.import_module("nurse_rcl.v2.inference_leakage")
    _, identity_manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="private-graph-case",
                incident_group_id="grp_" + "1" * 24,
                answer_services=("private-answer",),
            ),
        ),
        method="random_uuid_v4",
        uuid_factory=lambda: uuid.UUID("12345678-1234-4234-8234-1234567890ab"),
    )
    opaque_case_id = identity_manifest.records[0].opaque_case_id
    time_contract, universe, frozen, registry = graph_context(opaque_case_id)
    window_coverages = coverages(
        graph_module,
        time_contract,
        opaque_case_id=opaque_case_id,
    )
    normal = source(
        graph_module,
        time_contract,
        opaque_case_id=opaque_case_id,
    )
    manifest = freeze_graph_inputs(
        graph_module,
        time_contract,
        source_parents=(normal,),
        coverage_parents=window_coverages,
    )
    pair = graph_module.build_graph_snapshot_pair(
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )

    with pytest.raises(
        boundary.InferenceBoundaryLeakageError,
        match="unapproved_model_type",
    ):
        boundary.audit_inference_boundary(
            pair.normal_snapshot,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )
    with pytest.raises(
        boundary.InferenceBoundaryLeakageError,
        match="unapproved_model_type",
    ):
        boundary.audit_inference_boundary(
            pair,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )

    receipt = boundary.audit_replayed_graph_snapshot_boundary(
        pair,
        time_contract=time_contract,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
    )

    assert receipt.verify()
    assert receipt.opaque_case_ids == (opaque_case_id,)
    assert "GraphSnapshotPair" in receipt.public_model_types
