from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest

OPAQUE_CASE_ID = "opq_" + "a" * 32
DECISION_TIME = 100


def candidate_module() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.candidate_universe")
    except ModuleNotFoundError:
        pytest.fail("complete candidate-universe construction is not implemented")


def schemas() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.observation_schemas")


def registry_module() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.temporal_entity_registry")


def grouping_module() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.event_grouping")


def interval(start: int, end: int) -> Any:
    return schemas().TimeInterval(start_unix_ns=start, end_unix_ns=end)


def observed_summary() -> Any:
    return schemas().WindowSummary(
        record_state="observed",
        valid_point_count=5,
        coverage_fraction=1.0,
        median=1.0,
        mad=0.0,
        minimum=1.0,
        maximum=1.0,
        event_count=None,
    )


def no_record_summary() -> Any:
    return schemas().WindowSummary(
        record_state="no_record",
        valid_point_count=0,
        coverage_fraction=0.0,
        median=None,
        mad=None,
        minimum=None,
        maximum=None,
        event_count=None,
    )


def observation(
    *,
    suffix: str,
    evidence_state: str,
    formally_scorable: bool,
    start: int = 10,
    end: int = 20,
) -> Any:
    summary = no_record_summary() if evidence_state == "no_record" else observed_summary()
    anomaly_strength = {
        "abnormal": 0.8,
        "degraded": 0.6,
        "verified_normal": 0.0,
        "zero": 0.0,
        "unknown": None,
        "no_record": None,
    }[evidence_state]
    signal_type = "unknown" if evidence_state == "unknown" else "gauge"
    return schemas().Observation(
        observation_id="obs_" + suffix * 24,
        opaque_case_id=OPAQUE_CASE_ID,
        entity_binding_id=None,
        modality="metrics",
        signal_family="cpu_usage",
        signal_type=signal_type,
        unit="cores",
        interval=interval(start, end),
        decision_time_unix_ns=DECISION_TIME,
        aggregation="median",
        source_fingerprint=suffix * 64,
        baseline_summary=summary,
        fault_summary=summary,
        evidence_state=evidence_state,
        anomaly_strength=anomaly_strength,
        coverage_quality=0.5 if evidence_state == "degraded" else 1.0,
        minimum_valid_points=5,
        transform="identity",
        quality_flags=("insufficient_points",) if evidence_state == "degraded" else (),
        provenance_ids=(f"observation:{suffix}",),
        formal_scoring_eligible=formally_scorable,
        adapter_semantics_sha256="f" * 64 if formally_scorable else None,
        association_source_record_sha256=None,
    )


def metadata_record(
    *,
    suffix: str,
    entity: str,
    service: str,
    start: int = 0,
    end: int = DECISION_TIME,
) -> Any:
    return registry_module().build_entity_metadata_record(
        opaque_case_id=OPAQUE_CASE_ID,
        metadata_source="deployment",
        entity_kind="pod",
        observed_entity_key=entity,
        relation="explicit_service_binding",
        service_entity_key=service,
        valid_interval=interval(start, end),
        source_fingerprint=suffix * 64,
        provenance_ids=(f"deployment:{suffix}",),
    )


def collector_record() -> Any:
    return registry_module().build_entity_metadata_record(
        opaque_case_id=OPAQUE_CASE_ID,
        metadata_source="metrics",
        entity_kind="untyped",
        observed_entity_key="collector:kubernetes-endpoints",
        relation="collector_source",
        service_entity_key=None,
        valid_interval=interval(0, DECISION_TIME),
        source_fingerprint="e" * 64,
        provenance_ids=("metrics:collector",),
    )


def entity_input(item: Any, *, entity: str, kind: str = "pod") -> Any:
    return registry_module().ObservationEntityInput(
        observation=item,
        entity_kind=kind,
        observed_entity_key=entity,
    )


def build_registry_and_grouping(
    *,
    records: Sequence[Any],
    observation_inputs: Sequence[Any],
) -> tuple[Any, Any]:
    registry = registry_module().build_temporal_entity_registry(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        metadata_records=records,
        observations=observation_inputs,
    )
    grouping = grouping_module().build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=tuple(item.observation for item in registry.resolutions),
        association_claims=(),
        family_mapping=grouping_module().build_frozen_signal_family_mapping(
            mapping_version="candidate-test-map/1",
            entries=(),
        ),
    )
    return registry, grouping


def rich_fixture(*, reverse_inputs: bool = False) -> tuple[Any, Any]:
    supported = observation(suffix="1", evidence_state="abnormal", formally_scorable=True)
    degraded = observation(suffix="2", evidence_state="degraded", formally_scorable=False)
    unknown = observation(suffix="3", evidence_state="unknown", formally_scorable=False)
    no_record = observation(suffix="4", evidence_state="no_record", formally_scorable=False)
    normal = observation(suffix="5", evidence_state="verified_normal", formally_scorable=True)
    zero = observation(suffix="6", evidence_state="zero", formally_scorable=True)
    conflict = observation(suffix="7", evidence_state="abnormal", formally_scorable=True)
    collector = observation(suffix="8", evidence_state="abnormal", formally_scorable=True)

    records = [
        metadata_record(suffix="1", entity="pod:supported", service="service:supported"),
        metadata_record(suffix="2", entity="pod:degraded", service="service:degraded"),
        metadata_record(suffix="3", entity="pod:unknown", service="service:unknown"),
        metadata_record(suffix="4", entity="pod:no-record", service="service:no-record"),
        metadata_record(suffix="5", entity="pod:normal", service="service:normal"),
        metadata_record(suffix="6", entity="pod:zero", service="service:zero"),
        metadata_record(suffix="9", entity="pod:unobserved", service="service:unobserved"),
        metadata_record(suffix="a", entity="pod:conflict", service="service:conflict-a"),
        metadata_record(suffix="b", entity="pod:conflict", service="service:conflict-b"),
        collector_record(),
    ]
    inputs = [
        entity_input(supported, entity="pod:supported"),
        entity_input(degraded, entity="pod:degraded"),
        entity_input(unknown, entity="pod:unknown"),
        entity_input(no_record, entity="pod:no-record"),
        entity_input(normal, entity="pod:normal"),
        entity_input(zero, entity="pod:zero"),
        entity_input(conflict, entity="pod:conflict"),
        entity_input(
            collector,
            entity="collector:kubernetes-endpoints",
            kind="untyped",
        ),
    ]
    if reverse_inputs:
        records.reverse()
        inputs.reverse()
    return build_registry_and_grouping(records=records, observation_inputs=inputs)


def by_service(universe: Any) -> dict[str, Any]:
    return {item.service_entity_key: item for item in universe.candidates}


def test_complete_universe_retains_every_valid_service_and_maps_evidence_states() -> None:
    registry, grouping = rich_fixture()

    universe = candidate_module().build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=interval(0, DECISION_TIME),
    )

    candidates = by_service(universe)
    assert universe.verify()
    assert universe.candidate_count == 9
    assert tuple(candidates) == tuple(sorted(candidates))
    assert "collector:kubernetes-endpoints" not in candidates
    assert "kubernetes-endpoints" not in candidates
    assert {key: item.evidence_state for key, item in candidates.items()} == {
        "service:conflict-a": "unknown",
        "service:conflict-b": "unknown",
        "service:degraded": "degraded",
        "service:no-record": "unknown",
        "service:normal": "zero_support",
        "service:supported": "supported",
        "service:unknown": "unknown",
        "service:unobserved": "zero_support",
        "service:zero": "zero_support",
    }

    supported = candidates["service:supported"]
    assert supported.observation_ids == ("obs_" + "1" * 24,)
    assert len(supported.event_group_ids) == 1
    assert supported.binding_ids

    for key in ("service:normal", "service:zero", "service:unobserved"):
        assert candidates[key].observation_ids == ()
        assert candidates[key].event_group_ids == ()
        assert candidates[key].binding_ids

    for key in ("service:conflict-a", "service:conflict-b"):
        assert candidates[key].observation_ids == ()
        assert candidates[key].event_group_ids == ()
        assert candidates[key].binding_ids


def test_candidate_construction_is_order_independent_and_scope_hash_is_frozen() -> None:
    registry, grouping = rich_fixture()
    reversed_registry, reversed_grouping = rich_fixture(reverse_inputs=True)

    canonical = candidate_module().build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=interval(0, 90),
    )
    permuted = candidate_module().build_complete_candidate_universe(
        entity_registry=reversed_registry,
        event_grouping=reversed_grouping,
        scope_interval=interval(0, 90),
    )
    wider_scope = candidate_module().build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=interval(0, DECISION_TIME),
    )

    assert permuted == canonical
    assert wider_scope.candidates == canonical.candidates
    assert wider_scope.observation_contract_sha256 != canonical.observation_contract_sha256
    assert wider_scope.universe_sha256 != canonical.universe_sha256


def test_out_of_scope_metadata_does_not_create_a_candidate() -> None:
    in_scope = metadata_record(
        suffix="1",
        entity="pod:inside",
        service="service:inside",
        start=0,
        end=50,
    )
    out_of_scope = metadata_record(
        suffix="2",
        entity="pod:outside",
        service="service:outside",
        start=50,
        end=DECISION_TIME,
    )
    registry, grouping = build_registry_and_grouping(
        records=(out_of_scope, in_scope),
        observation_inputs=(),
    )

    universe = candidate_module().build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=interval(0, 50),
    )

    assert tuple(item.service_entity_key for item in universe.candidates) == ("service:inside",)
    assert universe.candidates[0].evidence_state == "zero_support"


def test_candidate_construction_rejects_inconsistent_or_unverified_inputs() -> None:
    registry, grouping = rich_fixture()
    missing_observation_grouping = grouping_module().build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=grouping.observations[:-1],
        association_claims=(),
        family_mapping=grouping.family_mapping,
        association_policy=grouping.association_policy,
    )

    with pytest.raises(ValueError, match="observation membership"):
        candidate_module().build_complete_candidate_universe(
            entity_registry=registry,
            event_grouping=missing_observation_grouping,
            scope_interval=interval(0, DECISION_TIME),
        )

    forged_registry = registry.model_copy(update={"registry_sha256": "0" * 64})
    with pytest.raises(ValueError, match="registry failed frozen verification"):
        candidate_module().build_complete_candidate_universe(
            entity_registry=forged_registry,
            event_grouping=grouping,
            scope_interval=interval(0, DECISION_TIME),
        )

    with pytest.raises(ValueError, match="scope extends beyond decision time"):
        candidate_module().build_complete_candidate_universe(
            entity_registry=registry,
            event_grouping=grouping,
            scope_interval=interval(0, DECISION_TIME + 1),
        )

    forged_scope = interval(0, DECISION_TIME).model_copy(update={"start_unix_ns": DECISION_TIME})
    with pytest.raises(ValueError, match="candidate scope is invalid"):
        candidate_module().build_complete_candidate_universe(
            entity_registry=registry,
            event_grouping=grouping,
            scope_interval=forged_scope,
        )
