from __future__ import annotations

import importlib
import unicodedata
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

OPAQUE_CASE_ID = "opq_" + "a" * 32
DECISION_TIME = 100


def grouping_module() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.event_grouping")
    except ModuleNotFoundError:
        pytest.fail("event grouping module is not implemented")


def interval(start: int, end: int) -> Any:
    schemas = importlib.import_module("nurse_rcl.v2.observation_schemas")
    return schemas.TimeInterval(start_unix_ns=start, end_unix_ns=end)


def association_source_fields(
    module: ModuleType,
    *,
    basis: str,
    key: str | None,
) -> tuple[Any, ...]:
    field_kind, default_value = {
        "shared_export_id": ("export_id", "export:fixture-123"),
        "explicit_source_relation": (
            "source_relation_id",
            "relation:fixture-123",
        ),
        "deterministic_association_key": (
            "log_record_id",
            "log:fixture-123",
        ),
    }[basis]
    return (
        module.AdapterAssociationSourceField(
            field_kind=field_kind,
            raw_value=default_value if key is None else key,
        ),
    )


def observation(
    *,
    suffix: str,
    modality: str,
    source_family: str,
    start: int,
    end: int,
    semantics: str = "f",
    binding: str | None = None,
    association_basis: str = "shared_export_id",
    association_key: str | None = None,
) -> Any:
    schemas = importlib.import_module("nurse_rcl.v2.observation_schemas")
    module = grouping_module()
    summary = schemas.WindowSummary(
        record_state="observed",
        valid_point_count=5,
        coverage_fraction=1.0,
        median=1.0,
        mad=0.0,
        minimum=1.0,
        maximum=1.0,
        event_count=None,
    )
    item = schemas.Observation(
        observation_id="obs_" + suffix * 24,
        opaque_case_id=OPAQUE_CASE_ID,
        entity_binding_id=binding,
        modality=modality,
        signal_family=source_family,
        signal_type="gauge",
        unit="count",
        interval=interval(start, end),
        decision_time_unix_ns=DECISION_TIME,
        aggregation="median",
        source_fingerprint=suffix * 64,
        baseline_summary=summary,
        fault_summary=summary,
        evidence_state="verified_normal",
        anomaly_strength=0.0,
        coverage_quality=1.0,
        minimum_valid_points=5,
        transform="identity",
        quality_flags=(),
        provenance_ids=(f"source:{suffix}",),
        formal_scoring_eligible=True,
        adapter_semantics_sha256=semantics * 64,
    )
    source_record = module.build_canonical_association_source_record(
        opaque_case_id=item.opaque_case_id,
        observation_id=item.observation_id,
        decision_time_unix_ns=item.decision_time_unix_ns,
        parent_source_fingerprint=item.source_fingerprint,
        adapter_semantics_sha256=item.adapter_semantics_sha256,
        association_basis=association_basis,
        source_fields=association_source_fields(
            module,
            basis=association_basis,
            key=association_key,
        ),
    )
    return schemas.Observation.model_validate(
        {
            **item.model_dump(mode="json"),
            "association_source_record_sha256": (source_record.association_source_record_sha256),
        }
    )


def family_mapping(module: ModuleType) -> Any:
    entries = (
        module.build_signal_family_mapping_entry(
            adapter_semantics_sha256="f" * 64,
            source_signal_family="cpu_usage",
            primary_signal_family="cpu",
            provenance_ids=("adapter:metrics/v1",),
        ),
        module.build_signal_family_mapping_entry(
            adapter_semantics_sha256="f" * 64,
            source_signal_family="span_failure",
            primary_signal_family="error",
            provenance_ids=("adapter:traces/v1",),
        ),
    )
    return module.build_frozen_signal_family_mapping(
        mapping_version="adapter-family-map/1",
        entries=entries,
    )


def claim(
    module: ModuleType,
    *observations: Any,
    basis: str = "shared_export_id",
    key: str | None = None,
) -> Any:
    if not observations:
        raise ValueError("test claim requires observations")
    proof_references = tuple(
        module.build_observation_association_proof(
            observation=item,
            evidence_time_unix_ns=90,
            adapter_identity=module.build_adapter_association_identity(
                observation=item,
                source_record=module.build_canonical_association_source_record(
                    opaque_case_id=item.opaque_case_id,
                    observation_id=item.observation_id,
                    decision_time_unix_ns=item.decision_time_unix_ns,
                    parent_source_fingerprint=item.source_fingerprint,
                    adapter_semantics_sha256=item.adapter_semantics_sha256,
                    association_basis=basis,
                    source_fields=association_source_fields(
                        module,
                        basis=basis,
                        key=key,
                    ),
                ),
            ),
        )
        for item in observations
    )
    return module.build_association_claim(
        opaque_case_id=observations[0].opaque_case_id,
        decision_time_unix_ns=observations[0].decision_time_unix_ns,
        association_basis=basis,
        proof_references=proof_references,
    )


def test_shared_identity_groups_duplicate_exports_across_modalities() -> None:
    module = grouping_module()
    metric = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        binding="bind_" + "1" * 24,
    )
    deployment = observation(
        suffix="2",
        modality="deployment",
        source_family="cpu_usage",
        start=12,
        end=24,
        binding="bind_" + "2" * 24,
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(metric, deployment),
        association_claims=(claim(module, metric, deployment),),
        family_mapping=family_mapping(module),
    )

    assert result.verify()
    assert len(result.event_groups) == 1
    group = result.event_groups[0]
    assert group.association_basis == "shared_export_id"
    assert group.observation_ids == tuple(
        sorted((metric.observation_id, deployment.observation_id))
    )
    assert group.primary_signal_family == "cpu"
    assert group.interval == interval(10, 24)
    assert group.entity_binding_ids == (
        "bind_" + "1" * 24,
        "bind_" + "2" * 24,
    )
    assert set(group.source_fingerprints) == {
        metric.source_fingerprint,
        deployment.source_fingerprint,
    }
    assert set(group.provenance_ids) == {
        "source:1",
        "source:2",
        *(proof.proof_id for proof in result.association_claims[0].proof_references),
    }


def test_temporal_proximity_never_groups_cross_modal_records() -> None:
    module = grouping_module()
    metric = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    log = observation(
        suffix="2",
        modality="logs",
        source_family="span_failure",
        start=10,
        end=20,
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(metric, log),
        association_claims=(),
        family_mapping=family_mapping(module),
    )

    assert len(result.event_groups) == 2
    assert all(
        group.association_basis == "singleton_no_association_evidence"
        for group in result.event_groups
    )
    assert {group.observation_ids for group in result.event_groups} == {
        (metric.observation_id,),
        (log.observation_id,),
    }


def test_every_observation_is_retained_once_and_unmapped_is_unknown_safe() -> None:
    module = grouping_module()
    observations = (
        observation(
            suffix="1",
            modality="metrics",
            source_family="cpu_usage",
            start=10,
            end=20,
        ),
        observation(
            suffix="2",
            modality="metrics",
            source_family="cpu_usage_total_filename_hint",
            start=10,
            end=20,
        ),
        observation(
            suffix="3",
            modality="logs",
            source_family="mystery",
            start=11,
            end=12,
        ),
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=observations,
        association_claims=(),
        family_mapping=family_mapping(module),
    )

    memberships = [
        observation_id for group in result.event_groups for observation_id in group.observation_ids
    ]
    assert sorted(memberships) == sorted(item.observation_id for item in observations)
    assert len(memberships) == len(set(memberships))
    family_by_observation = {
        group.observation_ids[0]: group.primary_signal_family for group in result.event_groups
    }
    assert family_by_observation[observations[0].observation_id] == "cpu"
    assert family_by_observation[observations[1].observation_id] == "unclassified"
    assert family_by_observation[observations[2].observation_id] == "unclassified"
    assert all(len(decision.observation_ids) == 1 for decision in result.decisions)


def test_grouping_is_input_order_invariant() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_basis="deterministic_association_key",
        association_key="log:stable",
    )
    second = observation(
        suffix="2",
        modality="metrics",
        source_family="cpu_usage",
        start=12,
        end=21,
        association_basis="deterministic_association_key",
        association_key="log:stable",
    )
    third = observation(
        suffix="3",
        modality="logs",
        source_family="mystery",
        start=15,
        end=16,
    )
    association = claim(
        module,
        first,
        second,
        basis="deterministic_association_key",
        key="log:stable",
    )
    mapping = family_mapping(module)

    canonical = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(first, second, third),
        association_claims=(association,),
        family_mapping=mapping,
    )
    permuted = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(third, second, first),
        association_claims=(association,),
        family_mapping=module.build_frozen_signal_family_mapping(
            mapping_version=mapping.mapping_version,
            entries=tuple(reversed(mapping.entries)),
        ),
    )

    assert permuted == canonical


@pytest.mark.parametrize(
    "basis",
    (
        "shared_export_id",
        "explicit_source_relation",
        "deterministic_association_key",
    ),
)
def test_only_the_three_frozen_association_bases_can_group(basis: str) -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="logs",
        source_family="span_failure",
        start=1,
        end=2,
        association_basis=basis,
    )
    second = observation(
        suffix="2",
        modality="traces",
        source_family="span_failure",
        start=80,
        end=90,
        association_basis=basis,
    )
    association = claim(module, first, second, basis=basis)

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(first, second),
        association_claims=(association,),
        family_mapping=family_mapping(module),
    )

    assert len(result.event_groups) == 1
    assert result.event_groups[0].association_basis == basis
    assert result.event_groups[0].association_claim_ids == (association.claim_id,)


def test_same_identity_with_conflicting_primary_families_is_rejected() -> None:
    module = grouping_module()
    cpu = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    error = observation(
        suffix="2",
        modality="traces",
        source_family="span_failure",
        start=10,
        end=20,
    )

    with pytest.raises(ValueError, match="conflicting primary signal families"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(cpu, error),
            association_claims=(claim(module, cpu, error),),
            family_mapping=family_mapping(module),
        )


def test_mixed_group_keeps_one_mapped_primary_and_audits_unmapped_auxiliary() -> None:
    module = grouping_module()
    mapped = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    auxiliary = observation(
        suffix="2",
        modality="logs",
        source_family="adapter_specific_auxiliary",
        start=10,
        end=20,
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(mapped, auxiliary),
        association_claims=(claim(module, mapped, auxiliary),),
        family_mapping=family_mapping(module),
    )

    group = result.event_groups[0]
    assert group.primary_signal_family == "cpu"
    assert group.family_assignment_status == "exact_mapped_with_unmapped_members"
    assignment_by_observation = {
        item.observation_id: item for item in group.member_family_assignments
    }
    auxiliary_assignment = assignment_by_observation[auxiliary.observation_id]
    assert auxiliary_assignment.assignment_basis == "missing_mapping_unknown_safe"
    assert auxiliary_assignment.primary_signal_family == "unclassified"
    assert auxiliary_assignment.provenance_ids == auxiliary.provenance_ids


def test_frozen_mapping_may_explicitly_assign_the_unclassified_family() -> None:
    module = grouping_module()
    item = observation(
        suffix="1",
        modality="metrics",
        source_family="adapter_declared_unknown",
        start=10,
        end=20,
    )
    entry = module.build_signal_family_mapping_entry(
        adapter_semantics_sha256="f" * 64,
        source_signal_family="adapter_declared_unknown",
        primary_signal_family="unclassified",
        provenance_ids=("adapter:explicit-unknown/v1",),
    )
    mapping = module.build_frozen_signal_family_mapping(
        mapping_version="adapter-family-map/1",
        entries=(entry,),
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(item,),
        association_claims=(),
        family_mapping=mapping,
    )

    group = result.event_groups[0]
    assert group.primary_signal_family == "unclassified"
    assert group.family_assignment_status == "exact_mapped"
    assert group.member_family_assignments[0].mapping_entry_id == entry.mapping_entry_id


def test_observation_with_two_incompatible_association_records_is_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_key="export:a",
    )
    second = observation(
        suffix="2",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_key="export:a",
    )
    third = observation(
        suffix="3",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_key="export:b",
    )

    assert claim(module, first, second, key="export:a").verify()
    with pytest.raises(ValueError, match="source record hash does not match observation"):
        claim(module, second, third, key="export:b")


def test_ambiguous_exact_family_mapping_is_rejected() -> None:
    module = grouping_module()
    cpu = module.build_signal_family_mapping_entry(
        adapter_semantics_sha256="f" * 64,
        source_signal_family="same_exact_key",
        primary_signal_family="cpu",
        provenance_ids=("adapter:a",),
    )
    memory = module.build_signal_family_mapping_entry(
        adapter_semantics_sha256="f" * 64,
        source_signal_family="same_exact_key",
        primary_signal_family="memory",
        provenance_ids=("adapter:b",),
    )

    with pytest.raises(ValueError, match="ambiguous exact key"):
        module.build_frozen_signal_family_mapping(
            mapping_version="adapter-family-map/1",
            entries=(cpu, memory),
        )


def test_duplicate_observation_and_association_claim_ids_are_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    association = claim(module, first)

    with pytest.raises(ValueError, match="observation IDs must be unique"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(first, first),
            association_claims=(),
            family_mapping=family_mapping(module),
        )
    with pytest.raises(ValueError, match="association claim IDs must be unique"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(first,),
            association_claims=(association, association),
            family_mapping=family_mapping(module),
        )


def test_duplicate_observation_references_inside_a_claim_are_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )

    proof = claim(module, first).proof_references[0]
    with pytest.raises(ValueError, match="observation references must be unique"):
        module.build_association_claim(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            association_basis="shared_export_id",
            proof_references=(proof, proof),
        )


def test_typed_adapter_field_rejects_invalid_normalized_hashes() -> None:
    module = grouping_module()

    with pytest.raises(ValueError, match="String should match pattern"):
        module.AssociationSourceFieldHash(
            field_kind="export_id",
            normalized_value_sha256="not-a-hash",
        )


def test_unknown_references_and_case_or_decision_drift_are_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    missing = observation(
        suffix="9",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_key="export:missing",
    )
    unknown = claim(module, missing, key="export:missing")
    with pytest.raises(ValueError, match="unknown observation"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(first,),
            association_claims=(unknown,),
            family_mapping=family_mapping(module),
        )

    wrong_case = first.model_copy(update={"opaque_case_id": "opq_" + "b" * 32})
    with pytest.raises(ValueError, match="different opaque case"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(wrong_case,),
            association_claims=(),
            family_mapping=family_mapping(module),
        )

    wrong_decision = first.model_copy(update={"decision_time_unix_ns": 99})
    with pytest.raises(ValueError, match="decision time"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(wrong_decision,),
            association_claims=(),
            family_mapping=family_mapping(module),
        )


def test_cross_case_drifted_and_post_decision_claims_are_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    with pytest.raises(ValueError, match="source record hash does not match observation"):
        claim(
            module,
            first.model_copy(update={"opaque_case_id": "opq_" + "b" * 32}),
        )

    with pytest.raises(ValueError, match="source record hash does not match observation"):
        claim(
            module,
            first.model_copy(update={"decision_time_unix_ns": 99}),
        )

    typed_identity = claim(module, first).proof_references[0].adapter_identity
    with pytest.raises(ValueError, match="post-decision"):
        module.build_observation_association_proof(
            observation=first,
            evidence_time_unix_ns=DECISION_TIME + 1,
            adapter_identity=typed_identity,
        )


def test_rehashed_invalid_interval_hash_and_schema_are_rejected_before_grouping() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    mapping = family_mapping(module)
    invalid_observations = (
        first.model_copy(
            update={
                "interval": first.interval.model_copy(
                    update={"start_unix_ns": 20, "end_unix_ns": 10}
                )
            }
        ),
        first.model_copy(update={"source_fingerprint": "not-a-hash"}),
        first.model_copy(update={"schema_version": "nurse.v2.observation/99"}),
    )

    for invalid in invalid_observations:
        with pytest.raises(ValueError):
            module.build_event_grouping_result(
                opaque_case_id=OPAQUE_CASE_ID,
                decision_time_unix_ns=DECISION_TIME,
                observations=(invalid,),
                association_claims=(),
                family_mapping=mapping,
            )


def built_result(module: ModuleType) -> Any:
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    second = observation(
        suffix="2",
        modality="metrics",
        source_family="cpu_usage",
        start=12,
        end=22,
    )
    return module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(first, second),
        association_claims=(claim(module, first, second),),
        family_mapping=family_mapping(module),
    )


def test_event_grouping_result_replays_exactly() -> None:
    module = grouping_module()
    result = built_result(module)

    assert module.replay_event_grouping_result(result) == result


def test_verify_and_replay_reject_hash_tampering() -> None:
    module = grouping_module()
    result = built_result(module)
    tampered = result.model_copy(update={"result_sha256": "0" * 64})

    assert tampered.verify() is False
    with pytest.raises(ValueError, match="cannot be replayed"):
        module.replay_event_grouping_result(tampered)


def test_verify_and_replay_reject_rehashed_semantic_tampering() -> None:
    module = grouping_module()
    result = built_result(module)
    document = result.model_dump(mode="json")
    changed_group = dict(document["event_groups"][0])
    changed_group["interval"] = {"start_unix_ns": 9, "end_unix_ns": 22}
    group_unsigned = {key: value for key, value in changed_group.items() if key != "event_group_id"}
    changed_group["event_group_id"] = "evt_" + sha256_hex(canonical_json_bytes(group_unsigned))[:24]
    document["event_groups"] = [changed_group]
    changed_decision = dict(document["decisions"][0])
    changed_decision["event_group_id"] = changed_group["event_group_id"]
    decision_unsigned = {
        key: value for key, value in changed_decision.items() if key != "decision_id"
    }
    changed_decision["decision_id"] = (
        "gdec_" + sha256_hex(canonical_json_bytes(decision_unsigned))[:24]
    )
    document["decisions"] = [changed_decision]
    result_unsigned = {key: value for key, value in document.items() if key != "result_sha256"}
    document["result_sha256"] = sha256_hex(canonical_json_bytes(result_unsigned))
    tampered = module.EventGroupingResult.model_validate(document)

    assert tampered.verify() is False
    with pytest.raises(ValueError, match="does not replay"):
        module.replay_event_grouping_result(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observations", ("invalid",)),
        ("association_claims", (object(),)),
        ("event_groups", ("invalid",)),
        ("decisions", (object(),)),
        ("family_mapping", object()),
        ("association_policy", object()),
    ),
)
def test_verify_is_exception_safe_and_replay_normalizes_malformed_values(
    field: str,
    value: object,
) -> None:
    module = grouping_module()
    malformed = built_result(module).model_copy(update={field: value})

    assert malformed.verify() is False
    with pytest.raises(ValueError, match="cannot be replayed"):
        module.replay_event_grouping_result(malformed)


def test_leaf_verifiers_are_exception_safe_for_wrong_nested_types() -> None:
    module = grouping_module()
    result = built_result(module)
    mapping = result.family_mapping.model_copy(update={"entries": (object(),)})
    policy = result.association_policy.model_copy(update={"rules": (object(),)})
    group = result.event_groups[0].model_copy(update={"member_family_assignments": (object(),)})

    assert mapping.verify() is False
    assert policy.verify() is False
    assert group.verify() is False


def test_event_grouping_artifacts_are_declared_in_schema_compatibility() -> None:
    module = grouping_module()
    compatibility = importlib.import_module("nurse_rcl.v2.schema_compatibility")

    assert compatibility.CURRENT_SCHEMA_VERSIONS["signal_family_mapping_entry"] == (
        module.SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["signal_family_mapping"] == (
        module.SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["event_association_claim"] == (
        module.ASSOCIATION_CLAIM_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["association_derivation_rule"] == (
        module.ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["association_policy"] == (
        module.ASSOCIATION_POLICY_SCHEMA_VERSION
    )
    assert (
        compatibility.CURRENT_SCHEMA_VERSIONS["canonical_association_source_record"]
        == module.ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["adapter_association_identity"] == (
        module.ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["observation_association_proof"] == (
        module.OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["event_group"] == (
        module.EVENT_GROUP_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["event_grouping_decision"] == (
        module.GROUPING_DECISION_SCHEMA_VERSION
    )
    assert compatibility.CURRENT_SCHEMA_VERSIONS["event_grouping_result"] == (
        module.EVENT_GROUPING_RESULT_SCHEMA_VERSION
    )


def test_public_current_event_group_is_accepted_by_schema_registry() -> None:
    module = grouping_module()
    compatibility = importlib.import_module("nurse_rcl.v2.schema_compatibility")
    built_group = built_result(module).event_groups[0]

    current_group = module.EventGroup.model_validate(built_group)
    artifact = compatibility.load_compatible_artifact(
        current_group.model_dump(mode="json"),
        artifact_kind="event_group",
    )

    assert current_group.schema_version == module.EVENT_GROUP_SCHEMA_VERSION
    assert artifact.schema_version == module.EVENT_GROUP_SCHEMA_VERSION
    assert artifact.document == current_group.model_dump(mode="json")


def test_free_form_filename_or_llm_inferred_association_is_not_accepted() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    second = observation(
        suffix="2",
        modality="logs",
        source_family="cpu_usage",
        start=10,
        end=20,
    )

    with pytest.raises(TypeError):
        module.build_association_claim(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            evidence_time_unix_ns=90,
            association_basis="deterministic_association_key",
            association_key="filename:nearby-cpu.log",
            observation_ids=(first.observation_id, second.observation_id),
            source_fingerprints=("9" * 64,),
            provenance_ids=("llm:inferred-same-event",),
        )


def test_typed_derivation_models_forbid_filename_and_llm_producers() -> None:
    module = grouping_module()
    with pytest.raises(ValueError, match="field_kind"):
        module.AdapterAssociationSourceField(
            field_kind="filename",
            normalized_value_sha256="9" * 64,
        )

    item = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    identity = claim(module, item).proof_references[0].adapter_identity
    invalid = identity.model_copy(update={"producer_kind": "llm"})
    unsigned = invalid.model_dump(mode="json", exclude={"adapter_identity_id"})
    invalid = invalid.model_copy(
        update={"adapter_identity_id": "aid_" + sha256_hex(canonical_json_bytes(unsigned))[:24]}
    )
    with pytest.raises(ValueError, match="producer_kind"):
        module.AdapterAssociationIdentity.model_validate(invalid.model_dump(mode="json"))


def test_non_singleton_decision_requires_an_association_claim_at_leaf_boundary() -> None:
    module = grouping_module()
    decision = built_result(module).decisions[0]
    changed = decision.model_copy(update={"association_claim_ids": ()})
    unsigned = changed.model_dump(mode="json", exclude={"decision_id"})
    changed = changed.model_copy(
        update={"decision_id": "gdec_" + sha256_hex(canonical_json_bytes(unsigned))[:24]}
    )

    with pytest.raises(ValueError, match="requires at least one association claim"):
        module.GroupingDecision.model_validate(changed.model_dump(mode="json"))


def test_typed_proofs_are_frozen_and_exactly_bound_to_each_observation() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_basis="deterministic_association_key",
        association_key="log:adapter-record-key-42",
    )
    second = observation(
        suffix="2",
        modality="logs",
        source_family="cpu_usage",
        start=10,
        end=20,
        association_basis="deterministic_association_key",
        association_key="log:adapter-record-key-42",
    )
    association = claim(
        module,
        first,
        second,
        basis="deterministic_association_key",
        key="log:adapter-record-key-42",
    )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(first, second),
        association_claims=(association,),
        family_mapping=family_mapping(module),
    )

    policy = result.association_policy
    assert policy.verify()
    assert association.association_policy_sha256 == policy.policy_sha256
    assert {item.observation_id for item in association.proof_references} == {
        first.observation_id,
        second.observation_id,
    }
    observation_by_id = {item.observation_id: item for item in (first, second)}
    for proof in association.proof_references:
        source = observation_by_id[proof.observation_id]
        assert proof.producer_kind == "frozen_adapter"
        assert proof.source_fingerprint == source.source_fingerprint
        assert proof.source_reference_sha256 == source.association_source_record_sha256
        assert proof.source_record.source_fields[0].raw_value == ("log:adapter-record-key-42")
        assert proof.adapter_semantics_sha256 == source.adapter_semantics_sha256
        assert proof.verify()


def test_content_addressed_proof_with_wrong_observation_source_binding_is_rejected() -> None:
    module = grouping_module()
    first = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    second = observation(
        suffix="2",
        modality="logs",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    first_identity = claim(module, first).proof_references[0].adapter_identity
    second_identity = claim(module, second).proof_references[0].adapter_identity
    first_proof = module.build_observation_association_proof(
        observation=first,
        evidence_time_unix_ns=90,
        adapter_identity=first_identity,
    )
    second_proof = module.build_observation_association_proof(
        observation=second,
        evidence_time_unix_ns=90,
        adapter_identity=second_identity,
    )
    forged_record = first_identity.source_record.model_copy(
        update={"parent_source_fingerprint": "9" * 64}
    )
    record_unsigned = forged_record.model_dump(
        mode="json",
        exclude={"source_record_id", "association_source_record_sha256"},
    )
    forged_record = forged_record.model_copy(
        update={
            "association_source_record_sha256": sha256_hex(canonical_json_bytes(record_unsigned))
        }
    )
    record_id_payload = forged_record.model_dump(mode="json", exclude={"source_record_id"})
    forged_record = forged_record.model_copy(
        update={
            "source_record_id": "asrc_" + sha256_hex(canonical_json_bytes(record_id_payload))[:24]
        }
    )
    forged_identity = first_identity.model_copy(update={"source_record": forged_record})
    identity_unsigned = forged_identity.model_dump(mode="json", exclude={"adapter_identity_id"})
    forged_identity = forged_identity.model_copy(
        update={
            "adapter_identity_id": "aid_" + sha256_hex(canonical_json_bytes(identity_unsigned))[:24]
        }
    )
    forged = first_proof.model_copy(update={"adapter_identity": forged_identity})
    proof_unsigned = forged.model_dump(mode="json", exclude={"proof_id"})
    forged = forged.model_copy(
        update={"proof_id": "aproof_" + sha256_hex(canonical_json_bytes(proof_unsigned))[:24]}
    )
    association = module.build_association_claim(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        association_basis="shared_export_id",
        proof_references=(forged, second_proof),
    )

    with pytest.raises(ValueError, match="source fingerprint does not match"):
        module.build_event_grouping_result(
            opaque_case_id=OPAQUE_CASE_ID,
            decision_time_unix_ns=DECISION_TIME,
            observations=(first, second),
            association_claims=(association,),
            family_mapping=family_mapping(module),
        )


def test_adapter_source_field_rejects_caller_supplied_final_digest() -> None:
    module = grouping_module()

    with pytest.raises(ValueError):
        module.AdapterAssociationSourceField(
            field_kind="trace_id",
            normalized_value_sha256="9" * 64,
        )


def test_source_field_hash_uses_canonical_utf8_after_nfc_precondition() -> None:
    module = grouping_module()
    item = observation(
        suffix="1",
        modality="traces",
        source_family="span_failure",
        start=10,
        end=20,
        association_basis="deterministic_association_key",
        association_key="log:record-cafe",
    )
    raw_value = "log:record-cafe"

    source_record = module.build_canonical_association_source_record(
        opaque_case_id=item.opaque_case_id,
        observation_id=item.observation_id,
        decision_time_unix_ns=item.decision_time_unix_ns,
        parent_source_fingerprint=item.source_fingerprint,
        adapter_semantics_sha256=item.adapter_semantics_sha256,
        association_basis="deterministic_association_key",
        source_fields=(
            module.AdapterAssociationSourceField(
                field_kind="log_record_id",
                raw_value=raw_value,
            ),
        ),
    )

    canonical_value = unicodedata.normalize("NFC", raw_value)

    assert canonical_value == raw_value  # The accepted opaque-ID grammar is ASCII-only.
    assert source_record.source_fields[0].raw_value == raw_value
    assert source_record.source_field_hashes[0].normalized_value_sha256 == sha256_hex(
        canonical_value.encode("utf-8")
    )


def test_unanchored_observation_can_only_remain_a_singleton() -> None:
    module = grouping_module()
    anchored = observation(
        suffix="1",
        modality="metrics",
        source_family="cpu_usage",
        start=10,
        end=20,
    )
    unanchored = anchored.model_copy(update={"association_source_record_sha256": None})
    source_record = claim(module, anchored).proof_references[0].source_record

    with pytest.raises(ValueError, match="requires an observation source record anchor"):
        module.build_adapter_association_identity(
            observation=unanchored,
            source_record=source_record,
        )

    result = module.build_event_grouping_result(
        opaque_case_id=OPAQUE_CASE_ID,
        decision_time_unix_ns=DECISION_TIME,
        observations=(unanchored,),
        association_claims=(),
        family_mapping=family_mapping(module),
    )
    assert result.event_groups[0].association_basis == "singleton_no_association_evidence"


def test_adapter_source_field_rejects_prehash_disguised_as_raw_trace_id() -> None:
    module = grouping_module()
    filename_digest = sha256_hex(b"nearby-cpu.log")

    with pytest.raises(ValueError, match="raw association identifiers cannot be pre-hashed"):
        module.AdapterAssociationSourceField(
            field_kind="trace_id",
            raw_value=filename_digest,
        )


@pytest.mark.parametrize(
    ("field_kind", "prefix"),
    (
        ("export_id", "export"),
        ("source_relation_id", "relation"),
        ("log_record_id", "log"),
        ("metric_exemplar_id", "exemplar"),
        ("counter_series_id", "series"),
    ),
)
def test_prefixed_source_field_rejects_sha256_payload(
    field_kind: str,
    prefix: str,
) -> None:
    module = grouping_module()
    filename_digest = sha256_hex(b"nearby-cpu.log")

    with pytest.raises(ValueError, match="raw association identifiers cannot be pre-hashed"):
        module.AdapterAssociationSourceField(
            field_kind=field_kind,
            raw_value=f"{prefix}:{filename_digest}",
        )


def test_identity_rejects_source_record_hash_not_anchored_by_observation() -> None:
    module = grouping_module()
    record_builder = getattr(module, "build_canonical_association_source_record", None)
    assert record_builder is not None, "canonical association source record builder is missing"
    item = observation(
        suffix="1",
        modality="traces",
        source_family="span_failure",
        start=10,
        end=20,
    ).model_copy(update={"association_source_record_sha256": "9" * 64})
    source_record = record_builder(
        opaque_case_id=item.opaque_case_id,
        observation_id=item.observation_id,
        decision_time_unix_ns=item.decision_time_unix_ns,
        parent_source_fingerprint=item.source_fingerprint,
        adapter_semantics_sha256=item.adapter_semantics_sha256,
        association_basis="deterministic_association_key",
        source_fields=(
            module.AdapterAssociationSourceField(
                field_kind="log_record_id",
                raw_value="log:record-42",
            ),
        ),
    )

    with pytest.raises(ValueError, match="source record hash does not match observation"):
        module.build_adapter_association_identity(
            observation=item,
            source_record=source_record,
        )


def test_reviewer_filename_disguised_as_trace_id_is_rejected_before_record_anchor() -> None:
    module = grouping_module()

    with pytest.raises(ValueError, match="trace_id.*canonical 32 lowercase hex"):
        module.AdapterAssociationSourceField(
            field_kind="trace_id",
            raw_value="nearby-cpu.log",
        )

    anchored_observations = (
        observation(
            suffix="1",
            modality="traces",
            source_family="span_failure",
            start=10,
            end=20,
        ),
        observation(
            suffix="2",
            modality="logs",
            source_family="span_failure",
            start=10,
            end=20,
        ),
    )
    valid = module.AdapterAssociationSourceField(
        field_kind="trace_id",
        raw_value="4bf92f3577b34da6a3ce929d0e0e4736",
    )
    disguised = valid.model_copy(update={"raw_value": "nearby-cpu.log"})
    for item in anchored_observations:
        with pytest.raises(ValueError, match="trace_id.*canonical 32 lowercase hex"):
            module.build_canonical_association_source_record(
                opaque_case_id=item.opaque_case_id,
                observation_id=item.observation_id,
                decision_time_unix_ns=item.decision_time_unix_ns,
                parent_source_fingerprint=item.source_fingerprint,
                adapter_semantics_sha256=item.adapter_semantics_sha256,
                association_basis="deterministic_association_key",
                source_fields=(disguised,),
            )


@pytest.mark.parametrize(
    "filename_or_path",
    (
        "nearby-cpu.log",
        "logs/nearby-cpu",
        r"logs\nearby-cpu",
    ),
)
def test_filename_or_path_disguised_as_log_record_id_is_rejected(
    filename_or_path: str,
) -> None:
    module = grouping_module()

    with pytest.raises(ValueError, match="log_record_id.*canonical opaque ID"):
        module.AdapterAssociationSourceField(
            field_kind="log_record_id",
            raw_value=filename_or_path,
        )


@pytest.mark.parametrize(
    "source_fields",
    (
        (("trace_id", "4bf92f3577b34da6a3ce929d0e0e4736"),),
        (("span_id", "00f067aa0ba902b7"),),
        (
            ("trace_id", "4bf92f3577b34da6a3ce929d0e0e4736"),
            ("log_record_id", "log:record-42"),
        ),
    ),
)
def test_deterministic_association_key_rejects_noncanonical_field_combinations(
    source_fields: tuple[tuple[str, str], ...],
) -> None:
    module = grouping_module()
    fields = tuple(
        module.AdapterAssociationSourceField(field_kind=field_kind, raw_value=raw_value)
        for field_kind, raw_value in source_fields
    )

    with pytest.raises(ValueError, match="source field combination"):
        module.build_canonical_association_source_record(
            opaque_case_id=OPAQUE_CASE_ID,
            observation_id="obs_" + "1" * 24,
            decision_time_unix_ns=DECISION_TIME,
            parent_source_fingerprint="1" * 64,
            adapter_semantics_sha256="f" * 64,
            association_basis="deterministic_association_key",
            source_fields=fields,
        )


@pytest.mark.parametrize(
    ("field_kind", "raw_value", "error"),
    (
        ("trace_id", "0" * 32, "cannot be all zero"),
        ("trace_id", "4BF92F3577B34DA6A3CE929D0E0E4736", "lowercase hex"),
        ("span_id", "0" * 16, "cannot be all zero"),
        ("span_id", "00f067aa0ba902b", "16 lowercase hex"),
        ("export_id", "export:file.name", "canonical opaque ID"),
        ("source_relation_id", "relation:two words", "canonical opaque ID"),
        ("log_record_id", "log:r\u00e9cord", "canonical opaque ID"),
        ("metric_exemplar_id", "metric:record-42", "canonical opaque ID"),
        ("counter_series_id", "series:/tmp/record", "canonical opaque ID"),
    ),
)
def test_each_typed_parser_rejects_noncanonical_identifiers(
    field_kind: str,
    raw_value: str,
    error: str,
) -> None:
    module = grouping_module()

    with pytest.raises(ValueError, match=error):
        module.AdapterAssociationSourceField(
            field_kind=field_kind,
            raw_value=raw_value,
        )


@pytest.mark.parametrize(
    ("field_kind", "raw_value"),
    (
        ("export_id", "export:batch-42"),
        ("source_relation_id", "relation:parent_42"),
        ("trace_id", "4bf92f3577b34da6a3ce929d0e0e4736"),
        ("span_id", "00f067aa0ba902b7"),
        ("log_record_id", "log:record-42"),
        ("metric_exemplar_id", "exemplar:record-42"),
        ("counter_series_id", "series:record-42"),
    ),
)
def test_each_typed_parser_accepts_its_canonical_identifier(
    field_kind: str,
    raw_value: str,
) -> None:
    module = grouping_module()

    field = module.AdapterAssociationSourceField(
        field_kind=field_kind,
        raw_value=raw_value,
    )
    assert field.raw_value == raw_value


def test_frozen_policy_retains_executed_parser_specs_and_exact_combinations() -> None:
    module = grouping_module()
    policy = module.build_frozen_association_policy()
    deterministic = policy.rule_for("deterministic_association_key")
    parser_specs = tuple(
        parser_spec for rule in policy.rules for parser_spec in rule.field_parser_specs
    )

    assert deterministic.allowed_source_field_combinations == (
        ("trace_id", "span_id"),
        ("log_record_id",),
        ("metric_exemplar_id",),
        ("counter_series_id",),
    )
    assert tuple(item.field_kind for item in deterministic.field_parser_specs) == (
        "trace_id",
        "span_id",
        "log_record_id",
        "metric_exemplar_id",
        "counter_series_id",
    )
    assert all(
        item.parser_semantics_version == "nurse-v2-association-field-parser/2"
        for item in parser_specs
    )
    assert {item.field_kind: item.reject_sha256_payload for item in parser_specs} == {
        "export_id": True,
        "source_relation_id": True,
        "trace_id": False,
        "span_id": False,
        "log_record_id": True,
        "metric_exemplar_id": True,
        "counter_series_id": True,
    }
    assert all(item.verify() for item in deterministic.field_parser_specs)
    assert policy.verify()


def test_frozen_association_builders_construct_one_identity_per_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = grouping_module()
    parser_builder = module._build_field_parser_spec
    rule_builder = module._build_association_derivation_rule
    policy_builder = module.build_frozen_association_policy
    for builder in (parser_builder, rule_builder, policy_builder):
        cache_clear = getattr(builder, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    parser_results: list[Any] = []
    rule_results: list[Any] = []

    def counting_parser(field_kind: str) -> Any:
        result = parser_builder(field_kind)
        parser_results.append(result)
        return result

    def counting_rule(basis: str) -> Any:
        result = rule_builder(basis)
        rule_results.append(result)
        return result

    monkeypatch.setattr(module, "_build_field_parser_spec", counting_parser)
    monkeypatch.setattr(module, "_build_association_derivation_rule", counting_rule)

    first = policy_builder()
    second = policy_builder()

    assert len({id(item) for item in parser_results}) == 7
    assert len({id(item) for item in rule_results}) == 3
    assert first is second
    assert parser_builder("trace_id") is parser_builder("trace_id")
    assert rule_builder("deterministic_association_key") is rule_builder(
        "deterministic_association_key"
    )
    with pytest.raises(ValidationError, match="frozen"):
        first.policy_sha256 = "0" * 64


def test_repeated_proof_and_result_verification_reuses_cached_frozen_builders() -> None:
    module = grouping_module()
    builders = (
        module._build_field_parser_spec,
        module._build_association_derivation_rule,
        module.build_frozen_association_policy,
    )
    for builder in builders:
        builder.cache_clear()
    result = built_result(module)
    proofs = tuple(
        proof for association in result.association_claims for proof in association.proof_references
    )
    misses_before = tuple(builder.cache_info().misses for builder in builders)
    hits_before = sum(builder.cache_info().hits for builder in builders)

    for _ in range(3):
        assert all(proof.verify() for proof in proofs)
        assert result.verify()

    assert misses_before == (7, 3, 1)
    assert tuple(builder.cache_info().misses for builder in builders) == misses_before
    assert sum(builder.cache_info().hits for builder in builders) > hits_before

    cached_policy = module.build_frozen_association_policy()
    tampered_policy = cached_policy.model_copy(update={"policy_sha256": "0" * 64})
    assert tampered_policy is not cached_policy
    assert tampered_policy.verify() is False
    assert module.build_frozen_association_policy() is cached_policy


def test_trace_and_span_pair_is_a_valid_deterministic_identity_record() -> None:
    module = grouping_module()
    record = module.build_canonical_association_source_record(
        opaque_case_id=OPAQUE_CASE_ID,
        observation_id="obs_" + "1" * 24,
        decision_time_unix_ns=DECISION_TIME,
        parent_source_fingerprint="1" * 64,
        adapter_semantics_sha256="f" * 64,
        association_basis="deterministic_association_key",
        source_fields=(
            module.AdapterAssociationSourceField(
                field_kind="trace_id",
                raw_value="4bf92f3577b34da6a3ce929d0e0e4736",
            ),
            module.AdapterAssociationSourceField(
                field_kind="span_id",
                raw_value="00f067aa0ba902b7",
            ),
        ),
    )

    assert record.verify()
    assert tuple(item.field_kind for item in record.source_field_hashes) == (
        "trace_id",
        "span_id",
    )
    assert all(item.parser_semantics_sha256 for item in record.source_field_hashes)
