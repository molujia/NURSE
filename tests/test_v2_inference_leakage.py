from __future__ import annotations

import hashlib
import importlib
import json
import uuid
from types import ModuleType
from typing import Any

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.event_grouping import (
    EVENT_GROUP_SCHEMA_VERSION,
    EventGroup,
    MemberSignalFamilyAssignment,
)
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    EntityBinding,
    EventGroupV1,
    Observation,
    ObservationV1,
    TimeInterval,
    WindowSummary,
    build_candidate_universe,
    build_dataset_manifest,
)
from nurse_rcl.v2.opaque_ids import (
    EvaluatorIdentitySource,
    InferenceIdentityManifest,
    InferenceIdentityRecord,
    build_identity_artifacts,
)
from nurse_rcl.v2.temporal_entity_registry import (
    build_entity_metadata_record,
    build_temporal_entity_registry,
)

ORIGINAL_CASE_ID = "case-private-enumerable-17"
PRIVATE_ANSWER = "payment-private-root"
PRIVATE_PATH = "C:/private/cases/case-private-enumerable-17/injection.json"
FIXED_UUID = uuid.UUID("12345678-1234-4234-8234-1234567890ab")


def leakage_guard() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.inference_leakage")
    except ModuleNotFoundError:
        pytest.fail("V2 inference-boundary leakage guard is not implemented", pytrace=False)


def identity_artifacts(
    *,
    method: str = "random_uuid_v4",
    answers: tuple[str, ...] = (PRIVATE_ANSWER,),
) -> tuple[Any, Any]:
    kwargs: dict[str, Any]
    if method == "random_uuid_v4":
        kwargs = {"uuid_factory": lambda: FIXED_UUID}
    else:
        kwargs = {"hmac_secret": b"h" * 32}
    return build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id=ORIGINAL_CASE_ID,
                incident_group_id="grp_" + "a" * 24,
                answer_services=answers,
            ),
        ),
        method=method,
        **kwargs,
    )


def typed_observation(
    opaque_case_id: str,
    *,
    quality_flags: tuple[str, ...] = (),
) -> Observation:
    summary = WindowSummary(
        record_state="observed",
        valid_point_count=1,
        coverage_fraction=1.0,
        median=1.0,
        mad=0.0,
        minimum=1.0,
        maximum=1.0,
        event_count=None,
    )
    return Observation(
        observation_id="obs_" + "e" * 24,
        opaque_case_id=opaque_case_id,
        entity_binding_id=None,
        modality="logs",
        signal_family="application_error",
        signal_type="sparse",
        unit="events",
        interval=TimeInterval(start_unix_ns=1, end_unix_ns=2),
        decision_time_unix_ns=2,
        aggregation="event_count",
        source_fingerprint="f" * 64,
        baseline_summary=summary,
        fault_summary=summary,
        evidence_state="verified_normal",
        anomaly_strength=0.0,
        coverage_quality=1.0,
        minimum_valid_points=1,
        transform="event_rate",
        quality_flags=quality_flags,
        provenance_ids=("telemetry:processed",),
        formal_scoring_eligible=True,
        adapter_semantics_sha256="1" * 64,
        association_source_record_sha256=None,
    )


def public_payload(opaque_case_id: str) -> dict[str, object]:
    return {
        "dataset": "rcabench",
        "opaque_case_id": opaque_case_id,
        "observations": [typed_observation(opaque_case_id)],
    }


def assert_known_secrets_absent(value: object, secrets: tuple[str, ...]) -> None:
    """Evaluator-side test oracle; production inference code must never receive it."""

    def serialize_model(item: object) -> object:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        raise TypeError

    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=serialize_model,
    )
    for secret in secrets:
        derivatives = (
            secret,
            hashlib.sha256(secret.encode()).hexdigest(),
            "opq_" + hashlib.sha256(secret.encode()).hexdigest()[:32],
        )
        assert all(derivative not in serialized for derivative in derivatives)


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "answer",
        "Ground.Truth",
        "target-services",
        "root_CAUSE",
        "fault-type",
        "Fault.Metadata",
        "INJECTION_type",
        "injection-metadata",
        "Original.Case-ID",
        "case_name",
        "source-path",
        "ORIGINAL path",
    ),
)
def test_recursive_guard_rejects_forbidden_field_provenance_without_secret_echo(
    forbidden_field: str,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    payload = public_payload(manifest.records[0].opaque_case_id)
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations.append(
        typed_observation(manifest.records[0].opaque_case_id).model_copy(
            update={forbidden_field: PRIVATE_ANSWER}
        )
    )

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    message = str(caught.value)
    assert "forbidden_field_provenance" in message
    assert PRIVATE_ANSWER not in message
    assert ORIGINAL_CASE_ID not in message
    assert PRIVATE_PATH not in message


@pytest.mark.parametrize(
    "nested_value",
    (
        {"wrapper": {"answer": PRIVATE_ANSWER}},
        {"wrapper": [{"ground-truth": PRIVATE_ANSWER}]},
        {"wrapper": {"injection": PRIVATE_ANSWER}},
        {"wrapper": [{"original_name": PRIVATE_ANSWER}]},
        {"wrapper": {"Injected.Fault": PRIVATE_ANSWER}},
        {"wrapper": [{"fault_name": PRIVATE_ANSWER}]},
    ),
)
def test_nested_untyped_mapping_keys_preserve_forbidden_provenance_classification(
    nested_value: object,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    payload = public_payload(manifest.records[0].opaque_case_id)
    payload["observations"] = [nested_value]

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="forbidden_field_provenance",
    ) as caught:
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert PRIVATE_ANSWER not in str(caught.value)


def test_private_evaluator_models_are_rejected_even_when_nested_in_lists() -> None:
    module = leakage_guard()
    evaluator_map, manifest = identity_artifacts()

    for artifact in (
        evaluator_map,
        {"observations": [[evaluator_map.records[0]]]},
    ):
        with pytest.raises(
            module.InferenceBoundaryLeakageError,
            match="private_evaluator_artifact",
        ):
            module.audit_inference_boundary(
                artifact,
                identity_manifest=manifest,
                expected_identity_manifest_sha256=manifest.manifest_sha256,
            )


def test_python_model_copy_cannot_bypass_recursive_field_allowlist() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    forged = manifest.records[0].model_copy(update={"Ground.Truth": PRIVATE_ANSWER})

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            forged,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert "forbidden_field_provenance" in str(caught.value)
    assert PRIVATE_ANSWER not in str(caught.value)


def test_non_string_model_copy_field_key_is_safely_rejected() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    forged = typed_observation(manifest.records[0].opaque_case_id).model_copy(
        update={1: "private-value"}  # type: ignore[dict-item]
    )

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            forged,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert "private-value" not in str(caught.value)


@pytest.mark.parametrize(
    ("misplaced_field", "private_value"),
    (
        ("source_file", PRIVATE_PATH),
        ("service_entity_key", PRIVATE_ANSWER),
    ),
)
def test_nested_untyped_mapping_cannot_reuse_a_model_field_out_of_context(
    misplaced_field: str,
    private_value: str,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    payload = public_payload(manifest.records[0].opaque_case_id)
    payload["observations"] = [{misplaced_field: private_value}]

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert "invalid_public_artifact" in str(caught.value)
    assert private_value not in str(caught.value)


def test_cyclic_untyped_artifact_is_rejected_deterministically() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    recursive: list[object] = []
    recursive.append(recursive)
    payload = {
        "opaque_case_id": manifest.records[0].opaque_case_id,
        "observations": recursive,
    }

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="cyclic_artifact",
    ):
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("artifact", ("raw-telemetry", ["raw-telemetry"]))
def test_untyped_root_values_are_not_public_artifacts(artifact: object) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_public_artifact",
    ):
        module.audit_inference_boundary(
            artifact,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_opaque_id_must_belong_to_trusted_manifest_not_self_declared_method() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    enumerable_id = "opq_" + hashlib.sha256(ORIGINAL_CASE_ID.encode()).hexdigest()[:32]
    payload = importlib.import_module("nurse_rcl.v2.opaque_ids").InferenceIdentityRecord(
        dataset="rcabench",
        opaque_case_id=enumerable_id,
        method="random_uuid_v4",
    )

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="untrusted_opaque_identity",
    ) as caught:
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert ORIGINAL_CASE_ID not in str(caught.value)
    with pytest.raises(AssertionError):
        assert_known_secrets_absent(payload, (ORIGINAL_CASE_ID,))


def test_opaque_id_cannot_be_reused_under_another_manifest_dataset() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    payload = public_payload(manifest.records[0].opaque_case_id)
    payload["dataset"] = "aiops25"

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="untrusted_opaque_identity",
    ):
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("dataset", (None, 7))
def test_root_envelope_requires_a_string_dataset(dataset: object) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    payload = public_payload(manifest.records[0].opaque_case_id)
    if dataset is None:
        payload.pop("dataset")
    else:
        payload["dataset"] = dataset

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_public_artifact",
    ):
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_identity_manifest_is_trust_input_not_an_audited_artifact() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_public_artifact",
    ):
        module.audit_inference_boundary(
            manifest,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def caller_claimed_identity_manifest(manifest: Any) -> Any:
    records = tuple(
        record.model_copy(update={"method": "caller_claimed"}) for record in manifest.records
    )
    unsigned_manifest = manifest.model_copy(update={"method": "caller_claimed", "records": records})
    unsigned = unsigned_manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    return unsigned_manifest.model_copy(
        update={"manifest_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def test_rehashed_caller_claimed_identity_method_is_not_a_trust_root() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method="hmac_sha256_128")
    forged = caller_claimed_identity_manifest(manifest)
    assert forged.verify()

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_identity_manifest",
    ):
        module.audit_inference_boundary(
            public_payload(forged.records[0].opaque_case_id),
            identity_manifest=forged,
            expected_identity_manifest_sha256=forged.manifest_sha256,
        )


def test_identity_manifest_must_match_protected_pinned_digest() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method="hmac_sha256_128")
    forged = caller_claimed_identity_manifest(manifest)
    assert forged.verify()

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_identity_manifest",
    ):
        module.audit_inference_boundary(
            public_payload(forged.records[0].opaque_case_id),
            identity_manifest=forged,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("malformed_digest", (object(), b"not-a-digest", None))
def test_malformed_identity_manifest_digest_is_safely_rejected(
    malformed_digest: object,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method="hmac_sha256_128")
    forged = manifest.model_copy(update={"manifest_sha256": malformed_digest})

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            public_payload(manifest.records[0].opaque_case_id),
            identity_manifest=forged,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert caught.value.code == "invalid_identity_manifest"
    assert "not-a-digest" not in str(caught.value)


def test_identity_manifest_missing_digest_is_safely_rejected() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method="hmac_sha256_128")
    fields = manifest.model_dump(mode="python")
    fields.pop("manifest_sha256")
    incomplete = type(manifest).model_construct(**fields)

    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            public_payload(manifest.records[0].opaque_case_id),
            identity_manifest=incomplete,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert caught.value.code == "invalid_identity_manifest"


@pytest.mark.parametrize(
    "artifact",
    (
        TimeInterval(start_unix_ns=1, end_unix_ns=2),
        CandidateEntry(
            service_entity_key="checkout",
            binding_ids=(),
            observation_ids=(),
            event_group_ids=(),
            evidence_state="zero_support",
        ),
    ),
)
def test_audit_requires_exactly_one_opaque_case_identity(artifact: object) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_public_artifact",
    ):
        module.audit_inference_boundary(
            artifact,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_dataset_manifest_is_not_a_single_case_inference_artifact() -> None:
    module = leakage_guard()
    _, identity_manifest = identity_artifacts()
    dataset_manifest = build_dataset_manifest(
        dataset="rcabench",
        adapter_version="test/1",
        target_granularity="service",
        source_inventory_sha256="a" * 64,
        adapter_configuration_sha256="b" * 64,
        cases=(),
    )

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="unapproved_model_type",
    ):
        module.audit_inference_boundary(
            dataset_manifest,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
        )


def test_envelope_scope_rejects_another_trusted_case_and_dataset() -> None:
    module = leakage_guard()
    _, manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="case-a",
                incident_group_id="grp_" + "a" * 24,
                answer_services=("root-a",),
            ),
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="case-b",
                incident_group_id="grp_" + "b" * 24,
                answer_services=("root-b",),
            ),
            EvaluatorIdentitySource(
                dataset="aiops25",
                original_case_id="case-c",
                incident_group_id="grp_" + "c" * 24,
                answer_services=("root-c",),
            ),
        ),
        method="hmac_sha256_128",
        hmac_secret=b"s" * 32,
    )
    by_dataset = {
        dataset: [record.opaque_case_id for record in manifest.records if record.dataset == dataset]
        for dataset in ("rcabench", "aiops25")
    }
    root_case_id = by_dataset["rcabench"][0]

    for nested_case_id in (
        by_dataset["rcabench"][1],
        by_dataset["aiops25"][0],
    ):
        payload = {
            "dataset": "rcabench",
            "opaque_case_id": root_case_id,
            "observations": [typed_observation(nested_case_id)],
        }
        with pytest.raises(
            module.InferenceBoundaryLeakageError,
            match="untrusted_opaque_identity",
        ):
            module.audit_inference_boundary(
                payload,
                identity_manifest=manifest,
                expected_identity_manifest_sha256=manifest.manifest_sha256,
            )


def current_event_group(opaque_case_id: str) -> EventGroup:
    member = MemberSignalFamilyAssignment(
        observation_id="obs_" + "e" * 24,
        adapter_semantics_sha256="1" * 64,
        source_signal_family="application_error",
        primary_signal_family="unclassified",
        assignment_basis="missing_mapping_unknown_safe",
        mapping_entry_id=None,
        provenance_ids=(),
    )
    unsigned: dict[str, object] = {
        "schema_version": EVENT_GROUP_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "association_basis": "singleton_no_association_evidence",
        "association_key_sha256": "2" * 64,
        "association_claim_ids": (),
        "observation_ids": (member.observation_id,),
        "entity_binding_ids": (),
        "primary_signal_family": "unclassified",
        "family_assignment_status": "degraded_unclassified",
        "family_mapping_version": "test/1",
        "family_mapping_sha256": "3" * 64,
        "member_family_assignments": (member.model_dump(mode="json"),),
        "interval": {"start_unix_ns": 1, "end_unix_ns": 2},
        "source_fingerprints": ("4" * 64,),
        "provenance_ids": ("telemetry:processed",),
    }
    return EventGroup.model_validate(
        {
            **unsigned,
            "event_group_id": "evt_" + sha256_hex(canonical_json_bytes(unsigned))[:24],
        }
    )


def test_current_event_group_is_accepted_but_legacy_public_artifacts_are_rejected() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    current = current_event_group(opaque_case_id)
    observation_document = typed_observation(opaque_case_id).model_dump(mode="json")
    observation_document.pop("association_source_record_sha256")
    observation_document["schema_version"] = "nurse.v2.observation/1"
    legacy_observation = ObservationV1.model_validate(observation_document)
    legacy_event_group = EventGroupV1(
        event_group_id="evt_" + "5" * 24,
        opaque_case_id=opaque_case_id,
        association_basis="deterministic_association_key",
        association_key_sha256="6" * 64,
        observation_ids=("obs_" + "e" * 24,),
        entity_binding_ids=(),
        signal_family="application_error",
        interval=TimeInterval(start_unix_ns=1, end_unix_ns=2),
        provenance_ids=("telemetry:processed",),
    )

    accepted = module.audit_inference_boundary(
        current,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert accepted.public_model_types == (
        "EventGroup",
        "MemberSignalFamilyAssignment",
        "TimeInterval",
    )
    for legacy in (legacy_observation, legacy_event_group):
        with pytest.raises(
            module.InferenceBoundaryLeakageError,
            match="unapproved_model_type",
        ):
            module.audit_inference_boundary(
                legacy,
                identity_manifest=manifest,
                expected_identity_manifest_sha256=manifest.manifest_sha256,
            )


@pytest.mark.parametrize("method", ("random_uuid_v4", "hmac_sha256_128"))
def test_trusted_uuid_and_hmac_identities_pass_without_scanning_telemetry_text(
    method: str,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method=method)
    payload = public_payload(manifest.records[0].opaque_case_id)
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations[0] = typed_observation(
        manifest.records[0].opaque_case_id,
        quality_flags=(
            "application log literally mentions answer, injection metadata, "
            "target_services, and /var/lib/work without carrying field provenance",
        ),
    )

    audit = module.audit_inference_boundary(
        payload,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert audit.opaque_case_ids == (manifest.records[0].opaque_case_id,)
    assert audit.artifact_count == 1
    assert len(audit.boundary_sha256) == 64
    assert_known_secrets_absent((manifest.model_dump(mode="json"), payload), (ORIGINAL_CASE_ID,))


def test_audit_receipt_is_bound_to_the_trusted_identity_manifest() -> None:
    module = leakage_guard()
    _, first_manifest = identity_artifacts(method="hmac_sha256_128")
    _, expanded_manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id=ORIGINAL_CASE_ID,
                incident_group_id="grp_" + "a" * 24,
                answer_services=(PRIVATE_ANSWER,),
            ),
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="case-second-private",
                incident_group_id="grp_" + "b" * 24,
                answer_services=("second-private-root",),
            ),
        ),
        method="hmac_sha256_128",
        hmac_secret=b"h" * 32,
    )
    opaque_case_id = first_manifest.records[0].opaque_case_id
    assert opaque_case_id in {record.opaque_case_id for record in expanded_manifest.records}
    payload = {
        "dataset": "rcabench",
        "opaque_case_id": opaque_case_id,
        "observations": [],
    }

    first = module.audit_inference_boundary(
        payload,
        identity_manifest=first_manifest,
        expected_identity_manifest_sha256=first_manifest.manifest_sha256,
    )
    expanded = module.audit_inference_boundary(
        payload,
        identity_manifest=expanded_manifest,
        expected_identity_manifest_sha256=expanded_manifest.manifest_sha256,
    )

    assert first.identity_manifest_sha256 == first_manifest.manifest_sha256
    assert expanded.identity_manifest_sha256 == expanded_manifest.manifest_sha256
    assert first.boundary_sha256 != expanded.boundary_sha256
    assert first.verify()
    assert not first.model_copy(update={"artifact_count": 2}).verify()
    assert len(first.audit_sha256) == 64


def test_rehashed_audit_receipt_cannot_bypass_current_schema() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts(method="hmac_sha256_128")
    audit = module.audit_inference_boundary(
        public_payload(manifest.records[0].opaque_case_id),
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    for update in (
        {"artifact_count": 0},
        {"schema_version": "evil"},
        {"boundary_sha256": "invalid"},
    ):
        tampered = audit.model_copy(update=update)
        unsigned = tampered.model_dump(mode="json", exclude={"audit_sha256"})
        rehashed = tampered.model_copy(
            update={"audit_sha256": sha256_hex(canonical_json_bytes(unsigned))}
        )
        assert rehashed.verify() is False

    unserializable = audit.model_copy(
        update={"opaque_case_ids": object(), "audit_sha256": "0" * 64}
    )
    assert unserializable.verify() is False


def collector_registry(
    opaque_case_id: str,
    *,
    include_explicit: bool = False,
    explicit_service_key: str = "kubernetes-endpoints",
) -> Any:
    records = [
        build_entity_metadata_record(
            opaque_case_id=opaque_case_id,
            metadata_source="metrics",
            entity_kind="untyped",
            observed_entity_key="kubernetes-endpoints",
            relation="collector_source",
            service_entity_key=None,
            valid_interval=TimeInterval(start_unix_ns=1, end_unix_ns=10),
            source_fingerprint="a" * 64,
            provenance_ids=("metrics:collector",),
        )
    ]
    if include_explicit:
        records.append(
            build_entity_metadata_record(
                opaque_case_id=opaque_case_id,
                metadata_source="deployment",
                entity_kind="service",
                observed_entity_key=explicit_service_key,
                relation="explicit_service_binding",
                service_entity_key=explicit_service_key,
                valid_interval=TimeInterval(start_unix_ns=1, end_unix_ns=10),
                source_fingerprint="b" * 64,
                provenance_ids=("deployment:service",),
            )
        )
    return build_temporal_entity_registry(
        opaque_case_id=opaque_case_id,
        decision_time_unix_ns=10,
        metadata_records=records,
        observations=(),
    )


def collector_candidate(opaque_case_id: str) -> Any:
    return build_candidate_universe(
        opaque_case_id=opaque_case_id,
        observation_contract_sha256="c" * 64,
        candidates=(
            CandidateEntry(
                service_entity_key="kubernetes-endpoints",
                binding_ids=(),
                observation_ids=(),
                event_group_ids=(),
                evidence_state="zero_support",
            ),
        ),
    )


def collector_chosen_binding(opaque_case_id: str) -> EntityBinding:
    return EntityBinding(
        binding_id="bind_" + "d" * 24,
        opaque_case_id=opaque_case_id,
        observed_entity_key="kubernetes-endpoints",
        entity_kind="untyped",
        status="bound",
        service_entity_key="kubernetes-endpoints",
        conflict_service_keys=(),
        valid_interval=TimeInterval(start_unix_ns=1, end_unix_ns=10),
        source_fingerprints=("a" * 64,),
        provenance_ids=("metrics:collector",),
    )


@pytest.mark.parametrize(
    "non_exact_service_key",
    (
        "KUBERNETES-ENDPOINTS",
        " kubernetes-endpoints ",
        "\uff4bubernetes-endpoints",
    ),
)
@pytest.mark.parametrize("derived_artifact", (collector_candidate, collector_chosen_binding))
def test_explicit_service_proof_requires_exact_entity_key_match(
    non_exact_service_key: str,
    derived_artifact: Any,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    registry = collector_registry(
        opaque_case_id,
        include_explicit=True,
        explicit_service_key=non_exact_service_key,
    )

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="collector_only_service",
    ):
        module.audit_inference_boundary(
            (registry, derived_artifact(opaque_case_id)),
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("derived_artifact", (collector_candidate, collector_chosen_binding))
def test_candidate_or_bound_binding_requires_explicit_registry_proof(
    derived_artifact: Any,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="unproven_service",
    ):
        module.audit_inference_boundary(
            derived_artifact(opaque_case_id),
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_candidate_envelope_requires_same_case_entity_registry() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    payload = public_payload(opaque_case_id)
    payload["candidate_directory"] = collector_candidate(opaque_case_id)

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="invalid_public_artifact",
    ):
        module.audit_inference_boundary(
            payload,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_candidate_envelope_accepts_exact_explicit_same_case_service() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    payload = public_payload(opaque_case_id)
    payload["candidate_directory"] = collector_candidate(opaque_case_id)
    payload["entities"] = collector_registry(opaque_case_id, include_explicit=True)

    audit = module.audit_inference_boundary(
        payload,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert audit.opaque_case_ids == (opaque_case_id,)


def test_collector_authorization_cannot_flow_between_cases() -> None:
    module = leakage_guard()
    _, manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="case-explicit-service",
                incident_group_id="grp_" + "a" * 24,
                answer_services=("root-a",),
            ),
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="case-collector-only",
                incident_group_id="grp_" + "b" * 24,
                answer_services=("root-b",),
            ),
        ),
        method="hmac_sha256_128",
        hmac_secret=b"c" * 32,
    )
    explicit_case_id, collector_case_id = (record.opaque_case_id for record in manifest.records)

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="untrusted_opaque_identity",
    ):
        module.audit_inference_boundary(
            (
                collector_registry(explicit_case_id, include_explicit=True),
                collector_registry(collector_case_id),
                collector_candidate(collector_case_id),
            ),
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


@pytest.mark.parametrize("derived_artifact", (collector_candidate, collector_chosen_binding))
def test_collector_only_label_cannot_be_candidate_or_chosen_binding(
    derived_artifact: Any,
) -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    registry = collector_registry(opaque_case_id)

    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="collector_only_service",
    ):
        module.audit_inference_boundary(
            (registry, derived_artifact(opaque_case_id)),
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )


def test_collector_label_remains_auditable_and_explicit_deployment_allows_service() -> None:
    module = leakage_guard()
    _, manifest = identity_artifacts()
    opaque_case_id = manifest.records[0].opaque_case_id
    collector_only = collector_registry(opaque_case_id)

    collector_audit = module.audit_inference_boundary(
        collector_only,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )
    explicitly_deployed_audit = module.audit_inference_boundary(
        (
            collector_registry(opaque_case_id, include_explicit=True),
            collector_candidate(opaque_case_id),
        ),
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert collector_audit.artifact_count == 1
    assert explicitly_deployed_audit.artifact_count == 2


def test_answer_mutation_cannot_change_public_inference_projection() -> None:
    module = leakage_guard()
    evaluator_before, manifest_before = identity_artifacts(answers=("root-before",))
    evaluator_after, manifest_after = identity_artifacts(answers=("root-after",))

    assert evaluator_before != evaluator_after
    assert manifest_before == manifest_after
    before = module.audit_inference_boundary(
        public_payload(manifest_before.records[0].opaque_case_id),
        identity_manifest=manifest_before,
        expected_identity_manifest_sha256=manifest_before.manifest_sha256,
    )
    after = module.audit_inference_boundary(
        public_payload(manifest_after.records[0].opaque_case_id),
        identity_manifest=manifest_after,
        expected_identity_manifest_sha256=manifest_after.manifest_sha256,
    )

    assert before == after
    assert_known_secrets_absent(
        manifest_before,
        (ORIGINAL_CASE_ID, "root-before", "root-after", PRIVATE_PATH),
    )


def test_inference_boundary_audit_schema_is_registered() -> None:
    module = leakage_guard()
    compatibility = importlib.import_module("nurse_rcl.v2.schema_compatibility")

    assert compatibility.CURRENT_SCHEMA_VERSIONS["inference_boundary_audit"] == (
        module.INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION
    )


def report_identity_manifest(*opaque_case_ids: str) -> InferenceIdentityManifest:
    records = tuple(
        InferenceIdentityRecord(
            dataset="rcabench",
            opaque_case_id=opaque_case_id,
            method="hmac_sha256_128",
        )
        for opaque_case_id in sorted(opaque_case_ids)
    )
    unsigned: dict[str, object] = {
        "schema_version": "nurse.v2.inference-identity-manifest/1",
        "method": "hmac_sha256_128",
        "case_count": len(records),
        "rcabench_case_count": len(records),
        "aiops25_case_count": 0,
        "records": tuple(item.model_dump(mode="json") for item in records),
    }
    return InferenceIdentityManifest.model_validate(
        {**unsigned, "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def current_frozen_report_context() -> Any:
    from tests.test_v2_report_context import _frozen_context

    return _frozen_context()


def resign_report_context_payload(context: Any, payload: Any) -> Any:
    from nurse_rcl.v2.report_context import FrozenReportContext

    payload_json = canonical_json_bytes(payload.model_dump(mode="json")).decode("utf-8")
    unsigned = context.model_dump(mode="json", exclude={"context_sha256"})
    unsigned.update(
        {
            "payload_json": payload_json,
            "payload_sha256": payload.payload_sha256,
        }
    )
    return FrozenReportContext.model_validate(
        {**unsigned, "context_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def test_current_frozen_report_context_scans_its_typed_payload() -> None:
    module = leakage_guard()
    context = current_frozen_report_context()
    manifest = report_identity_manifest(context.opaque_case_id)

    audit = module.audit_inference_boundary(
        context,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert audit.opaque_case_ids == (context.opaque_case_id,)
    assert {
        "FrozenReportContext",
        "FrozenReportPayload",
        "ReportHypothesisFact",
        "ReportCompetitionDelta",
        "EventGroupingResult",
        "EventGroupBaseStrength",
        "CandidateEventRoleAdjustment",
        "EventStrengthResult",
        "FamilyAggregationResult",
        "GraphSnapshot",
        "RelationSCCCondensation",
        "OnsetInterval",
        "TemporalPrecedence",
        "RoleCompilationResult",
        "ReferenceRankingResult",
    } <= set(audit.public_model_types)


def test_resigned_report_payload_cannot_cross_the_single_case_scope() -> None:
    module = leakage_guard()
    context = current_frozen_report_context()
    payload = context.payload
    other_case_id = "opq_" + "b" * 32
    original_onset = payload.onsets[0]
    onset_unsigned = original_onset.model_dump(
        mode="json",
        exclude={"onset_id", "onset_sha256"},
    )
    onset_unsigned["opaque_case_id"] = other_case_id
    onset_digest = sha256_hex(canonical_json_bytes(onset_unsigned))
    crossed_onset = type(original_onset).model_validate(
        {
            **onset_unsigned,
            "onset_id": "onset_" + onset_digest[:24],
            "onset_sha256": onset_digest,
        }
    )
    crossed_candidate = payload.model_copy(
        update={
            "onsets": tuple(
                sorted(
                    (crossed_onset, *payload.onsets[1:]),
                    key=lambda item: item.onset_id,
                )
            )
        }
    )
    payload_unsigned = crossed_candidate.model_dump(
        mode="json",
        exclude={"payload_sha256"},
    )
    payload_digest = sha256_hex(canonical_json_bytes(payload_unsigned))
    crossed_payload = crossed_candidate.model_copy(update={"payload_sha256": payload_digest})
    crossed_context = resign_report_context_payload(context, crossed_payload)
    assert not crossed_payload.verify()
    assert not crossed_context.verify()
    for manifest in (
        report_identity_manifest(context.opaque_case_id),
        report_identity_manifest(context.opaque_case_id, other_case_id),
    ):
        with pytest.raises(
            module.InferenceBoundaryLeakageError,
            match="invalid_public_artifact",
        ):
            module.audit_inference_boundary(
                crossed_context,
                identity_manifest=manifest,
                expected_identity_manifest_sha256=manifest.manifest_sha256,
            )


@pytest.mark.parametrize(
    ("forbidden_field", "private_value"),
    (
        ("answer", PRIVATE_ANSWER),
        ("injection_provenance", PRIVATE_ANSWER),
        ("original_case_id", ORIGINAL_CASE_ID),
        ("source_path", PRIVATE_PATH),
    ),
)
def test_resigned_report_payload_forbidden_fields_never_cross_the_boundary(
    forbidden_field: str,
    private_value: str,
) -> None:
    from nurse_rcl.v2.report_context import FrozenReportContext

    module = leakage_guard()
    context = current_frozen_report_context()
    payload_unsigned = context.payload.model_dump(mode="json", exclude={"payload_sha256"})
    payload_unsigned[forbidden_field] = private_value
    payload_digest = sha256_hex(canonical_json_bytes(payload_unsigned))
    payload_document = {**payload_unsigned, "payload_sha256": payload_digest}
    unsigned = context.model_dump(mode="json", exclude={"context_sha256"})
    unsigned.update(
        {
            "payload_json": canonical_json_bytes(payload_document).decode("utf-8"),
            "payload_sha256": payload_digest,
        }
    )
    resigned = FrozenReportContext.model_validate(
        {**unsigned, "context_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    manifest = report_identity_manifest(context.opaque_case_id)

    assert not resigned.verify()
    with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
        module.audit_inference_boundary(
            resigned,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert private_value not in str(caught.value)


@pytest.mark.parametrize(
    "forbidden_provenance",
    (
        PRIVATE_PATH,
        ORIGINAL_CASE_ID,
        "answer:payment-private-root",
        "injection:fault-metadata",
    ),
)
def test_resigned_report_payload_rejects_forbidden_provenance_values(
    forbidden_provenance: str,
) -> None:
    module = leakage_guard()
    context = current_frozen_report_context()
    payload = context.payload
    onset = payload.onsets[0]
    onset_unsigned = onset.model_dump(
        mode="json",
        exclude={"onset_id", "onset_sha256"},
    )
    onset_unsigned["provenance_ids"] = tuple(sorted({*onset.provenance_ids, forbidden_provenance}))
    onset_digest = sha256_hex(canonical_json_bytes(onset_unsigned))
    poisoned_onset = type(onset).model_validate(
        {
            **onset_unsigned,
            "onset_id": "onset_" + onset_digest[:24],
            "onset_sha256": onset_digest,
        }
    )
    poisoned_candidate = payload.model_copy(
        update={
            "onsets": tuple(
                sorted(
                    (poisoned_onset, *payload.onsets[1:]),
                    key=lambda item: item.onset_id,
                )
            )
        }
    )
    payload_unsigned = poisoned_candidate.model_dump(
        mode="json",
        exclude={"payload_sha256"},
    )
    payload_digest = sha256_hex(canonical_json_bytes(payload_unsigned))
    poisoned_payload = poisoned_candidate.model_copy(update={"payload_sha256": payload_digest})
    poisoned_context = resign_report_context_payload(context, poisoned_payload)
    manifest = report_identity_manifest(context.opaque_case_id)

    assert poisoned_payload.verify()
    assert poisoned_context.verify()
    assert forbidden_provenance in poisoned_context.payload_json
    with pytest.raises(
        module.InferenceBoundaryLeakageError,
        match="forbidden_field_provenance",
    ) as caught:
        module.audit_inference_boundary(
            poisoned_context,
            identity_manifest=manifest,
            expected_identity_manifest_sha256=manifest.manifest_sha256,
        )

    assert forbidden_provenance not in str(caught.value)


def test_report_payload_and_replay_internals_are_not_standalone_public_roots() -> None:
    from tests.test_v2_case_audit import build_real_replay_bundle

    module = leakage_guard()
    context = current_frozen_report_context()
    payload = context.payload
    score_impact = next(
        result.score_impact
        for result in build_real_replay_bundle().evidence.role_guard_results
        if result.score_impact is not None
    )
    manifest = report_identity_manifest(context.opaque_case_id)
    report_root_only_artifacts = (
        payload,
        payload.hypotheses[0],
        payload.competition_deltas[0],
        payload.event_grouping,
        payload.event_strength,
        payload.event_strength.group_strengths[0],
        payload.event_strength.candidate_event_adjustments[0],
        payload.family_aggregation,
        payload.graph_snapshots[0],
        payload.condensations[0],
        payload.onsets[0],
        payload.precedences[0],
        payload.role_compilation,
        score_impact,
    )

    for artifact in report_root_only_artifacts:
        with pytest.raises(module.InferenceBoundaryLeakageError) as caught:
            module.audit_inference_boundary(
                artifact,
                identity_manifest=manifest,
                expected_identity_manifest_sha256=manifest.manifest_sha256,
            )
        assert caught.value.code in {"invalid_public_artifact", "unapproved_model_type"}


def test_legacy_v2_report_context_remains_a_compatible_public_artifact() -> None:
    from nurse_rcl.v2 import ranking_schemas
    from tests.test_v2_ranking_schemas import score

    module = leakage_guard()
    opaque_case_id = "opq_" + "2" * 32
    alpha = score(ranking_schemas, "alpha", 0.4)
    tie = ranking_schemas.build_tie_group(scores=(alpha,))
    top_k = ranking_schemas.build_top_k(
        opaque_case_id=opaque_case_id,
        candidate_universe_sha256="1" * 64,
        scores=(alpha,),
        tie_groups=(tie,),
        ordered_service_keys=("alpha",),
        k=1,
        tie_break_seed_sha256="c" * 64,
    )
    legacy_context = ranking_schemas.build_frozen_report_context(
        opaque_case_id=opaque_case_id,
        candidate_universe_sha256="1" * 64,
        focus_set_sha256="2" * 64,
        observation_set_sha256="3" * 64,
        entity_binding_set_sha256="4" * 64,
        event_group_set_sha256="5" * 64,
        hypothesis_set_sha256="6" * 64,
        graph_snapshot_sha256_by_window={"normal": "7" * 64, "fault": "8" * 64},
        scc_provenance_sha256="9" * 64,
        scores=(alpha,),
        tie_groups=(tie,),
        top_k=top_k,
        rule_set_sha256="a" * 64,
        history_context_sha256="b" * 64,
        frozen_at_utc="2026-09-05T09:00:00Z",
    )
    manifest = report_identity_manifest(opaque_case_id)

    audit = module.audit_inference_boundary(
        legacy_context,
        identity_manifest=manifest,
        expected_identity_manifest_sha256=manifest.manifest_sha256,
    )

    assert legacy_context.schema_version == "nurse.v2.frozen-report-context/2"
    assert audit.opaque_case_ids == (opaque_case_id,)
    assert "FrozenReportPayload" not in audit.public_model_types
