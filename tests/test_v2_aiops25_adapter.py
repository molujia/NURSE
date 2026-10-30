from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

OPAQUE_CASE_ID = "opq_" + "d" * 32
SOURCE_CASE_ID = "009be6db-313"
T0 = 1_750_097_405
NORMAL_FILES = (
    "normal_logs.parquet",
    "normal_metrics.parquet",
    "normal_traces.parquet",
)
FAULT_FILES = tuple(name.replace("normal_", "abnormal_") for name in NORMAL_FILES)
ALL_FILES = tuple(sorted((*NORMAL_FILES, *FAULT_FILES)))


def adapter() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.aiops25_adapter")


def inventory() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.aiops25_inventory")


def pyarrow_modules() -> tuple[ModuleType, ModuleType]:
    return importlib.import_module("pyarrow"), importlib.import_module(
        "pyarrow.parquet"
    )


def write_time_parquet(path: Path, timestamps: list[int]) -> None:
    pa, pq = pyarrow_modules()
    values = [datetime.fromtimestamp(value, tz=timezone.utc) for value in timestamps]
    pq.write_table(
        pa.table({"time": pa.array(values, type=pa.timestamp("ns", tz="UTC"))}),
        path,
    )


def write_prepared_case(
    root: Path,
    *,
    files: dict[str, list[int] | None] | None = None,
    marker_update: dict[str, object] | None = None,
    env_update: dict[str, object] | None = None,
) -> Path:
    case = root / SOURCE_CASE_ID
    case.mkdir(parents=True)
    selected = files
    if selected is None:
        selected = {
            **{name: [T0 - 1] for name in NORMAL_FILES},
            **{name: [T0] for name in FAULT_FILES},
        }
    row_counts = {name.removesuffix(".parquet"): 0 for name in ALL_FILES}
    row_counts.update(
        {
            name.removesuffix(".parquet"): len(timestamps or ())
            for name, timestamps in selected.items()
        }
    )
    marker: dict[str, object] = {
        "schema_version": 1,
        "case": SOURCE_CASE_ID,
        "rows": row_counts,
        "services": ["paymentservice", "frontend", "checkoutservice"],
        "metrics": ["request", "error_ratio"],
    }
    marker.update(marker_update or {})
    (case / ".prepared.json").write_text(json.dumps(marker), encoding="utf-8")
    env: dict[str, object] = {
        "NORMAL_START": T0 - 600,
        "NORMAL_END": T0,
        "ABNORMAL_START": T0,
        "ABNORMAL_END": T0 + 840,
        "ORIGINAL_ABNORMAL_END": T0 + 840,
    }
    env.update(env_update or {})
    (case / "env.json").write_text(json.dumps(env), encoding="utf-8")
    for name, timestamps in selected.items():
        if timestamps is None:
            (case / name).write_bytes(b"not-a-parquet-file")
        else:
            write_time_parquet(case / name, timestamps)
    (case / "injection.json").write_text("private-invalid-label-data", encoding="utf-8")
    return case


def authorized_queue(
    *,
    case_ids: tuple[str, ...] = (SOURCE_CASE_ID,),
    destination: str = "inference",
) -> object:
    module = inventory()
    unsigned = {
        "schema_version": module.AIOPS25_QUEUE_SCHEMA_VERSION,
        "dataset": "aiops25",
        "destination": destination,
        "inventory_sha256": "a" * 64,
        "case_ids": case_ids,
    }
    return module.AuthorizedAIOps25Queue.model_validate(
        {**unsigned, "artifact_hash": sha256_hex(canonical_json_bytes(unsigned))}
    )


def configuration(module: ModuleType) -> object:
    return module.build_aiops25_adapter_config(
        window_seconds=480,
        expected_cadence_seconds=30,
        clock_uncertainty_seconds=1,
    )


def test_adapter_uses_prepared_case_with_the_shared_decision_contract(
    tmp_path: Path,
) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path)
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )

    contract = result.time_contract
    assert contract.t0_unix_ns == T0 * 1_000_000_000
    assert contract.t0_source == "benchmark_fault_time"
    assert contract.localization_scope == "given_fault_time"
    assert contract.timezone == "UTC"
    assert contract.expected_cadence_ns == 30 * 1_000_000_000
    assert contract.clock_uncertainty_ns == 1_000_000_000
    assert contract.baseline_window.start_unix_ns == (T0 - 480) * 1_000_000_000
    assert contract.decision_time_unix_ns == (T0 + 480) * 1_000_000_000
    assert result.source_baseline_window.start_unix_ns == (T0 - 600) * 1_000_000_000
    assert result.source_fault_window.end_unix_ns == (T0 + 840) * 1_000_000_000
    assert result.target_granularity == "service"
    assert result.service_entity_keys == (
        "checkoutservice",
        "frontend",
        "paymentservice",
    )
    assert result.verify()


def test_adapter_requires_a_verified_inference_queue_membership(tmp_path: Path) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path)

    with pytest.raises(module.AIOps25AdapterScopeError, match="inference queue"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(destination="ark"),
            configuration=configuration(module),
        )
    with pytest.raises(module.AIOps25AdapterScopeError, match="not authorized"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(case_ids=("0410d710-226",)),
            configuration=configuration(module),
        )


def test_post_decision_and_partial_modalities_use_the_same_audit_semantics(
    tmp_path: Path,
) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        files={
            "normal_metrics.parquet": [T0 - 480, T0 - 1, T0],
            "abnormal_metrics.parquet": [T0, T0 + 479, T0 + 480],
        },
    )
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )
    audits = {item.source_file: item for item in result.files}

    assert audits["normal_metrics.parquet"].accepted_rows == 2
    assert audits["normal_metrics.parquet"].outside_contract_rows == 1
    assert audits["abnormal_metrics.parquet"].accepted_rows == 2
    assert audits["abnormal_metrics.parquet"].post_decision_rows == 1
    assert result.time_contract.late_data_observed is True
    assert result.time_contract.missing_modalities == (
        "deployment",
        "logs",
        "traces",
    )


def test_output_contains_no_source_case_id_path_or_label_payload(tmp_path: Path) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path)
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )
    serialized = result.model_dump_json().lower()

    assert SOURCE_CASE_ID not in serialized
    assert "injection" not in serialized
    assert "private-invalid-label-data" not in serialized
    assert str(case).lower().replace("\\", "/") not in serialized.replace("\\", "/")
    assert result.inventory_sha256 == "a" * 64


@pytest.mark.parametrize(
    "forbidden_service",
    (
        SOURCE_CASE_ID,
        f"svc-{SOURCE_CASE_ID}",
        f" {SOURCE_CASE_ID} ",
        "source/case",
        "source\\case",
    ),
)
def test_service_directory_cannot_reintroduce_source_identity_or_path(
    tmp_path: Path,
    forbidden_service: str,
) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        marker_update={"services": ["frontend", forbidden_service]},
    )

    with pytest.raises(module.AIOps25AdapterError, match="service directory"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(),
            configuration=configuration(module),
        )


def test_prepared_marker_rejects_fields_outside_the_unlabeled_allowlist(
    tmp_path: Path,
) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        marker_update={"injection": "must-not-affect-inference"},
    )

    with pytest.raises(module.AIOps25AdapterError, match="prepared marker"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(),
            configuration=configuration(module),
        )


def test_env_rejects_fields_outside_the_time_allowlist(tmp_path: Path) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        env_update={"CASE_ID": SOURCE_CASE_ID},
    )

    with pytest.raises(module.AIOps25AdapterError, match="env time contract"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(),
            configuration=configuration(module),
        )


def test_result_verification_recomputes_configuration_derived_time_fields(
    tmp_path: Path,
) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path)
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )
    tampered_contract = result.time_contract.model_copy(
        update={"expected_cadence_ns": 1}
    )
    unsigned = result.model_dump(mode="json", exclude={"result_sha256"})
    unsigned["time_contract"] = tampered_contract.model_dump(mode="json")
    tampered = result.model_copy(
        update={
            "time_contract": tampered_contract,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )

    assert tampered.verify() is False


def test_result_verification_rejects_rehashed_decision_time_tampering(
    tmp_path: Path,
) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path)
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )
    tampered_contract = result.time_contract.model_copy(
        update={"decision_time_unix_ns": result.time_contract.t0_unix_ns + 1}
    )
    unsigned = result.model_dump(mode="json", exclude={"result_sha256"})
    unsigned["time_contract"] = tampered_contract.model_dump(mode="json")
    tampered = result.model_copy(
        update={
            "time_contract": tampered_contract,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )

    assert tampered.verify() is False


def test_prepared_marker_row_mismatch_degrades_only_that_source(
    tmp_path: Path,
) -> None:
    module = adapter()
    mismatched_rows = {name.removesuffix(".parquet"): 0 for name in ALL_FILES}
    mismatched_rows.update({"normal_metrics": 1, "abnormal_metrics": 1})
    case = write_prepared_case(
        tmp_path,
        files={
            "normal_metrics.parquet": [T0 - 2, T0 - 1],
            "abnormal_metrics.parquet": [T0],
        },
        marker_update={"rows": mismatched_rows},
    )

    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )

    assert tuple(
        (item.source_file, item.reason) for item in result.source_rejections
    ) == (("normal_metrics.parquet", "prepared_row_count_mismatch"),)
    assert tuple(item.source_file for item in result.files) == (
        "abnormal_metrics.parquet",
    )
    assert result.verify()


def test_case_without_valid_in_scope_telemetry_is_rejected(tmp_path: Path) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path, files={})

    with pytest.raises(module.AIOps25AdapterError, match="valid in-scope telemetry"):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(),
            configuration=configuration(module),
        )


def test_one_os_unreadable_source_degrades_without_losing_other_modalities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = adapter()
    rcabench = importlib.import_module("nurse_rcl.v2.rcabench_adapter")
    case = write_prepared_case(
        tmp_path,
        files={
            "normal_metrics.parquet": [T0 - 1],
            "abnormal_metrics.parquet": [T0],
        },
    )
    original_hash = rcabench._file_sha256

    def fail_one_source(path: Path) -> str:
        if path.name == "normal_metrics.parquet":
            raise PermissionError(path)
        return original_hash(path)

    monkeypatch.setattr(rcabench, "_file_sha256", fail_one_source)
    result = module.adapt_aiops25_case(
        case,
        source_case_id=SOURCE_CASE_ID,
        opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(),
        configuration=configuration(module),
    )

    assert result.unavailable_source_files == ("normal_metrics.parquet",)
    assert tuple(
        (item.source_file, item.reason) for item in result.source_rejections
    ) == (("normal_metrics.parquet", "os_unreadable"),)
    assert result.missing_source_files == tuple(
        sorted(set(ALL_FILES).difference({"abnormal_metrics.parquet"}))
    )
    assert result.time_contract.missing_modalities == (
        "deployment",
        "logs",
        "traces",
    )
    assert result.verify()


@pytest.mark.parametrize(
    ("marker_update", "env_update", "message"),
    (
        ({"case": "different-case"}, {}, "prepared marker"),
        ({"services": []}, {}, "service directory"),
        ({}, {"ORIGINAL_ABNORMAL_END": T0 - 1}, "env time contract"),
    ),
)
def test_invalid_prepared_contract_is_rejected(
    tmp_path: Path,
    marker_update: dict[str, object],
    env_update: dict[str, object],
    message: str,
) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        marker_update=marker_update,
        env_update=env_update,
    )

    with pytest.raises(module.AIOps25AdapterError, match=message):
        module.adapt_aiops25_case(
            case,
            source_case_id=SOURCE_CASE_ID,
            opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(),
            configuration=configuration(module),
        )


@pytest.mark.parametrize("original_end_offset", (1, 600, 840))
def test_export_window_can_extend_or_shorten_without_changing_decision_gate(
    tmp_path: Path, original_end_offset: int,
) -> None:
    module = adapter()
    case = write_prepared_case(
        tmp_path,
        env_update={
            "ABNORMAL_END": T0 + 600,
            "ORIGINAL_ABNORMAL_END": T0 + original_end_offset,
        },
        files={
            "normal_metrics.parquet": [T0 - 481, T0 - 480, T0 - 1],
            "abnormal_metrics.parquet": [T0, T0 + 1, T0 + 479, T0 + 480, T0 + 599],
        },
    )
    result = module.adapt_aiops25_case(
        case, source_case_id=SOURCE_CASE_ID, opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=authorized_queue(), configuration=configuration(module),
    )
    assert result.verify()
    assert result.source_fault_window.end_unix_ns == (T0 + 600) * 1_000_000_000
    assert result.time_contract.decision_time_unix_ns == (T0 + 480) * 1_000_000_000
    audits = {item.source_file: item for item in result.files}
    assert audits["normal_metrics.parquet"].accepted_rows == 2
    assert audits["abnormal_metrics.parquet"].accepted_rows == 3
    assert audits["abnormal_metrics.parquet"].post_decision_rows == 2
    assert result.time_contract.late_data_observed


@pytest.mark.parametrize("field", ("ABNORMAL_END", "ORIGINAL_ABNORMAL_END"))
@pytest.mark.parametrize("offset", (0, -1))
def test_both_export_and_original_end_must_remain_after_start(
    tmp_path: Path, field: str, offset: int,
) -> None:
    module = adapter()
    case = write_prepared_case(tmp_path, env_update={field: T0 + offset})
    with pytest.raises(module.AIOps25AdapterError, match="env time contract"):
        module.adapt_aiops25_case(
            case, source_case_id=SOURCE_CASE_ID, opaque_case_id=OPAQUE_CASE_ID,
            authorized_queue=authorized_queue(), configuration=configuration(module),
        )


def test_adapter_artifact_versions_are_in_the_compatibility_registry() -> None:
    module = adapter()
    compatibility = importlib.import_module("nurse_rcl.v2.schema_compatibility")

    assert compatibility.CURRENT_SCHEMA_VERSIONS[
        "aiops25_adapter_configuration"
    ] == module.AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION
    assert compatibility.CURRENT_SCHEMA_VERSIONS[
        "aiops25_adapter_result"
    ] == module.AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION
