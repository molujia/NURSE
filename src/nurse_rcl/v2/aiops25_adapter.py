"""Decision-bounded AIOps25 adapter over the approved prepared-case format."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.aiops25_inventory import (
    AIOPS25_PARQUET_FILES,
    AuthorizedAIOps25Queue,
)
from nurse_rcl.v2.observation_schemas import (
    CaseTimeContract,
    TimeInterval,
)
from nurse_rcl.v2.rcabench_adapter import (
    NANOSECONDS_PER_SECOND,
    ParquetSourceContract,
    TelemetrySourceAudit,
    _audit_source_values,
    audit_parquet_source,
    build_parquet_source_contract,
    missing_telemetry_modalities,
)

AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION: Literal[
    "nurse.v2.aiops25-adapter-configuration/1"
] = "nurse.v2.aiops25-adapter-configuration/1"
AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.aiops25-adapter-result/1"
] = "nurse.v2.aiops25-adapter-result/1"
AIOPS25_ADAPTER_VERSION: Literal["nurse-v2-aiops25/1"] = "nurse-v2-aiops25/1"
AIOPS25_TIMEZONE: Literal["UTC"] = "UTC"

_SOURCE_CONTRACTS: dict[str, ParquetSourceContract] = {
    "normal_logs.parquet": build_parquet_source_contract(
        "logs", "baseline", "logs"
    ),
    "abnormal_logs.parquet": build_parquet_source_contract(
        "logs", "fault", "logs"
    ),
    "normal_metrics.parquet": build_parquet_source_contract(
        "metrics", "baseline", "metrics"
    ),
    "abnormal_metrics.parquet": build_parquet_source_contract(
        "metrics", "fault", "metrics"
    ),
    "normal_traces.parquet": build_parquet_source_contract(
        "traces", "baseline", "traces"
    ),
    "abnormal_traces.parquet": build_parquet_source_contract(
        "traces", "fault", "traces"
    ),
}
_ROW_KEYS = tuple(name.removesuffix(".parquet") for name in AIOPS25_PARQUET_FILES)
_MARKER_KEYS = frozenset({"schema_version", "case", "rows", "services", "metrics"})
_ENV_KEYS = frozenset(
    {
        "NORMAL_START",
        "NORMAL_END",
        "ABNORMAL_START",
        "ABNORMAL_END",
        "ORIGINAL_ABNORMAL_END",
    }
)
SourceRejectionReason = Literal["os_unreadable", "prepared_row_count_mismatch"]


class AIOps25AdapterError(ValueError):
    """Raised when a prepared AIOps25 case violates its adapter contract."""


class AIOps25AdapterScopeError(AIOps25AdapterError):
    """Raised when a case lacks a valid prepared-data inference authorization."""


class AIOps25AdapterConfiguration(StrictFrozenModel):
    """Versioned AIOps25 time assumptions that are never fitted from labels."""

    schema_version: Literal[
        "nurse.v2.aiops25-adapter-configuration/1"
    ] = AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION
    adapter_version: Literal["nurse-v2-aiops25/1"] = AIOPS25_ADAPTER_VERSION
    window_seconds: int = Field(gt=0)
    expected_cadence_seconds: int = Field(gt=0)
    clock_uncertainty_seconds: int = Field(ge=0)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"configuration_sha256"})
        return self.configuration_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


def build_aiops25_adapter_config(
    *,
    window_seconds: int = 480,
    expected_cadence_seconds: int = 30,
    clock_uncertainty_seconds: int = 0,
) -> AIOps25AdapterConfiguration:
    unsigned: dict[str, object] = {
        "schema_version": AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION,
        "adapter_version": AIOPS25_ADAPTER_VERSION,
        "window_seconds": window_seconds,
        "expected_cadence_seconds": expected_cadence_seconds,
        "clock_uncertainty_seconds": clock_uncertainty_seconds,
    }
    configuration = AIOps25AdapterConfiguration.model_validate(
        {
            **unsigned,
            "configuration_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not configuration.verify():  # pragma: no cover - construction invariant
        raise AssertionError("AIOps25 adapter configuration verification failed")
    return configuration


class AIOps25SourceRejection(StrictFrozenModel):
    """One prepared source excluded from evidence without failing the case."""

    source_file: str = Field(pattern=r"^(normal|abnormal)_[a-z_]+\.parquet$")
    reason: SourceRejectionReason
    declared_rows: int = Field(ge=0)
    observed_rows: int | None = Field(default=None, ge=0)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_reason_specific_evidence(self) -> Self:
        if self.reason == "os_unreadable":
            if self.observed_rows is not None or self.source_sha256 is not None:
                raise ValueError("OS-unreadable source cannot claim inspected content")
        elif (
            self.observed_rows is None
            or self.source_sha256 is None
            or self.observed_rows == self.declared_rows
        ):
            raise ValueError("row-count rejection requires a verified mismatch")
        return self


class AIOps25AdapterResult(StrictFrozenModel):
    """Inference-safe audit with no source identity, path, or injection payload."""

    schema_version: Literal["nurse.v2.aiops25-adapter-result/1"] = (
        AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    target_granularity: Literal["service"] = "service"
    configuration: AIOps25AdapterConfiguration
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    env_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    service_entity_keys: tuple[str, ...] = Field(min_length=1)
    time_contract: CaseTimeContract
    source_baseline_window: TimeInterval
    source_fault_window: TimeInterval
    files: tuple[TelemetrySourceAudit, ...]
    missing_source_files: tuple[str, ...]
    unavailable_source_files: tuple[str, ...]
    source_rejections: tuple[AIOps25SourceRejection, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_inference_safe_contract(self) -> Self:
        if (
            self.service_entity_keys != tuple(sorted(set(self.service_entity_keys)))
            or any(
                item != item.strip() or "/" in item or "\\" in item
                for item in self.service_entity_keys
            )
        ):
            raise ValueError("service directory must be sorted and unique")
        return self

    def verify(self) -> bool:
        if not self.configuration.verify():
            return False
        if (
            self.time_contract.opaque_case_id != self.opaque_case_id
            or self.time_contract.timezone != AIOPS25_TIMEZONE
            or self.time_contract.t0_source != "benchmark_fault_time"
            or self.time_contract.localization_scope != "given_fault_time"
            or self.target_granularity != "service"
        ):
            return False
        width_ns = self.configuration.window_seconds * NANOSECONDS_PER_SECOND
        if self.time_contract.t0_unix_ns < width_ns:
            return False
        if self.time_contract.baseline_window != TimeInterval(
            start_unix_ns=self.time_contract.t0_unix_ns - width_ns,
            end_unix_ns=self.time_contract.t0_unix_ns,
        ):
            return False
        if self.time_contract.fault_window != TimeInterval(
            start_unix_ns=self.time_contract.t0_unix_ns,
            end_unix_ns=self.time_contract.t0_unix_ns + width_ns,
        ):
            return False
        if self.time_contract.decision_time_unix_ns != (
            self.time_contract.t0_unix_ns + width_ns
        ):
            return False
        if self.time_contract.expected_cadence_ns != (
            self.configuration.expected_cadence_seconds * NANOSECONDS_PER_SECOND
        ):
            return False
        if self.time_contract.clock_uncertainty_ns != (
            self.configuration.clock_uncertainty_seconds * NANOSECONDS_PER_SECOND
        ):
            return False
        if self.source_baseline_window.end_unix_ns != self.time_contract.t0_unix_ns:
            return False
        if self.source_fault_window.start_unix_ns != self.time_contract.t0_unix_ns:
            return False
        if self.files != tuple(sorted(self.files, key=lambda item: item.source_file)):
            return False
        names = tuple(item.source_file for item in self.files)
        if len(names) != len(set(names)):
            return False
        if self.source_rejections != tuple(
            sorted(self.source_rejections, key=lambda item: item.source_file)
        ):
            return False
        rejected_names = tuple(item.source_file for item in self.source_rejections)
        if len(rejected_names) != len(set(rejected_names)) or set(names).intersection(
            rejected_names
        ):
            return False
        expected_missing = tuple(sorted(set(_SOURCE_CONTRACTS).difference(names)))
        if self.missing_source_files != expected_missing:
            return False
        if not set(rejected_names).issubset(expected_missing):
            return False
        if self.unavailable_source_files != rejected_names:
            return False
        if not any(
            item.status == "valid" and item.accepted_rows > 0 for item in self.files
        ):
            return False
        if self.time_contract.late_data_observed != any(
            item.post_decision_rows > 0 for item in self.files
        ):
            return False
        if self.time_contract.missing_modalities != missing_telemetry_modalities(
            self.files
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_sha256"})
        return self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def adapt_aiops25_case(
    case_root: Path,
    *,
    source_case_id: str,
    opaque_case_id: str,
    authorized_queue: AuthorizedAIOps25Queue,
    configuration: AIOps25AdapterConfiguration,
) -> AIOps25AdapterResult:
    """Audit one authorized prepared case without opening injection metadata."""
    return _adapt_aiops25_case_inputs(
        case_root, source_case_id=source_case_id, opaque_case_id=opaque_case_id,
        authorized_queue=authorized_queue, configuration=configuration,
    )


def _adapt_aiops25_case_inputs(
    case_root: Path, *, source_case_id: str, opaque_case_id: str,
    authorized_queue: AuthorizedAIOps25Queue, configuration: AIOps25AdapterConfiguration,
    input_bytes: Mapping[str, bytes] | None = None,
) -> AIOps25AdapterResult:
    """Shared authorization/window/accounting rules over files or pinned bytes."""

    _require_inference_authorization(authorized_queue, source_case_id)
    if not configuration.verify():
        raise AIOps25AdapterError("adapter configuration hash is invalid")
    try:
        root = Path(case_root).resolve(strict=True)
    except OSError as error:
        raise AIOps25AdapterError("prepared case root is unavailable") from error
    if not root.is_dir():
        raise AIOps25AdapterError("prepared case root is not a directory")

    marker_contract, marker_rows, service_keys = (
        _read_prepared_marker(root / ".prepared.json", source_case_id)
        if input_bytes is None
        else _decode_prepared_marker(input_bytes[".prepared.json"], source_case_id)
    )
    env_contract, source_baseline, source_fault = (
        _read_env_contract(root / "env.json")
        if input_bytes is None else _decode_env_contract(input_bytes["env.json"])
    )
    t0_ns = source_fault.start_unix_ns
    width_ns = configuration.window_seconds * NANOSECONDS_PER_SECOND
    decision_time_ns = t0_ns + width_ns
    provisional_contract = CaseTimeContract(
        opaque_case_id=opaque_case_id,
        t0_unix_ns=t0_ns,
        t0_source="benchmark_fault_time",
        localization_scope="given_fault_time",
        baseline_window=TimeInterval(
            start_unix_ns=t0_ns - width_ns,
            end_unix_ns=t0_ns,
        ),
        fault_window=TimeInterval(
            start_unix_ns=t0_ns,
            end_unix_ns=decision_time_ns,
        ),
        decision_time_unix_ns=decision_time_ns,
        timezone=AIOPS25_TIMEZONE,
        expected_cadence_ns=(
            configuration.expected_cadence_seconds * NANOSECONDS_PER_SECOND
        ),
        clock_uncertainty_ns=(
            configuration.clock_uncertainty_seconds * NANOSECONDS_PER_SECOND
        ),
        missing_modalities=(),
        late_data_observed=False,
    )
    audits, source_rejections = _audit_available_sources(
        root=root,
        time_contract=provisional_contract,
        marker_rows=marker_rows,
        input_bytes=input_bytes,
    )
    if not any(
        item.status == "valid" and item.accepted_rows > 0 for item in audits
    ):
        raise AIOps25AdapterError("prepared case has no valid in-scope telemetry")
    time_contract = provisional_contract.model_copy(
        update={
            "missing_modalities": missing_telemetry_modalities(audits),
            "late_data_observed": any(
                item.post_decision_rows > 0 for item in audits
            ),
        }
    )
    unsigned: dict[str, object] = {
        "schema_version": AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "target_granularity": "service",
        "configuration": configuration.model_dump(mode="json"),
        "inventory_sha256": authorized_queue.inventory_sha256,
        # The source case identity is validated, then excluded before hashing.
        "prepared_contract_sha256": sha256_hex(
            canonical_json_bytes(marker_contract)
        ),
        "env_contract_sha256": sha256_hex(canonical_json_bytes(env_contract)),
        "service_entity_keys": service_keys,
        "time_contract": time_contract.model_dump(mode="json"),
        "source_baseline_window": source_baseline.model_dump(mode="json"),
        "source_fault_window": source_fault.model_dump(mode="json"),
        "files": [item.model_dump(mode="json") for item in audits],
        "missing_source_files": tuple(
            sorted(
                set(_SOURCE_CONTRACTS).difference(
                    item.source_file for item in audits
                )
            )
        ),
        "unavailable_source_files": tuple(
            item.source_file for item in source_rejections
        ),
        "source_rejections": [
            item.model_dump(mode="json") for item in source_rejections
        ],
    }
    result = AIOps25AdapterResult.model_validate(
        {**unsigned, "result_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("AIOps25 adapter result verification failed")
    return result


def _require_inference_authorization(
    queue: AuthorizedAIOps25Queue,
    source_case_id: str,
) -> None:
    if not isinstance(queue, AuthorizedAIOps25Queue) or not queue.verify():
        raise AIOps25AdapterScopeError("verified inference queue is required")
    if queue.destination != "inference":
        raise AIOps25AdapterScopeError("AIOps25 adapter requires an inference queue")
    if source_case_id not in queue.case_ids:
        raise AIOps25AdapterScopeError("case is not authorized for inference")


def _read_prepared_marker(
    path: Path,
    source_case_id: str,
) -> tuple[dict[str, object], dict[str, int], tuple[str, ...]]:
    try:
        return _decode_prepared_marker(path.read_bytes(), source_case_id)
    except OSError as error:
        raise AIOps25AdapterError("AIOps25 prepared marker is invalid") from error


def _decode_prepared_marker(
    data: bytes, source_case_id: str,
) -> tuple[dict[str, object], dict[str, int], tuple[str, ...]]:
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise TypeError("marker must be an object")
        if set(value) != _MARKER_KEYS:
            raise ValueError("marker fields are outside the allowlist")
        if value.get("schema_version") != 1 or value.get("case") != source_case_id:
            raise ValueError("marker identity mismatch")
        rows = value.get("rows")
        if not isinstance(rows, dict) or not all(
            isinstance(rows.get(key), int)
            and not isinstance(rows.get(key), bool)
            and cast(int, rows.get(key)) >= 0
            for key in _ROW_KEYS
        ):
            raise ValueError("marker row directory is invalid")
        marker_rows = {key: cast(int, rows[key]) for key in _ROW_KEYS}
        service_keys = _canonical_marker_directory(
            value.get("services"),
            source_case_id=source_case_id,
            directory_name="service",
            require_nonempty=True,
        )
        metric_keys = _canonical_marker_directory(
            value.get("metrics"),
            source_case_id=source_case_id,
            directory_name="metric",
            require_nonempty=False,
        )
        marker_contract: dict[str, object] = {
            "schema_version": 1,
            "rows": dict(sorted(marker_rows.items())),
            "services": service_keys,
            "metrics": metric_keys,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = (
            "AIOps25 service directory is invalid"
            if "service directory" in str(error)
            else "AIOps25 prepared marker is invalid"
        )
        raise AIOps25AdapterError(message) from error
    return marker_contract, marker_rows, service_keys


def _read_env_contract(
    path: Path,
) -> tuple[dict[str, int], TimeInterval, TimeInterval]:
    try:
        return _decode_env_contract(path.read_bytes())
    except OSError as error:
        raise AIOps25AdapterError("AIOps25 env time contract is invalid") from error


def _decode_env_contract(data: bytes) -> tuple[dict[str, int], TimeInterval, TimeInterval]:
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise TypeError("env must be an object")
        if set(value) != _ENV_KEYS:
            raise ValueError("env fields are outside the time allowlist")
        normal_start = _unix_seconds(value, "NORMAL_START")
        normal_end = _unix_seconds(value, "NORMAL_END")
        abnormal_start = _unix_seconds(value, "ABNORMAL_START")
        abnormal_end = _unix_seconds(value, "ABNORMAL_END")
        original_abnormal_end = _unix_seconds(value, "ORIGINAL_ABNORMAL_END")
        # The approved preparer may extend a short fault observation window or
        # cap it at the next fault. Neither end sets the fixed decision gate.
        if not (
            normal_start < normal_end == abnormal_start < abnormal_end
            and abnormal_start < original_abnormal_end
        ):
            raise ValueError("source windows are not contiguous and positive")
        baseline = TimeInterval(
            start_unix_ns=normal_start * NANOSECONDS_PER_SECOND,
            end_unix_ns=normal_end * NANOSECONDS_PER_SECOND,
        )
        fault = TimeInterval(
            start_unix_ns=abnormal_start * NANOSECONDS_PER_SECOND,
            end_unix_ns=abnormal_end * NANOSECONDS_PER_SECOND,
        )
        env_contract = {
            "NORMAL_START": normal_start,
            "NORMAL_END": normal_end,
            "ABNORMAL_START": abnormal_start,
            "ABNORMAL_END": abnormal_end,
            "ORIGINAL_ABNORMAL_END": original_abnormal_end,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIOps25AdapterError("AIOps25 env time contract is invalid") from error
    return env_contract, baseline, fault


def _audit_available_sources(
    *,
    root: Path,
    time_contract: CaseTimeContract,
    marker_rows: Mapping[str, int],
    input_bytes: Mapping[str, bytes] | None = None,
) -> tuple[
    tuple[TelemetrySourceAudit, ...],
    tuple[AIOps25SourceRejection, ...],
]:
    audits: list[TelemetrySourceAudit] = []
    rejections: list[AIOps25SourceRejection] = []
    for source_file, contract in sorted(_SOURCE_CONTRACTS.items()):
        path = root / source_file
        if (
            (not path.is_file() or path.is_symlink()) if input_bytes is None
            else source_file not in input_bytes
        ):
            continue
        try:
            audit = audit_parquet_source(
                path=path,
                source_file=source_file,
                contract=contract,
                time_contract=time_contract,
            ) if input_bytes is None else _audit_source_values(
                source_input=pa.BufferReader(input_bytes[source_file]),
                source_sha256=sha256_hex(input_bytes[source_file]),
                source_file=source_file, contract=contract, time_contract=time_contract,
            )
        except OSError:
            rejections.append(
                AIOps25SourceRejection(
                    source_file=source_file,
                    reason="os_unreadable",
                    declared_rows=marker_rows[
                        source_file.removesuffix(".parquet")
                    ],
                    observed_rows=None,
                    source_sha256=None,
                )
            )
            continue
        marker_key = source_file.removesuffix(".parquet")
        if audit.status == "valid" and marker_rows[marker_key] != audit.input_rows:
            rejections.append(
                AIOps25SourceRejection(
                    source_file=source_file,
                    reason="prepared_row_count_mismatch",
                    declared_rows=marker_rows[marker_key],
                    observed_rows=audit.input_rows,
                    source_sha256=audit.source_sha256,
                )
            )
            continue
        audits.append(audit)
    return tuple(audits), tuple(rejections)


def _canonical_marker_directory(
    value: object,
    *,
    source_case_id: str,
    directory_name: Literal["service", "metric"],
    require_nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (require_nonempty and not value):
        raise ValueError(f"{directory_name} directory is invalid")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{directory_name} directory is invalid")
        normalized = unicodedata.normalize("NFKC", raw)
        if (
            normalized != raw
            or normalized != normalized.strip()
            or source_case_id.casefold() in normalized.casefold()
            or "/" in normalized
            or "\\" in normalized
        ):
            raise ValueError(
                f"{directory_name} directory contains source identity or path"
            )
        result.append(normalized)
    return tuple(sorted(set(result)))


def _unix_seconds(value: Mapping[str, object], field: str) -> int:
    raw = value.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        raise TypeError(f"{field} must be Unix seconds")
    result = int(raw)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result
