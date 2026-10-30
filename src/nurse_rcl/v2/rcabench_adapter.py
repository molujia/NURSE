"""Decision-bounded, label-isolated RCABench source adapter for NURSE V2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import (
    CaseTimeContract,
    TelemetryModality,
    TimeInterval,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
RCABENCH_ADAPTER_CONFIGURATION_SCHEMA_VERSION: Literal[
    "nurse.v2.rcabench-adapter-configuration/1"
] = "nurse.v2.rcabench-adapter-configuration/1"
RCABENCH_ADAPTER_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.rcabench-adapter-result/1"
] = "nurse.v2.rcabench-adapter-result/1"
RCABENCH_ADAPTER_VERSION: Literal[
    "nurse-v2-rcabench/1"
] = "nurse-v2-rcabench/1"
RCABENCH_TIMEZONE: Literal["Asia/Shanghai"] = "Asia/Shanghai"
_TELEMETRY_MODALITIES: tuple[TelemetryModality, ...] = (
    "deployment",
    "metrics",
    "logs",
    "traces",
)

SourceWindow = Literal["baseline", "fault"]
SourceStatus = Literal["valid", "invalid"]
SourceErrorCode = Literal[
    "unreadable_parquet",
    "missing_time_column",
    "unsupported_time_type",
]


class RCABenchAdapterError(ValueError):
    """Raised when the case-level RCABench source contract is unusable."""


@dataclass(frozen=True, slots=True)
class _SourceContract:
    modality: Literal["metrics", "logs", "traces"]
    window: SourceWindow
    source_kind: Literal[
        "logs",
        "metrics",
        "metrics_histogram",
        "metrics_sum",
        "traces",
    ]


def _source(
    modality: Literal["metrics", "logs", "traces"],
    window: SourceWindow,
    source_kind: Literal[
        "logs",
        "metrics",
        "metrics_histogram",
        "metrics_sum",
        "traces",
    ],
) -> _SourceContract:
    return _SourceContract(
        modality=modality,
        window=window,
        source_kind=source_kind,
    )


_SOURCE_CONTRACTS: dict[str, _SourceContract] = {
    "normal_logs.parquet": _source("logs", "baseline", "logs"),
    "abnormal_logs.parquet": _source("logs", "fault", "logs"),
    "normal_metrics.parquet": _source("metrics", "baseline", "metrics"),
    "abnormal_metrics.parquet": _source("metrics", "fault", "metrics"),
    "normal_metrics_histogram.parquet": _source(
        "metrics", "baseline", "metrics_histogram"
    ),
    "abnormal_metrics_histogram.parquet": _source(
        "metrics", "fault", "metrics_histogram"
    ),
    "normal_metrics_sum.parquet": _source(
        "metrics", "baseline", "metrics_sum"
    ),
    "abnormal_metrics_sum.parquet": _source(
        "metrics", "fault", "metrics_sum"
    ),
    "normal_traces.parquet": _source("traces", "baseline", "traces"),
    "abnormal_traces.parquet": _source("traces", "fault", "traces"),
}


class RCABenchAdapterConfiguration(StrictFrozenModel):
    """Versioned time and quality assumptions; never fitted from labels."""

    schema_version: Literal[
        "nurse.v2.rcabench-adapter-configuration/1"
    ] = RCABENCH_ADAPTER_CONFIGURATION_SCHEMA_VERSION
    adapter_version: Literal["nurse-v2-rcabench/1"] = RCABENCH_ADAPTER_VERSION
    window_seconds: int = Field(gt=0)
    expected_cadence_seconds: int = Field(gt=0)
    clock_uncertainty_seconds: int = Field(ge=0)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"configuration_sha256"})
        return self.configuration_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


def build_rcabench_adapter_config(
    *,
    window_seconds: int = 480,
    expected_cadence_seconds: int = 30,
    clock_uncertainty_seconds: int = 0,
) -> RCABenchAdapterConfiguration:
    unsigned: dict[str, object] = {
        "schema_version": RCABENCH_ADAPTER_CONFIGURATION_SCHEMA_VERSION,
        "adapter_version": RCABENCH_ADAPTER_VERSION,
        "window_seconds": window_seconds,
        "expected_cadence_seconds": expected_cadence_seconds,
        "clock_uncertainty_seconds": clock_uncertainty_seconds,
    }
    configuration = RCABenchAdapterConfiguration.model_validate(
        {
            **unsigned,
            "configuration_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not configuration.verify():  # pragma: no cover - construction invariant
        raise AssertionError("RCABench adapter configuration verification failed")
    return configuration


class RCABenchSourceAudit(StrictFrozenModel):
    """Content identity and timestamp accounting for one allowlisted source."""

    source_file: str = Field(pattern=r"^(normal|abnormal)_[a-z_]+\.parquet$")
    modality: Literal["metrics", "logs", "traces"]
    window: SourceWindow
    source_kind: Literal[
        "logs",
        "metrics",
        "metrics_histogram",
        "metrics_sum",
        "traces",
    ]
    status: SourceStatus
    error_code: SourceErrorCode | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    outside_contract_rows: int = Field(ge=0)
    post_decision_rows: int = Field(ge=0)
    invalid_timestamp_rows: int = Field(ge=0)
    earliest_accepted_unix_ns: int | None = Field(default=None, ge=0)
    latest_accepted_unix_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_complete_safe_accounting(self) -> Self:
        if self.status == "invalid":
            if (
                self.error_code is None
                or self.schema_sha256 is not None
                or self.input_rows
                or self.accepted_rows
                or self.outside_contract_rows
                or self.post_decision_rows
                or self.invalid_timestamp_rows
                or self.earliest_accepted_unix_ns is not None
                or self.latest_accepted_unix_ns is not None
            ):
                raise ValueError("invalid source must not expose untrusted row metadata")
            return self
        if self.error_code is not None or self.schema_sha256 is None:
            raise ValueError("valid source requires schema and no error code")
        if self.input_rows != (
            self.accepted_rows
            + self.outside_contract_rows
            + self.post_decision_rows
            + self.invalid_timestamp_rows
        ):
            raise ValueError("source audit must account for every timestamp row")
        bounds = (
            self.earliest_accepted_unix_ns,
            self.latest_accepted_unix_ns,
        )
        if self.accepted_rows == 0 and bounds != (None, None):
            raise ValueError("empty accepted source cannot have timestamp bounds")
        if self.accepted_rows > 0 and (
            bounds[0] is None
            or bounds[1] is None
            or bounds[0] > bounds[1]
        ):
            raise ValueError("accepted source requires ordered timestamp bounds")
        return self


class RCABenchAdapterResult(StrictFrozenModel):
    """Inference-safe source audit with no case name, path, label, or payload."""

    schema_version: Literal["nurse.v2.rcabench-adapter-result/1"] = (
        RCABENCH_ADAPTER_RESULT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    configuration: RCABenchAdapterConfiguration
    env_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    time_contract: CaseTimeContract
    source_baseline_window: TimeInterval
    source_fault_window: TimeInterval
    files: tuple[RCABenchSourceAudit, ...]
    missing_source_files: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if not self.configuration.verify():
            return False
        if self.time_contract.opaque_case_id != self.opaque_case_id:
            return False
        if self.files != tuple(sorted(self.files, key=lambda item: item.source_file)):
            return False
        names = tuple(item.source_file for item in self.files)
        if len(names) != len(set(names)):
            return False
        expected_missing = tuple(sorted(set(_SOURCE_CONTRACTS).difference(names)))
        if self.missing_source_files != expected_missing:
            return False
        if self.time_contract.late_data_observed != any(
            item.post_decision_rows > 0 for item in self.files
        ):
            return False
        expected_modalities = _missing_modalities(self.files)
        if self.time_contract.missing_modalities != expected_modalities:
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_sha256"})
        return self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def adapt_rcabench_case(
    case_root: Path,
    *,
    opaque_case_id: str,
    configuration: RCABenchAdapterConfiguration,
) -> RCABenchAdapterResult:
    """Audit one RCABench case without opening label- or answer-bearing files."""
    return _adapt_rcabench_case_inputs(
        case_root, opaque_case_id=opaque_case_id, configuration=configuration,
    )


def _adapt_rcabench_case_inputs(
    case_root: Path, *, opaque_case_id: str,
    configuration: RCABenchAdapterConfiguration,
    input_bytes: Mapping[str, bytes] | None = None,
) -> RCABenchAdapterResult:
    """Shared adapter rules over files or a private, already pinned byte snapshot."""

    if not configuration.verify():
        raise RCABenchAdapterError("adapter configuration hash is invalid")
    root = Path(case_root).resolve(strict=True)
    if not root.is_dir():
        raise RCABenchAdapterError("RCABench case root is not a directory")
    env_bytes, source_baseline, source_fault, timezone_name = (
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
        timezone=timezone_name,
        expected_cadence_ns=(
            configuration.expected_cadence_seconds * NANOSECONDS_PER_SECOND
        ),
        clock_uncertainty_ns=(
            configuration.clock_uncertainty_seconds * NANOSECONDS_PER_SECOND
        ),
        missing_modalities=(),
        late_data_observed=False,
    )
    audits = tuple(
        _audit_source(
            path=root / source_file,
            source_file=source_file,
            contract=contract,
            time_contract=provisional_contract,
        ) if input_bytes is None else _audit_source_values(
            source_input=pa.BufferReader(input_bytes[source_file]),
            source_sha256=sha256_hex(input_bytes[source_file]),
            source_file=source_file, contract=contract, time_contract=provisional_contract,
        )
        for source_file, contract in sorted(_SOURCE_CONTRACTS.items())
        if ((root / source_file).is_file() if input_bytes is None else source_file in input_bytes)
    )
    time_contract = provisional_contract.model_copy(
        update={
            "missing_modalities": _missing_modalities(audits),
            "late_data_observed": any(
                item.post_decision_rows > 0 for item in audits
            ),
        }
    )
    unsigned: dict[str, object] = {
        "schema_version": RCABENCH_ADAPTER_RESULT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "configuration": configuration.model_dump(mode="json"),
        "env_sha256": sha256_hex(env_bytes),
        "time_contract": time_contract.model_dump(mode="json"),
        "source_baseline_window": source_baseline.model_dump(mode="json"),
        "source_fault_window": source_fault.model_dump(mode="json"),
        "files": [item.model_dump(mode="json") for item in audits],
        "missing_source_files": tuple(
            sorted(set(_SOURCE_CONTRACTS).difference(item.source_file for item in audits))
        ),
    }
    result = RCABenchAdapterResult.model_validate(
        {**unsigned, "result_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("RCABench adapter result verification failed")
    return result


def _read_env_contract(
    path: Path,
) -> tuple[bytes, TimeInterval, TimeInterval, str]:
    try:
        return _decode_env_contract(path.read_bytes())
    except OSError as error:
        raise RCABenchAdapterError("RCABench env time contract is invalid") from error


def _decode_env_contract(data: bytes) -> tuple[bytes, TimeInterval, TimeInterval, str]:
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise TypeError("env must be an object")
        normal_start = _unix_seconds(value, "NORMAL_START")
        normal_end = _unix_seconds(value, "NORMAL_END")
        abnormal_start = _unix_seconds(value, "ABNORMAL_START")
        abnormal_end = _unix_seconds(value, "ABNORMAL_END")
        timezone_name = value.get("TIMEZONE")
        if timezone_name != RCABENCH_TIMEZONE:
            raise ValueError("TIMEZONE is outside the frozen RCABench contract")
        if not normal_start < normal_end == abnormal_start < abnormal_end:
            raise ValueError("source windows are not contiguous and positive")
        baseline = TimeInterval(
            start_unix_ns=normal_start * NANOSECONDS_PER_SECOND,
            end_unix_ns=normal_end * NANOSECONDS_PER_SECOND,
        )
        fault = TimeInterval(
            start_unix_ns=abnormal_start * NANOSECONDS_PER_SECOND,
            end_unix_ns=abnormal_end * NANOSECONDS_PER_SECOND,
        )
    except (OSError, TypeError, ValueError) as error:
        raise RCABenchAdapterError("RCABench env time contract is invalid") from error
    return data, baseline, fault, timezone_name


def _unix_seconds(value: Mapping[str, object], field: str) -> int:
    raw = value.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        raise TypeError(f"{field} must be Unix seconds")
    result = int(raw)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _audit_source(
    *,
    path: Path,
    source_file: str,
    contract: _SourceContract,
    time_contract: CaseTimeContract,
) -> RCABenchSourceAudit:
    return _audit_source_values(
        source_input=path, source_sha256=_file_sha256(path), source_file=source_file,
        contract=contract, time_contract=time_contract,
    )


def _audit_source_values(
    *, source_input: Any, source_sha256: str, source_file: str,
    contract: _SourceContract, time_contract: CaseTimeContract,
) -> RCABenchSourceAudit:
    try:
        schema = pq.read_schema(source_input)
    except (OSError, pa.ArrowException):
        return _invalid_source(
            source_file=source_file,
            contract=contract,
            source_sha256=source_sha256,
            error_code="unreadable_parquet",
        )
    if "time" not in schema.names:
        return _invalid_source(
            source_file=source_file,
            contract=contract,
            source_sha256=source_sha256,
            error_code="missing_time_column",
        )
    time_type = schema.field("time").type
    if not pa.types.is_timestamp(time_type):
        return _invalid_source(
            source_file=source_file,
            contract=contract,
            source_sha256=source_sha256,
            error_code="unsupported_time_type",
        )
    try:
        values = pq.read_table(source_input, columns=["time"]).column("time").cast(
            pa.int64()
        ).to_pylist()
    except (OSError, pa.ArrowException):
        return _invalid_source(
            source_file=source_file,
            contract=contract,
            source_sha256=source_sha256,
            error_code="unreadable_parquet",
        )
    factor = _timestamp_unit_factor(time_type.unit)
    accepted: list[int] = []
    outside = 0
    post_decision = 0
    invalid = 0
    interval = (
        time_contract.baseline_window
        if contract.window == "baseline"
        else time_contract.fault_window
    )
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            invalid += 1
            continue
        timestamp_ns = raw * factor
        if timestamp_ns >= time_contract.decision_time_unix_ns:
            post_decision += 1
        elif interval.start_unix_ns <= timestamp_ns < interval.end_unix_ns:
            accepted.append(timestamp_ns)
        else:
            outside += 1
    return RCABenchSourceAudit(
        source_file=source_file,
        modality=contract.modality,
        window=contract.window,
        source_kind=contract.source_kind,
        status="valid",
        error_code=None,
        source_sha256=source_sha256,
        schema_sha256=_schema_sha256(schema),
        input_rows=len(values),
        accepted_rows=len(accepted),
        outside_contract_rows=outside,
        post_decision_rows=post_decision,
        invalid_timestamp_rows=invalid,
        earliest_accepted_unix_ns=min(accepted) if accepted else None,
        latest_accepted_unix_ns=max(accepted) if accepted else None,
    )


def _invalid_source(
    *,
    source_file: str,
    contract: _SourceContract,
    source_sha256: str,
    error_code: SourceErrorCode,
) -> RCABenchSourceAudit:
    return RCABenchSourceAudit(
        source_file=source_file,
        modality=contract.modality,
        window=contract.window,
        source_kind=contract.source_kind,
        status="invalid",
        error_code=error_code,
        source_sha256=source_sha256,
        schema_sha256=None,
        input_rows=0,
        accepted_rows=0,
        outside_contract_rows=0,
        post_decision_rows=0,
        invalid_timestamp_rows=0,
        earliest_accepted_unix_ns=None,
        latest_accepted_unix_ns=None,
    )


def _missing_modalities(
    audits: tuple[RCABenchSourceAudit, ...],
) -> tuple[TelemetryModality, ...]:
    available = {item.modality for item in audits if item.status == "valid"}
    return tuple(
        sorted(
            modality
            for modality in _TELEMETRY_MODALITIES
            if modality not in available
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    fields = tuple((field.name, str(field.type)) for field in schema)
    return sha256_hex(canonical_json_bytes(fields))


def _timestamp_unit_factor(unit: str) -> int:
    try:
        return {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}[
            unit
        ]
    except KeyError as error:  # pragma: no cover - Arrow has a closed unit enum
        raise RCABenchAdapterError("unsupported timestamp unit") from error


# Public, dataset-neutral names used by adapters that share the same Parquet
# decision-boundary accounting.  The RCABench names remain the canonical
# implementation for schema compatibility in this first adapter release.
TelemetrySourceAudit = RCABenchSourceAudit
ParquetSourceContract = _SourceContract
build_parquet_source_contract = _source
audit_parquet_source = _audit_source
missing_telemetry_modalities = _missing_modalities
