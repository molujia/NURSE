"""Strict Parquet normalization for fixed paper V1 case telemetry."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.alarm import (
    canonicalize_service_name,
    canonicalize_signal_name,
)
from nurse_rcl.schemas import (
    DatasetName,
    InferenceCase,
    StrictFrozenModel,
    TelemetryModality,
    TelemetryReference,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
WINDOW_SECONDS = 480
TELEMETRY_SCHEMA_VERSION: Literal[1] = 1

Window = Literal["normal", "fault"]
FileKind = Literal["metric", "metric_sum", "metric_histogram", "trace", "log"]
RejectionCode = Literal[
    "invalid_timestamp",
    "missing_metric",
    "missing_service",
    "missing_trace_identity",
    "non_finite_value",
    "out_of_window",
]
MetricKey = tuple[str, str, Window]


class RowRejectionCount(StrictFrozenModel):
    """Stable per-file count for one rejected-row reason."""

    code: RejectionCode
    count: int = Field(ge=1)


class TelemetryFileCoverage(StrictFrozenModel):
    """Schema identity and complete row accounting for one authorized file."""

    file_name: str = Field(min_length=1)
    modality: TelemetryModality
    path: Path
    schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    rejections: tuple[RowRejectionCount, ...]

    @model_validator(mode="after")
    def require_complete_row_accounting(self) -> Self:
        if self.accepted_rows + sum(item.count for item in self.rejections) != self.input_rows:
            raise ValueError("telemetry coverage must account for every input row")
        return self


class MetricSample(StrictFrozenModel):
    """One finite scalar observation at an exact UTC Unix nanosecond."""

    timestamp_ns: int
    value: float


class MetricSeries(StrictFrozenModel):
    """One canonical service-signal series split at the frozen case time."""

    service: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    baseline: tuple[MetricSample, ...]
    fault: tuple[MetricSample, ...]


class TelemetryTraceSpan(StrictFrozenModel):
    """Only the trace fields authorized to influence fixed V1 topology."""

    timestamp_ns: int
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_span_id: str | None
    service: str = Field(min_length=1)


class LogTelemetryAudit(StrictFrozenModel):
    """Counts only; log messages and levels never enter the normalized case."""

    normal_input_rows: int = Field(ge=0)
    normal_in_window_rows: int = Field(ge=0)
    fault_input_rows: int = Field(ge=0)
    fault_in_window_rows: int = Field(ge=0)


class CaseTelemetry(StrictFrozenModel):
    """Immutable, label-free input shared by all fixed V1 deterministic stages."""

    schema_version: Literal[1] = TELEMETRY_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    dataset: DatasetName
    t0: int = Field(ge=0)
    window_start_ns: int
    t0_ns: int
    window_end_ns: int
    metric_series: tuple[MetricSeries, ...]
    normal_spans: tuple[TelemetryTraceSpan, ...]
    fault_spans: tuple[TelemetryTraceSpan, ...]
    log_audit: LogTelemetryAudit
    coverage: tuple[TelemetryFileCoverage, ...]
    telemetry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"telemetry_hash"})
        return self.telemetry_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class _FileContract:
    modality: TelemetryModality
    window: Window
    kind: FileKind

    @property
    def required_columns(self) -> tuple[str, ...]:
        if self.kind == "metric" or self.kind == "metric_sum":
            return "time", "service_name", "metric", "value"
        if self.kind == "metric_histogram":
            return "time", "service_name", "metric", "count", "sum", "min", "max"
        if self.kind == "trace":
            return "time", "trace_id", "span_id", "parent_span_id", "service_name"
        return "time", "service_name"


def _contract(
    modality: TelemetryModality,
    window: Window,
    kind: FileKind,
) -> _FileContract:
    return _FileContract(modality=modality, window=window, kind=kind)


_AIOPS25_CONTRACTS: dict[str, _FileContract] = {
    "normal_metrics.parquet": _contract("metrics", "normal", "metric"),
    "abnormal_metrics.parquet": _contract("metrics", "fault", "metric"),
    "normal_traces.parquet": _contract("normal_traces", "normal", "trace"),
    "abnormal_traces.parquet": _contract("fault_traces", "fault", "trace"),
    "normal_logs.parquet": _contract("logs", "normal", "log"),
    "abnormal_logs.parquet": _contract("logs", "fault", "log"),
}
_RCABENCH_CONTRACTS: dict[str, _FileContract] = {
    **_AIOPS25_CONTRACTS,
    "normal_metrics_sum.parquet": _contract("metrics", "normal", "metric_sum"),
    "abnormal_metrics_sum.parquet": _contract("metrics", "fault", "metric_sum"),
    "normal_metrics_histogram.parquet": _contract(
        "metrics", "normal", "metric_histogram"
    ),
    "abnormal_metrics_histogram.parquet": _contract(
        "metrics", "fault", "metric_histogram"
    ),
}


def load_case_telemetry(case: InferenceCase) -> CaseTelemetry:
    """Read one authorized inference case into the fixed immutable telemetry view."""

    if not isinstance(case, InferenceCase):
        raise TypeError("CaseTelemetry loading accepts only an InferenceCase")
    if case.t0 < 0:
        raise ValueError("case t0 must be non-negative")

    contracts = _contracts(case.dataset)
    references = tuple(
        sorted(case.telemetry, key=lambda item: (item.path.name, item.path.as_posix()))
    )
    names = tuple(reference.path.name for reference in references)
    if len(names) != len(set(names)):
        raise ValueError("telemetry filenames must be unique within one case")

    t0_ns = case.t0 * NANOSECONDS_PER_SECOND
    start_ns = (case.t0 - WINDOW_SECONDS) * NANOSECONDS_PER_SECOND
    end_ns = (case.t0 + WINDOW_SECONDS) * NANOSECONDS_PER_SECOND
    metric_points: defaultdict[MetricKey, list[MetricSample]] = defaultdict(list)
    normal_spans: list[TelemetryTraceSpan] = []
    fault_spans: list[TelemetryTraceSpan] = []
    coverage: list[TelemetryFileCoverage] = []
    log_input: Counter[Window] = Counter()
    log_accepted: Counter[Window] = Counter()

    for reference in references:
        contract = contracts.get(reference.path.name)
        if contract is None:
            raise ValueError(
                f"unsupported {case.dataset} telemetry filename: {reference.path.name}"
            )
        if reference.modality != contract.modality:
            raise ValueError(
                f"{reference.path.name}: modality {reference.modality!r} does not match "
                f"the fixed {contract.modality!r} contract"
            )
        table, schema_hash = _read_contract_table(reference, contract)
        rejections: Counter[RejectionCode] = Counter()
        if contract.kind in {"metric", "metric_sum", "metric_histogram"}:
            accepted = _normalize_metric_rows(
                table=table,
                contract=contract,
                t0_ns=t0_ns,
                start_ns=start_ns,
                end_ns=end_ns,
                output=metric_points,
                rejections=rejections,
            )
        elif contract.kind == "trace":
            target = normal_spans if contract.window == "normal" else fault_spans
            accepted = _normalize_trace_rows(
                table=table,
                contract=contract,
                t0_ns=t0_ns,
                start_ns=start_ns,
                end_ns=end_ns,
                output=target,
                rejections=rejections,
            )
        else:
            log_input[contract.window] += table.num_rows
            accepted = _audit_log_rows(
                table=table,
                contract=contract,
                t0_ns=t0_ns,
                start_ns=start_ns,
                end_ns=end_ns,
                rejections=rejections,
            )
            log_accepted[contract.window] += accepted
        coverage.append(
            TelemetryFileCoverage(
                file_name=reference.path.name,
                modality=reference.modality,
                path=reference.path,
                schema_hash=schema_hash,
                input_rows=table.num_rows,
                accepted_rows=accepted,
                rejections=_rejection_records(rejections),
            )
        )

    series = _metric_series(metric_points)
    ordered_normal_spans = tuple(sorted(normal_spans, key=_span_order))
    ordered_fault_spans = tuple(sorted(fault_spans, key=_span_order))
    log_audit = LogTelemetryAudit(
        normal_input_rows=log_input["normal"],
        normal_in_window_rows=log_accepted["normal"],
        fault_input_rows=log_input["fault"],
        fault_in_window_rows=log_accepted["fault"],
    )
    unsigned = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "case_id": case.case_id,
        "dataset": case.dataset,
        "t0": case.t0,
        "window_start_ns": start_ns,
        "t0_ns": t0_ns,
        "window_end_ns": end_ns,
        "metric_series": [item.model_dump(mode="json") for item in series],
        "normal_spans": [item.model_dump(mode="json") for item in ordered_normal_spans],
        "fault_spans": [item.model_dump(mode="json") for item in ordered_fault_spans],
        "log_audit": log_audit.model_dump(mode="json"),
        "coverage": [item.model_dump(mode="json") for item in coverage],
    }
    result = CaseTelemetry.model_validate(
        {
            **unsigned,
            "telemetry_hash": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("CaseTelemetry hash verification failed")
    return result


def _contracts(dataset: DatasetName) -> Mapping[str, _FileContract]:
    if dataset == "rcabench":
        return _RCABENCH_CONTRACTS
    return _AIOPS25_CONTRACTS


def _read_contract_table(
    reference: TelemetryReference,
    contract: _FileContract,
) -> tuple[pa.Table, str]:
    path = reference.path
    if path.is_symlink():
        raise ValueError(f"{path.name}: telemetry symlinks are forbidden")
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = pq.read_schema(path)
    missing = tuple(name for name in contract.required_columns if name not in schema.names)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {', '.join(missing)}")
    _validate_column_types(path.name, schema, contract)
    table = pq.read_table(path, columns=list(contract.required_columns))
    schema_hash = sha256_hex(str(schema).encode("utf-8"))
    return table, schema_hash


def _validate_column_types(
    file_name: str,
    schema: pa.Schema,
    contract: _FileContract,
) -> None:
    time_type = schema.field("time").type
    if not pa.types.is_timestamp(time_type):
        raise ValueError(f"{file_name}: time must be an Arrow timestamp")
    timestamp_type = cast(pa.TimestampType, time_type)
    if timestamp_type.unit != "ns" or timestamp_type.tz != "UTC":
        raise ValueError(f"{file_name}: time must be timestamp[ns, tz=UTC]")

    text_columns = set(contract.required_columns) - {
        "time",
        "value",
        "count",
        "sum",
        "min",
        "max",
    }
    for name in sorted(text_columns):
        data_type = schema.field(name).type
        if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
            raise ValueError(f"{file_name}: {name} must be string or large_string")
    numeric_columns = set(contract.required_columns) & {
        "value",
        "count",
        "sum",
        "min",
        "max",
    }
    for name in sorted(numeric_columns):
        data_type = schema.field(name).type
        if not (pa.types.is_integer(data_type) or pa.types.is_floating(data_type)):
            raise ValueError(f"{file_name}: {name} must be numeric")


def _normalize_metric_rows(
    *,
    table: pa.Table,
    contract: _FileContract,
    t0_ns: int,
    start_ns: int,
    end_ns: int,
    output: defaultdict[MetricKey, list[MetricSample]],
    rejections: Counter[RejectionCode],
) -> int:
    times = _timestamp_values(table)
    services = _column_values(table, "service_name")
    metrics = _column_values(table, "metric")
    value_columns = (
        ("value", "")
        if contract.kind == "metric"
        else ("value", ".sum")
        if contract.kind == "metric_sum"
        else None
    )
    if value_columns is None:
        scalar_columns = tuple(
            (name, f".histogram-{name}") for name in ("count", "sum", "min", "max")
        )
    else:
        scalar_columns = (value_columns,)
    scalars = {name: _column_values(table, name) for name, _ in scalar_columns}

    accepted = 0
    for index, timestamp_ns in enumerate(times):
        rejection = _common_rejection(
            timestamp_ns=timestamp_ns,
            service=services[index],
            window=contract.window,
            t0_ns=t0_ns,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if rejection is not None:
            rejections[rejection] += 1
            continue
        metric = _non_empty_text(metrics[index])
        if metric is None:
            rejections["missing_metric"] += 1
            continue
        values = tuple(_finite_number(scalars[name][index]) for name, _ in scalar_columns)
        if any(value is None for value in values):
            rejections["non_finite_value"] += 1
            continue
        service = canonicalize_service_name(cast(str, services[index]))
        for (_, suffix), numeric in zip(scalar_columns, values, strict=True):
            signal = canonicalize_signal_name(f"{metric}{suffix}")
            output[(service, signal, contract.window)].append(
                MetricSample(timestamp_ns=cast(int, timestamp_ns), value=cast(float, numeric))
            )
        accepted += 1
    return accepted


def _normalize_trace_rows(
    *,
    table: pa.Table,
    contract: _FileContract,
    t0_ns: int,
    start_ns: int,
    end_ns: int,
    output: list[TelemetryTraceSpan],
    rejections: Counter[RejectionCode],
) -> int:
    times = _timestamp_values(table)
    services = _column_values(table, "service_name")
    trace_ids = _column_values(table, "trace_id")
    span_ids = _column_values(table, "span_id")
    parent_ids = _column_values(table, "parent_span_id")
    accepted = 0
    for index, timestamp_ns in enumerate(times):
        rejection = _common_rejection(
            timestamp_ns=timestamp_ns,
            service=services[index],
            window=contract.window,
            t0_ns=t0_ns,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if rejection is not None:
            rejections[rejection] += 1
            continue
        trace_id = _non_empty_text(trace_ids[index])
        span_id = _non_empty_text(span_ids[index])
        if trace_id is None or span_id is None:
            rejections["missing_trace_identity"] += 1
            continue
        parent_raw = parent_ids[index]
        parent_span_id = _non_empty_text(parent_raw) if parent_raw is not None else None
        output.append(
            TelemetryTraceSpan(
                timestamp_ns=cast(int, timestamp_ns),
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                service=canonicalize_service_name(cast(str, services[index])),
            )
        )
        accepted += 1
    return accepted


def _audit_log_rows(
    *,
    table: pa.Table,
    contract: _FileContract,
    t0_ns: int,
    start_ns: int,
    end_ns: int,
    rejections: Counter[RejectionCode],
) -> int:
    times = _timestamp_values(table)
    services = _column_values(table, "service_name")
    accepted = 0
    for index, timestamp_ns in enumerate(times):
        rejection = _common_rejection(
            timestamp_ns=timestamp_ns,
            service=services[index],
            window=contract.window,
            t0_ns=t0_ns,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if rejection is None:
            accepted += 1
        else:
            rejections[rejection] += 1
    return accepted


def _common_rejection(
    *,
    timestamp_ns: object,
    service: object,
    window: Window,
    t0_ns: int,
    start_ns: int,
    end_ns: int,
) -> RejectionCode | None:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        return "invalid_timestamp"
    if window == "normal":
        if not start_ns <= timestamp_ns < t0_ns:
            return "out_of_window"
    elif not t0_ns <= timestamp_ns < end_ns:
        return "out_of_window"
    if _non_empty_text(service) is None:
        return "missing_service"
    return None


def _timestamp_values(table: pa.Table) -> tuple[object, ...]:
    values = table.column("time").cast(pa.int64()).to_pylist()
    return tuple(cast(list[object], values))


def _column_values(table: pa.Table, name: str) -> tuple[object, ...]:
    values = table.column(name).to_pylist()
    return tuple(cast(list[object], values))


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _metric_series(
    points: Mapping[MetricKey, list[MetricSample]],
) -> tuple[MetricSeries, ...]:
    pairs = sorted({(service, signal) for service, signal, _ in points})
    return tuple(
        MetricSeries(
            service=service,
            signal=signal,
            baseline=tuple(
                sorted(points.get((service, signal, "normal"), ()), key=_sample_order)
            ),
            fault=tuple(
                sorted(points.get((service, signal, "fault"), ()), key=_sample_order)
            ),
        )
        for service, signal in pairs
    )


def _sample_order(sample: MetricSample) -> tuple[int, float]:
    return sample.timestamp_ns, sample.value


def _span_order(span: TelemetryTraceSpan) -> tuple[int, str, str, str, str]:
    return (
        span.timestamp_ns,
        span.trace_id,
        span.span_id,
        span.service,
        span.parent_span_id or "",
    )


def _rejection_records(
    counts: Mapping[RejectionCode, int],
) -> tuple[RowRejectionCount, ...]:
    return tuple(
        RowRejectionCount(code=code, count=count)
        for code, count in sorted(counts.items())
        if count
    )
