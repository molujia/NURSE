"""Raw telemetry adapters for the public NURSE datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .aiops25 import aiops25_service_from_pod, date_dirs_for_window
from .matching import canonical_entity
from .schemas import _assert_no_forbidden


Series = List[Tuple[Union[int, float], float]]


@dataclass
class RawIncident:
    dataset: str
    case_id: str
    case_dir: Optional[Path]
    available_modalities: List[str]
    file_index: Dict[str, Path]
    metric_series: Dict[str, Series] = field(default_factory=dict)
    topology_edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_no_forbidden(self.metadata)


def load_raw_incident(
    case_dir: str | Path,
    dataset: str | None = None,
    *,
    window_width_seconds: int = 480,
) -> RawIncident:
    path = Path(case_dir)
    if (path / "case.json").exists():
        descriptor = _load_json(path / "case.json")
        if descriptor.get("source_layout") == "AIOps25":
            return _load_aiops25_case(
                path,
                dataset=dataset or "aiops25-retained-nonnode",
                window_width_seconds=window_width_seconds,
            )
    if (path / "metrics.json").exists():
        return _load_rcaeval_tt_case(path, dataset=dataset or "rcaeval-tt")
    raise ValueError(f"Unsupported public NURSE incident layout: {path}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_metric_key(metric_key: str) -> Tuple[str, str]:
    if "_" not in metric_key:
        return metric_key, "value"
    entity, metric = metric_key.rsplit("_", 1)
    return entity, metric


def _load_rcaeval_tt_case(path: Path, dataset: str = "rcaeval-tt") -> RawIncident:
    metrics = _load_json(path / "metrics.json")
    metric_series: Dict[str, Series] = {}
    for metric_key, points in metrics.items():
        entity, metric = _split_metric_key(str(metric_key))
        series: Series = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    series.append((int(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        if series:
            metric_series[f"{entity}.{metric}"] = series

    file_index = {"metrics": path / "metrics.json"}
    modalities = ["metrics"]
    for name in ("logs", "traces"):
        file_path = path / f"{name}.csv"
        if file_path.exists():
            file_index[name] = file_path
            modalities.append(name)

    return RawIncident(
        dataset=dataset,
        case_id=path.name,
        case_dir=path,
        available_modalities=modalities,
        file_index=file_index,
        metric_series=metric_series,
        topology_edges=[],
        metadata={"source_layout": "RCAEval-TT"},
    )


def _parse_aiops25_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_nullish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "null", "none", "nan"}


def _aiops25_entity_from_row(row: Any, file_path: Path) -> str:
    object_type = str(row.get("object_type", "") or "").strip().lower()
    object_id = str(row.get("object_id", "") or "").strip()
    pod = str(row.get("pod", "") or "").strip()
    instance = str(row.get("instance", "") or "").strip()
    node = str(row.get("kubernetes_node", "") or "").strip()
    if object_type == "node" or "infra_node" in str(file_path):
        for value in (node, instance):
            if not _is_nullish(value):
                return canonical_entity(value)
    if not _is_nullish(pod):
        service = aiops25_service_from_pod(pod)
        if service:
            return service
    if object_type == "pod" and not _is_nullish(object_id):
        return aiops25_service_from_pod(object_id)
    if object_type == "service" and not _is_nullish(object_id):
        return canonical_entity(object_id)
    if not _is_nullish(object_id):
        return aiops25_service_from_pod(object_id)
    if not _is_nullish(instance) and str(instance).startswith("aiops-k8s-"):
        return canonical_entity(instance)
    if not _is_nullish(object_type):
        return canonical_entity(object_type)
    return ""


def _aiops25_numeric_columns(frame: Any) -> list[str]:
    import pandas as pd  # type: ignore

    excluded = {
        "time",
        "cf",
        "device",
        "instance",
        "kpi_key",
        "kpi_name",
        "kubernetes_node",
        "mountpoint",
        "namespace",
        "object_id",
        "object_type",
        "pod",
        "sql_type",
        "type",
    }
    return [
        str(col)
        for col in frame.columns
        if str(col) not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _datetime_series_to_epoch_seconds(values: Any):
    raw = values.astype("int64")
    max_abs = float(raw.abs().max()) if len(raw) else 0.0
    if max_abs > 1e17:
        return raw / 1_000_000_000
    if max_abs > 1e14:
        return raw / 1_000_000
    if max_abs > 1e11:
        return raw / 1_000
    return raw.astype(float)


def _load_aiops25_metric_series(root: Path, start: datetime, end: datetime) -> tuple[Dict[str, Series], List[Tuple[str, str]], Dict[str, Any]]:
    import pandas as pd  # type: ignore

    metric_series: Dict[str, Series] = {}
    topology_edges: set[Tuple[str, str]] = set()
    files_read = 0
    rows_kept = 0
    errors: list[str] = []
    for day_dir in date_dirs_for_window(root, start, end):
        metric_root = day_dir / "metric-parquet"
        if not metric_root.exists():
            continue
        for file_path in sorted(metric_root.rglob("*.parquet")):
            try:
                frame = pd.read_parquet(file_path)
            except Exception as exc:  # pragma: no cover - depends on optional parquet files
                errors.append(f"{file_path.name}: {exc}")
                continue
            files_read += 1
            if frame.empty or "time" not in frame.columns:
                continue
            timestamps = pd.to_datetime(frame["time"], errors="coerce", utc=True)
            mask = (timestamps >= start) & (timestamps <= end)
            if not mask.any():
                continue
            selected = frame.loc[mask].copy()
            selected["__timestamp"] = _datetime_series_to_epoch_seconds(timestamps.loc[mask]).astype(float)
            numeric_cols = _aiops25_numeric_columns(selected)
            if not numeric_cols:
                continue
            for _, row in selected.iterrows():
                entity = _aiops25_entity_from_row(row, file_path)
                if not entity:
                    continue
                pod = str(row.get("pod", "") or "").strip()
                node = str(row.get("instance", "") or row.get("kubernetes_node", "") or "").strip()
                if pod and node and node.startswith("aiops-k8s-"):
                    topology_edges.add((entity, canonical_entity(node)))
                timestamp = float(row["__timestamp"])
                for metric in numeric_cols:
                    try:
                        value = float(row[metric])
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(value):
                        continue
                    metric_series.setdefault(f"{entity}.{metric}", []).append((timestamp, value))
                    rows_kept += 1
    for key, values in metric_series.items():
        metric_series[key] = sorted(values, key=lambda item: item[0])
    return metric_series, sorted(topology_edges), {
        "status": "loaded" if metric_series else "empty",
        "files_read": files_read,
        "series": len(metric_series),
        "points": rows_kept,
        "errors": errors[:20],
    }


def _aiops25_files_for_window(root: Path, start: datetime, end: datetime, subdir: str) -> list[Path]:
    files: list[Path] = []
    for day_dir in date_dirs_for_window(root, start, end):
        parent = day_dir / subdir
        if parent.exists():
            files.extend(sorted(parent.rglob("*.parquet")))
    return files


def _load_aiops25_case(
    path: Path,
    dataset: str = "aiops25-retained-nonnode",
    *,
    window_width_seconds: int = 480,
) -> RawIncident:
    descriptor = _load_json(path / "case.json")
    root = Path(descriptor["raw_paths"]["root"])
    start_time = _parse_aiops25_time(descriptor.get("start_time"))
    end_time = _parse_aiops25_time(descriptor.get("end_time"))
    if start_time is None:
        raise ValueError(f"AIOps25 descriptor missing start_time: {path}")
    if end_time is None:
        end_time = start_time
    width_seconds = max(int(window_width_seconds), 1)
    normal_start = start_time - timedelta(seconds=width_seconds)
    fault_end = start_time + timedelta(seconds=width_seconds)
    metric_series, topology_edges, metric_adapter = _load_aiops25_metric_series(root, normal_start, fault_end)
    log_files = _aiops25_files_for_window(root, normal_start, fault_end, "log-parquet")
    trace_files = _aiops25_files_for_window(root, normal_start, fault_end, "trace-parquet")
    file_index: Dict[str, Path] = {"case": path / "case.json", "metrics": root}
    modalities = ["case", "metrics"]
    if log_files:
        file_index["logs"] = log_files[0]
        modalities.append("logs")
    if trace_files:
        file_index["traces"] = trace_files[0]
        modalities.append("traces")
    metadata = {
        "source_layout": "AIOps25",
        "raw_root": str(root),
        "start_time": start_time.isoformat().replace("+00:00", "Z"),
        "end_time": end_time.isoformat().replace("+00:00", "Z"),
        "benchmark_trigger_time": start_time.isoformat().replace("+00:00", "Z"),
        "metric_adapter": metric_adapter,
        "log_files": [str(item) for item in log_files],
        "trace_files": [str(item) for item in trace_files],
        "selected_date_dirs": [str(item) for item in date_dirs_for_window(root, normal_start, fault_end)],
        "window_width_seconds": width_seconds,
        "normal_window_seconds": width_seconds,
        "fault_window_seconds": width_seconds,
    }
    return RawIncident(
        dataset=dataset,
        case_id=str(descriptor.get("case_id") or descriptor.get("uuid") or path.name),
        case_dir=path,
        available_modalities=modalities,
        file_index=file_index,
        metric_series=metric_series,
        topology_edges=topology_edges,
        metadata=metadata,
    )
