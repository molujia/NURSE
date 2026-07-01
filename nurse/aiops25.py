"""AIOpsChallenge 2025 dataset helpers.

The helpers in this module keep inference-facing case descriptors separate from
answer keys. Raw adapters may use descriptor timing and raw paths, while labels
are only exposed through the explicit evaluation loader.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence


AIOPS25_DIR = "25\u6311\u6218\u8d5b"
AIOPS25_RELATIVE_ROOT = ("datasets", AIOPS25_DIR, "aiopschallenge2025-main")
AIOPS25_PUBLIC_RELATIVE_ROOT = ("datasets", "AIOps25", "aiopschallenge2025-main")


def _join_relative(root: Path, parts: Sequence[str]) -> Path:
    result = root
    for part in parts:
        result = result / part
    return result

def aiops25_root(lab_root: str | Path) -> Path:
    root = Path(lab_root)
    candidates = (
        _join_relative(root, AIOPS25_RELATIVE_ROOT),
        _join_relative(root, AIOPS25_PUBLIC_RELATIVE_ROOT),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def aiops25_case_index_root(lab_root: str | Path) -> Path:
    return _repo_root() / "outputs" / "experiments" / "aiops25_case_index"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_aiops25_groundtruth_map(lab_root: str | Path) -> dict[str, dict]:
    root = aiops25_root(lab_root)
    return {
        str(row.get("uuid")): row
        for row in _read_jsonl(root / "groundtruth.jsonl")
        if row.get("uuid")
    }


def _parse_utc(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_descriptor(row: dict, root: Path, dataset: str = "aiops25") -> dict:
    start = _parse_utc(row.get("start_time"))
    end = _parse_utc(row.get("end_time"))
    return {
        "source_layout": "AIOps25",
        "dataset": dataset,
        "uuid": str(row.get("uuid")),
        "case_id": str(row.get("uuid")),
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "end_time": end.isoformat().replace("+00:00", "Z"),
        "raw_paths": {
            "root": str(root),
            "metrics_glob": "*/metric-parquet/**/*.parquet",
            "logs_glob": "*/log-parquet/*.parquet",
            "traces_glob": "*/trace-parquet/*.parquet",
            "pod_instance_mapping": str(root / "pod_instance_mapping.json"),
        },
        "window": {
            "trigger_time": start.isoformat().replace("+00:00", "Z"),
        },
    }


def _aiops25_retained_manifest_root(lab_root: str | Path, dataset: str) -> Path:
    if dataset == "aiops25-retained-nonnode":
        return _repo_root() / "outputs" / "experiments" / "aiops25_retained_nonnode_manifest"
    safe_name = dataset.replace("-", "_")
    return _repo_root() / "outputs" / "experiments" / f"{safe_name}_manifest"


def _include_aiops25_row(row: dict, exclude_instance_types: Sequence[str]) -> bool:
    excluded = {str(item).strip().lower() for item in exclude_instance_types if str(item).strip()}
    instance_type = str(row.get("instance_type") or "").strip().lower()
    return instance_type not in excluded


def _write_retained_manifest(
    lab_root: str | Path,
    *,
    dataset: str,
    source_dataset: str,
    rows: list[dict],
    exclude_instance_types: Sequence[str],
) -> None:
    output_root = _aiops25_retained_manifest_root(lab_root, dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    excluded = sorted({str(item).strip().lower() for item in exclude_instance_types if str(item).strip()})
    payload = {
        "dataset": dataset,
        "source_dataset": source_dataset,
        "excluded_instance_types": excluded,
        "case_count": len(rows),
        "cases": [
            {
                "case_id": str(row.get("uuid")),
                "source_dataset": source_dataset,
                "retention_reason": "excluded_node_faults" if excluded else "all_cases",
            }
            for row in rows
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_aiops25_case_index(
    lab_root: str | Path,
    *,
    dataset: str = "aiops25",
    exclude_instance_types: Sequence[str] = (),
) -> list[Path]:
    root = aiops25_root(lab_root)
    output_root = aiops25_case_index_root(lab_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in sorted(load_aiops25_groundtruth_map(lab_root).values(), key=lambda item: str(item.get("uuid")))
        if _include_aiops25_row(row, exclude_instance_types)
    ]
    paths: list[Path] = []
    for row in rows:
        case_id = str(row.get("uuid"))
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case.json").write_text(
            json.dumps(_safe_descriptor(row, root, dataset=dataset), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(case_dir)
    if dataset != "aiops25" or exclude_instance_types:
        _write_retained_manifest(
            lab_root,
            dataset=dataset,
            source_dataset="aiops25",
            rows=rows,
            exclude_instance_types=exclude_instance_types,
        )
    return paths


def aiops25_case_dirs(
    lab_root: str | Path,
    *,
    dataset: str = "aiops25",
    exclude_instance_types: Sequence[str] = (),
) -> list[Path]:
    return build_aiops25_case_index(
        lab_root,
        dataset=dataset,
        exclude_instance_types=exclude_instance_types,
    )


_POD_INDEX_RE = re.compile(r"^(?P<service>.+)-(?P<index>\d+)$")
_POD_HASH_RE = re.compile(r"^(?P<service>.+)-(?P<hash>[a-z0-9]{8,})-(?P<suffix>[a-z0-9]{4,8})$")


def aiops25_service_from_pod(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "null":
        return ""
    for pattern in (_POD_HASH_RE, _POD_INDEX_RE):
        match = pattern.match(text)
        if match:
            return match.group("service")
    return text


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def load_aiops25_labels(case_id: str, lab_root: str | Path) -> List[str]:
    row = load_aiops25_groundtruth_map(lab_root).get(str(case_id))
    if not row:
        return []
    source = str(row.get("source") or "").strip()
    destination = str(row.get("destination") or "").strip()
    if source or destination:
        return _unique([source, destination])

    instance_type = str(row.get("instance_type") or "").lower()
    service = str(row.get("service") or "").strip()
    instance = row.get("instance")
    if instance_type == "node":
        if isinstance(instance, list):
            return _unique(str(item) for item in instance)
        return _unique([str(instance or "")])
    if service:
        return _unique([service])
    if isinstance(instance, list):
        return _unique(aiops25_service_from_pod(item) for item in instance)
    return _unique([aiops25_service_from_pod(instance)])


def date_dirs_for_window(root: Path, start: datetime, end: datetime) -> list[Path]:
    """Return local-date directories that may overlap a UTC telemetry window."""
    cst = timezone(timedelta(hours=8))
    local_start = start.astimezone(cst).date()
    local_end = end.astimezone(cst).date()
    days = (local_end - local_start).days
    dirs = []
    for offset in range(days + 1):
        path = root / (local_start + timedelta(days=offset)).isoformat()
        if path.exists():
            dirs.append(path)
    return dirs


