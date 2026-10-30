"""RCABench preparation with a hard inference/evaluation boundary."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from nurse_rcl.schemas import (
    EvaluationCase,
    EvaluationManifest,
    InferenceManifest,
    TelemetryModality,
)

_TELEMETRY_FILES: dict[str, TelemetryModality] = {
    "abnormal_logs.parquet": "logs",
    "abnormal_metrics.parquet": "metrics",
    "abnormal_metrics_histogram.parquet": "metrics",
    "abnormal_metrics_sum.parquet": "metrics",
    "abnormal_traces.parquet": "fault_traces",
    "normal_logs.parquet": "logs",
    "normal_metrics.parquet": "metrics",
    "normal_metrics_histogram.parquet": "metrics",
    "normal_metrics_sum.parquet": "metrics",
    "normal_traces.parquet": "normal_traces",
}
_SERVICE_SEPARATOR = re.compile(r"[-_\s]+")


def canonicalize_service_name(raw: str) -> str:
    """Return a stable service identifier without changing service semantics."""

    normalized = unicodedata.normalize("NFKC", raw).strip().casefold()
    canonical = _SERVICE_SEPARATOR.sub("-", normalized).strip("-")
    if not canonical:
        raise ValueError("service name must contain a non-separator character")
    return canonical


def build_inference_manifest(dataset_root: Path) -> InferenceManifest:
    """Build the telemetry-only view without opening any label-bearing file."""

    root = _validated_root(dataset_root)
    cases: list[dict[str, object]] = []
    for case_path in _finished_cases(root):
        telemetry = _telemetry_payload(case_path)
        cases.append(
            {
                "case_id": case_path.name,
                "dataset": "rcabench",
                "t0": _read_t0(case_path / "env.json"),
                "telemetry": telemetry,
            }
        )

    return InferenceManifest.model_validate(
        {
            "schema_version": 1,
            "dataset": "rcabench",
            "method_version": "paper_v1",
            "cases": cases,
        },
        context={"allowed_roots": [root]},
    )


def build_evaluation_manifest(dataset_root: Path) -> EvaluationManifest:
    """Build label references for the isolated post-seal evaluator."""

    root = _validated_root(dataset_root)
    cases: list[EvaluationCase] = []
    for case_path in _finished_cases(root):
        label_path = case_path / "injection.json"
        if not label_path.is_file():
            raise FileNotFoundError(f"{case_path.name}: missing injection.json")
        cases.append(
            EvaluationCase(
                case_id=case_path.name,
                label_path=label_path.resolve(),
                label_field="ground_truth.service",
            )
        )
    return EvaluationManifest(dataset="rcabench", cases=tuple(cases))


def _validated_root(dataset_root: Path) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def _finished_cases(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and (path / ".finished").is_file()
            ),
            key=lambda path: path.name,
        )
    )


def _read_t0(env_path: Path) -> int:
    if not env_path.is_file():
        raise FileNotFoundError(f"{env_path.parent.name}: missing env.json")
    with env_path.open(encoding="utf-8") as handle:
        env: object = json.load(handle)
    if not isinstance(env, dict) or "ABNORMAL_START" not in env:
        raise ValueError(f"{env_path.parent.name}: env.json has no ABNORMAL_START")
    raw_t0 = env["ABNORMAL_START"]
    if isinstance(raw_t0, bool) or not isinstance(raw_t0, int | str):
        raise ValueError(f"{env_path.parent.name}: ABNORMAL_START must be Unix seconds")
    try:
        t0 = int(raw_t0)
    except ValueError as exc:
        raise ValueError(
            f"{env_path.parent.name}: ABNORMAL_START must be Unix seconds"
        ) from exc
    if t0 < 0:
        raise ValueError(f"{env_path.parent.name}: ABNORMAL_START must be non-negative")
    return t0


def _telemetry_payload(case_path: Path) -> list[dict[str, object]]:
    missing = [name for name in _TELEMETRY_FILES if not (case_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{case_path.name}: missing required telemetry: {', '.join(missing)}"
        )
    return [
        {
            "modality": _TELEMETRY_FILES[name],
            "path": (case_path / name).resolve(),
        }
        for name in sorted(_TELEMETRY_FILES)
    ]
