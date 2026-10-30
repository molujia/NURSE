"""Fingerprint and validate the mandated AIOps25 case preparation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import (
    EvaluationCase,
    EvaluationManifest,
    InferenceManifest,
    StrictFrozenModel,
    TelemetryModality,
)

MANDATED_DATA_ROOT = Path("/home/wangrunzhou/25挑战赛/aiopschallenge2025-main")
MANDATED_PREPROCESSOR = Path(
    "/home/wangrunzhou/25挑战赛/reproduction_hd4_baselines/hd4_baseline_repro.py"
)
MANDATED_PREPROCESSOR_SHA256 = (
    "1b3e8643ccd8c252037adda2739240d67897edd1343cbc7fe4554ef9fa0410c1"
)
PREPARED_TELEMETRY = (
    "abnormal_logs.parquet",
    "abnormal_metrics.parquet",
    "abnormal_traces.parquet",
    "normal_logs.parquet",
    "normal_metrics.parquet",
    "normal_traces.parquet",
)
_AIOPS25_MODALITIES: dict[str, TelemetryModality] = {
    "abnormal_logs.parquet": "logs",
    "abnormal_metrics.parquet": "metrics",
    "abnormal_traces.parquet": "fault_traces",
    "normal_logs.parquet": "logs",
    "normal_metrics.parquet": "metrics",
    "normal_traces.parquet": "normal_traces",
}
_ROW_KEYS = tuple(filename.removesuffix(".parquet") for filename in PREPARED_TELEMETRY)
_REQUIRED_ENV_KEYS = (
    "NORMAL_START",
    "NORMAL_END",
    "ABNORMAL_START",
    "ABNORMAL_END",
    "ORIGINAL_ABNORMAL_END",
)
_RECEIPT_NAME = "nurse_preparation_receipt.json"

CommandRunner = Callable[[tuple[str, ...]], None]


@dataclass(frozen=True)
class PreparationConfig:
    """Frozen authority and locations for one preparation invocation."""

    data_root: Path
    work_root: Path
    preprocessor: Path
    approved_preprocessor: Path
    expected_preprocessor_sha256: str
    python_executable: Path


class PreparationReceipt(StrictFrozenModel):
    """Label-free fingerprints proving which converter produced which cases."""

    schema_version: Literal[1] = 1
    data_root: Path
    work_root: Path
    preprocessor: Path
    preprocessor_sha256: str
    source_fingerprint: str
    source_file_count: int
    prepared_fingerprint: str
    prepared_file_count: int
    prepared_case_count: int
    command: tuple[str, ...]


def server_preparation_config(
    *, work_root: Path, python_executable: Path
) -> PreparationConfig:
    """Return the fixed server policy for the user-mandated converter."""

    return PreparationConfig(
        data_root=MANDATED_DATA_ROOT,
        work_root=work_root,
        preprocessor=MANDATED_PREPROCESSOR,
        approved_preprocessor=MANDATED_PREPROCESSOR,
        expected_preprocessor_sha256=MANDATED_PREPROCESSOR_SHA256,
        python_executable=python_executable,
    )


def build_aiops25_inference_manifest(work_root: Path) -> InferenceManifest:
    """Build inference cases without opening label manifests or injections."""

    root = Path(work_root).resolve(strict=True)
    canonical_root = root / "canonical"
    if not canonical_root.is_dir():
        raise FileNotFoundError("prepared output has no canonical directory")
    case_paths = tuple(
        sorted(
            path
            for path in canonical_root.iterdir()
            if path.is_dir() and (path / ".prepared.json").is_file()
        )
    )
    cases: list[dict[str, object]] = []
    for case_path in case_paths:
        telemetry = _aiops25_telemetry_payload(case_path)
        cases.append(
            {
                "case_id": case_path.name,
                "dataset": "aiops25",
                "t0": _aiops25_t0(case_path / "env.json"),
                "telemetry": telemetry,
            }
        )
    return InferenceManifest.model_validate(
        {
            "schema_version": 1,
            "dataset": "aiops25",
            "method_version": "paper_v1",
            "cases": cases,
        },
        context={"allowed_roots": [canonical_root]},
    )


def build_aiops25_evaluation_manifest(work_root: Path) -> EvaluationManifest:
    """Build service-level label references and explicit node exclusions."""

    root = Path(work_root).resolve(strict=True)
    manifest = _load_json_object(root / "manifests" / "manifest.json")
    included_ids = _included_case_ids(manifest)
    excluded_ids = _excluded_node_case_ids(manifest)

    cases: list[EvaluationCase] = []
    for case_id in included_ids:
        label_path = root / "canonical" / case_id / "injection.json"
        if not label_path.is_file():
            raise FileNotFoundError(f"{case_id}: missing injection.json")
        cases.append(
            EvaluationCase(
                case_id=case_id,
                label_path=label_path.resolve(),
                label_field="target_services",
            )
        )
    cases.extend(
        EvaluationCase(
            case_id=case_id,
            excluded=True,
            exclusion_reason="node-level case",
        )
        for case_id in excluded_ids
    )
    return EvaluationManifest(dataset="aiops25", cases=tuple(cases))


def prepare_aiops25(
    config: PreparationConfig,
    *,
    run_command: CommandRunner | None = None,
) -> PreparationReceipt:
    """Run only the approved prepare mode, or reuse an intact prior receipt."""

    data_root, work_root, preprocessor, script_digest = _validate_config(config)
    source_digest, source_file_count = _fingerprint_files(
        data_root,
        (path for path in data_root.rglob("*") if path.is_file()),
    )
    command = _prepare_command(config)
    receipt_path = work_root / "manifests" / _RECEIPT_NAME

    if receipt_path.is_file():
        receipt = PreparationReceipt.model_validate_json(receipt_path.read_bytes())
        prepared = _validate_prepared_output(work_root)
        prepared_digest, prepared_file_count = _fingerprint_prepared_output(
            work_root, prepared
        )
        expected = _receipt(
            data_root=data_root,
            work_root=work_root,
            preprocessor=preprocessor,
            script_digest=script_digest,
            source_digest=source_digest,
            source_file_count=source_file_count,
            prepared_digest=prepared_digest,
            prepared_file_count=prepared_file_count,
            prepared_case_count=len(prepared),
            command=command,
        )
        if receipt != expected:
            raise ValueError("AIOps25 preparation receipt does not match current inputs")
        return receipt

    runner = _run_subprocess if run_command is None else run_command
    runner(command)
    prepared = _validate_prepared_output(work_root)
    prepared_digest, prepared_file_count = _fingerprint_prepared_output(work_root, prepared)
    receipt = _receipt(
        data_root=data_root,
        work_root=work_root,
        preprocessor=preprocessor,
        script_digest=script_digest,
        source_digest=source_digest,
        source_file_count=source_file_count,
        prepared_digest=prepared_digest,
        prepared_file_count=prepared_file_count,
        prepared_case_count=len(prepared),
        command=command,
    )
    atomic_write_bytes(
        receipt_path,
        canonical_json_bytes(receipt.model_dump(mode="json")),
    )
    return receipt


def _aiops25_t0(env_path: Path) -> int:
    env = _load_json_object(env_path)
    value = env.get("ABNORMAL_START")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{env_path.parent.name}: invalid ABNORMAL_START")
    return value


def _aiops25_telemetry_payload(case_path: Path) -> list[dict[str, object]]:
    missing = [name for name in PREPARED_TELEMETRY if not (case_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{case_path.name}: missing required telemetry: {', '.join(missing)}"
        )
    return [
        {
            "modality": _AIOPS25_MODALITIES[name],
            "path": (case_path / name).resolve(),
        }
        for name in sorted(PREPARED_TELEMETRY)
    ]


def _included_case_ids(manifest: dict[str, object]) -> tuple[str, ...]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("AIOps25 manifest cases must be an array")
    identifiers: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("AIOps25 manifest case must be an object")
        case_id = case.get("uuid")
        instance_type = case.get("instance_type")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("AIOps25 manifest case has invalid uuid")
        if instance_type not in {"service", "pod"}:
            raise ValueError(f"{case_id}: included case is not service-level")
        identifiers.append(case_id)
    return tuple(sorted(identifiers))


def _excluded_node_case_ids(manifest: dict[str, object]) -> tuple[str, ...]:
    values = manifest.get("excluded_node_case_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError("AIOps25 excluded_node_case_ids must be a string array")
    return tuple(sorted(cast(list[str], values)))


def _validate_config(config: PreparationConfig) -> tuple[Path, Path, Path, str]:
    data_root = config.data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    preprocessor = config.preprocessor.resolve(strict=True)
    approved = config.approved_preprocessor.resolve(strict=True)
    if preprocessor != approved:
        raise ValueError("preprocessor does not match the approved preprocessor path")
    script_digest = _sha256_file(preprocessor)
    if script_digest != config.expected_preprocessor_sha256.casefold():
        raise ValueError("preprocessor SHA-256 does not match the approved fingerprint")
    work_root = config.work_root.resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    return data_root, work_root, preprocessor, script_digest


def _prepare_command(config: PreparationConfig) -> tuple[str, ...]:
    return (
        str(config.python_executable),
        str(config.preprocessor),
        "--data-root",
        str(config.data_root),
        "--work-root",
        str(config.work_root),
        "--mode",
        "prepare",
    )


def _run_subprocess(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


def _validate_prepared_output(work_root: Path) -> tuple[Path, ...]:
    manifest = work_root / "manifests" / "manifest.json"
    _load_json_object(manifest)
    canonical_root = work_root / "canonical"
    if not canonical_root.is_dir():
        raise FileNotFoundError("prepared output has no canonical directory")
    case_paths = tuple(sorted(path for path in canonical_root.iterdir() if path.is_dir()))
    if not case_paths:
        raise ValueError("prepared output contains no cases")
    for case_path in case_paths:
        _validate_prepared_case(case_path)
    return case_paths


def _validate_prepared_case(case_path: Path) -> None:
    marker = _load_json_object(case_path / ".prepared.json")
    if marker.get("schema_version") != 1 or marker.get("case") != case_path.name:
        raise ValueError(f"{case_path.name}: invalid .prepared.json identity")
    rows = marker.get("rows")
    if not isinstance(rows, dict):
        raise ValueError(f"{case_path.name}: .prepared.json rows must be an object")
    for row_key in _ROW_KEYS:
        value = rows.get(row_key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{case_path.name}: invalid row count for {row_key}")

    env = _load_json_object(case_path / "env.json")
    for key in _REQUIRED_ENV_KEYS:
        value = env.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{case_path.name}: env.json has invalid {key}")

    injection_path = case_path / "injection.json"
    if not injection_path.is_file() or injection_path.stat().st_size == 0:
        raise FileNotFoundError(f"{case_path.name}: missing injection.json")
    for filename in PREPARED_TELEMETRY:
        path = case_path / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{case_path.name}: missing {filename}")


def _fingerprint_prepared_output(
    work_root: Path, case_paths: tuple[Path, ...]
) -> tuple[str, int]:
    files = [work_root / "manifests" / "manifest.json"]
    for case_path in case_paths:
        files.extend(
            [
                case_path / ".prepared.json",
                case_path / "env.json",
                case_path / "injection.json",
                *(case_path / filename for filename in PREPARED_TELEMETRY),
            ]
        )
    return _fingerprint_files(work_root, files)


def _fingerprint_files(root: Path, files: Iterable[Path]) -> tuple[str, int]:
    records = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"fingerprinted data cannot contain symlinks: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"cannot fingerprint empty data view: {root}")
    return sha256_hex(canonical_json_bytes(records)), len(records)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _receipt(
    *,
    data_root: Path,
    work_root: Path,
    preprocessor: Path,
    script_digest: str,
    source_digest: str,
    source_file_count: int,
    prepared_digest: str,
    prepared_file_count: int,
    prepared_case_count: int,
    command: tuple[str, ...],
) -> PreparationReceipt:
    return PreparationReceipt(
        data_root=data_root,
        work_root=work_root,
        preprocessor=preprocessor,
        preprocessor_sha256=script_digest,
        source_fingerprint=source_digest,
        source_file_count=source_file_count,
        prepared_fingerprint=prepared_digest,
        prepared_file_count=prepared_file_count,
        prepared_case_count=prepared_case_count,
        command=command,
    )
