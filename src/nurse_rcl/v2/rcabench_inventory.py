"""Server-side RCABench filesystem inventory without telemetry download."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

RCABENCH_INVENTORY_SCHEMA_VERSION: Literal[
    "nurse.v2.rcabench-inventory/1"
] = "nurse.v2.rcabench-inventory/1"
RCABENCH_EXPECTED_CASES = 1_422
RCABENCH_PARQUET_FILES = (
    "abnormal_logs.parquet",
    "abnormal_metrics.parquet",
    "abnormal_metrics_histogram.parquet",
    "abnormal_metrics_sum.parquet",
    "abnormal_traces.parquet",
    "normal_logs.parquet",
    "normal_metrics.parquet",
    "normal_metrics_histogram.parquet",
    "normal_metrics_sum.parquet",
    "normal_traces.parquet",
)
RCABENCH_REQUIRED_FILES = (
    ".finished",
    *RCABENCH_PARQUET_FILES,
    "env.json",
    "injection.json",
)

ParquetProbe = Callable[[Path], None]
InventoryStatus = Literal["valid", "corrupt", "duplicate", "excluded"]
RootEntryKind = Literal["directory", "file", "symlink", "other"]


class RCABenchInventoryEntry(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    normalized_case_id: str = Field(min_length=1)
    status: InventoryStatus
    reasons: tuple[str, ...]
    duplicate_of: str | None = None
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("inventory reasons must be sorted and unique")
        if self.status == "valid" and self.reasons:
            raise ValueError("valid inventory entries cannot have failure reasons")
        if self.status != "valid" and not self.reasons:
            raise ValueError("non-valid inventory entries require a reason")
        if (self.status == "duplicate") != (self.duplicate_of is not None):
            raise ValueError("only duplicate entries declare duplicate_of")
        return self


class ExcludedRootEntry(StrictFrozenModel):
    name: str = Field(min_length=1)
    kind: RootEntryKind
    reason: str = Field(min_length=1)


class RCABenchInventoryReport(StrictFrozenModel):
    schema_version: Literal["nurse.v2.rcabench-inventory/1"] = (
        RCABENCH_INVENTORY_SCHEMA_VERSION
    )
    dataset: Literal["rcabench"] = "rcabench"
    root: str = Field(min_length=1)
    expected_cases: int = Field(ge=0)
    discovered_case_directories: int = Field(ge=0)
    valid_cases: int = Field(ge=0)
    corrupt_cases: int = Field(ge=0)
    duplicate_cases: int = Field(ge=0)
    excluded_case_directories: int = Field(ge=0)
    classified_case_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[RCABenchInventoryEntry, ...]
    excluded_root_entries: tuple[ExcludedRootEntry, ...]
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def acceptable(self) -> bool:
        return (
            self.discovered_case_directories == self.expected_cases
            and self.valid_cases == self.expected_cases
            and self.corrupt_cases == 0
            and self.duplicate_cases == 0
            and self.excluded_case_directories == 0
        )

    def verify(self) -> bool:
        if self.entries != tuple(sorted(self.entries, key=lambda item: item.case_id)):
            return False
        if self.excluded_root_entries != tuple(
            sorted(self.excluded_root_entries, key=lambda item: item.name)
        ):
            return False
        counts = {
            status: sum(entry.status == status for entry in self.entries)
            for status in ("valid", "corrupt", "duplicate", "excluded")
        }
        if (
            self.discovered_case_directories != len(self.entries)
            or self.valid_cases != counts["valid"]
            or self.corrupt_cases != counts["corrupt"]
            or self.duplicate_cases != counts["duplicate"]
            or self.excluded_case_directories != counts["excluded"]
        ):
            return False
        if self.classified_case_ids_sha256 != _classified_digest(self.entries):
            return False
        unsigned = self.model_dump(mode="json", exclude={"inventory_sha256"})
        return self.inventory_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def audit_rcabench_inventory(
    dataset_root: Path,
    *,
    expected_cases: int = RCABENCH_EXPECTED_CASES,
    parquet_probe: ParquetProbe | None = None,
) -> RCABenchInventoryReport:
    """Classify every top-level entry and validate every candidate case independently."""

    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    probe = _probe_parquet_metadata if parquet_probe is None else parquet_probe

    case_paths: list[Path] = []
    excluded_root_entries: list[ExcludedRootEntry] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.name.startswith("."):
            excluded_root_entries.append(
                ExcludedRootEntry(
                    name=child.name,
                    kind=_root_entry_kind(child),
                    reason="hidden_non_case_entry",
                )
            )
        elif child.is_symlink():
            excluded_root_entries.append(
                ExcludedRootEntry(
                    name=child.name,
                    kind="symlink",
                    reason="symlink_not_allowed",
                )
            )
        elif child.is_dir():
            case_paths.append(child)
        else:
            excluded_root_entries.append(
                ExcludedRootEntry(
                    name=child.name,
                    kind=_root_entry_kind(child),
                    reason="non_case_root_file",
                )
            )

    entries: list[RCABenchInventoryEntry] = []
    canonical_owner: dict[str, str] = {}
    for case_path in case_paths:
        normalized_case_id = _normalize_case_id(case_path.name)
        duplicate_of = canonical_owner.get(normalized_case_id)
        if duplicate_of is not None:
            file_count, total_bytes = _case_size(case_path)
            entries.append(
                RCABenchInventoryEntry(
                    case_id=case_path.name,
                    normalized_case_id=normalized_case_id,
                    status="duplicate",
                    reasons=("normalized_case_id_duplicate",),
                    duplicate_of=duplicate_of,
                    file_count=file_count,
                    total_bytes=total_bytes,
                )
            )
            continue
        canonical_owner[normalized_case_id] = case_path.name
        entries.append(_audit_case(case_path, normalized_case_id, probe))

    frozen_entries = tuple(entries)
    status_counts = {
        status: sum(entry.status == status for entry in frozen_entries)
        for status in ("valid", "corrupt", "duplicate", "excluded")
    }
    unsigned: dict[str, object] = {
        "schema_version": RCABENCH_INVENTORY_SCHEMA_VERSION,
        "dataset": "rcabench",
        "root": root.as_posix(),
        "expected_cases": expected_cases,
        "discovered_case_directories": len(frozen_entries),
        "valid_cases": status_counts["valid"],
        "corrupt_cases": status_counts["corrupt"],
        "duplicate_cases": status_counts["duplicate"],
        "excluded_case_directories": status_counts["excluded"],
        "classified_case_ids_sha256": _classified_digest(frozen_entries),
        "entries": [entry.model_dump(mode="json") for entry in frozen_entries],
        "excluded_root_entries": [
            entry.model_dump(mode="json") for entry in excluded_root_entries
        ],
    }
    report = RCABenchInventoryReport.model_validate(
        {
            **unsigned,
            "inventory_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not report.verify():  # pragma: no cover - construction invariant
        raise AssertionError("RCABench inventory report verification failed")
    return report


def write_inventory_report(path: Path, report: RCABenchInventoryReport) -> None:
    """Write once; permit an exact replay but reject replacement by different bytes."""

    output = Path(path)
    data = canonical_json_bytes(report.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable inventory artifact already exists with other content")
        return
    if not report.verify():
        raise ValueError("inventory report verification failed")
    atomic_write_bytes(output, data)


def _audit_case(
    case_path: Path,
    normalized_case_id: str,
    parquet_probe: ParquetProbe,
) -> RCABenchInventoryEntry:
    reasons: list[str] = []
    available: dict[str, Path] = {}
    for name in RCABENCH_REQUIRED_FILES:
        path = case_path / name
        if not path.exists():
            reasons.append(f"missing_required_file:{name}")
        elif path.is_symlink() or not path.is_file():
            reasons.append(f"invalid_required_file:{name}")
        else:
            available[name] = path

    env_path = available.get("env.json")
    if env_path is not None and not _valid_env_contract(env_path):
        reasons.append("invalid_env_contract")
    injection_path = available.get("injection.json")
    if injection_path is not None and not _valid_injection_contract(injection_path):
        reasons.append("invalid_injection_contract")

    for name in RCABENCH_PARQUET_FILES:
        parquet_path = available.get(name)
        if parquet_path is None:
            continue
        try:
            if parquet_path.stat().st_size == 0:
                raise ValueError("empty parquet file")
            parquet_probe(parquet_path)
        except Exception as error:
            reasons.append(f"invalid_parquet:{name}:{type(error).__name__}")

    file_count, total_bytes = _case_size(case_path)
    frozen_reasons = tuple(sorted(set(reasons)))
    return RCABenchInventoryEntry(
        case_id=case_path.name,
        normalized_case_id=normalized_case_id,
        status="corrupt" if frozen_reasons else "valid",
        reasons=frozen_reasons,
        file_count=file_count,
        total_bytes=total_bytes,
    )


def _valid_env_contract(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_t0 = payload["ABNORMAL_START"] if isinstance(payload, dict) else None
        if isinstance(raw_t0, bool) or not isinstance(raw_t0, int | str):
            return False
        return int(raw_t0) >= 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _valid_injection_contract(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return False
    service = ground_truth.get("service")
    if isinstance(service, str):
        return bool(service.strip())
    return (
        isinstance(service, list)
        and bool(service)
        and all(isinstance(item, str) and bool(item.strip()) for item in service)
    )


def _probe_parquet_metadata(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata is None or parquet.schema_arrow is None:
        raise ValueError("Parquet metadata is unavailable")


def _case_size(case_path: Path) -> tuple[int, int]:
    files = tuple(path for path in case_path.iterdir() if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)


def _normalize_case_id(case_id: str) -> str:
    return unicodedata.normalize("NFKC", case_id).casefold()


def _classified_digest(entries: Sequence[RCABenchInventoryEntry]) -> str:
    return sha256_hex(
        canonical_json_bytes(
            [
                {
                    "case_id": entry.case_id,
                    "duplicate_of": entry.duplicate_of,
                    "status": entry.status,
                }
                for entry in entries
            ]
        )
    )


def _root_entry_kind(path: Path) -> RootEntryKind:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the RCABench case inventory")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=RCABENCH_EXPECTED_CASES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_rcabench_inventory(
        args.data_root,
        expected_cases=args.expected_cases,
    )
    write_inventory_report(args.output, report)
    print(
        json.dumps(
            {
                "acceptable": report.acceptable,
                "classified_case_ids_sha256": report.classified_case_ids_sha256,
                "corrupt_cases": report.corrupt_cases,
                "discovered_case_directories": report.discovered_case_directories,
                "duplicate_cases": report.duplicate_cases,
                "excluded_case_directories": report.excluded_case_directories,
                "excluded_root_entries": len(report.excluded_root_entries),
                "inventory_sha256": report.inventory_sha256,
                "valid_cases": report.valid_cases,
            },
            sort_keys=True,
        )
    )
    return 0 if report.acceptable else 2


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
