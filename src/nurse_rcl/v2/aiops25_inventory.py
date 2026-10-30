"""Server-side AIOps25 preparation audit and queue-scope authorization."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

AIOPS25_INVENTORY_SCHEMA_VERSION: Literal[
    "nurse.v2.aiops25-inventory/1"
] = "nurse.v2.aiops25-inventory/1"
AIOPS25_QUEUE_SCHEMA_VERSION: Literal[
    "nurse.v2.aiops25-authorized-queue/1"
] = "nurse.v2.aiops25-authorized-queue/1"
AIOPS25_EXPECTED_SOURCE_CASES = 400
AIOPS25_EXPECTED_INCLUDED_CASES = 318
AIOPS25_EXPECTED_EXCLUDED_NODE_CASES = 82
AIOPS25_MANDATED_PREPROCESSOR = Path(
    "/home/wangrunzhou/25挑战赛/reproduction_hd4_baselines/hd4_baseline_repro.py"
)
AIOPS25_MANDATED_PREPROCESSOR_SHA256 = (
    "1b3e8643ccd8c252037adda2739240d67897edd1343cbc7fe4554ef9fa0410c1"
)
AIOPS25_PARQUET_FILES = (
    "abnormal_logs.parquet",
    "abnormal_metrics.parquet",
    "abnormal_traces.parquet",
    "normal_logs.parquet",
    "normal_metrics.parquet",
    "normal_traces.parquet",
)
AIOPS25_REQUIRED_PREPARED_FILES = (
    ".prepared.json",
    "env.json",
    "injection.json",
    *AIOPS25_PARQUET_FILES,
)
_REQUIRED_ENV_KEYS = (
    "NORMAL_START",
    "NORMAL_END",
    "ABNORMAL_START",
    "ABNORMAL_END",
    "ORIGINAL_ABNORMAL_END",
)
_ROW_KEYS = tuple(path.removesuffix(".parquet") for path in AIOPS25_PARQUET_FILES)

ParquetProbe = Callable[[Path], None]
InventoryStatus = Literal["included", "excluded", "corrupt", "duplicate"]
InstanceType = Literal["service", "pod", "node", "unknown"]
QueueDestination = Literal["inference", "ark"]


class AIOps25CaseScopeError(ValueError):
    """Raised before a case outside the audited included scope can form a queue."""


class AIOps25InventoryEntry(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    normalized_case_id: str = Field(min_length=1)
    instance_type: InstanceType
    status: InventoryStatus
    reasons: tuple[str, ...]
    duplicate_of: str | None = None
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("inventory reasons must be sorted and unique")
        if self.status == "included" and self.reasons:
            raise ValueError("included inventory entries cannot have failure reasons")
        if self.status != "included" and not self.reasons:
            raise ValueError("non-included inventory entries require a reason")
        if (self.status == "duplicate") != (self.duplicate_of is not None):
            raise ValueError("only duplicate entries declare duplicate_of")
        if self.status == "excluded" and self.instance_type != "node":
            raise ValueError("only node-level cases may be excluded")
        return self


class AIOps25InventoryReport(StrictFrozenModel):
    schema_version: Literal["nurse.v2.aiops25-inventory/1"] = (
        AIOPS25_INVENTORY_SCHEMA_VERSION
    )
    dataset: Literal["aiops25"] = "aiops25"
    source_root: str = Field(min_length=1)
    work_root: str = Field(min_length=1)
    expected_source_cases: int = Field(ge=0)
    expected_included_cases: int = Field(ge=0)
    expected_excluded_node_cases: int = Field(ge=0)
    source_case_records: int = Field(ge=0)
    included_cases: int = Field(ge=0)
    excluded_node_cases: int = Field(ge=0)
    corrupt_cases: int = Field(ge=0)
    duplicate_cases: int = Field(ge=0)
    orphan_prepared_cases: tuple[str, ...]
    preparation_contract_errors: tuple[str, ...]
    preparation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessor: str = Field(min_length=1)
    preprocessor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_case_count: int = Field(ge=0)
    validated_prepared_markers: int = Field(ge=0)
    validated_time_anchor_documents: int = Field(ge=0)
    validated_injection_documents: int = Field(ge=0)
    validated_parquet_metadata_files: int = Field(ge=0)
    classified_case_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[AIOps25InventoryEntry, ...]
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def acceptable(self) -> bool:
        return (
            self.source_case_records == self.expected_source_cases
            and self.included_cases == self.expected_included_cases
            and self.excluded_node_cases == self.expected_excluded_node_cases
            and self.corrupt_cases == 0
            and self.duplicate_cases == 0
            and not self.orphan_prepared_cases
            and not self.preparation_contract_errors
            and self.prepared_case_count == self.expected_included_cases
            and self.validated_prepared_markers == self.expected_included_cases
            and self.validated_time_anchor_documents == self.expected_included_cases
            and self.validated_injection_documents == self.expected_included_cases
            and self.validated_parquet_metadata_files
            == self.expected_included_cases * len(AIOPS25_PARQUET_FILES)
        )

    def verify(self) -> bool:
        if self.entries != tuple(sorted(self.entries, key=lambda item: item.case_id)):
            return False
        if self.orphan_prepared_cases != tuple(sorted(set(self.orphan_prepared_cases))):
            return False
        if self.preparation_contract_errors != tuple(
            sorted(set(self.preparation_contract_errors))
        ):
            return False
        counts = {
            status: sum(entry.status == status for entry in self.entries)
            for status in ("included", "excluded", "corrupt", "duplicate")
        }
        if (
            self.source_case_records != len(self.entries)
            or self.included_cases != counts["included"]
            or self.excluded_node_cases != counts["excluded"]
            or self.corrupt_cases != counts["corrupt"]
            or self.duplicate_cases != counts["duplicate"]
            or self.classified_case_ids_sha256 != _classified_digest(self.entries)
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"inventory_sha256"})
        return self.inventory_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class AuthorizedAIOps25Queue(StrictFrozenModel):
    schema_version: Literal["nurse.v2.aiops25-authorized-queue/1"] = (
        AIOPS25_QUEUE_SCHEMA_VERSION
    )
    dataset: Literal["aiops25"] = "aiops25"
    destination: QueueDestination
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_ids: tuple[str, ...] = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if len(self.case_ids) != len(set(self.case_ids)):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    case_id: str
    instance_type: InstanceType
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedValidation:
    reasons: tuple[str, ...]
    file_count: int
    total_bytes: int
    prepared_markers: int
    time_anchor_documents: int
    injection_documents: int
    parquet_metadata_files: int


def audit_aiops25_inventory(
    source_root: Path,
    work_root: Path,
    *,
    expected_source_cases: int = AIOPS25_EXPECTED_SOURCE_CASES,
    expected_included_cases: int = AIOPS25_EXPECTED_INCLUDED_CASES,
    expected_excluded_node_cases: int = AIOPS25_EXPECTED_EXCLUDED_NODE_CASES,
    parquet_probe: ParquetProbe | None = None,
) -> AIOps25InventoryReport:
    """Audit source classification and every prepared case without emitting labels."""

    source = Path(source_root).resolve(strict=True)
    prepared = Path(work_root).resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not prepared.is_dir():
        raise NotADirectoryError(prepared)
    probe = _probe_parquet_metadata if parquet_probe is None else parquet_probe

    records = _load_source_records(source / "groundtruth.jsonl")
    manifest_path = prepared / "manifests" / "manifest.json"
    receipt_path = prepared / "manifests" / "nurse_preparation_receipt.json"
    manifest = _load_json_object(manifest_path)
    receipt = _load_json_object(receipt_path)
    included_manifest, excluded_manifest, metadata_errors = _preparation_contract(
        manifest=manifest,
        receipt=receipt,
        source_root=source,
        work_root=prepared,
        expected_source_cases=expected_source_cases,
        expected_included_cases=expected_included_cases,
        expected_excluded_node_cases=expected_excluded_node_cases,
    )

    canonical_root = prepared / "canonical"
    canonical_paths = {
        path.name: path
        for path in canonical_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    source_ids = {record.case_id for record in records}
    orphan_prepared = tuple(
        sorted(set(canonical_paths).difference(included_manifest).union(
            set(canonical_paths).difference(source_ids)
        ))
    )
    unknown_manifest_ids = set(included_manifest).union(excluded_manifest).difference(
        source_ids
    )
    if unknown_manifest_ids:
        metadata_errors.append("manifest_references_unknown_source_case")

    entries: list[AIOps25InventoryEntry] = []
    canonical_owner: dict[str, str] = {}
    validated_prepared_markers = 0
    validated_time_anchor_documents = 0
    validated_injection_documents = 0
    validated_parquet_metadata_files = 0
    for record in records:
        normalized = _normalize_case_id(record.case_id)
        duplicate_of = canonical_owner.get(normalized)
        if duplicate_of is not None:
            case_path = canonical_paths.get(record.case_id)
            file_count, total_bytes = _case_size(case_path)
            entries.append(
                AIOps25InventoryEntry(
                    case_id=record.case_id,
                    normalized_case_id=normalized,
                    instance_type=record.instance_type,
                    status="duplicate",
                    reasons=("normalized_case_id_duplicate",),
                    duplicate_of=duplicate_of,
                    file_count=file_count,
                    total_bytes=total_bytes,
                )
            )
            continue
        canonical_owner[normalized] = record.case_id

        reasons = list(record.reasons)
        file_count = 0
        total_bytes = 0
        if record.instance_type == "node":
            if record.case_id not in excluded_manifest:
                reasons.append("node_missing_from_exclusion_manifest")
            if record.case_id in included_manifest:
                reasons.append("node_present_in_inclusion_manifest")
            if record.case_id in canonical_paths:
                reasons.append("excluded_node_has_prepared_directory")
                file_count, total_bytes = _case_size(canonical_paths[record.case_id])
            status: InventoryStatus = "corrupt" if reasons else "excluded"
            if status == "excluded":
                reasons.append("node_level_target_not_service_granularity")
        elif record.instance_type in {"service", "pod"}:
            manifest_type = included_manifest.get(record.case_id)
            if manifest_type is None:
                reasons.append("missing_from_inclusion_manifest")
            elif manifest_type != record.instance_type:
                reasons.append("manifest_instance_type_mismatch")
            if record.case_id in excluded_manifest:
                reasons.append("service_level_case_in_node_exclusion")
            case_path = canonical_paths.get(record.case_id)
            if case_path is None:
                reasons.append("missing_prepared_case_directory")
            else:
                validation = _audit_prepared_case(case_path, probe)
                reasons.extend(validation.reasons)
                file_count = validation.file_count
                total_bytes = validation.total_bytes
                validated_prepared_markers += validation.prepared_markers
                validated_time_anchor_documents += validation.time_anchor_documents
                validated_injection_documents += validation.injection_documents
                validated_parquet_metadata_files += validation.parquet_metadata_files
            status = "corrupt" if reasons else "included"
        else:
            reasons.append("unsupported_instance_type")
            status = "corrupt"

        entries.append(
            AIOps25InventoryEntry(
                case_id=record.case_id,
                normalized_case_id=normalized,
                instance_type=record.instance_type,
                status=status,
                reasons=tuple(sorted(set(reasons))),
                file_count=file_count,
                total_bytes=total_bytes,
            )
        )

    frozen_entries = tuple(sorted(entries, key=lambda item: item.case_id))
    counts = {
        status: sum(entry.status == status for entry in frozen_entries)
        for status in ("included", "excluded", "corrupt", "duplicate")
    }
    preprocessor_value = receipt.get("preprocessor")
    preprocessor = preprocessor_value if isinstance(preprocessor_value, str) else "invalid"
    preprocessor_digest_value = receipt.get("preprocessor_sha256")
    preprocessor_digest = (
        preprocessor_digest_value
        if isinstance(preprocessor_digest_value, str)
        and _is_sha256(preprocessor_digest_value)
        else "0" * 64
    )
    prepared_count_value = receipt.get("prepared_case_count")
    prepared_count = (
        prepared_count_value
        if isinstance(prepared_count_value, int)
        and not isinstance(prepared_count_value, bool)
        and prepared_count_value >= 0
        else 0
    )
    unsigned: dict[str, object] = {
        "schema_version": AIOPS25_INVENTORY_SCHEMA_VERSION,
        "dataset": "aiops25",
        "source_root": source.as_posix(),
        "work_root": prepared.as_posix(),
        "expected_source_cases": expected_source_cases,
        "expected_included_cases": expected_included_cases,
        "expected_excluded_node_cases": expected_excluded_node_cases,
        "source_case_records": len(records),
        "included_cases": counts["included"],
        "excluded_node_cases": counts["excluded"],
        "corrupt_cases": counts["corrupt"],
        "duplicate_cases": counts["duplicate"],
        "orphan_prepared_cases": orphan_prepared,
        "preparation_contract_errors": tuple(sorted(set(metadata_errors))),
        "preparation_manifest_sha256": _sha256_file(manifest_path),
        "preparation_receipt_sha256": _sha256_file(receipt_path),
        "preprocessor": preprocessor,
        "preprocessor_sha256": preprocessor_digest,
        "prepared_case_count": prepared_count,
        "validated_prepared_markers": validated_prepared_markers,
        "validated_time_anchor_documents": validated_time_anchor_documents,
        "validated_injection_documents": validated_injection_documents,
        "validated_parquet_metadata_files": validated_parquet_metadata_files,
        "classified_case_ids_sha256": _classified_digest(frozen_entries),
        "entries": [entry.model_dump(mode="json") for entry in frozen_entries],
    }
    report = AIOps25InventoryReport.model_validate(
        {
            **unsigned,
            "inventory_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not report.verify():  # pragma: no cover - construction invariant
        raise AssertionError("AIOps25 inventory report verification failed")
    return report


def authorize_aiops25_queue(
    report: AIOps25InventoryReport,
    *,
    case_ids: Sequence[str],
    destination: QueueDestination | str,
) -> AuthorizedAIOps25Queue:
    """Atomically authorize a complete batch before inference or ARK queue creation."""

    if destination not in {"inference", "ark"}:
        raise AIOps25CaseScopeError(f"unsupported queue destination: {destination}")
    if not report.verify() or not report.acceptable:
        raise AIOps25CaseScopeError("AIOps25 inventory is not acceptable for queue use")
    frozen_case_ids = tuple(case_ids)
    if not frozen_case_ids:
        raise AIOps25CaseScopeError("AIOps25 queue cannot be empty")
    if len(frozen_case_ids) != len(set(frozen_case_ids)):
        raise AIOps25CaseScopeError("AIOps25 queue case identities must be unique")

    by_id = {entry.case_id: entry for entry in report.entries}
    for case_id in frozen_case_ids:
        entry = by_id.get(case_id)
        if entry is None:
            raise AIOps25CaseScopeError(
                f"{case_id}: case is outside the audited AIOps25 population"
            )
        if entry.status == "excluded" and entry.instance_type == "node":
            raise AIOps25CaseScopeError(
                f"{case_id}: excluded node-level case cannot enter {destination} queue"
            )
        if entry.status != "included":
            raise AIOps25CaseScopeError(
                f"{case_id}: {entry.status} case cannot enter {destination} queue"
            )

    destination_value = cast(QueueDestination, destination)
    unsigned: dict[str, object] = {
        "schema_version": AIOPS25_QUEUE_SCHEMA_VERSION,
        "dataset": "aiops25",
        "destination": destination_value,
        "inventory_sha256": report.inventory_sha256,
        "case_ids": frozen_case_ids,
    }
    queue = AuthorizedAIOps25Queue.model_validate(
        {
            **unsigned,
            "artifact_hash": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not queue.verify():  # pragma: no cover - construction invariant
        raise AssertionError("authorized AIOps25 queue verification failed")
    return queue


def prove_aiops25_queue_scope(report: AIOps25InventoryReport) -> dict[str, int]:
    """Count authorized included cases and verify every excluded node stays blocked."""

    if not report.verify() or not report.acceptable:
        raise AIOps25CaseScopeError("AIOps25 inventory is not acceptable for queue proof")
    included_ids = tuple(
        entry.case_id for entry in report.entries if entry.status == "included"
    )
    excluded_node_ids = tuple(
        entry.case_id
        for entry in report.entries
        if entry.status == "excluded" and entry.instance_type == "node"
    )
    proof: dict[str, int] = {}
    for destination in ("inference", "ark"):
        authorized = authorize_aiops25_queue(
            report,
            case_ids=included_ids,
            destination=destination,
        )
        proof[f"{destination}_included_cases_authorized"] = len(authorized.case_ids)
        excluded_authorized = 0
        for case_id in excluded_node_ids:
            try:
                authorize_aiops25_queue(
                    report,
                    case_ids=(case_id,),
                    destination=destination,
                )
            except AIOps25CaseScopeError:
                continue
            excluded_authorized += 1  # pragma: no cover - guarded invariant
        proof[f"{destination}_excluded_node_cases_authorized"] = excluded_authorized
    return proof


def write_inventory_report(path: Path, report: AIOps25InventoryReport) -> None:
    """Write once; permit exact replay and reject replacement by different bytes."""

    output = Path(path)
    data = canonical_json_bytes(report.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable inventory artifact already exists with other content")
        return
    if not report.verify():
        raise ValueError("inventory report verification failed")
    atomic_write_bytes(output, data)


def _load_source_records(path: Path) -> tuple[_SourceRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[_SourceRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        reasons: list[str] = []
        try:
            value: object = json.loads(raw_line)
        except json.JSONDecodeError:
            value = None
            reasons.append("invalid_source_json")
        if not isinstance(value, dict):
            records.append(
                _SourceRecord(
                    case_id=f"source-line-{line_number:06d}",
                    instance_type="unknown",
                    reasons=tuple(sorted(set((*reasons, "invalid_source_record")))),
                )
            )
            continue
        case_id_value = value.get("uuid")
        if not isinstance(case_id_value, str) or not case_id_value.strip():
            case_id = f"source-line-{line_number:06d}"
            reasons.append("invalid_source_case_id")
        else:
            case_id = case_id_value
        raw_type = value.get("instance_type")
        if raw_type in {"service", "pod", "node"}:
            instance_type = cast(InstanceType, raw_type)
        else:
            instance_type = "unknown"
            reasons.append("unsupported_instance_type")
        records.append(
            _SourceRecord(
                case_id=case_id,
                instance_type=instance_type,
                reasons=tuple(sorted(set(reasons))),
            )
        )
    return tuple(records)


def _preparation_contract(
    *,
    manifest: dict[str, object],
    receipt: dict[str, object],
    source_root: Path,
    work_root: Path,
    expected_source_cases: int,
    expected_included_cases: int,
    expected_excluded_node_cases: int,
) -> tuple[dict[str, InstanceType], set[str], list[str]]:
    errors: list[str] = []
    included: dict[str, InstanceType] = {}
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        cases = []
        errors.append("invalid_preparation_manifest_cases")
    for case in cases:
        if not isinstance(case, dict):
            errors.append("invalid_preparation_manifest_case")
            continue
        case_id = case.get("uuid")
        instance_type = case.get("instance_type")
        if not isinstance(case_id, str) or not case_id:
            errors.append("invalid_preparation_manifest_case_id")
            continue
        if instance_type not in {"service", "pod"}:
            errors.append("non_service_case_in_inclusion_manifest")
            continue
        if case_id in included:
            errors.append("duplicate_preparation_manifest_case_id")
            continue
        included[case_id] = cast(InstanceType, instance_type)

    raw_excluded = manifest.get("excluded_node_case_ids")
    if not isinstance(raw_excluded, list) or not all(
        isinstance(item, str) and item for item in raw_excluded
    ):
        excluded: set[str] = set()
        errors.append("invalid_excluded_node_case_ids")
    else:
        excluded_values = cast(list[str], raw_excluded)
        excluded = set(excluded_values)
        if len(excluded) != len(excluded_values):
            errors.append("duplicate_excluded_node_case_id")
    if set(included).intersection(excluded):
        errors.append("included_excluded_manifest_overlap")

    _check_integer_field(manifest, "case_count", expected_included_cases, errors)
    _check_integer_field(
        manifest,
        "excluded_node_cases",
        expected_excluded_node_cases,
        errors,
    )
    _check_integer_field(
        manifest,
        "all_groundtruth_cases",
        expected_source_cases,
        errors,
    )
    raw_unmapped = manifest.get("excluded_unmapped_case_ids")
    if not isinstance(raw_unmapped, list) or raw_unmapped:
        errors.append("unmapped_source_cases_present")

    if len(included) != expected_included_cases:
        errors.append("unexpected_included_manifest_count")
    if len(excluded) != expected_excluded_node_cases:
        errors.append("unexpected_excluded_node_manifest_count")
    _check_receipt(
        receipt,
        source_root=source_root,
        work_root=work_root,
        expected_included_cases=expected_included_cases,
        errors=errors,
    )
    return included, excluded, errors


def _check_integer_field(
    payload: dict[str, object],
    field: str,
    expected: int,
    errors: list[str],
) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        errors.append(f"invalid_preparation_manifest_{field}")


def _check_receipt(
    receipt: dict[str, object],
    *,
    source_root: Path,
    work_root: Path,
    expected_included_cases: int,
    errors: list[str],
) -> None:
    if receipt.get("schema_version") != 1:
        errors.append("invalid_preparation_receipt_schema")
    if receipt.get("data_root") != source_root.as_posix():
        errors.append("preparation_receipt_source_root_mismatch")
    if receipt.get("work_root") != work_root.as_posix():
        errors.append("preparation_receipt_work_root_mismatch")
    if receipt.get("preprocessor") != AIOPS25_MANDATED_PREPROCESSOR.as_posix():
        errors.append("unapproved_preprocessor_path")
    if receipt.get("preprocessor_sha256") != AIOPS25_MANDATED_PREPROCESSOR_SHA256:
        errors.append("unapproved_preprocessor_sha256")
    if receipt.get("prepared_case_count") != expected_included_cases:
        errors.append("preparation_receipt_case_count_mismatch")
    command = receipt.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        errors.append("invalid_preparation_command")
        return
    command_values = cast(list[str], command)
    required_tail = (
        AIOPS25_MANDATED_PREPROCESSOR.as_posix(),
        "--data-root",
        source_root.as_posix(),
        "--work-root",
        work_root.as_posix(),
        "--mode",
        "prepare",
    )
    if len(command_values) < 8 or tuple(command_values[1:]) != required_tail:
        errors.append("preparation_command_not_approved_prepare_mode")


def _audit_prepared_case(
    case_path: Path,
    parquet_probe: ParquetProbe,
) -> _PreparedValidation:
    reasons: list[str] = []
    available: dict[str, Path] = {}
    for name in AIOPS25_REQUIRED_PREPARED_FILES:
        path = case_path / name
        if not path.exists():
            reasons.append(f"missing_required_file:{name}")
        elif path.is_symlink() or not path.is_file():
            reasons.append(f"invalid_required_file:{name}")
        else:
            available[name] = path

    prepared_markers = 0
    marker_path = available.get(".prepared.json")
    if marker_path is not None:
        if _valid_prepared_marker(marker_path, case_path.name):
            prepared_markers = 1
        else:
            reasons.append("invalid_prepared_marker")
    time_anchor_documents = 0
    env_path = available.get("env.json")
    if env_path is not None:
        if _valid_env_contract(env_path):
            time_anchor_documents = 1
        else:
            reasons.append("invalid_env_contract")
    injection_documents = 0
    injection_path = available.get("injection.json")
    if injection_path is not None:
        if _valid_injection_contract(injection_path):
            injection_documents = 1
        else:
            reasons.append("invalid_injection_contract")

    parquet_metadata_files = 0
    for name in AIOPS25_PARQUET_FILES:
        parquet_path = available.get(name)
        if parquet_path is None:
            continue
        try:
            if parquet_path.stat().st_size == 0:
                raise ValueError("empty parquet file")
            parquet_probe(parquet_path)
        except Exception as error:
            reasons.append(f"invalid_parquet:{name}:{type(error).__name__}")
        else:
            parquet_metadata_files += 1

    file_count, total_bytes = _case_size(case_path)
    return _PreparedValidation(
        reasons=tuple(sorted(set(reasons))),
        file_count=file_count,
        total_bytes=total_bytes,
        prepared_markers=prepared_markers,
        time_anchor_documents=time_anchor_documents,
        injection_documents=injection_documents,
        parquet_metadata_files=parquet_metadata_files,
    )


def _valid_prepared_marker(path: Path, case_id: str) -> bool:
    try:
        marker = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if marker.get("schema_version") != 1 or marker.get("case") != case_id:
        return False
    rows = marker.get("rows")
    if not isinstance(rows, dict):
        return False
    return all(
        isinstance(rows.get(key), int)
        and not isinstance(rows.get(key), bool)
        and cast(int, rows.get(key)) >= 0
        for key in _ROW_KEYS
    )


def _valid_env_contract(path: Path) -> bool:
    try:
        env = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return all(
        isinstance(env.get(key), int) and not isinstance(env.get(key), bool)
        for key in _REQUIRED_ENV_KEYS
    )


def _valid_injection_contract(path: Path) -> bool:
    try:
        injection = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    targets = injection.get("target_services")
    return (
        isinstance(targets, list)
        and bool(targets)
        and all(isinstance(item, str) and bool(item.strip()) for item in targets)
    )


def _probe_parquet_metadata(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata is None or parquet.schema_arrow is None:
        raise ValueError("Parquet metadata is unavailable")


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _case_size(case_path: Path | None) -> tuple[int, int]:
    if case_path is None:
        return 0, 0
    files = tuple(path for path in case_path.iterdir() if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)


def _normalize_case_id(case_id: str) -> str:
    return unicodedata.normalize("NFKC", case_id).casefold()


def _classified_digest(entries: Sequence[AIOps25InventoryEntry]) -> str:
    return sha256_hex(
        canonical_json_bytes(
            [
                {
                    "case_id": entry.case_id,
                    "duplicate_of": entry.duplicate_of,
                    "instance_type": entry.instance_type,
                    "reasons": entry.reasons,
                    "status": entry.status,
                }
                for entry in entries
            ]
        )
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the prepared AIOps25 inventory")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-cases", type=int, default=AIOPS25_EXPECTED_SOURCE_CASES)
    parser.add_argument(
        "--expected-included-cases",
        type=int,
        default=AIOPS25_EXPECTED_INCLUDED_CASES,
    )
    parser.add_argument(
        "--expected-excluded-node-cases",
        type=int,
        default=AIOPS25_EXPECTED_EXCLUDED_NODE_CASES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_aiops25_inventory(
        args.source_root,
        args.work_root,
        expected_source_cases=args.expected_source_cases,
        expected_included_cases=args.expected_included_cases,
        expected_excluded_node_cases=args.expected_excluded_node_cases,
    )
    write_inventory_report(args.output, report)
    queue_scope_proof = prove_aiops25_queue_scope(report)
    print(
        json.dumps(
            {
                "acceptable": report.acceptable,
                "corrupt_cases": report.corrupt_cases,
                "duplicate_cases": report.duplicate_cases,
                "excluded_node_cases": report.excluded_node_cases,
                "included_cases": report.included_cases,
                "inventory_sha256": report.inventory_sha256,
                "orphan_prepared_cases": len(report.orphan_prepared_cases),
                "queue_scope_proof": queue_scope_proof,
            },
            sort_keys=True,
        )
    )
    return 0 if report.acceptable else 2


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
