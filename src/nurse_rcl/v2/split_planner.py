"""Server-side evaluator planner for private splits and safe review summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel
from nurse_rcl.v2.exposure_registry import ExposureEvent, load_exposure_events
from nurse_rcl.v2.opaque_ids import (
    EvaluatorIdentityMap,
    load_evaluator_identity_map,
    load_incident_group_index,
)
from nurse_rcl.v2.split_allocator import (
    RESERVATION_ORDER,
    EvaluatorSplitCase,
    PartitionSummary,
    SplitAllocation,
    SplitTargets,
    allocate_group_atomic_split,
)

SPLIT_REVIEW_SCHEMA_VERSION: Literal[
    "nurse.v2.split-review/1"
] = "nurse.v2.split-review/1"


class PrivateAllocationArtifact(StrictFrozenModel):
    dataset: DatasetName
    path: str = Field(min_length=1)
    mode: str = Field(pattern=r"^(0600|windows)$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetSplitReview(StrictFrozenModel):
    """Aggregate-only review data; no case, answer, or fault values."""

    dataset: DatasetName
    case_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    exposed_case_count: int = Field(ge=0)
    exposed_group_count: int = Field(ge=0)
    targets: SplitTargets
    partition_summaries: tuple[PartitionSummary, ...]
    stratum_count_by_level: dict[str, int]
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_allocation: PrivateAllocationArtifact


class SplitReviewReport(StrictFrozenModel):
    schema_version: Literal["nurse.v2.split-review/1"] = SPLIT_REVIEW_SCHEMA_VERSION
    seed: str = Field(min_length=1)
    minimum_groups_per_stratum: int = Field(ge=1)
    reservation_order: tuple[
        Literal["final"], Literal["validation"], Literal["dev"]
    ] = RESERVATION_ORDER
    identity_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_registry_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    incident_group_artifact_sha256_by_dataset: dict[DatasetName, str]
    contains_private_case_ids: Literal[False] = False
    contains_answer_values: Literal[False] = False
    contains_fault_type_values: Literal[False] = False
    datasets: tuple[DatasetSplitReview, ...]
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.reservation_order != RESERVATION_ORDER:
            return False
        if self.datasets != tuple(sorted(self.datasets, key=lambda item: item.dataset)):
            return False
        if len({item.dataset for item in self.datasets}) != len(self.datasets):
            return False
        if set(self.incident_group_artifact_sha256_by_dataset) != {
            item.dataset for item in self.datasets
        }:
            return False
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.incident_group_artifact_sha256_by_dataset.values()
        ):
            return False
        for item in self.datasets:
            if item.private_allocation.dataset != item.dataset:
                return False
            if item.private_allocation.allocation_sha256 != item.allocation_sha256:
                return False
            if item.case_count != sum(
                summary.actual_case_count for summary in item.partition_summaries
            ):
                return False
            if item.group_count != sum(
                summary.group_count for summary in item.partition_summaries
            ):
                return False
            if item.exposed_case_count != sum(
                summary.exposed_case_count for summary in item.partition_summaries
            ):
                return False
            if item.exposed_group_count != sum(
                summary.exposed_group_count for summary in item.partition_summaries
            ):
                return False
        unsigned = self.model_dump(mode="json", exclude={"report_sha256"})
        return self.report_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_evaluator_split_cases(
    *,
    evaluator_map: EvaluatorIdentityMap,
    exposure_events: Sequence[ExposureEvent],
    group_ids_by_dataset: Mapping[DatasetName, Mapping[str, str]],
    rcabench_data_root: Path,
    aiops25_work_root: Path,
) -> tuple[EvaluatorSplitCase, ...]:
    """Join private labels, group membership, exposure, and fault strata."""

    if not evaluator_map.verify():
        raise ValueError("evaluator identity map verification failed")
    identity_by_key = {
        (record.dataset, record.original_case_id): record
        for record in evaluator_map.records
    }
    expected_by_dataset: dict[DatasetName, dict[str, str]] = {
        "rcabench": {},
        "aiops25": {},
    }
    for record in evaluator_map.records:
        index = group_ids_by_dataset.get(record.dataset)
        if index is None or index.get(record.original_case_id) != record.incident_group_id:
            raise ValueError("identity and incident-group manifests disagree")
        expected_by_dataset[record.dataset][record.original_case_id] = (
            record.incident_group_id
        )
    for dataset, index in group_ids_by_dataset.items():
        if dict(index) != expected_by_dataset[dataset]:
            raise ValueError("incident-group and identity populations disagree")

    exposed_groups: set[tuple[DatasetName, str]] = set()
    for event in exposure_events:
        if not event.verify():
            raise ValueError("invalid exposure event")
        exposure_record = identity_by_key.get((event.dataset, event.case_id))
        if exposure_record is None:
            raise ValueError("exposure case is outside the identity map")
        if exposure_record.incident_group_id != event.incident_group_id:
            raise ValueError("exposure and identity incident groups disagree")
        exposed_groups.add((event.dataset, event.incident_group_id))

    rcabench_root = Path(rcabench_data_root).resolve(strict=True)
    aiops25_root = Path(aiops25_work_root).resolve(strict=True)
    cases = []
    for record in evaluator_map.records:
        if record.dataset == "rcabench":
            injection_path = _case_file(
                rcabench_root,
                record.original_case_id,
                suffix=("injection.json",),
            )
        else:
            injection_path = _case_file(
                aiops25_root,
                record.original_case_id,
                suffix=("canonical", "injection.json"),
            )
        injection = _load_json_object(injection_path)
        cases.append(
            EvaluatorSplitCase(
                dataset=record.dataset,
                case_id=record.original_case_id,
                incident_group_id=record.incident_group_id,
                target_services=record.answer_services,
                fault_type=_normalize_fault_type(
                    injection.get("fault_type"),
                    context=f"{record.dataset}/{record.original_case_id}",
                ),
                exposed=(record.dataset, record.incident_group_id)
                in exposed_groups,
            )
        )
    return tuple(sorted(cases, key=lambda item: (item.dataset, item.case_id)))


def write_private_split_allocation(path: Path, allocation: SplitAllocation) -> None:
    """Persist private case membership as a mode-0600 immutable artifact."""

    output = Path(path)
    data = canonical_json_bytes(allocation.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable private split allocation already exists")
        if not allocation.verify():
            raise ValueError("split allocation verification failed")
        _require_private_mode(output)
        return
    if not allocation.verify():
        raise ValueError("split allocation verification failed")
    _atomic_write_private(output, data)
    _require_private_mode(output)


def build_split_review_report(
    *,
    allocations: Sequence[SplitAllocation],
    private_allocation_paths: Mapping[DatasetName, Path],
    identity_artifact_sha256: str,
    exposure_registry_artifact_sha256: str,
    incident_group_artifact_sha256_by_dataset: Mapping[DatasetName, str],
) -> SplitReviewReport:
    """Project private allocations into an aggregate label-free review report."""

    frozen = tuple(sorted(allocations, key=lambda item: item.dataset))
    if not frozen or any(not allocation.verify() for allocation in frozen):
        raise ValueError("all split allocations must verify")
    if len({allocation.dataset for allocation in frozen}) != len(frozen):
        raise ValueError("one allocation per dataset is required")
    seeds = {allocation.seed for allocation in frozen}
    minimums = {allocation.minimum_groups_per_stratum for allocation in frozen}
    if len(seeds) != 1 or len(minimums) != 1:
        raise ValueError("split allocations must share seed and stratum policy")
    dataset_names = {allocation.dataset for allocation in frozen}
    if set(private_allocation_paths) != dataset_names:
        raise ValueError("private allocation paths do not match datasets")
    if set(incident_group_artifact_sha256_by_dataset) != dataset_names:
        raise ValueError("incident-group hashes do not match datasets")

    dataset_reviews = []
    for allocation in frozen:
        path = Path(private_allocation_paths[allocation.dataset])
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_private_mode(path)
        stratum_sets: dict[str, set[str]] = {
            "target_fault": set(),
            "fault_type": set(),
            "dataset": set(),
        }
        for assignment in allocation.assignments:
            stratum_sets[assignment.stratum_level].add(assignment.stratum_id)
        dataset_reviews.append(
            DatasetSplitReview(
                dataset=allocation.dataset,
                case_count=allocation.case_count,
                group_count=allocation.group_count,
                exposed_case_count=allocation.exposed_case_count,
                exposed_group_count=allocation.exposed_group_count,
                targets=allocation.targets,
                partition_summaries=allocation.partition_summaries,
                stratum_count_by_level={
                    level: len(values) for level, values in stratum_sets.items()
                },
                allocation_sha256=allocation.allocation_sha256,
                private_allocation=PrivateAllocationArtifact(
                    dataset=allocation.dataset,
                    path=path.as_posix(),
                    mode=_private_mode(path),
                    artifact_sha256=_sha256_file(path),
                    allocation_sha256=allocation.allocation_sha256,
                ),
            )
        )
    unsigned: dict[str, object] = {
        "schema_version": SPLIT_REVIEW_SCHEMA_VERSION,
        "seed": next(iter(seeds)),
        "minimum_groups_per_stratum": next(iter(minimums)),
        "reservation_order": RESERVATION_ORDER,
        "identity_artifact_sha256": identity_artifact_sha256,
        "exposure_registry_artifact_sha256": exposure_registry_artifact_sha256,
        "incident_group_artifact_sha256_by_dataset": dict(
            incident_group_artifact_sha256_by_dataset
        ),
        "contains_private_case_ids": False,
        "contains_answer_values": False,
        "contains_fault_type_values": False,
        "datasets": [item.model_dump(mode="json") for item in dataset_reviews],
    }
    report = SplitReviewReport.model_validate(
        {
            **unsigned,
            "report_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not report.verify():  # pragma: no cover - construction invariant
        raise AssertionError("split review report verification failed")
    return report


def write_split_review_report(path: Path, report: SplitReviewReport) -> None:
    if not report.verify():
        raise ValueError("split review report verification failed")
    output = Path(path)
    data = canonical_json_bytes(report.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable split review report already exists")
        return
    atomic_write_bytes(output, data)


def _case_file(root: Path, case_id: str, *, suffix: tuple[str, ...]) -> Path:
    if not case_id or case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
        raise ValueError("case ID cannot contain a path")
    if suffix[0] == "canonical":
        candidate = root / suffix[0] / case_id / suffix[1]
    else:
        candidate = root / case_id / suffix[0]
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("case path escapes dataset root")
    return resolved


def _normalize_fault_type(value: object, *, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{context}: invalid evaluator fault type")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{context}: invalid evaluator fault type")
    return normalized


def _load_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _atomic_write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _require_private_mode(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("private split allocation must be a regular file")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("private split allocation must have mode 0600")


def _private_mode(path: Path) -> str:
    if os.name == "nt":
        return "windows"
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create evaluator-only V2 splits")
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--rcabench-groups", type=Path, required=True)
    parser.add_argument("--aiops25-groups", type=Path, required=True)
    parser.add_argument("--rcabench-data-root", type=Path, required=True)
    parser.add_argument("--aiops25-work-root", type=Path, required=True)
    parser.add_argument("--rcabench-output", type=Path, required=True)
    parser.add_argument("--aiops25-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--minimum-groups-per-stratum", type=int, default=3)
    parser.add_argument("--rcabench-dev", type=int, default=284)
    parser.add_argument("--rcabench-validation", type=int, default=142)
    parser.add_argument("--rcabench-final", type=int, default=284)
    parser.add_argument("--aiops25-dev", type=int, default=64)
    parser.add_argument("--aiops25-validation", type=int, default=32)
    parser.add_argument("--aiops25-final", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluator_map = load_evaluator_identity_map(args.identity_map)
    group_ids_by_dataset: dict[DatasetName, dict[str, str]] = {
        "rcabench": load_incident_group_index(
            args.rcabench_groups,
            dataset="rcabench",
        ),
        "aiops25": load_incident_group_index(
            args.aiops25_groups,
            dataset="aiops25",
        ),
    }
    cases = build_evaluator_split_cases(
        evaluator_map=evaluator_map,
        exposure_events=load_exposure_events(args.exposure_registry),
        group_ids_by_dataset=group_ids_by_dataset,
        rcabench_data_root=args.rcabench_data_root,
        aiops25_work_root=args.aiops25_work_root,
    )
    allocations = (
        allocate_group_atomic_split(
            tuple(item for item in cases if item.dataset == "rcabench"),
            targets=SplitTargets(
                dev_case_count=args.rcabench_dev,
                validation_case_count=args.rcabench_validation,
                final_case_count=args.rcabench_final,
            ),
            seed=args.seed,
            minimum_groups_per_stratum=args.minimum_groups_per_stratum,
        ),
        allocate_group_atomic_split(
            tuple(item for item in cases if item.dataset == "aiops25"),
            targets=SplitTargets(
                dev_case_count=args.aiops25_dev,
                validation_case_count=args.aiops25_validation,
                final_case_count=args.aiops25_final,
            ),
            seed=args.seed,
            minimum_groups_per_stratum=args.minimum_groups_per_stratum,
        ),
    )
    output_paths: dict[DatasetName, Path] = {
        "rcabench": args.rcabench_output,
        "aiops25": args.aiops25_output,
    }
    for allocation in allocations:
        write_private_split_allocation(output_paths[allocation.dataset], allocation)
    report = build_split_review_report(
        allocations=allocations,
        private_allocation_paths=output_paths,
        identity_artifact_sha256=_sha256_file(args.identity_map),
        exposure_registry_artifact_sha256=_sha256_file(args.exposure_registry),
        incident_group_artifact_sha256_by_dataset={
            "rcabench": _sha256_file(args.rcabench_groups),
            "aiops25": _sha256_file(args.aiops25_groups),
        },
    )
    write_split_review_report(args.review_output, report)
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
