"""Freeze an approved evaluator split into private and inference-safe artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel
from nurse_rcl.v2.exposure_registry import (
    ExposureCaseSummary,
    ExposureEvent,
    load_exposure_events,
    summarize_exposure_events,
)
from nurse_rcl.v2.opaque_ids import (
    EvaluatorIdentityMap,
    load_evaluator_identity_map,
)
from nurse_rcl.v2.split_allocator import (
    RESERVATION_ORDER,
    GroupAssignment,
    PartitionName,
    SplitAllocation,
)
from nurse_rcl.v2.split_planner import SplitReviewReport

EXPOSURE_SNAPSHOT_SCHEMA_VERSION: Literal[
    "nurse.v2.approved-exposure-snapshot/1"
] = "nurse.v2.approved-exposure-snapshot/1"
APPROVED_INFERENCE_SPLIT_SCHEMA_VERSION: Literal[
    "nurse.v2.approved-inference-split/1"
] = "nurse.v2.approved-inference-split/1"
APPROVED_SPLIT_SEAL_SCHEMA_VERSION: Literal[
    "nurse.v2.approved-split-seal/1"
] = "nurse.v2.approved-split-seal/1"

_PARTITION_ORDER: tuple[PartitionName, ...] = (
    "dev",
    "validation",
    "final",
    "remainder",
)
_DATASET_ORDER: tuple[DatasetName, ...] = ("aiops25", "rcabench")


def assert_partition_disjoint(assignments: Sequence[GroupAssignment]) -> None:
    """Reject a case or incident group assigned to more than one partition."""

    case_partitions: dict[str, PartitionName] = {}
    group_partitions: dict[str, PartitionName] = {}
    for assignment in assignments:
        prior_group = group_partitions.get(assignment.incident_group_id)
        if prior_group is not None and prior_group != assignment.partition:
            raise ValueError(
                "incident group crosses partitions: "
                f"{assignment.incident_group_id}"
            )
        group_partitions[assignment.incident_group_id] = assignment.partition
        for case_id in assignment.case_ids:
            prior_case = case_partitions.get(case_id)
            if prior_case is not None and prior_case != assignment.partition:
                raise ValueError(f"case crosses partitions: {case_id}")
            case_partitions[case_id] = assignment.partition


class ApprovedExposureSnapshot(StrictFrozenModel):
    """Evaluator-only snapshot of the exact append-only exposure ledger state."""

    schema_version: Literal["nurse.v2.approved-exposure-snapshot/1"] = (
        EXPOSURE_SNAPSHOT_SCHEMA_VERSION
    )
    registry_path: str = Field(min_length=1)
    registry_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    events: tuple[ExposureEvent, ...]
    records: tuple[ExposureCaseSummary, ...]
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.event_count != len(self.events):
            return False
        if len({event.event_id for event in self.events}) != len(self.events):
            return False
        if any(not event.verify() for event in self.events):
            return False
        expected_records = summarize_exposure_events(self.events)
        if self.records != expected_records or self.case_count != len(self.records):
            return False
        group_keys = {
            (record.dataset, record.incident_group_id) for record in self.records
        }
        if self.group_count != len(group_keys):
            return False
        event_ids = tuple(
            event_id for record in self.records for event_id in record.event_ids
        )
        if len(event_ids) != self.event_count or set(event_ids) != {
            event.event_id for event in self.events
        }:
            return False
        content = [event.model_dump(mode="json") for event in self.events]
        if self.registry_content_sha256 != sha256_hex(canonical_json_bytes(content)):
            return False
        unsigned = self.model_dump(mode="json", exclude={"snapshot_sha256"})
        return self.snapshot_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_exposure_snapshot(
    *,
    exposure_events: Sequence[ExposureEvent],
    registry_path: Path,
) -> ApprovedExposureSnapshot:
    """Bind the exact validated registry bytes and its private exposure records."""

    registry = Path(registry_path)
    events = tuple(exposure_events)
    if load_exposure_events(registry) != events:
        raise ValueError("exposure events do not match the exact registry")
    records = summarize_exposure_events(events)
    content = [event.model_dump(mode="json") for event in events]
    unsigned: dict[str, object] = {
        "schema_version": EXPOSURE_SNAPSHOT_SCHEMA_VERSION,
        "registry_path": registry.as_posix(),
        "registry_artifact_sha256": _sha256_file(registry),
        "registry_content_sha256": sha256_hex(canonical_json_bytes(content)),
        "event_count": len(events),
        "case_count": len(records),
        "group_count": len(
            {(record.dataset, record.incident_group_id) for record in records}
        ),
        "events": [event.model_dump(mode="json") for event in events],
        "records": [record.model_dump(mode="json") for record in records],
    }
    snapshot = ApprovedExposureSnapshot.model_validate(
        {
            **unsigned,
            "snapshot_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not snapshot.verify():  # pragma: no cover - construction invariant
        raise AssertionError("exposure snapshot verification failed")
    return snapshot


def write_private_exposure_snapshot(
    path: Path,
    snapshot: ApprovedExposureSnapshot,
) -> None:
    """Write an immutable evaluator snapshot with POSIX mode 0600."""

    output = Path(path)
    data = canonical_json_bytes(snapshot.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable private exposure snapshot already exists")
        if not snapshot.verify():
            raise ValueError("exposure snapshot verification failed")
        _require_private_mode(output, artifact="private exposure snapshot")
        return
    if not snapshot.verify():
        raise ValueError("exposure snapshot verification failed")
    _atomic_write_private(output, data)
    _require_private_mode(output, artifact="private exposure snapshot")


class ApprovedInferencePartition(StrictFrozenModel):
    """Inference-visible membership containing no reversible identity fields."""

    dataset: DatasetName
    partition: PartitionName
    case_count: int = Field(ge=0)
    opaque_case_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_membership(self) -> Self:
        if self.opaque_case_ids != tuple(sorted(set(self.opaque_case_ids))):
            raise ValueError("opaque partition membership must be sorted and unique")
        if self.case_count != len(self.opaque_case_ids):
            raise ValueError("opaque partition case count does not match membership")
        return self


class ApprovedInferenceSplitManifest(StrictFrozenModel):
    """Approved split projection safe to load inside inference workers."""

    schema_version: Literal["nurse.v2.approved-inference-split/1"] = (
        APPROVED_INFERENCE_SPLIT_SCHEMA_VERSION
    )
    identity_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_reference: str = Field(min_length=1)
    approved_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )
    case_count: int = Field(ge=0)
    partitions: tuple[ApprovedInferencePartition, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        expected_order = tuple(
            sorted(
                self.partitions,
                key=lambda item: (
                    item.dataset,
                    _PARTITION_ORDER.index(item.partition),
                ),
            )
        )
        if self.partitions != expected_order:
            return False
        datasets = {item.dataset for item in self.partitions}
        if {(item.dataset, item.partition) for item in self.partitions} != {
            (dataset, partition)
            for dataset in datasets
            for partition in _PARTITION_ORDER
        }:
            return False
        opaque_ids = tuple(
            opaque_id
            for partition in self.partitions
            for opaque_id in partition.opaque_case_ids
        )
        if self.case_count != len(opaque_ids) or len(opaque_ids) != len(
            set(opaque_ids)
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        return self.manifest_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_approved_inference_split_manifest(
    *,
    evaluator_map: EvaluatorIdentityMap,
    allocations: Sequence[SplitAllocation],
    identity_artifact_sha256: str,
    exposure_snapshot_sha256: str,
    approval_reference: str,
    approved_at_utc: str,
) -> ApprovedInferenceSplitManifest:
    """Project private split membership onto nonreversible opaque identities."""

    if not evaluator_map.verify():
        raise ValueError("evaluator identity map verification failed")
    frozen_allocations = tuple(sorted(allocations, key=lambda item: item.dataset))
    if not frozen_allocations or any(
        not allocation.verify() for allocation in frozen_allocations
    ):
        raise ValueError("all split allocations must verify")
    if len({allocation.dataset for allocation in frozen_allocations}) != len(
        frozen_allocations
    ):
        raise ValueError("one split allocation per dataset is required")
    all_assignments = tuple(
        assignment
        for allocation in frozen_allocations
        for assignment in allocation.assignments
    )
    assert_partition_disjoint(all_assignments)
    identity_by_key = {
        (record.dataset, record.original_case_id): record
        for record in evaluator_map.records
    }
    expected_keys = set(identity_by_key)
    allocated_keys = {
        (allocation.dataset, case_id)
        for allocation in frozen_allocations
        for assignment in allocation.assignments
        for case_id in assignment.case_ids
    }
    if allocated_keys != expected_keys:
        raise ValueError("split allocation and evaluator identity populations disagree")

    partitions: list[ApprovedInferencePartition] = []
    for allocation in frozen_allocations:
        membership = {
            partition: tuple(
                sorted(
                    identity_by_key[(allocation.dataset, case_id)].opaque_case_id
                    for assignment in allocation.assignments
                    if assignment.partition == partition
                    for case_id in assignment.case_ids
                )
            )
            for partition in _PARTITION_ORDER
        }
        partitions.extend(
            ApprovedInferencePartition(
                dataset=allocation.dataset,
                partition=partition,
                case_count=len(membership[partition]),
                opaque_case_ids=membership[partition],
            )
            for partition in _PARTITION_ORDER
        )
    unsigned: dict[str, object] = {
        "schema_version": APPROVED_INFERENCE_SPLIT_SCHEMA_VERSION,
        "identity_artifact_sha256": identity_artifact_sha256,
        "exposure_snapshot_sha256": exposure_snapshot_sha256,
        "approval_reference": approval_reference,
        "approved_at_utc": approved_at_utc,
        "case_count": sum(item.case_count for item in partitions),
        "partitions": [item.model_dump(mode="json") for item in partitions],
    }
    manifest = ApprovedInferenceSplitManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not manifest.verify():  # pragma: no cover - construction invariant
        raise AssertionError("approved inference split verification failed")
    return manifest


def write_approved_inference_split_manifest(
    path: Path,
    manifest: ApprovedInferenceSplitManifest,
) -> None:
    """Write an immutable inference-safe approved split manifest."""

    _write_immutable_model(
        path,
        manifest,
        verified=manifest.verify(),
        error_prefix="approved inference split",
    )


class ApprovedDatasetSeal(StrictFrozenModel):
    dataset: DatasetName
    private_allocation_path: str = Field(min_length=1)
    private_allocation_mode: str = Field(pattern=r"^(0600|windows)$")
    private_allocation_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    partition_case_counts: dict[PartitionName, int]
    partition_group_counts: dict[PartitionName, int]

    @model_validator(mode="after")
    def require_complete_partition_counts(self) -> Self:
        if set(self.partition_case_counts) != set(_PARTITION_ORDER):
            raise ValueError("partition case counts are incomplete")
        if set(self.partition_group_counts) != set(_PARTITION_ORDER):
            raise ValueError("partition group counts are incomplete")
        if any(value < 0 for value in self.partition_case_counts.values()):
            raise ValueError("partition case counts cannot be negative")
        if any(value < 0 for value in self.partition_group_counts.values()):
            raise ValueError("partition group counts cannot be negative")
        return self


class ApprovedSplitSeal(StrictFrozenModel):
    """Aggregate-only approval chain over private and inference split artifacts."""

    schema_version: Literal["nurse.v2.approved-split-seal/1"] = (
        APPROVED_SPLIT_SEAL_SCHEMA_VERSION
    )
    server: Literal["10.10.1.226"] = "10.10.1.226"
    approval_reference: str = Field(min_length=1)
    approved_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )
    approval_verified: Literal[True] = True
    seed: str = Field(min_length=1)
    minimum_groups_per_stratum: int = Field(ge=1)
    reservation_order: tuple[
        Literal["final"], Literal["validation"], Literal["dev"]
    ] = RESERVATION_ORDER
    identity_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_review_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_review_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_registry_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_registry_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_snapshot_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_inference_split_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    approved_inference_split_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    datasets: tuple[ApprovedDatasetSeal, ...]
    case_cross_partition_overlap: Literal[0] = 0
    group_cross_partition_overlap: Literal[0] = 0
    opaque_cross_partition_overlap: Literal[0] = 0
    inference_contains_original_case_ids: Literal[False] = False
    inference_contains_answer_values: Literal[False] = False
    inference_contains_incident_group_ids: Literal[False] = False
    seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.reservation_order != RESERVATION_ORDER:
            return False
        if self.datasets != tuple(sorted(self.datasets, key=lambda item: item.dataset)):
            return False
        if len({item.dataset for item in self.datasets}) != len(self.datasets):
            return False
        unsigned = self.model_dump(mode="json", exclude={"seal_sha256"})
        return self.seal_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_approved_split_seal(
    *,
    evaluator_map: EvaluatorIdentityMap,
    allocations: Sequence[SplitAllocation],
    private_allocation_paths: Mapping[DatasetName, Path],
    split_review: SplitReviewReport,
    split_review_path: Path,
    exposure_snapshot: ApprovedExposureSnapshot,
    exposure_snapshot_path: Path,
    inference_manifest: ApprovedInferenceSplitManifest,
    inference_manifest_path: Path,
    approval_reference: str,
    approved_at_utc: str,
) -> ApprovedSplitSeal:
    """Seal the approved review, allocations, exposure state, and safe projection."""

    frozen_allocations = tuple(sorted(allocations, key=lambda item: item.dataset))
    if not evaluator_map.verify() or any(
        not allocation.verify() for allocation in frozen_allocations
    ):
        raise ValueError("identity map and split allocations must verify")
    if not split_review.verify() or not exposure_snapshot.verify():
        raise ValueError("split review and exposure snapshot must verify")
    if not inference_manifest.verify():
        raise ValueError("approved inference split must verify")
    if approval_reference != inference_manifest.approval_reference:
        raise ValueError("approval reference does not match inference split")
    if approved_at_utc != inference_manifest.approved_at_utc:
        raise ValueError("approval time does not match inference split")
    if split_review.identity_artifact_sha256 != (
        inference_manifest.identity_artifact_sha256
    ):
        raise ValueError("identity artifact binding does not match")
    if exposure_snapshot.snapshot_sha256 != (
        inference_manifest.exposure_snapshot_sha256
    ):
        raise ValueError("exposure snapshot binding does not match")
    if set(private_allocation_paths) != {
        allocation.dataset for allocation in frozen_allocations
    }:
        raise ValueError("private allocation paths do not match datasets")
    all_assignments = tuple(
        assignment
        for allocation in frozen_allocations
        for assignment in allocation.assignments
    )
    assert_partition_disjoint(all_assignments)
    opaque_ids = tuple(
        opaque_id
        for partition in inference_manifest.partitions
        for opaque_id in partition.opaque_case_ids
    )
    if len(opaque_ids) != len(set(opaque_ids)):
        raise ValueError("opaque case crosses partitions")

    review_by_dataset = {item.dataset: item for item in split_review.datasets}
    datasets = []
    for allocation in frozen_allocations:
        review = review_by_dataset.get(allocation.dataset)
        if review is None or review.allocation_sha256 != allocation.allocation_sha256:
            raise ValueError("split review and private allocation disagree")
        path = Path(private_allocation_paths[allocation.dataset])
        _require_private_mode(path, artifact="private split allocation")
        if review.private_allocation.artifact_sha256 != _sha256_file(path):
            raise ValueError("private split allocation artifact hash changed")
        datasets.append(
            ApprovedDatasetSeal(
                dataset=allocation.dataset,
                private_allocation_path=path.as_posix(),
                private_allocation_mode=_private_mode(path),
                private_allocation_artifact_sha256=_sha256_file(path),
                allocation_sha256=allocation.allocation_sha256,
                partition_case_counts={
                    item.partition: item.actual_case_count
                    for item in allocation.partition_summaries
                },
                partition_group_counts={
                    item.partition: item.group_count
                    for item in allocation.partition_summaries
                },
            )
        )
    unsigned: dict[str, object] = {
        "schema_version": APPROVED_SPLIT_SEAL_SCHEMA_VERSION,
        "server": "10.10.1.226",
        "approval_reference": approval_reference,
        "approved_at_utc": approved_at_utc,
        "approval_verified": True,
        "seed": split_review.seed,
        "minimum_groups_per_stratum": split_review.minimum_groups_per_stratum,
        "reservation_order": RESERVATION_ORDER,
        "identity_artifact_sha256": split_review.identity_artifact_sha256,
        "split_review_artifact_sha256": _sha256_file(split_review_path),
        "split_review_report_sha256": split_review.report_sha256,
        "exposure_registry_artifact_sha256": (
            exposure_snapshot.registry_artifact_sha256
        ),
        "exposure_registry_content_sha256": (
            exposure_snapshot.registry_content_sha256
        ),
        "exposure_snapshot_artifact_sha256": _sha256_file(
            exposure_snapshot_path
        ),
        "exposure_snapshot_sha256": exposure_snapshot.snapshot_sha256,
        "approved_inference_split_artifact_sha256": _sha256_file(
            inference_manifest_path
        ),
        "approved_inference_split_manifest_sha256": (
            inference_manifest.manifest_sha256
        ),
        "datasets": [item.model_dump(mode="json") for item in datasets],
        "case_cross_partition_overlap": 0,
        "group_cross_partition_overlap": 0,
        "opaque_cross_partition_overlap": 0,
        "inference_contains_original_case_ids": False,
        "inference_contains_answer_values": False,
        "inference_contains_incident_group_ids": False,
    }
    seal = ApprovedSplitSeal.model_validate(
        {**unsigned, "seal_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not seal.verify():  # pragma: no cover - construction invariant
        raise AssertionError("approved split seal verification failed")
    return seal


def write_approved_split_seal(path: Path, seal: ApprovedSplitSeal) -> None:
    """Write the immutable aggregate-only approved split seal."""

    _write_immutable_model(
        path,
        seal,
        verified=seal.verify(),
        error_prefix="approved split seal",
    )


def load_private_split_allocation(path: Path) -> SplitAllocation:
    """Load a mode-0600 private split and verify its internal content hash."""

    source = Path(path)
    _require_private_mode(source, artifact="private split allocation")
    try:
        allocation = SplitAllocation.model_validate_json(source.read_bytes())
    except Exception as error:
        raise ValueError("invalid private split allocation") from error
    if not allocation.verify():
        raise ValueError("private split allocation verification failed")
    assert_partition_disjoint(allocation.assignments)
    return allocation


def load_split_review(path: Path) -> SplitReviewReport:
    """Load an aggregate split review and verify its internal content hash."""

    source = Path(path)
    try:
        review = SplitReviewReport.model_validate_json(source.read_bytes())
    except Exception as error:
        raise ValueError("invalid split review") from error
    if not review.verify():
        raise ValueError("split review verification failed")
    return review


def _write_immutable_model(
    path: Path,
    value: StrictFrozenModel,
    *,
    verified: bool,
    error_prefix: str,
) -> None:
    if not verified:
        raise ValueError(f"{error_prefix} verification failed")
    output = Path(path)
    data = canonical_json_bytes(value.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError(f"immutable {error_prefix} already exists")
        return
    atomic_write_bytes(output, data)


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


def _require_private_mode(path: Path, *, artifact: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{artifact} must be a regular file")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError(f"{artifact} must have mode 0600")


def _private_mode(path: Path) -> str:
    if os.name == "nt":
        return "windows"
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze an approved V2 split")
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--rcabench-allocation", type=Path, required=True)
    parser.add_argument("--aiops25-allocation", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--exposure-snapshot-output", type=Path, required=True)
    parser.add_argument("--inference-split-output", type=Path, required=True)
    parser.add_argument("--split-seal-output", type=Path, required=True)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--approved-at-utc", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluator_map = load_evaluator_identity_map(args.identity_map)
    allocation_paths: dict[DatasetName, Path] = {
        "rcabench": args.rcabench_allocation,
        "aiops25": args.aiops25_allocation,
    }
    allocations = tuple(
        load_private_split_allocation(allocation_paths[dataset])
        for dataset in _DATASET_ORDER
    )
    if any(
        allocation.dataset != dataset
        for allocation, dataset in zip(
            allocations,
            _DATASET_ORDER,
            strict=True,
        )
    ):
        raise ValueError("private split allocation dataset mismatch")
    split_review = load_split_review(args.split_review)
    exposure_snapshot = build_exposure_snapshot(
        exposure_events=load_exposure_events(args.exposure_registry),
        registry_path=args.exposure_registry,
    )
    write_private_exposure_snapshot(
        args.exposure_snapshot_output,
        exposure_snapshot,
    )
    inference_manifest = build_approved_inference_split_manifest(
        evaluator_map=evaluator_map,
        allocations=allocations,
        identity_artifact_sha256=_sha256_file(args.identity_map),
        exposure_snapshot_sha256=exposure_snapshot.snapshot_sha256,
        approval_reference=args.approval_reference,
        approved_at_utc=args.approved_at_utc,
    )
    write_approved_inference_split_manifest(
        args.inference_split_output,
        inference_manifest,
    )
    seal = build_approved_split_seal(
        evaluator_map=evaluator_map,
        allocations=allocations,
        private_allocation_paths=allocation_paths,
        split_review=split_review,
        split_review_path=args.split_review,
        exposure_snapshot=exposure_snapshot,
        exposure_snapshot_path=args.exposure_snapshot_output,
        inference_manifest=inference_manifest,
        inference_manifest_path=args.inference_split_output,
        approval_reference=args.approval_reference,
        approved_at_utc=args.approved_at_utc,
    )
    write_approved_split_seal(args.split_seal_output, seal)
    print(json.dumps(seal.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
