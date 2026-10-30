"""Deterministic incident and near-duplicate grouping for evaluator-side cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

INCIDENT_GROUP_POLICY_VERSION: Literal[
    "nurse.v2.incident-group-policy/1"
] = "nurse.v2.incident-group-policy/1"
INCIDENT_GROUP_MANIFEST_SCHEMA_VERSION: Literal[
    "nurse.v2.incident-group-manifest/1"
] = "nurse.v2.incident-group-manifest/1"
RCABENCH_GROUPING_TELEMETRY = (
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
AIOPS25_GROUPING_TELEMETRY = (
    "abnormal_logs.parquet",
    "abnormal_metrics.parquet",
    "abnormal_traces.parquet",
    "normal_logs.parquet",
    "normal_metrics.parquet",
    "normal_traces.parquet",
)
_CONTENT_SAMPLE_BYTES = 64 * 1024

GroupingReason = Literal[
    "confirmed_duplicate",
    "explicit_incident",
    "overlapping_injection_run",
    "overlapping_matching_capture_content",
]


class CaseGroupingRecord(StrictFrozenModel):
    """Label-free grouping inputs produced inside the isolated evaluator."""

    case_id: str = Field(min_length=1)
    interval_start_ns: int
    interval_end_ns: int
    capture_context_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    explicit_incident_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    injection_run_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    confirmed_duplicate_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def require_nonempty_interval(self) -> Self:
        if self.interval_start_ns >= self.interval_end_ns:
            raise ValueError("grouping interval must have positive duration")
        return self


class GroupingEvidenceEdge(StrictFrozenModel):
    left_case_id: str = Field(min_length=1)
    right_case_id: str = Field(min_length=1)
    reasons: tuple[GroupingReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_edge(self) -> Self:
        if self.left_case_id >= self.right_case_id:
            raise ValueError("grouping edge case IDs must be strictly ordered")
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("grouping edge reasons must be sorted and unique")
        return self


class IncidentGroup(StrictFrozenModel):
    group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    case_ids: tuple[str, ...] = Field(min_length=1)
    evidence_reasons: tuple[GroupingReason, ...]
    edges: tuple[GroupingEvidenceEdge, ...]

    @model_validator(mode="after")
    def require_consistent_group(self) -> Self:
        if self.case_ids != tuple(sorted(set(self.case_ids))):
            raise ValueError("group case IDs must be sorted and unique")
        if self.evidence_reasons != tuple(sorted(set(self.evidence_reasons))):
            raise ValueError("group evidence reasons must be sorted and unique")
        if self.edges != tuple(
            sorted(self.edges, key=lambda edge: (edge.left_case_id, edge.right_case_id))
        ):
            raise ValueError("group edges must be canonically ordered")
        member_set = set(self.case_ids)
        if any(
            edge.left_case_id not in member_set or edge.right_case_id not in member_set
            for edge in self.edges
        ):
            raise ValueError("group edge references a case outside its group")
        edge_reasons = tuple(
            sorted({reason for edge in self.edges for reason in edge.reasons})
        )
        if edge_reasons != self.evidence_reasons:
            raise ValueError("group reasons must equal the union of edge reasons")
        if len(self.case_ids) == 1 and (self.edges or self.evidence_reasons):
            raise ValueError("singleton groups cannot have grouping evidence")
        if len(self.case_ids) > 1 and not self.edges:
            raise ValueError("multi-case groups require grouping evidence")
        return self


class IncidentGroupManifest(StrictFrozenModel):
    schema_version: Literal["nurse.v2.incident-group-manifest/1"] = (
        INCIDENT_GROUP_MANIFEST_SCHEMA_VERSION
    )
    grouping_policy_version: Literal["nurse.v2.incident-group-policy/1"] = (
        INCIDENT_GROUP_POLICY_VERSION
    )
    dataset: DatasetName
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grouping_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    grouped_case_count: int = Field(ge=0)
    singleton_group_count: int = Field(ge=0)
    multi_case_group_count: int = Field(ge=0)
    max_group_size: int = Field(ge=0)
    groups: tuple[IncidentGroup, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.groups != tuple(sorted(self.groups, key=lambda group: group.case_ids)):
            return False
        all_case_ids = tuple(case_id for group in self.groups for case_id in group.case_ids)
        if len(all_case_ids) != len(set(all_case_ids)):
            return False
        if any(
            group.group_id != _group_id(self.dataset, group.case_ids)
            for group in self.groups
        ):
            return False
        singleton_count = sum(len(group.case_ids) == 1 for group in self.groups)
        multi_count = sum(len(group.case_ids) > 1 for group in self.groups)
        max_size = max((len(group.case_ids) for group in self.groups), default=0)
        if (
            self.case_count != len(all_case_ids)
            or self.grouped_case_count != len(all_case_ids)
            or self.group_count != len(self.groups)
            or self.singleton_group_count != singleton_count
            or self.multi_case_group_count != multi_count
            or self.max_group_size != max_size
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        return self.manifest_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class _DisjointSet:
    def __init__(self, case_ids: Sequence[str]) -> None:
        self._parent = {case_id: case_id for case_id in case_ids}

    def find(self, case_id: str) -> str:
        parent = self._parent[case_id]
        if parent != case_id:
            self._parent[case_id] = self.find(parent)
        return self._parent[case_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def build_incident_group_manifest(
    *,
    dataset: DatasetName,
    records: Sequence[CaseGroupingRecord],
    source_inventory_sha256: str,
) -> IncidentGroupManifest:
    """Build transitive, case-atomic groups from only predeclared evidence keys."""

    frozen_records = tuple(sorted(records, key=lambda record: record.case_id))
    case_ids = tuple(record.case_id for record in frozen_records)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case identities must be unique before grouping")
    disjoint = _DisjointSet(case_ids)
    edges: list[GroupingEvidenceEdge] = []
    for left_index, left in enumerate(frozen_records):
        for right in frozen_records[left_index + 1 :]:
            reasons = _pair_reasons(left, right)
            if not reasons:
                continue
            edge = GroupingEvidenceEdge(
                left_case_id=left.case_id,
                right_case_id=right.case_id,
                reasons=reasons,
            )
            edges.append(edge)
            disjoint.union(left.case_id, right.case_id)

    members: dict[str, list[str]] = {}
    for case_id in case_ids:
        members.setdefault(disjoint.find(case_id), []).append(case_id)
    groups: list[IncidentGroup] = []
    for member_values in members.values():
        member_ids = tuple(sorted(member_values))
        member_set = set(member_ids)
        member_edges = tuple(
            edge
            for edge in edges
            if edge.left_case_id in member_set and edge.right_case_id in member_set
        )
        reasons = tuple(
            sorted({reason for edge in member_edges for reason in edge.reasons})
        )
        groups.append(
            IncidentGroup(
                group_id=_group_id(dataset, member_ids),
                case_ids=member_ids,
                evidence_reasons=reasons,
                edges=member_edges,
            )
        )

    frozen_groups = tuple(sorted(groups, key=lambda group: group.case_ids))
    records_sha256 = sha256_hex(
        canonical_json_bytes(
            [record.model_dump(mode="json") for record in frozen_records]
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": INCIDENT_GROUP_MANIFEST_SCHEMA_VERSION,
        "grouping_policy_version": INCIDENT_GROUP_POLICY_VERSION,
        "dataset": dataset,
        "source_inventory_sha256": source_inventory_sha256,
        "grouping_records_sha256": records_sha256,
        "case_count": len(case_ids),
        "group_count": len(frozen_groups),
        "grouped_case_count": sum(len(group.case_ids) for group in frozen_groups),
        "singleton_group_count": sum(
            len(group.case_ids) == 1 for group in frozen_groups
        ),
        "multi_case_group_count": sum(
            len(group.case_ids) > 1 for group in frozen_groups
        ),
        "max_group_size": max(
            (len(group.case_ids) for group in frozen_groups),
            default=0,
        ),
        "groups": [group.model_dump(mode="json") for group in frozen_groups],
    }
    manifest = IncidentGroupManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not manifest.verify():  # pragma: no cover - construction invariant
        raise AssertionError("incident-group manifest verification failed")
    return manifest


def build_rcabench_grouping_records(
    dataset_root: Path,
    *,
    case_ids: Sequence[str],
) -> tuple[CaseGroupingRecord, ...]:
    """Derive label-free grouping keys from audited RCABench case metadata."""

    root = Path(dataset_root).resolve(strict=True)
    records: list[CaseGroupingRecord] = []
    for case_id in sorted(case_ids):
        case_root = root / case_id
        if not case_root.is_dir() or case_root.is_symlink():
            raise FileNotFoundError(f"{case_id}: audited case directory is unavailable")
        env = _load_json_object(case_root / "env.json")
        injection = _load_json_object(case_root / "injection.json")
        namespace = _required_string(env, "NAMESPACE", case_id=case_id)
        benchmark = _required_string(injection, "benchmark", case_id=case_id)
        task_id = _optional_string(injection.get("task_id"))
        records.append(
            CaseGroupingRecord(
                case_id=case_id,
                interval_start_ns=_timestamp_ns(
                    env.get("NORMAL_START"),
                    field="NORMAL_START",
                    case_id=case_id,
                ),
                interval_end_ns=_timestamp_ns(
                    env.get("ABNORMAL_END"),
                    field="ABNORMAL_END",
                    case_id=case_id,
                ),
                capture_context_key=_metadata_key(
                    "rcabench-capture-context",
                    {"benchmark": benchmark, "namespace": namespace},
                ),
                content_fingerprint=_bounded_case_content_fingerprint(
                    case_root,
                    filenames=RCABENCH_GROUPING_TELEMETRY,
                    extra_files=(),
                ),
                explicit_incident_key=None,
                injection_run_key=(
                    _metadata_key("rcabench-injection-run", task_id)
                    if task_id is not None
                    else None
                ),
                confirmed_duplicate_key=None,
            )
        )
    return tuple(records)


def build_aiops25_grouping_records(
    work_root: Path,
    *,
    case_ids: Sequence[str],
) -> tuple[CaseGroupingRecord, ...]:
    """Derive label-free grouping keys from audited AIOps25 preparation metadata."""

    root = Path(work_root).resolve(strict=True)
    manifest = _load_json_object(root / "manifests" / "manifest.json")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("AIOps25 manifest cases must be an array")
    by_id: dict[str, dict[str, object]] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("AIOps25 manifest case must be an object")
        case_id = raw_case.get("uuid")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("AIOps25 manifest case has invalid uuid")
        if case_id in by_id:
            raise ValueError(f"duplicate AIOps25 manifest case ID: {case_id}")
        by_id[case_id] = raw_case

    records: list[CaseGroupingRecord] = []
    for case_id in sorted(case_ids):
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"{case_id}: case is absent from AIOps25 preparation manifest")
        case_root = root / "canonical" / case_id
        if not case_root.is_dir() or case_root.is_symlink():
            raise FileNotFoundError(f"{case_id}: audited prepared case is unavailable")
        normal_days = _string_array(case.get("normal_days"), field="normal_days", case_id=case_id)
        abnormal_days = _string_array(
            case.get("abnormal_days"),
            field="abnormal_days",
            case_id=case_id,
        )
        records.append(
            CaseGroupingRecord(
                case_id=case_id,
                interval_start_ns=_timestamp_ns(
                    case.get("effective_abnormal_start"),
                    field="effective_abnormal_start",
                    case_id=case_id,
                ),
                interval_end_ns=_timestamp_ns(
                    case.get("effective_abnormal_end"),
                    field="effective_abnormal_end",
                    case_id=case_id,
                ),
                capture_context_key=_metadata_key(
                    "aiops25-capture-context",
                    {"abnormal_days": abnormal_days, "normal_days": normal_days},
                ),
                content_fingerprint=_bounded_case_content_fingerprint(
                    case_root,
                    filenames=AIOPS25_GROUPING_TELEMETRY,
                    extra_files=(".prepared.json",),
                ),
                explicit_incident_key=None,
                injection_run_key=None,
                confirmed_duplicate_key=None,
            )
        )
    return tuple(records)


def write_incident_group_manifest(path: Path, manifest: IncidentGroupManifest) -> None:
    """Write one immutable group manifest while permitting byte-identical replay."""

    output = Path(path)
    data = canonical_json_bytes(manifest.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable incident-group artifact already exists")
        return
    if not manifest.verify():
        raise ValueError("incident-group manifest verification failed")
    atomic_write_bytes(output, data)


def _pair_reasons(
    left: CaseGroupingRecord,
    right: CaseGroupingRecord,
) -> tuple[GroupingReason, ...]:
    reasons: list[GroupingReason] = []
    if _matching_key(left.explicit_incident_key, right.explicit_incident_key):
        reasons.append("explicit_incident")
    if _matching_key(left.confirmed_duplicate_key, right.confirmed_duplicate_key):
        reasons.append("confirmed_duplicate")
    overlaps = _intervals_overlap(left, right)
    if overlaps and _matching_key(left.injection_run_key, right.injection_run_key):
        reasons.append("overlapping_injection_run")
    if (
        overlaps
        and left.capture_context_key == right.capture_context_key
        and left.content_fingerprint == right.content_fingerprint
    ):
        reasons.append("overlapping_matching_capture_content")
    return tuple(sorted(reasons))


def _matching_key(left: str | None, right: str | None) -> bool:
    return left is not None and left == right


def _intervals_overlap(left: CaseGroupingRecord, right: CaseGroupingRecord) -> bool:
    return max(left.interval_start_ns, right.interval_start_ns) < min(
        left.interval_end_ns,
        right.interval_end_ns,
    )


def _group_id(dataset: DatasetName, case_ids: Sequence[str]) -> str:
    digest = sha256_hex(
        canonical_json_bytes({"dataset": dataset, "case_ids": tuple(case_ids)})
    )
    return f"grp_{digest[:24]}"


def _bounded_case_content_fingerprint(
    case_root: Path,
    *,
    filenames: Sequence[str],
    extra_files: Sequence[str],
) -> str:
    descriptors = []
    for name in sorted((*filenames, *extra_files)):
        path = case_root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{case_root.name}: missing audited grouping input {name}")
        size = path.stat().st_size
        digest = hashlib.sha256()
        digest.update(str(size).encode("ascii"))
        with path.open("rb") as stream:
            digest.update(stream.read(_CONTENT_SAMPLE_BYTES))
            if size > _CONTENT_SAMPLE_BYTES:
                stream.seek(max(0, size - _CONTENT_SAMPLE_BYTES))
                digest.update(stream.read(_CONTENT_SAMPLE_BYTES))
        descriptors.append(
            {"name": name, "sample_sha256": digest.hexdigest(), "size": size}
        )
    return sha256_hex(canonical_json_bytes(descriptors))


def _metadata_key(kind: str, value: object) -> str:
    return sha256_hex(canonical_json_bytes({"kind": kind, "value": value}))


def _timestamp_ns(value: object, *, field: str, case_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{case_id}: invalid {field}")
    if isinstance(value, int | float):
        return int(float(value) * 1_000_000_000)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id}: invalid {field}")
    stripped = value.strip()
    try:
        numeric = float(stripped)
    except ValueError:
        normalized = stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(f"{case_id}: invalid {field}") from error
        if timestamp.tzinfo is None:
            raise ValueError(
                f"{case_id}: {field} must include a timezone"
            ) from None
        return int(timestamp.timestamp() * 1_000_000_000)
    return int(numeric * 1_000_000_000)


def _required_string(payload: dict[str, object], field: str, *, case_id: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id}: invalid {field}")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and bool(value.strip()) else None


def _string_array(value: object, *, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{case_id}: invalid {field}")
    return tuple(sorted(value))


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_inventory(path: Path, *, dataset: DatasetName) -> dict[str, object]:
    inventory = _load_json_object(path)
    if inventory.get("dataset") != dataset:
        raise ValueError("inventory dataset does not match grouping dataset")
    inventory_sha256 = inventory.get("inventory_sha256")
    if not isinstance(inventory_sha256, str):
        raise ValueError("inventory has no content hash")
    unsigned = dict(inventory)
    del unsigned["inventory_sha256"]
    if inventory_sha256 != sha256_hex(canonical_json_bytes(unsigned)):
        raise ValueError("inventory content hash verification failed")
    return inventory


def _inventory_case_ids(inventory: dict[str, object], *, dataset: DatasetName) -> tuple[str, ...]:
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("inventory entries must be an array")
    expected_status = "valid" if dataset == "rcabench" else "included"
    case_ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("inventory entry must be an object")
        if entry.get("status") != expected_status:
            continue
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("inventory entry has invalid case_id")
        case_ids.append(case_id)
    expected_count = 1_422 if dataset == "rcabench" else 318
    if len(case_ids) != expected_count or len(case_ids) != len(set(case_ids)):
        raise ValueError("inventory does not contain the expected unique grouping population")
    return tuple(sorted(case_ids))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build atomic incident-group manifests")
    parser.add_argument("--dataset", choices=("rcabench", "aiops25"), required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset: DatasetName = args.dataset
    inventory = _load_inventory(args.inventory, dataset=dataset)
    case_ids = _inventory_case_ids(inventory, dataset=dataset)
    inventory_sha256 = inventory["inventory_sha256"]
    if not isinstance(inventory_sha256, str):  # pragma: no cover - validated above
        raise AssertionError("validated inventory hash changed type")
    if dataset == "rcabench":
        if args.data_root is None:
            raise ValueError("--data-root is required for RCABench")
        records = build_rcabench_grouping_records(args.data_root, case_ids=case_ids)
    else:
        if args.work_root is None:
            raise ValueError("--work-root is required for AIOps25")
        records = build_aiops25_grouping_records(args.work_root, case_ids=case_ids)
    manifest = build_incident_group_manifest(
        dataset=dataset,
        records=records,
        source_inventory_sha256=inventory_sha256,
    )
    write_incident_group_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "case_count": manifest.case_count,
                "group_count": manifest.group_count,
                "grouped_case_count": manifest.grouped_case_count,
                "manifest_sha256": manifest.manifest_sha256,
                "max_group_size": manifest.max_group_size,
                "multi_case_group_count": manifest.multi_case_group_count,
                "singleton_group_count": manifest.singleton_group_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
