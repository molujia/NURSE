"""Evaluator-only, group-atomic and exposure-aware dataset allocation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

SPLIT_ALLOCATION_SCHEMA_VERSION: Literal[
    "nurse.v2.split-allocation/1"
] = "nurse.v2.split-allocation/1"

PartitionName = Literal["dev", "validation", "final", "remainder"]
AllocationPhase = Literal[
    "reserved_final",
    "reserved_validation",
    "forced_exposed_dev",
    "dev_fill",
    "remainder",
]
StratumLevel = Literal["target_fault", "fault_type", "dataset"]

RESERVATION_ORDER: tuple[Literal["final"], Literal["validation"], Literal["dev"]] = (
    "final",
    "validation",
    "dev",
)
_SUMMARY_ORDER: tuple[PartitionName, ...] = (
    "dev",
    "validation",
    "final",
    "remainder",
)


class EvaluatorSplitCase(StrictFrozenModel):
    """Private allocation metadata that must never enter inference."""

    dataset: DatasetName
    case_id: str = Field(min_length=1)
    incident_group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    target_services: tuple[str, ...] = Field(min_length=1)
    fault_type: str = Field(min_length=1)
    exposed: bool = False

    @model_validator(mode="after")
    def require_canonical_private_labels(self) -> Self:
        if self.target_services != tuple(sorted(set(self.target_services))):
            raise ValueError("target services must be sorted and unique")
        if any(not value.strip() for value in self.target_services):
            raise ValueError("target services cannot be blank")
        if self.fault_type != self.fault_type.strip():
            raise ValueError("fault type must be stripped")
        return self


class SplitTargets(StrictFrozenModel):
    """Requested case counts; group integrity can cause recorded deviations."""

    dev_case_count: int = Field(ge=0)
    validation_case_count: int = Field(ge=0)
    final_case_count: int = Field(ge=0)


class StratumCount(StrictFrozenModel):
    stratum_level: StratumLevel
    stratum_id: str = Field(pattern=r"^str_[0-9a-f]{24}$")
    case_count: int = Field(ge=1)
    group_count: int = Field(ge=1)


class GroupAssignment(StrictFrozenModel):
    """One indivisible incident group and its single assigned partition."""

    incident_group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    case_ids: tuple[str, ...] = Field(min_length=1)
    partition: PartitionName
    allocation_phase: AllocationPhase
    exposed: bool
    stratum_level: StratumLevel
    stratum_id: str = Field(pattern=r"^str_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def require_atomic_consistent_assignment(self) -> Self:
        if self.case_ids != tuple(sorted(set(self.case_ids))):
            raise ValueError("group case IDs must be sorted and unique")
        allowed_phases: Mapping[PartitionName, tuple[AllocationPhase, ...]] = {
            "dev": ("forced_exposed_dev", "dev_fill"),
            "validation": ("reserved_validation",),
            "final": ("reserved_final",),
            "remainder": ("remainder",),
        }
        if self.allocation_phase not in allowed_phases[self.partition]:
            raise ValueError("allocation phase does not match partition")
        if self.exposed != (self.allocation_phase == "forced_exposed_dev"):
            raise ValueError("exposure state does not match allocation phase")
        return self


class PartitionSummary(StrictFrozenModel):
    partition: PartitionName
    requested_case_count: int | None = Field(default=None, ge=0)
    actual_case_count: int = Field(ge=0)
    case_count_deviation: int | None = None
    group_count: int = Field(ge=0)
    exposed_case_count: int = Field(ge=0)
    exposed_group_count: int = Field(ge=0)
    strata: tuple[StratumCount, ...]

    @model_validator(mode="after")
    def require_consistent_deviation(self) -> Self:
        if self.partition == "remainder":
            if self.requested_case_count is not None or self.case_count_deviation is not None:
                raise ValueError("remainder has no requested case count")
        elif (
            self.requested_case_count is None
            or self.case_count_deviation
            != self.actual_case_count - self.requested_case_count
        ):
            raise ValueError("partition deviation does not match requested count")
        if self.exposed_case_count > self.actual_case_count:
            raise ValueError("exposed case count exceeds partition case count")
        if self.exposed_group_count > self.group_count:
            raise ValueError("exposed group count exceeds partition group count")
        if self.strata != tuple(
            sorted(self.strata, key=lambda item: (item.stratum_level, item.stratum_id))
        ):
            raise ValueError("partition strata must be canonically ordered")
        return self


class SplitAllocation(StrictFrozenModel):
    """Hashed evaluator allocation with all groups represented exactly once."""

    schema_version: Literal["nurse.v2.split-allocation/1"] = (
        SPLIT_ALLOCATION_SCHEMA_VERSION
    )
    dataset: DatasetName
    seed: str = Field(min_length=1)
    minimum_groups_per_stratum: int = Field(ge=1)
    reservation_order: tuple[
        Literal["final"], Literal["validation"], Literal["dev"]
    ] = RESERVATION_ORDER
    targets: SplitTargets
    case_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    exposed_case_count: int = Field(ge=0)
    exposed_group_count: int = Field(ge=0)
    assignments: tuple[GroupAssignment, ...]
    partition_summaries: tuple[PartitionSummary, ...]
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.reservation_order != RESERVATION_ORDER:
            return False
        if self.assignments != tuple(
            sorted(self.assignments, key=lambda item: item.incident_group_id)
        ):
            return False
        group_ids = tuple(item.incident_group_id for item in self.assignments)
        case_ids = tuple(case_id for item in self.assignments for case_id in item.case_ids)
        if len(group_ids) != len(set(group_ids)) or len(case_ids) != len(set(case_ids)):
            return False
        if self.group_count != len(group_ids) or self.case_count != len(case_ids):
            return False
        exposed_assignments = tuple(item for item in self.assignments if item.exposed)
        if any(item.partition != "dev" for item in exposed_assignments):
            return False
        if self.exposed_group_count != len(exposed_assignments):
            return False
        if self.exposed_case_count != sum(
            len(item.case_ids) for item in exposed_assignments
        ):
            return False
        if tuple(item.partition for item in self.partition_summaries) != _SUMMARY_ORDER:
            return False
        requested = {
            "dev": self.targets.dev_case_count,
            "validation": self.targets.validation_case_count,
            "final": self.targets.final_case_count,
            "remainder": None,
        }
        for summary in self.partition_summaries:
            members = tuple(
                item for item in self.assignments if item.partition == summary.partition
            )
            if summary.requested_case_count != requested[summary.partition]:
                return False
            if summary.actual_case_count != sum(len(item.case_ids) for item in members):
                return False
            if summary.group_count != len(members):
                return False
            if summary.exposed_case_count != sum(
                len(item.case_ids) for item in members if item.exposed
            ):
                return False
            if summary.exposed_group_count != sum(item.exposed for item in members):
                return False
            if summary.strata != _summarize_strata(members):
                return False
        unsigned = self.model_dump(mode="json", exclude={"allocation_sha256"})
        return self.allocation_sha256 == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True)
class _Group:
    incident_group_id: str
    case_ids: tuple[str, ...]
    exposed: bool
    stratum_level: StratumLevel
    stratum_id: str

    @property
    def case_count(self) -> int:
        return len(self.case_ids)


def allocate_group_atomic_split(
    cases: Sequence[EvaluatorSplitCase],
    *,
    targets: SplitTargets,
    seed: str,
    minimum_groups_per_stratum: int = 3,
) -> SplitAllocation:
    """Reserve Final, then Validation, then expose/fill Dev without splitting groups."""

    if not seed:
        raise ValueError("split seed cannot be empty")
    if minimum_groups_per_stratum < 1:
        raise ValueError("minimum groups per stratum must be positive")
    ordered_cases = tuple(
        sorted(cases, key=lambda item: (item.dataset, item.case_id))
    )
    case_keys = tuple((item.dataset, item.case_id) for item in ordered_cases)
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("case identities must be unique")
    datasets = {item.dataset for item in ordered_cases}
    if len(datasets) != 1:
        raise ValueError("one allocation must contain exactly one dataset")
    dataset = next(iter(datasets))
    groups = _build_groups(
        ordered_cases,
        dataset=dataset,
        minimum_groups_per_stratum=minimum_groups_per_stratum,
    )
    exposed_groups = tuple(group for group in groups if group.exposed)
    unexposed_groups = tuple(group for group in groups if not group.exposed)

    final_groups = _select_groups(
        unexposed_groups,
        target_case_count=targets.final_case_count,
        seed=seed,
        partition="final",
        prefer_at_least=False,
    )
    final_ids = {group.incident_group_id for group in final_groups}
    after_final = tuple(
        group for group in unexposed_groups if group.incident_group_id not in final_ids
    )
    validation_groups = _select_groups(
        after_final,
        target_case_count=targets.validation_case_count,
        seed=seed,
        partition="validation",
        prefer_at_least=False,
    )
    validation_ids = {group.incident_group_id for group in validation_groups}
    after_validation = tuple(
        group
        for group in after_final
        if group.incident_group_id not in validation_ids
    )
    exposed_case_count = sum(group.case_count for group in exposed_groups)
    dev_fill_groups = _select_groups(
        after_validation,
        target_case_count=max(0, targets.dev_case_count - exposed_case_count),
        seed=seed,
        partition="dev",
        prefer_at_least=True,
    )
    dev_fill_ids = {group.incident_group_id for group in dev_fill_groups}
    remainder_groups = tuple(
        group
        for group in after_validation
        if group.incident_group_id not in dev_fill_ids
    )

    assignments = tuple(
        sorted(
            (
                *(
                    _assignment(group, "final", "reserved_final")
                    for group in final_groups
                ),
                *(
                    _assignment(group, "validation", "reserved_validation")
                    for group in validation_groups
                ),
                *(
                    _assignment(group, "dev", "forced_exposed_dev")
                    for group in exposed_groups
                ),
                *(
                    _assignment(group, "dev", "dev_fill")
                    for group in dev_fill_groups
                ),
                *(
                    _assignment(group, "remainder", "remainder")
                    for group in remainder_groups
                ),
            ),
            key=lambda item: item.incident_group_id,
        )
    )
    partition_summaries = tuple(
        _partition_summary(
            partition,
            assignments,
            requested_case_count={
                "dev": targets.dev_case_count,
                "validation": targets.validation_case_count,
                "final": targets.final_case_count,
                "remainder": None,
            }[partition],
        )
        for partition in _SUMMARY_ORDER
    )
    unsigned: dict[str, object] = {
        "schema_version": SPLIT_ALLOCATION_SCHEMA_VERSION,
        "dataset": dataset,
        "seed": seed,
        "minimum_groups_per_stratum": minimum_groups_per_stratum,
        "reservation_order": RESERVATION_ORDER,
        "targets": targets.model_dump(mode="json"),
        "case_count": len(ordered_cases),
        "group_count": len(groups),
        "exposed_case_count": exposed_case_count,
        "exposed_group_count": len(exposed_groups),
        "assignments": [item.model_dump(mode="json") for item in assignments],
        "partition_summaries": [
            item.model_dump(mode="json") for item in partition_summaries
        ],
    }
    allocation = SplitAllocation.model_validate(
        {
            **unsigned,
            "allocation_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not allocation.verify():  # pragma: no cover - construction invariant
        raise AssertionError("split allocation verification failed")
    return allocation


def _build_groups(
    cases: Sequence[EvaluatorSplitCase],
    *,
    dataset: DatasetName,
    minimum_groups_per_stratum: int,
) -> tuple[_Group, ...]:
    grouped: dict[str, list[EvaluatorSplitCase]] = defaultdict(list)
    for item in cases:
        grouped[item.incident_group_id].append(item)

    fine_values: dict[str, object | None] = {}
    fault_values: dict[str, str | None] = {}
    fine_counts: Counter[bytes] = Counter()
    fault_counts: Counter[str] = Counter()
    for group_id, members in grouped.items():
        if any(item.dataset != dataset for item in members):
            raise ValueError("one incident group cannot cross datasets")
        fine_signatures = {
            (item.target_services, item.fault_type) for item in members
        }
        fault_types = {item.fault_type for item in members}
        fine_value: object | None = None
        if len(fine_signatures) == 1:
            target_services, fault_type = next(iter(fine_signatures))
            fine_value = {
                "target_services": target_services,
                "fault_type": fault_type,
            }
            fine_counts[canonical_json_bytes(fine_value)] += 1
        fault_value = next(iter(fault_types)) if len(fault_types) == 1 else None
        if fault_value is not None:
            fault_counts[fault_value] += 1
        fine_values[group_id] = fine_value
        fault_values[group_id] = fault_value

    result: list[_Group] = []
    for group_id, members in grouped.items():
        fine_value = fine_values[group_id]
        fault_value = fault_values[group_id]
        if (
            fine_value is not None
            and fine_counts[canonical_json_bytes(fine_value)]
            >= minimum_groups_per_stratum
        ):
            level: StratumLevel = "target_fault"
            value = fine_value
        elif (
            fault_value is not None
            and fault_counts[fault_value] >= minimum_groups_per_stratum
        ):
            level = "fault_type"
            value = fault_value
        else:
            level = "dataset"
            value = dataset
        result.append(
            _Group(
                incident_group_id=group_id,
                case_ids=tuple(sorted(item.case_id for item in members)),
                exposed=any(item.exposed for item in members),
                stratum_level=level,
                stratum_id=_stratum_id(dataset=dataset, level=level, value=value),
            )
        )
    return tuple(sorted(result, key=lambda item: item.incident_group_id))


def _select_groups(
    groups: Sequence[_Group],
    *,
    target_case_count: int,
    seed: str,
    partition: Literal["dev", "validation", "final"],
    prefer_at_least: bool,
) -> tuple[_Group, ...]:
    if target_case_count == 0 or not groups:
        return ()
    pool_case_count = sum(group.case_count for group in groups)
    stratum_case_counts: Counter[str] = Counter()
    for group in groups:
        stratum_case_counts[group.stratum_id] += group.case_count
    selected_stratum_counts: Counter[str] = Counter()
    selected: list[_Group] = []
    remaining = list(groups)
    selected_case_count = 0
    while selected_case_count < target_case_count and remaining:
        space = target_case_count - selected_case_count
        fitting = tuple(group for group in remaining if group.case_count <= space)
        if fitting:
            candidates = fitting
        else:
            candidates = tuple(
                group
                for group in remaining
                if prefer_at_least
                or abs(selected_case_count + group.case_count - target_case_count)
                <= abs(selected_case_count - target_case_count)
            )
            if not candidates:
                break

        def choice_key(
            group: _Group,
            available_space: int = space,
        ) -> tuple[Fraction, int, str, str]:
            quota = Fraction(
                target_case_count * stratum_case_counts[group.stratum_id],
                pool_case_count,
            )
            deficit = quota - selected_stratum_counts[group.stratum_id]
            overshoot = max(0, group.case_count - available_space)
            return (
                -deficit,
                overshoot,
                _selection_priority(seed, partition, group.incident_group_id),
                group.incident_group_id,
            )

        chosen = min(candidates, key=choice_key)
        selected.append(chosen)
        selected_case_count += chosen.case_count
        selected_stratum_counts[chosen.stratum_id] += chosen.case_count
        remaining.remove(chosen)
    return tuple(selected)


def _assignment(
    group: _Group,
    partition: PartitionName,
    phase: AllocationPhase,
) -> GroupAssignment:
    return GroupAssignment(
        incident_group_id=group.incident_group_id,
        case_ids=group.case_ids,
        partition=partition,
        allocation_phase=phase,
        exposed=group.exposed,
        stratum_level=group.stratum_level,
        stratum_id=group.stratum_id,
    )


def _partition_summary(
    partition: PartitionName,
    assignments: Sequence[GroupAssignment],
    *,
    requested_case_count: int | None,
) -> PartitionSummary:
    members = tuple(item for item in assignments if item.partition == partition)
    actual_case_count = sum(len(item.case_ids) for item in members)
    return PartitionSummary(
        partition=partition,
        requested_case_count=requested_case_count,
        actual_case_count=actual_case_count,
        case_count_deviation=(
            None
            if requested_case_count is None
            else actual_case_count - requested_case_count
        ),
        group_count=len(members),
        exposed_case_count=sum(
            len(item.case_ids) for item in members if item.exposed
        ),
        exposed_group_count=sum(item.exposed for item in members),
        strata=_summarize_strata(members),
    )


def _summarize_strata(
    assignments: Sequence[GroupAssignment],
) -> tuple[StratumCount, ...]:
    by_stratum: dict[tuple[StratumLevel, str], list[GroupAssignment]] = defaultdict(
        list
    )
    for item in assignments:
        by_stratum[(item.stratum_level, item.stratum_id)].append(item)
    return tuple(
        StratumCount(
            stratum_level=level,
            stratum_id=stratum_id,
            case_count=sum(len(item.case_ids) for item in members),
            group_count=len(members),
        )
        for (level, stratum_id), members in sorted(by_stratum.items())
    )


def _stratum_id(*, dataset: DatasetName, level: StratumLevel, value: object) -> str:
    digest = sha256_hex(
        canonical_json_bytes(
            {"dataset": dataset, "level": level, "private_value": value}
        )
    )
    return f"str_{digest[:24]}"


def _selection_priority(
    seed: str,
    partition: str,
    incident_group_id: str,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "domain": "nurse.v2.split-selection/1",
                "seed": seed,
                "partition": partition,
                "incident_group_id": incident_group_id,
            }
        )
    )
