"""Public dataset registry for NURSE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .aiops25 import aiops25_case_dirs


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    layout: str
    relative_root: Tuple[str, ...]
    prefix: Optional[str] = None
    source_dataset: Optional[str] = None
    exclude_instance_types: Tuple[str, ...] = ()


MAIN_DATASETS: Tuple[str, ...] = (
    "aiops25-retained-nonnode",
    "rcaeval-tt",
)

DATASET_SPECS = {
    "aiops25-retained-nonnode": DatasetSpec(
        name="aiops25-retained-nonnode",
        layout="aiops25",
        relative_root=("datasets", "25挑战赛", "aiopschallenge2025-main"),
        source_dataset="aiops25",
        exclude_instance_types=("node",),
    ),
    "rcaeval-tt": DatasetSpec(
        name="rcaeval-tt",
        layout="rcaeval",
        relative_root=("datasets", "RCAEval-v2", "data"),
        prefix="re1tt_",
    ),
}


def dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc


def dataset_root(dataset: str, lab_root: str | Path) -> Path:
    spec = dataset_spec(dataset)
    root = Path(lab_root)
    for part in spec.relative_root:
        root = root / part
    return root


def case_dirs_for_dataset(dataset: str, lab_root: str | Path) -> list[Path]:
    spec = dataset_spec(dataset)
    if spec.layout == "aiops25":
        return aiops25_case_dirs(
            lab_root,
            dataset=spec.name,
            exclude_instance_types=spec.exclude_instance_types,
        )
    root = dataset_root(dataset, lab_root)
    paths = sorted(path for path in root.iterdir() if path.is_dir())
    if spec.prefix:
        paths = [path for path in paths if path.name.startswith(spec.prefix)]
    return paths


def is_rcaeval_dataset(dataset: str) -> bool:
    return dataset_spec(dataset).layout == "rcaeval"


def dataset_choices() -> list[str]:
    return list(MAIN_DATASETS)