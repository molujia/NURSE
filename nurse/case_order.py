"""Deterministic case ordering for shared experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .datasets import case_dirs_for_dataset


@dataclass(frozen=True)
class CaseOrderEntry:
    dataset: str
    case_id: str
    case_dir: Path
    order_index: int

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "case_dir": str(self.case_dir),
            "order_index": self.order_index,
        }


def _ordered_case_dirs(
    dataset: str,
    *,
    lab_root: str | Path,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Path]:
    paths = list(case_dirs_for_dataset(dataset, lab_root))
    if shuffle:
        rng = random.Random(f"{int(seed)}:{dataset}")
        rng.shuffle(paths)
    return paths


def build_case_order(
    dataset: str,
    *,
    lab_root: str | Path,
    seed: int = 0,
    shuffle: bool = False,
    sample_size: int = 0,
    case_limit: int = 0,
) -> list[CaseOrderEntry]:
    paths = _ordered_case_dirs(dataset, lab_root=lab_root, seed=seed, shuffle=shuffle)
    limit = int(sample_size or case_limit or 0)
    if limit > 0:
        paths = paths[:limit]
    return [
        CaseOrderEntry(
            dataset=dataset,
            case_id=path.name,
            case_dir=path,
            order_index=index,
        )
        for index, path in enumerate(paths, start=1)
    ]


def write_case_order_manifest(
    path: str | Path,
    *,
    datasets: Sequence[str],
    lab_root: str | Path,
    seed: int = 0,
    shuffle: bool = False,
    sample_size: int = 0,
    case_limit: int = 0,
    name: str = "case_order",
) -> dict:
    output = Path(path)
    payload = {
        "version": 1,
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "shuffle": bool(shuffle),
        "sample_size": int(sample_size or 0),
        "case_limit": int(case_limit or 0),
        "datasets": {},
        "entries": {},
    }
    for dataset in datasets:
        entries = build_case_order(
            dataset,
            lab_root=lab_root,
            seed=seed,
            shuffle=shuffle,
            sample_size=sample_size,
            case_limit=case_limit,
        )
        payload["datasets"][dataset] = [entry.case_id for entry in entries]
        payload["entries"][dataset] = [entry.to_dict() for entry in entries]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_case_order_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_ids_from_file(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if text and not text.startswith("#"):
            values.append(text)
    return values


def _case_ids_from_manifest(dataset: str, manifest: Mapping | str | Path | None) -> list[str]:
    if manifest is None or manifest == "":
        return []
    payload = load_case_order_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), Mapping) else {}
    values = datasets.get(dataset, [])
    if not isinstance(values, list):
        raise ValueError(f"case-order manifest has invalid case list for dataset={dataset}")
    return [str(value) for value in values]


def select_case_dirs_from_order(
    dataset: str,
    *,
    lab_root: str | Path,
    manifest: Mapping | str | Path | None = None,
    case_ids: Iterable[str] | None = None,
    case_ids_file: str | Path | None = None,
    case_limit: int = 0,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Path]:
    by_id = {path.name: path for path in case_dirs_for_dataset(dataset, lab_root)}
    selected_ids: list[str] = []
    if case_ids_file:
        selected_ids = _case_ids_from_file(case_ids_file)
    elif manifest is not None:
        selected_ids = _case_ids_from_manifest(dataset, manifest)
    elif case_ids is not None:
        selected_ids = [str(value) for value in case_ids]

    if selected_ids:
        missing = [case_id for case_id in selected_ids if case_id not in by_id]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"case ids not found for dataset={dataset}: {preview}")
        paths = [by_id[case_id] for case_id in selected_ids]
    else:
        paths = _ordered_case_dirs(dataset, lab_root=lab_root, seed=seed, shuffle=shuffle)

    if case_limit and case_limit > 0:
        paths = paths[:case_limit]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a shared case-order manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--lab-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--name", default="case_order")
    args = parser.parse_args()
    manifest = write_case_order_manifest(
        args.output,
        datasets=args.datasets,
        lab_root=args.lab_root,
        seed=args.seed,
        shuffle=args.shuffle,
        sample_size=args.sample_size,
        case_limit=args.case_limit,
        name=args.name,
    )
    total = sum(len(values) for values in manifest["datasets"].values())
    print(f"[Nurse] Case-order manifest: {args.output} ({total} cases)")


if __name__ == "__main__":
    main()


