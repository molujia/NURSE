"""Shared helpers for NURSE agent tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from nurse.datasets import case_dirs_for_dataset
from nurse.schemas import _assert_no_forbidden


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(os.environ.get("NURSE_LAB_ROOT", REPO_ROOT.parent)).expanduser().resolve()
SAMPLE_CASE_DIR = Path(__file__).resolve().parents[1] / "sample_cases"


def load_sample_case(name: str) -> Dict[str, Any]:
    path = SAMPLE_CASE_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_forbidden(payload)
    return payload


def find_case_dir(dataset: str, case_id: str) -> Path:
    for case_dir in case_dirs_for_dataset(dataset, LAB_ROOT):
        if case_dir.name == case_id:
            return case_dir
    raise FileNotFoundError(f"case {case_id!r} not found for dataset {dataset!r}")


def dataclass_dicts(items: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        else:
            rows.append(dict(item))
    _assert_no_forbidden(rows)
    return rows
