"""Evaluation-only labels for NURSE.

These helpers must never be called from raw adapters or evidence construction.
They are used only after inference to build metrics and reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .aiops25 import load_aiops25_labels


def load_labels_for_dataset(dataset: str, case_id: str, lab_root: str | Path) -> List[str]:
    if dataset == "aiops25-retained-nonnode":
        return load_aiops25_labels(case_id, lab_root)
    return []