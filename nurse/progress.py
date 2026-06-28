"""Small progress rendering helpers for Nurse CLI runs."""

from __future__ import annotations


def format_progress_line(
    *,
    dataset: str,
    llm: str,
    index: int,
    total: int,
    case_id: str,
    status: str,
    top1: str = "",
    width: int = 24,
) -> str:
    total = max(total, 1)
    index = min(max(index, 0), total)
    filled = int(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = index / total * 100
    suffix = f" top1={top1}" if top1 else ""
    return (
        f"[Nurse][{dataset}][llm={llm}] "
        f"{index}/{total} [{bar}] {pct:5.1f}% "
        f"case={case_id} status={status}{suffix}"
    )
