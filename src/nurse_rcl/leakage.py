"""Byte-level label-mutation invariance checks for prediction trees."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel


class LabelMutationInvariantError(RuntimeError):
    """Raised when changing labels changes prediction artifacts."""


class LabelMutationAudit(StrictFrozenModel):
    """Label-free receipt for two byte-identical prediction trees."""

    schema_version: Literal[1] = 1
    file_count: int = Field(gt=0)
    prediction_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def verify_label_mutation_invariance(
    baseline_root: Path,
    mutated_root: Path,
) -> LabelMutationAudit:
    """Require equal relative paths, sizes, and SHA-256 values in both trees."""

    baseline = _prediction_snapshot(baseline_root)
    mutated = _prediction_snapshot(mutated_root)
    if baseline != mutated:
        baseline_by_path = {entry[0]: entry for entry in baseline}
        mutated_by_path = {entry[0]: entry for entry in mutated}
        changed_paths = sorted(
            path
            for path in baseline_by_path.keys() | mutated_by_path.keys()
            if baseline_by_path.get(path) != mutated_by_path.get(path)
        )
        detail = changed_paths[0] if changed_paths else "<unknown>"
        raise LabelMutationInvariantError(
            f"prediction bytes changed after label mutation: {detail}"
        )
    records = [
        {"path": path, "size": size, "sha256": digest}
        for path, size, digest in baseline
    ]
    return LabelMutationAudit(
        file_count=len(records),
        prediction_tree_sha256=sha256_hex(canonical_json_bytes(records)),
    )


def _prediction_snapshot(root: Path) -> tuple[tuple[str, int, str], ...]:
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    files = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    if not files:
        raise ValueError("prediction tree must contain at least one file")
    snapshot = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f"prediction tree cannot contain symlinks: {path}")
        snapshot.append(
            (
                path.relative_to(resolved).as_posix(),
                path.stat().st_size,
                _sha256_file(path),
            )
        )
    return tuple(snapshot)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
