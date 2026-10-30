"""Prediction-tree sealing enforced before evaluation can read labels."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel


class PredictionSealMismatch(RuntimeError):
    """Raised when current prediction artifacts differ from their frozen seal."""


class SealedFile(StrictFrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PredictionSeal(StrictFrozenModel):
    schema_version: int = 1
    hash_algorithm: str = "sha256"
    files: tuple[SealedFile, ...]
    tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_seal(prediction_root: Path) -> PredictionSeal:
    root = prediction_root.resolve(strict=True)
    files = tuple(
        SealedFile(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=_hash_file(path),
        )
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    )
    serialized_files = [entry.model_dump(mode="json") for entry in files]
    return PredictionSeal(
        files=files,
        tree_hash=sha256_hex(canonical_json_bytes(serialized_files)),
    )


def create_prediction_seal(prediction_root: Path, seal_path: Path) -> PredictionSeal:
    """Hash a prediction tree and persist its immutable seal outside that tree."""

    root = prediction_root.resolve(strict=True)
    destination = seal_path.resolve(strict=False)
    if destination.is_relative_to(root):
        raise ValueError("prediction seal must be stored outside the sealed tree")
    seal = _build_seal(root)
    atomic_write_bytes(destination, canonical_json_bytes(seal.model_dump(mode="json")))
    return seal


def verify_prediction_seal(prediction_root: Path, seal_path: Path) -> PredictionSeal:
    """Return the stored seal only when every current prediction byte matches."""

    expected = PredictionSeal.model_validate_json(seal_path.read_text(encoding="utf-8"))
    actual = _build_seal(prediction_root)
    if actual != expected:
        raise PredictionSealMismatch("prediction tree does not match its frozen seal")
    return expected
