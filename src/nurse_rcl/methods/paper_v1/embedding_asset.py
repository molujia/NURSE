"""Immutable embedding asset lock and offline preflight verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

EMBEDDING_MODEL_ID: Literal["sentence-transformers/all-MiniLM-L6-v2"] = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_WEIGHT_FILE: Literal["model.safetensors"] = "model.safetensors"
EMBEDDING_WEIGHT_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
EMBEDDING_WEIGHT_BYTES = 90_868_376
_HASH_CHUNK_BYTES = 1024 * 1024


class EmbeddingAssetLock(StrictFrozenModel):
    """Exact repository revision and primary safetensors object identity."""

    model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_file: Literal["model.safetensors"]
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_bytes: int = Field(gt=0)


FIXED_EMBEDDING_LOCK = EmbeddingAssetLock(
    model_id=EMBEDDING_MODEL_ID,
    revision=EMBEDDING_REVISION,
    weight_file=EMBEDDING_WEIGHT_FILE,
    weight_sha256=EMBEDDING_WEIGHT_SHA256,
    weight_bytes=EMBEDDING_WEIGHT_BYTES,
)


class EmbeddingPreflightReceipt(StrictFrozenModel):
    """Path-free proof that the selected local snapshot matches its lock."""

    model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_file: Literal["model.safetensors"]
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_bytes: int = Field(gt=0)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


def verify_embedding_asset(
    *,
    snapshot_dir: Path,
    resolved_revision: str,
    lock: EmbeddingAssetLock = FIXED_EMBEDDING_LOCK,
) -> EmbeddingPreflightReceipt:
    """Reject a floating, drifted, missing, or relocated embedding snapshot."""

    if resolved_revision != lock.revision:
        raise ValueError("embedding revision does not match the fixed lock")
    if not snapshot_dir.is_dir():
        raise ValueError("embedding snapshot directory does not exist")
    if snapshot_dir.name != lock.revision:
        raise ValueError("embedding snapshot directory does not match the revision")

    weight_path = snapshot_dir / lock.weight_file
    if not weight_path.is_file():
        raise ValueError("embedding weight file is missing")
    actual_size = weight_path.stat().st_size
    if actual_size != lock.weight_bytes:
        raise ValueError("embedding weight size does not match the fixed lock")
    actual_hash = _file_sha256(weight_path)
    if actual_hash != lock.weight_sha256:
        raise ValueError("embedding weight SHA-256 does not match the fixed lock")

    unsigned = {
        "model_id": lock.model_id,
        "revision": lock.revision,
        "weight_file": lock.weight_file,
        "weight_sha256": actual_hash,
        "weight_bytes": actual_size,
    }
    receipt = EmbeddingPreflightReceipt(
        model_id=lock.model_id,
        revision=lock.revision,
        weight_file=lock.weight_file,
        weight_sha256=actual_hash,
        weight_bytes=actual_size,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("embedding preflight receipt verification failed")
    return receipt


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
