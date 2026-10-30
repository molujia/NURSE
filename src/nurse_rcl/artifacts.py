"""Canonical, content-addressed artifacts used by every NURSE stage."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-standard numeric values."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(data).hexdigest()


def stable_identifier(namespace: str, payload: Any) -> str:
    """Build a short stable identifier scoped by a caller-provided namespace."""

    digest = sha256_hex(canonical_json_bytes(payload))[:16]
    return f"{namespace}_{digest}"


class ArtifactEnvelope(BaseModel):
    """Immutable payload plus the hashes needed to verify a stage chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    kind: str
    payload: dict[str, Any]
    input_hashes: tuple[str, ...] = ()
    previous_hash: str | None = None
    payload_hash: str
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        payload: dict[str, Any],
        input_hashes: tuple[str, ...] = (),
        previous_hash: str | None = None,
    ) -> ArtifactEnvelope:
        payload_hash = sha256_hex(canonical_json_bytes(payload))
        unsigned = {
            "schema_version": 1,
            "kind": kind,
            "payload": payload,
            "input_hashes": list(input_hashes),
            "previous_hash": previous_hash,
            "payload_hash": payload_hash,
        }
        return cls(
            schema_version=1,
            kind=kind,
            payload=payload,
            input_hashes=input_hashes,
            previous_hash=previous_hash,
            payload_hash=payload_hash,
            artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )

    def verify(self) -> bool:
        expected_payload_hash = sha256_hex(canonical_json_bytes(self.payload))
        if self.payload_hash != expected_payload_hash:
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def verify_artifact_chain(envelopes: list[ArtifactEnvelope]) -> bool:
    """Verify every envelope and each predecessor link in order."""

    for index, envelope in enumerate(envelopes):
        if not envelope.verify():
            return False
        expected_previous = None if index == 0 else envelopes[index - 1].artifact_hash
        if envelope.previous_hash != expected_previous:
            return False
    return True


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably write bytes before atomically replacing the target path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_artifact(
    path: Path,
    *,
    kind: str,
    payload: dict[str, Any],
    input_hashes: tuple[str, ...] = (),
    previous_hash: str | None = None,
) -> ArtifactEnvelope:
    """Create, verify, and atomically persist an artifact envelope."""

    envelope = ArtifactEnvelope.create(
        kind=kind,
        payload=payload,
        input_hashes=input_hashes,
        previous_hash=previous_hash,
    )
    atomic_write_bytes(path, canonical_json_bytes(envelope.model_dump(mode="json")))
    return envelope
