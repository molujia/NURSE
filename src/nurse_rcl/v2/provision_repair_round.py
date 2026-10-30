"""PRIVATE provisioning only; exit this process before starting inference.

The pinned evaluator mapping is used solely to recover existing opaque IDs.
Answers are never selected, returned, logged or passed to a model runner.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live.staged_case import read_inference_identity, require_plain_path
from nurse_rcl.v2.opaque_ids import EvaluatorIdentityMap, inference_manifest_from_evaluator


def resolve_identities(
    *,
    evaluator_map_path: Path,
    expected_evaluator_artifact_sha256: str,
    inference_manifest_path: Path,
    expected_inference_artifact_sha256: str,
    requested: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if not requested or len(requested) > 4 or len(set(requested)) != len(requested):
        raise ValueError("private provisioning requires distinct requested identities")
    source = require_plain_path(evaluator_map_path)
    if os.name == "posix" and stat.S_IMODE(source.stat().st_mode) != 0o600:
        raise ValueError("private evaluator map mode must be 0600")
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("private evaluator map exceeds bounded import size")
    data = source.read_bytes()
    if sha256_hex(data) != expected_evaluator_artifact_sha256:
        raise ValueError("private evaluator map differs from independent pin")
    mapping = EvaluatorIdentityMap.model_validate_json(data)
    manifest = read_inference_identity(inference_manifest_path, expected_inference_artifact_sha256)
    if not mapping.verify() or inference_manifest_from_evaluator(mapping) != manifest:
        raise ValueError("private and inference identity artifacts disagree")
    result = []
    for dataset, original_id in requested:
        matches = [
            r.opaque_case_id
            for r in mapping.records
            if r.dataset == dataset and r.original_case_id == original_id
        ]
        if len(matches) != 1:
            raise ValueError("requested identity is not authorized")
        result.append(matches[0])
    return tuple(result)
