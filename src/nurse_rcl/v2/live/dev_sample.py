"""Fixed, parent-bound RCABench pilot selection; no data or label access."""

from __future__ import annotations

import platform
import random
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.split_freeze import ApprovedInferenceSplitManifest

SCHEMA = "nurse.v2.dev-pilot-selection/1"
SMOKE = (
    "opq_d7573450f2a04691984547e38e51dfc6",
    "opq_796c8a0ccf754475924830092bd4d880",
)
SEED = 20260926
ALGORITHM = "python.random.Random.sample(sorted_pool,20);sorted_execution/1"


def sample_rcabench_dev20(parent_path: Path, parent_pin: str) -> dict[str, Any]:
    split = ApprovedInferenceSplitManifest.model_validate_json(_pinned(parent_path, parent_pin))
    if not split.verify():
        raise ValueError("pilot parent split integrity differs")
    members = [
        opaque
        for part in split.partitions
        if part.dataset == "rcabench" and part.partition == "dev"
        for opaque in part.opaque_case_ids
    ]
    if len(members) != 284 or len(set(members)) != 284 or not set(SMOKE) <= set(members):
        raise ValueError("pilot parent must contain 284 RCABench Dev cases and both smoke cases")
    pool = sorted(set(members) - set(SMOKE))
    body = dict(
        schema_version=SCHEMA,
        parent_split_path=str(parent_path),
        parent_split_sha256=parent_pin,
        identity_artifact_sha256=split.identity_artifact_sha256,
        dataset="rcabench",
        seed=SEED,
        algorithm=ALGORITHM,
        python_version=platform.python_version(),
        excluded_case_ids=sorted(SMOKE),
        eligible_count=len(pool),
        case_count=20,
        opaque_case_ids=sorted(random.Random(SEED).sample(pool, 20)),
    )
    return dict(body, selection_sha256=sha256_hex(canonical_json_bytes(body)))


def read_pilot_selection(path: Path, pin: str) -> dict[str, Any]:
    value = _metadata(_pinned(path, pin))
    if (
        value.get("schema_version") != SCHEMA
        or type(value.get("parent_split_path")) is not str
        or type(value.get("parent_split_sha256")) is not str
    ):
        raise ValueError("pilot selection parent fields differ")
    expected = sample_rcabench_dev20(Path(value["parent_split_path"]), value["parent_split_sha256"])
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError("pilot selection differs from deterministic parent-bound population")
    return value
