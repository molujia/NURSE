"""Bounded server execution records, not a new inference or trust engine.

The complete-case query, score and report entry points retain all source checks.
Clocks and publications are immutable operational records. Reading a bundle
proves file integrity only, never scientific source/score authority.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding
from nurse_rcl.v2.live.repair_report_prompt import REPAIR_EXCHANGE_VERSION
from nurse_rcl.v2.live.repair_report_request import REPAIR_REPORT_CONFIGURATION
from nurse_rcl.v2.live.repair_timing import (
    CASE_SECONDS,
    METHOD_SECONDS,
    PREPARATION_SECONDS,
    TIMING_POLICY,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path

STAGES = ("queries", "audit", "report")
ROUND_SECONDS = 4 * (CASE_SECONDS + 30)
PREPARATION_EXECUTION_VERSION = "live-repair-invocation-reuse/1"


def _profile_constraints() -> dict[str, Any]:
    return dict(
        schema_version="nurse.v2.live-evidence-repair-profile/1",
        profile="v2-live-evidence-repair",
        dynamic_verification=True,
        pre_ranking_semantics=False,
        history="off",
        worker_count=1,
        cases_in_flight=1,
        llm_in_flight=1,
        max_request_attempts=5,
        endpoint="https://ark.cn-beijing.volces.com/api/v3",
        model="deepseek-v4-flash-ga-260731",
        credential_variable="ARK_API_KEY2",
        report_logical_requests_per_case=1,
        round_report_logical_requests=4,
        round_max_network_attempts=20,
        report_thinking={"type": "disabled"},
        report_input_token_limit=100000,
        report_output_token_limit=4096,
        report_temperature=0,
        request_timeout_seconds=180,
        timing_policy_version=TIMING_POLICY,
        preparation_case_budget_seconds=PREPARATION_SECONDS,
        method_case_budget_seconds=METHOD_SECONDS,
        case_timeout_seconds=CASE_SECONDS,
        round_timeout_seconds=ROUND_SECONDS,
        stage_order=list(STAGES),
        address_space_limit_bytes=8589934592,
        terminate_grace_seconds=30,
        heartbeat_seconds=30,
        rcabench_window_seconds=240,
        aiops25_window_seconds=480,
        expected_cadence_seconds=30,
        max_dynamic_queries_per_case=8,
        q_base=8,
        q_max=12,
        candidate_truncation=False,
        case_splitting=False,
        modality_filtering=False,
        b0_capacity_profile="live-b0-6m-256m/1",
        repair_capacity_profile="live-repair-6m-256m/1",
        repair_max_serialization_nodes=6000000,
        repair_max_serialization_json_bytes=268435456,
        b0_max_serialization_nodes=6000000,
        b0_max_serialization_json_bytes=268435456,
        b0_max_serialization_depth=64,
        b0_max_serialization_string_length=8192,
        statistical_scale=dict(
            mad_multiplier=1.4826,
            baseline_relative_floor=0.05,
            threshold_multiplier=3,
            strength_divisor=3,
        ),
        continuous_episode=dict(minimum_points=3, minimum_cadences=2),
        score_weights="inherit_existing_transparent_scoring_unchanged",
        report_configuration_version=REPAIR_REPORT_CONFIGURATION,
        report_exchange_version=REPAIR_EXCHANGE_VERSION,
        preparation_execution_version=PREPARATION_EXECUTION_VERSION,
    )


def require_profile_common(profile: dict[str, Any]) -> None:
    """Shared numerical, disclosure and resource limits, never timing authority."""
    versioned = {
        "schema_version", "timing_policy_version", "preparation_case_budget_seconds",
        "method_case_budget_seconds", "case_timeout_seconds", "round_timeout_seconds",
        "stage_order", "preparation_execution_version", "round_report_logical_requests",
        "round_max_network_attempts",
    }
    if type(profile) is not dict or any(
        type(profile.get(key)) is not type(value)
        or canonical_json_bytes(profile[key]) != canonical_json_bytes(value)
        for key, value in _profile_constraints().items() if key not in versioned
    ):
        raise ValueError("repair profile differs from shared non-time constraints")


def require_profile(profile: dict[str, Any]) -> dict[str, int]:
    fixed = _profile_constraints()
    if (
        type(profile) is not dict
        or any(
            type(profile.get(k)) is not type(v)
            or canonical_json_bytes(profile[k]) != canonical_json_bytes(v)
            for k, v in fixed.items()
        )
        or any(
            k in profile
            for k in ("engineering_stage_timeout_seconds", "report_stage_timeout_seconds")
        )
    ):
        raise ValueError("repair profile differs from registered four-case execution limits")
    return dict(
        preparation=PREPARATION_SECONDS, method=METHOD_SECONDS, case=CASE_SECONDS, cleanup=30
    )


def _pin(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid independent identity pin")


def _name(name: str) -> None:
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]*\.json", name) is None:
        raise ValueError("invalid publication filename")


def _fsync(root: Path) -> None:
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path, limit: int = 4 * 1024 * 1024) -> dict[str, Any]:
    source = require_plain_path(path)
    if not source.is_file() or source.stat().st_size > limit:
        raise ValueError("operational record is not a bounded regular file")
    return decode_json_object(source.read_bytes(), ExchangeLimits(limit, 64, 100000))


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Publish once with an exclusive hard link; same bytes are resume-idempotent."""
    _name(path.name)
    parent = require_plain_path(path.parent)
    raw = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        if require_plain_path(path).read_bytes() != raw:
            raise ValueError("immutable operational record differs")
        return
    fd, name = tempfile.mkstemp(prefix=".record-", dir=parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync(parent)
    finally:
        temporary.unlink(missing_ok=True)


def open_clock(path: Path, *, identity: str, seconds: int, now: float) -> dict[str, Any]:
    _pin(identity)
    # Historical helper remains readable; new case execution uses TimingJournal.
    if type(seconds) is not int or seconds not in (
        1100,
        1800,
        4120,
        16480,
        CASE_SECONDS,
        ROUND_SECONDS,
    ):
        raise ValueError("unregistered clock budget")
    if type(now) is not float or not math.isfinite(now) or now < 0:
        raise ValueError("invalid clock")
    if path.exists() or path.is_symlink():
        value = read_json(path)
        if (
            set(value) != {"identity", "started", "deadline", "seconds"}
            or value["identity"] != identity
            or type(value["seconds"]) is not int
            or value["seconds"] != seconds
        ):
            raise ValueError("clock identity differs")
        if (
            type(value["started"]) is not float
            or not math.isfinite(value["started"])
            or value["started"] < 0
            or now < value["started"]
            or type(value["deadline"]) is not float
            or value["deadline"] != value["started"] + seconds
        ):
            raise ValueError("invalid persisted clock")
        return value
    value = dict(identity=identity, started=now, deadline=now + seconds, seconds=seconds)
    publish_json(path, value)
    return value


def remaining_seconds(clock: dict[str, Any], *, now: float) -> float:
    if not math.isfinite(now) or now < clock["started"]:
        raise ValueError("clock moved backwards")
    return max(0.0, float(clock["deadline"] - now))


def publish_bundle(target: Path, files: dict[str, bytes], *, identity: str) -> dict[str, Any]:
    """Atomically expose all stage outputs plus their exact inventory, or none."""
    _pin(identity)
    parent = require_plain_path(target.parent)
    if target.exists() or target.is_symlink():
        raise ValueError("immutable stage publication already exists")
    if not files or "receipt.json" in files:
        raise ValueError("invalid stage publication inventory")
    for name, data in files.items():
        _name(name)
        if type(data) is not bytes:
            raise ValueError("publication needs exact bytes")
    receipt = dict(
        identity=identity, files={name: sha256_hex(data) for name, data in sorted(files.items())}
    )
    temporary = Path(tempfile.mkdtemp(prefix=".incomplete-stage-", dir=parent))
    for name, data in {**files, "receipt.json": canonical_json_bytes(receipt)}.items():
        with (temporary / name).open("xb") as stream:
            os.chmod(temporary / name, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    _fsync(temporary)
    # Caller owns the fixed round execution lock. A populated target cannot be
    # replaced by rename; failed/incomplete temporary directories are retained.
    if target.exists() or target.is_symlink():
        raise ValueError("immutable stage publication already exists")
    temporary.rename(target)
    _fsync(parent)
    return receipt


def read_bundle(target: Path, *, identity: str) -> dict[str, Any]:
    _pin(identity)
    root = require_plain_path(target)
    receipt = read_json(root / "receipt.json")
    if (
        set(receipt) != {"identity", "files"}
        or receipt["identity"] != identity
        or type(receipt["files"]) is not dict
        or not receipt["files"]
    ):
        raise ValueError("stage publication identity differs")
    if {p.name for p in root.iterdir()} != {"receipt.json", *receipt["files"]}:
        raise ValueError("stage publication inventory differs")
    for name, pin in receipt["files"].items():
        _name(name)
        _pin(pin)
        source = require_plain_path(root / name)
        if file_sha256(source) != pin:
            raise ValueError("stage publication integrity differs")
    return receipt


def report_binding(
    payload: bytes, *, opaque_case_id: str, source_sha256: str, audit_sha256: str
) -> LiveRequestBinding:
    request = decode_json_object(payload, ExchangeLimits())
    public = decode_json_object(request["messages"][1]["content"].encode(), ExchangeLimits())
    view = public["evidence_view"]
    if (
        public.get("profile") != "v2-live-evidence-repair"
        or view["opaque_case_id"] != opaque_case_id
        or view["context_sha256"] != audit_sha256
    ):
        raise ValueError("report request differs from frozen case or audit")
    return LiveRequestBinding(
        opaque_case_id=opaque_case_id,
        variant="v2-live-evidence-repair",
        phase="report",
        input_sha256=source_sha256,
        context_sha256=audit_sha256,
        prompt_schema_sha256=sha256_hex(request["messages"][0]["content"].encode()),
        request_sha256=sha256_hex(payload),
    )


def require_queue(cases: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    if type(cases) is not tuple or len(cases) != 4:
        raise ValueError("repair requires exactly the frozen four-case queue")
    for case in cases:
        if (
            type(case) is not dict
            or set(case) != {"dataset", "opaque_case_id", "input_path", "input_sha256"}
            or case["dataset"] not in ("rcabench", "aiops25")
            or type(case["opaque_case_id"]) is not str
            or re.fullmatch(r"opq_[0-9a-f]{32}", case["opaque_case_id"]) is None
            or type(case["input_path"]) is not str
            or not Path(case["input_path"]).is_absolute()
        ):
            raise ValueError("invalid opaque queue member")
        _pin(case["input_sha256"])
    if (
        len({c["opaque_case_id"] for c in cases}) != 4
        or len({c["input_path"] for c in cases}) != 4
        or sum(c["dataset"] == "rcabench" for c in cases) != 2
    ):
        raise ValueError("repair requires two distinct whole cases per dataset")
    return cases
