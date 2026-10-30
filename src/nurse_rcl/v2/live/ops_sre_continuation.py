"""One report-only owner for a source-admitted, paid terminal investigation.

The fixed first-case importer owns source/projection admission. This module
freezes that proof and checks it again before use; it is not a sandbox for
arbitrary Python callers or a general cross-version migration facility.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import sre_run_id
from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus
from nurse_rcl.v2.live.ops_empirical_policy import SRE_REPORT_PROTOCOL
from nurse_rcl.v2.live.ops_plan import code_inventory, plain_location
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext
from nurse_rcl.v2.live.ops_sre_report_exchange import sre_report_projection
from nurse_rcl.v2.live.ops_sre_report_run import OpsSreReportRun, _check_wire
from nurse_rcl.v2.live.ops_sre_report_storage import (
    decode_sre_report_storage,
    encode_sre_report_storage,
)
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path

REGISTRY_ROOT = Path("/home/wangrunzhou/NURSE/development/report-continuations")
SCHEMA = "nurse.v2.sre-report-continuation/1"
RECEIPT_LIMITS = ExchangeLimits(4_194_304, 32, 100_000)


@contextmanager
def continuation_lock(directory: Path) -> Iterator[None]:
    require_plain_path(directory)
    fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def write_once(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        if _read_bytes(path, 64 * 1024 * 1024) != raw:
            raise ValueError("continuation frozen artifact differs")
        return
    atomic_write_bytes(path, raw)
    _fsync(path.parent)


def _target(source: dict[str, Any]) -> Path:
    # New code/context do not buy a new lifetime request. The fixed importer
    # admits only the original plan at its original path (no copied source).
    identity = {key: source[key] for key in ("root", "plan_sha256", "opaque_case_id")}
    return plain_location(REGISTRY_ROOT) / sha256_hex(canonical_json_bytes(identity))


def _source(source: dict[str, Any]) -> None:
    for name in ("plan_sha256", "code_sha256", "case_snapshot_id", "core_result_digest"):
        require_pin(source[name])
    root = require_plain_path(Path(source["root"]))
    if str(root) != source["root"] or source["termination"] != "model_stop" or not (
        type(source["decision_count"]) is int and 1 <= source["decision_count"] <= 8
        and type(source["action_count"]) is int and 0 <= source["action_count"] <= 8
    ):
        raise ValueError("continuation requires admitted terminal source")
    files = source["source_files"]
    if not files or len(files) > 10000:
        raise ValueError("continuation source inventory differs")
    for text, pin in files.items():
        path = require_plain_path(Path(text))
        if [path.stat().st_size, file_sha256(path)] != pin:
            raise ValueError("continuation source changed")
    for name in ("conversation.json", "state.json", "result.json"):
        if str(root / "investigation" / name) not in files:
            raise ValueError("continuation source proof missing")
    for folder in (root / "investigation", root / "provider"):
        if any(str(p) not in files for p in folder.rglob("*.json")):
            raise ValueError("continuation source inventory changed")
    if list((root / "provider/ark-report").glob("request-*.json")) or (
        (root / "report/result.json").exists() or (root / "report/result.json").is_symlink()
    ):
        raise ValueError("source report is no longer zero-attempt")
    usage = source["usage"]
    rows = usage["attempt_usage"]
    if any(row["phase"] != "investigation" for row in rows):
        raise ValueError("source report attempts cannot be inherited")
    summary = summarize_attempts(rows, confirmed_no_send=not rows)
    if any(usage.get(k) != v for k, v in summary.items()):
        raise ValueError("continuation source usage differs")


def report_code_inventory(root: Path) -> dict[str, str]:
    root = require_plain_path(root)
    if not Path(__file__).resolve().is_relative_to(root):
        raise ValueError("continuation running code root differs")
    result = code_inventory(root)
    for name in (".audit_scripts/sre_report_storage_preflight_20260929.py",
                 ".audit_scripts/sre_report_continuation_20260929.py"):
        if (root / name).exists():
            result[name] = file_sha256(require_plain_path(root / name))
    return result


def inherit_report_book(
    state: dict[str, Any], *, source: dict[str, Any], target: Path, report_code: str,
) -> dict[str, Any]:
    snapshot = source["case_snapshot_id"]
    old = SreReferenceBook.from_state(state, snapshot=snapshot,
        run_id=sre_run_id(Path(source["root"]), snapshot, source["code_sha256"]))
    raw = _read_bytes(Path(source["root"]) / "investigation/conversation.json",
                      128 * 1024 * 1024)
    if json.loads(raw)["reference_state"] != old.to_state():
        raise ValueError("continuation historical reference prefix differs")
    new = SreReferenceBook(snapshot=snapshot, run_id=sre_run_id(target, snapshot, report_code))
    for kind, handle, original, visible_at in state["entries"]:
        if visible_at > 8 or new.issue(kind, original, visible_at=visible_at) != handle:
            raise ValueError("continuation historical reference prefix differs")
    return new.to_state()


def freeze_continuation(
    *, source: dict[str, Any], context: OpsSreReportContext,
    reference_state: dict[str, Any], code_root: Path,
) -> dict[str, Any]:
    _source(source)
    if (context.evidence.case_snapshot_id, context.evidence.opaque_case_id,
        context.core_result_digest, context.evidence.investigation_status) != (
        source["case_snapshot_id"], source["opaque_case_id"], source["core_result_digest"],
        "completed",
    ):
        raise ValueError("continuation source/context identity differs")
    if (context.termination != source["termination"] or len(context.steps) != (
        source["decision_count"]
    ) or len(context.decision_seconds) != source["decision_count"] or (
        len(context.action_seconds) != source["action_count"]
        or sum(s["action_index"] is not None for s in context.steps) != source["action_count"]
    )):
        raise ValueError("continuation source/context history differs")
    target = _target(source)
    if any(target.is_relative_to(Path(p)) or Path(p).is_relative_to(target)
           for p in (source["root"], str(code_root))):
        raise ValueError("continuation target overlaps sources")
    files = report_code_inventory(code_root)
    code = sha256_hex(canonical_json_bytes(files))
    references = inherit_report_book(reference_state, source=source, target=target,
                                     report_code=code)
    projection = sre_report_projection(context, references=references)
    wire = projection.wire_bytes
    if wire is None:
        raise ValueError("continuation report projection not ready: " + projection.status)
    state = OpsSreReportRun(code_sha256=code, reference_state=references,
        context_sha256=context.context_sha256, core_result_digest=context.core_result_digest,
        channels=OpsChannelStatus(core="completed", investigation="completed", report="pending"),
        input_wire_json=wire.decode(), input_wire_sha256=sha256_hex(wire),
        provided_fact_refs=projection.provided_fact_refs,
        provided_step_refs=tuple(s["step_ref"] for s in context.steps),
        target_exceeded=projection.target_exceeded, expansion_reasons=projection.expansion_reasons)
    state = state.model_copy(update={"state_sha256": sha256_hex(canonical_json_bytes(
        state.model_dump(mode="json", exclude={"state_sha256"})))})
    _check_wire(state, context)
    artifacts = {"context.json": encode_sre_report_storage(context),
                 "initial-state.json": canonical_json_bytes(state.model_dump(mode="json")),
                 "report-wire.json": wire}
    receipt = dict(schema_version=SCHEMA, target=str(target), source=source,
        code_root=str(code_root), code_files=files, code_sha256=code,
        context_sha256=context.context_sha256, request_sha256=sha256_hex(wire),
        source_reference_sha256=reference_state["state_sha256"],
        source_reference_count=len(reference_state["entries"]),
        source_reference_state=reference_state, report_reference_sha256=references["state_sha256"],
        files={name: [len(raw), sha256_hex(raw)] for name, raw in artifacts.items()},
        report_protocol=SRE_REPORT_PROTOCOL, logical_requests=1, max_attempts=5)
    raw_receipt = canonical_json_bytes(receipt)
    decode_json_object(raw_receipt, RECEIPT_LIMITS)
    # One fixed registry lock also serializes the very first creation.
    registry = plain_location(REGISTRY_ROOT)
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    with continuation_lock(registry):
        target.mkdir(exist_ok=True, mode=0o700)
        require_plain_path(target)
        # Claim is durable before other artifacts; a crash resumes the SAME claim.
        write_once(target / "receipt.json", raw_receipt)
        for name, raw in artifacts.items():
            write_once(target / name, raw)
    _source(source)
    return receipt


def read_continuation(
    path: Path, receipt_sha256: str,
) -> tuple[dict[str, Any], OpsSreReportContext, OpsSreReportRun]:
    require_pin(receipt_sha256)
    raw = _read_bytes(path, RECEIPT_LIMITS.max_bytes)
    if sha256_hex(raw) != receipt_sha256:
        raise ValueError("continuation receipt pin differs")
    receipt = decode_json_object(raw, RECEIPT_LIMITS)
    if receipt["schema_version"] != SCHEMA or (receipt["logical_requests"],
        receipt["max_attempts"], receipt["report_protocol"]) != (
        1, 5, SRE_REPORT_PROTOCOL,
    ) or raw != canonical_json_bytes(receipt):
        raise ValueError("continuation receipt schema differs")
    target = _target(receipt["source"])
    if str(target) != receipt["target"] or require_plain_path(path) != target / "receipt.json":
        raise ValueError("continuation target differs")
    _source(receipt["source"])
    files = report_code_inventory(Path(receipt["code_root"]))
    if files != receipt["code_files"] or sha256_hex(canonical_json_bytes(files)) != (
        receipt["code_sha256"]
    ):
        raise ValueError("continuation report code changed")
    if set(receipt["files"]) != {"context.json", "initial-state.json", "report-wire.json"}:
        raise ValueError("continuation frozen files differ")
    artifacts = {}
    for name, pin in receipt["files"].items():
        content = _read_bytes(target / name, 64 * 1024 * 1024)
        if [len(content), sha256_hex(content)] != pin:
            raise ValueError("continuation frozen artifact changed")
        artifacts[name] = content
    context = decode_sre_report_storage(artifacts["context.json"])
    initial = OpsSreReportRun.model_validate_json(artifacts["initial-state.json"])
    old = receipt["source_reference_state"]
    prefix = inherit_report_book(old, source=receipt["source"], target=target,
                                 report_code=receipt["code_sha256"])
    new = SreReferenceBook.from_state(initial.reference_state,
        snapshot=context.evidence.case_snapshot_id, run_id=prefix["run_id"]).to_state()
    count = receipt["source_reference_count"]
    if count != len(prefix["entries"]) or new["entries"][:count] != prefix["entries"] or (
        any(row[3] != 9 for row in new["entries"][count:])
        or receipt["source_reference_sha256"] != old["state_sha256"]
        or receipt["report_reference_sha256"] != new["state_sha256"]
        or initial.context_sha256 != receipt["context_sha256"]
        or initial.code_sha256 != receipt["code_sha256"]
        or initial.input_wire_json is None
        or initial.input_wire_json.encode() != artifacts["report-wire.json"]
        or initial.input_wire_sha256 != receipt["request_sha256"]
    ):
        raise ValueError("continuation owner or reference prefix differs")
    _check_wire(initial, context)
    return receipt, context, initial
