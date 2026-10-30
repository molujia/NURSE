"""Explicit normal G+ report owner and keyless, non-mutating saved reader.

The owner reserves before external work; no complete native answer is repaired,
re-sampled or re-dispatched. Mechanical acceptance never certifies diagnosis.
"""

# ruff: noqa: RUF001 -- Chinese user-facing report qualification.

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import decode_json_object
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail, failure_detail
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_render import render_brief_report_pair
from nurse_rcl.v2.live.ops_sre_brief_report import (
    BriefSreReportContext,
    decode_brief_report_storage,
    encode_brief_report_storage,
    validate_brief_report,
)
from nurse_rcl.v2.live.ops_sre_gplus_contract import bind_gplus_response
from nurse_rcl.v2.live.ops_sre_gplus_provider import GplusDelivery, read_gplus_delivery
from nurse_rcl.v2.live.ops_sre_gplus_report import gplus_report_body, project_gplus_report
from nurse_rcl.v2.live.ops_sre_model_view import _book, _prefix
from nurse_rcl.v2.live.ops_sre_references import _translate
from nurse_rcl.v2.live.ops_sre_report_check_policy import parse_checked_brief, response_admission
from nurse_rcl.v2.live.ops_sre_typed_report_provider import _durable


def _members(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("G+ owner cannot contain symlinks")
        if not path.is_file() or path == root / "result.json":
            continue
        raw = path.read_bytes()
        rows.append(
            dict(path=path.relative_to(root).as_posix(), bytes=len(raw), sha256=sha256_hex(raw))
        )
    return rows


def _save(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    value = {k: v for k, v in state.items() if k != "state_sha256"}
    value["members"] = _members(root)
    value["state_sha256"] = sha256_hex(canonical_json_bytes(value))
    atomic_write_bytes(root / "result.json", canonical_json_bytes(value))
    return value


def _evaluate(
    context: BriefSreReportContext, references: dict[str, Any], wire: bytes, delivery: GplusDelivery
) -> dict[str, bytes]:
    body, store = gplus_report_body(wire)
    receipt, short = bind_gplus_response(
        delivery.native,
        body["context"]["supply"],
        store,
        request_sha256=sha256_hex(wire),
        graph_sha256=body["graph_sha256"],
        graph_representation=body["graph_representation"],
        response_origin=dict(
            request_sha256=delivery.origin["request_sha256"],
            graph_sha256=body["graph_sha256"],
            native_sha256=delivery.origin["native_sha256"],
        ),
    )
    parsed = parse_checked_brief(context, short)
    _, book = _prefix(_book(references, context.metadata.evidence.case_snapshot_id), 9)
    resolved = canonical_json_bytes(
        _translate(
            parsed.model_dump(mode="json"),
            book=book,
            visible_at=9,
            case_id=context.metadata.evidence.opaque_case_id,
            encode=False,
            limits=REPORT_LIMITS,
            brief_fields=True,
        )
    )
    admitted = response_admission(context, resolved.decode())
    checks = canonical_json_bytes([c.model_dump(mode="json") for c in admitted.checks])
    pair = render_brief_report_pair(
        context=context,
        response=parse_checked_brief(context, resolved),
        references=references,
        raw_response_json=None,
    )
    note = (
        "\n\n程序限定：G+ 来源、字段与上游闭包为程序绑定；"
        "模型原生解释保持未认证，绑定成功不等于根因正确。\n"
    )
    return {
        "binding.json": canonical_json_bytes(receipt),
        "checked-short.json": short,
        "resolved.json": resolved,
        "checks.json": checks,
        "report.md": (pair["report"] + note).encode(),
        "evidence.md": (pair["evidence"] + note).encode(),
    }


def run_gplus_report(
    *,
    root: Path,
    context: BriefSreReportContext,
    references: dict[str, Any],
    code_sha256: str,
    execution_sha256: str,
    approved_execution_sha256: str | None,
    complete: Callable[[bytes], GplusDelivery],
) -> dict[str, Any]:
    for pin in (code_sha256, execution_sha256):
        require_pin(pin)
    if approved_execution_sha256 != execution_sha256:
        raise ValueError("G+ owner requires explicit execution approval")
    validate_brief_report(context)
    root = Path(root)
    if (root / "result.json").exists():
        return inspect_gplus_report(
            root=root,
            expected_context_sha256=context.context_sha256,
            expected_code_sha256=code_sha256,
            expected_execution_sha256=execution_sha256,
        )
    working = deepcopy(references)
    projection = project_gplus_report(context, references=working)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _durable(root / "context.json", encode_brief_report_storage(context))
    _durable(root / "references.json", canonical_json_bytes(working))
    state = dict(
        schema_version="nurse.v2.sre-gplus-report-run/2",
        code_sha256=code_sha256,
        execution_sha256=execution_sha256,
        context_sha256=context.context_sha256,
        core_result_digest=context.core_result_digest,
        status="pending",
        call_started=False,
        semantic_certified=False,
        request_sha256=None,
        failure_type=None,
        failure_detail=None,
        decision_or_action_dispatches=0,
        preparation_load_imports=0,
    )
    if projection.wire_bytes is None:
        detail = getattr(projection, "failure_detail", None)
        state.update(
            status="unavailable",
            failure_type=projection.status,
            failure_detail=detail.model_dump(mode="json") if detail else None,
        )
        return _save(root, state)
    wire = projection.wire_bytes
    _durable(root / "request.bin", wire)
    state["request_sha256"] = sha256_hex(wire)
    references.clear()
    references.update(working)
    _save(root, state)
    state["call_started"] = True
    _save(root, state)  # Durable reservation BEFORE invoking any external callback.
    try:
        delivered = complete(wire)
        actual = read_gplus_delivery(
            root=root / "provider",
            wire=wire,
            code_sha256=code_sha256,
            execution_sha256=execution_sha256,
        )
        if type(delivered) is not GplusDelivery or delivered != actual:
            raise ValueError("G+ callback differs from actual original delivery")
        _durable(root / "native.bin", actual.native)
        _durable(root / "delivery-origin.json", canonical_json_bytes(actual.origin))
    except Exception as exc:
        state.update(
            status="unavailable",
            failure_type=type(exc).__name__,
            failure_detail=failure_detail("report_provider", exc).model_dump(mode="json"),
        )
        return _save(root, state)
    try:
        outputs = _evaluate(context, working, wire, actual)
        for name, raw in outputs.items():
            _durable(root / name, raw)
        checks = decode_json_object(b'{"checks":' + outputs["checks.json"] + b"}", REPORT_LIMITS)[
            "checks"
        ]
        accepted = sum(not row["reason_codes"] for row in checks)
        state.update(status="completed" if checks and accepted == len(checks) else "partial")
    except Exception as exc:
        state.update(
            status="rejected",
            failure_type=type(exc).__name__,
            failure_detail=failure_detail("report_validation", exc).model_dump(mode="json"),
        )
    return _save(root, state)


def inspect_gplus_report(
    *,
    root: Path,
    expected_context_sha256: str,
    expected_code_sha256: str,
    expected_execution_sha256: str,
) -> dict[str, Any]:
    """Recheck saved files/source/selection/check/render with zero writes or sends."""
    root = Path(root)
    state_raw = (root / "result.json").read_bytes()
    state = decode_json_object(state_raw, REPORT_LIMITS)
    unsigned = {k: v for k, v in state.items() if k != "state_sha256"}
    if (
        state["schema_version"]
        not in {
            "nurse.v2.sre-gplus-report-run/1",
            "nurse.v2.sre-gplus-report-run/2",
        }
        or state["state_sha256"] != sha256_hex(canonical_json_bytes(unsigned))
        or canonical_json_bytes(state) != state_raw
        or state["context_sha256"] != expected_context_sha256
        or state["code_sha256"] != expected_code_sha256
        or state["execution_sha256"] != expected_execution_sha256
        or state["members"] != _members(root)
    ):
        raise ValueError("G+ saved owner identity or complete member set differs")
    base_fields = {
        "schema_version",
        "code_sha256",
        "execution_sha256",
        "context_sha256",
        "core_result_digest",
        "status",
        "call_started",
        "semantic_certified",
        "request_sha256",
        "failure_type",
        "decision_or_action_dispatches",
        "preparation_load_imports",
        "members",
        "state_sha256",
    }
    new_version = state["schema_version"] == "nurse.v2.sre-gplus-report-run/2"
    if set(state) != base_fields | ({"failure_detail"} if new_version else set()):
        raise ValueError("G+ saved diagnostic schema differs")
    if new_version and state["failure_detail"] is not None:
        detail = OpsFailureDetail.model_validate(state["failure_detail"], strict=True)
        expected_stage = (
            "report_projection"
            if state["request_sha256"] is None
            else "report_validation"
            if state["status"] == "rejected"
            else "report_provider"
        )
        if detail.stage != expected_stage or state["failure_type"] is None:
            raise ValueError("G+ saved failure stage differs")
    context = decode_brief_report_storage((root / "context.json").read_bytes())
    if (
        context.context_sha256 != expected_context_sha256
        or context.core_result_digest != state["core_result_digest"]
    ):
        raise ValueError("G+ saved source context/core differs")
    references = decode_json_object((root / "references.json").read_bytes(), REPORT_LIMITS)
    if state["request_sha256"] is None:
        if state["status"] != "unavailable" or state["call_started"]:
            raise ValueError("G+ missing request owner differs")
        return state
    wire = (root / "request.bin").read_bytes()
    body, _ = gplus_report_body(wire, references=references)
    if (
        sha256_hex(wire) != state["request_sha256"]
        or body["context_sha256"] != expected_context_sha256
    ):
        raise ValueError("G+ saved request differs")
    if (root / "native.bin").exists():
        actual = read_gplus_delivery(
            root=root / "provider",
            wire=wire,
            code_sha256=expected_code_sha256,
            execution_sha256=expected_execution_sha256,
        )
        if (
            actual.native != (root / "native.bin").read_bytes()
            or canonical_json_bytes(actual.origin) != (root / "delivery-origin.json").read_bytes()
        ):
            raise ValueError("G+ native actual-byte delivery differs")
        try:
            outputs = _evaluate(context, references, wire, actual)
        except Exception as exc:
            if state["status"] != "rejected" or type(exc).__name__ != state["failure_type"]:
                raise ValueError("G+ saved rejection differs") from None
            if new_version and state["failure_detail"] != failure_detail(
                "report_validation",
                exc,
            ).model_dump(mode="json"):
                raise ValueError("G+ saved rejection diagnostic differs") from None
        else:
            if state["status"] not in {"completed", "partial"}:
                raise ValueError("G+ saved acceptance status differs")
            if any((root / name).read_bytes() != raw for name, raw in outputs.items()):
                raise ValueError("G+ saved expansion/check/render differs")
    elif state["status"] not in {"pending", "unavailable"}:
        raise ValueError("G+ saved native missing")
    return state
