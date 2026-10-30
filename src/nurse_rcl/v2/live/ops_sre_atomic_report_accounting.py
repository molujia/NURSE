"""Pure rendering view of pinned history and the original identity-checked ledger.

This is not a ledger, budget, send permission or inferred all-experiment total.
"""

from __future__ import annotations

import json
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin


def history_json(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    if set(value) != {"scope", "sources"} or (
        value["scope"] != "selected_source_and_previous_reports_only"
        or type(value["sources"]) is not list
        or not value["sources"]
    ):
        raise ValueError("report accounting history scope differs")
    pins = set()
    for source in value["sources"]:
        if set(source) != {"label", "artifact_sha256", "usage"} or (
            type(source["label"]) is not str or not source["label"].strip()
        ):
            raise ValueError("report accounting history source differs")
        pin = source["artifact_sha256"]
        require_pin(pin)
        if pin in pins:
            raise ValueError("report accounting history duplicate source")
        pins.add(pin)
        _validate_usage(source["usage"])
    return canonical_json_bytes(value).decode()


def _validate_usage(usage: dict[str, Any]) -> None:
    for key in ("attempts", "logical_requests"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError("report accounting usage inventory differs")
    for part in ("input", "output", "total", "cached"):
        known, unknown, total = (
            usage.get("reported_" + part + "_tokens"),
            usage.get("unknown_" + part + "_attempts"),
            usage.get(part + "_tokens"),
        )
        if (
            type(known) is not int
            or known < 0
            or type(unknown) is not int
            or (not 0 <= unknown <= usage["attempts"])
            or (total is not None and (type(total) is not int or total != known or unknown))
        ):
            raise ValueError("report accounting usage values differ")


def combine_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        _validate_usage(row)
    result = {k: sum(r[k] for r in rows) for k in ("attempts", "logical_requests")}
    for part in ("input", "output", "total", "cached"):
        for name in ("reported_" + part + "_tokens", "unknown_" + part + "_attempts"):
            result[name] = sum(r[name] for r in rows)
        result[part + "_tokens"] = (
            result["reported_" + part + "_tokens"]
            if all(r[part + "_tokens"] is not None for r in rows)
            else None
        )
    return result


def render_accounting(state: Any) -> dict[str, Any]:
    result = dict(
        report_status=state.status,
        provider_seconds=state.provider_seconds,
        validation_seconds=state.validation_seconds,
    )
    history = state.accounting_history_json
    current = state.report_usage_json
    if history is not None:
        packet = json.loads(history)
        if history_json(packet) != history:
            raise ValueError("report accounting history is not canonical")
        result["source_usage"] = combine_usage([s["usage"] for s in packet["sources"]])
        result["usage_scope"] = packet["scope"]
        result["history_sources"] = packet["sources"]
    if current is not None:
        usage = json.loads(current)
        _validate_usage(usage)
        result["report_usage"] = usage
        if history is not None:
            result["usage"] = combine_usage([result["source_usage"], usage])
        result["untracked_send_unknown"] = usage.get("untracked_send_unknown", False)
    return result


def require_history_pin(raw: str | None, expected: str | None) -> None:
    actual = None if raw is None else sha256_hex(raw.encode())
    if actual != expected:
        raise ValueError("report accounting history external pin differs")
