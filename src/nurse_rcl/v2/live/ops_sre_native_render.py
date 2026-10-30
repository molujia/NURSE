"""D12 presentation reads local facts and delivers a bound native-tree appendix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nurse_rcl.v2.live.ops_sre_context import WORKING_INPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_native_report import NativeSreReportContext, validate_native_report
from nurse_rcl.v2.live.ops_sre_report import OpsSreReportResponse
from nurse_rcl.v2.live.ops_sre_report_delivery import _inspect_report_pair, _save_report_pair
from nurse_rcl.v2.live.ops_sre_report_render import _render_report_pair
from nurse_rcl.v2.live.staged_case import require_plain_path

NATIVE_RENDER_POLICY = "sre-readable-pair/3"


def render_native_report_pair(
    *, context: NativeSreReportContext, response: OpsSreReportResponse | None,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
    references: dict[str, Any] | None = None, accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None, preview: bool = False,
    raw_response_json: str | None = None,
) -> dict[str, str]:
    validate_native_report(context)
    if context.metadata.investigation_input_schema != WORKING_INPUT_SCHEMA and not preview:
        raise ValueError("old investigation material requires diagnostic preview")
    return _render_report_pair(context=context, response=response,
        provided_fact_refs=provided_fact_refs, provided_step_refs=provided_step_refs,
        references=references, accounting=accounting, public_history=public_history,
        preview=preview, raw_response_json=raw_response_json, policy=NATIVE_RENDER_POLICY)


def save_native_report_pair(directory: Path, **kwargs: Any) -> dict[str, Any]:
    require_plain_path(directory)
    return _save_report_pair(directory, pair=render_native_report_pair(**kwargs),
                             policy=NATIVE_RENDER_POLICY)


def inspect_native_report_pair(directory: Path) -> dict[str, Any]:
    return _inspect_report_pair(directory, policy=NATIVE_RENDER_POLICY)
