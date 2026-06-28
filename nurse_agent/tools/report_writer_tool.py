"""Agent report writer tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from nurse.schemas import NursePrediction, VerifiedRCLContext, _assert_no_forbidden
from nurse_agent.report_contracts import build_agent_report, write_report_files


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(payload["output_dir"]))
    prediction = NursePrediction.from_dict(dict(payload["prediction"]))
    context = VerifiedRCLContext.from_dict(dict(payload["context"]))
    trace_path = Path(str(payload["trace_path"]))
    report = build_agent_report(
        dataset=str(payload["dataset"]),
        case_id=str(payload["case_id"]),
        model=str(payload.get("model") or "gpt-5-nano"),
        prediction=prediction,
        context=context,
        incident_prior=dict(payload.get("incident_prior") or {}),
        recovery_advice=payload.get("recovery_advice") or [],
        trace_path=trace_path,
        artifact_paths=dict(payload.get("artifact_paths") or {}),
    )
    if payload.get("llm_explanation"):
        report["llm_explanation"] = str(payload.get("llm_explanation") or "")
    _assert_no_forbidden(report)
    paths = write_report_files(report, output_dir)
    return {
        "dataset": str(payload["dataset"]),
        "case_id": str(payload["case_id"]),
        "report": report,
        "artifact_paths": {
            "agent_report_json": str(paths["json"]),
            "agent_report_markdown": str(paths["markdown"]),
            "agent_trace": str(trace_path),
        },
    }
