"""Fixed report contracts for agentic NURSE outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from nurse.schemas import NursePrediction, VerifiedRCLContext, _assert_no_forbidden


E_KEYS = {
    "root_evidence",
    "source_evidence",
    "symptom_penalty",
    "generic_evidence",
    "semantic_fact_evidence",
}
P_KEYS = {
    "propagation_consistency",
    "upstream_source_penalty",
}
H_KEYS = {"history_prior"}
R_KEYS = {"sre_rule"}


def score_breakdown_ephr(parts: Dict[str, float]) -> Dict[str, float]:
    """Collapse implementation-specific score parts into E/P/H/R."""

    buckets = {"E": 0.0, "P": 0.0, "H": 0.0, "R": 0.0}
    for key, raw_value in dict(parts or {}).items():
        value = float(raw_value)
        if key in E_KEYS:
            buckets["E"] += value
        elif key in P_KEYS:
            buckets["P"] += value
        elif key in H_KEYS:
            buckets["H"] += value
        elif key in R_KEYS:
            buckets["R"] += value
        else:
            buckets["E"] += value
    return {key: round(value, 6) for key, value in buckets.items()}


def score_breakdown_escpi(parts: Dict[str, float]) -> Dict[str, float]:
    """Backward-compatible alias kept for older callers."""

    return score_breakdown_ephr(parts)


def _evidence_lookup(context: VerifiedRCLContext) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("evidence_id")): dict(item)
        for item in context.evidence_summaries
        if item.get("evidence_id")
    }


def top_k_rows(prediction: NursePrediction, *, limit: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, entity in enumerate(prediction.ranked[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "entity": entity,
                "score": round(float(prediction.scores.get(entity, 0.0)), 6),
                "score_breakdown": score_breakdown_ephr(
                    prediction.score_breakdown.get(entity, {})
                ),
                "supporting_evidence_ids": list(prediction.evidence_ids.get(entity, [])),
            }
        )
    return rows


def evidence_paths(context: VerifiedRCLContext, *, limit: int = 8) -> List[Dict[str, Any]]:
    lookup = _evidence_lookup(context)
    paths: List[Dict[str, Any]] = []
    for relation in context.closure_paths[:limit]:
        src = str(relation.get("src") or "")
        dst = str(relation.get("dst") or "")
        rtype = str(relation.get("type") or "relation")
        evidence_ids = [
            evidence_id
            for evidence_id, item in lookup.items()
            if item.get("entity") in {src, dst}
        ][:4]
        paths.append(
            {
                "path_id": f"P{len(paths) + 1}",
                "candidate": src or dst,
                "relation_chain": [f"{rtype}({src}, {dst})"],
                "evidence_ids": evidence_ids,
                "interpretation": f"{rtype} relation links {src or '?'} to {dst or '?'} under the verified Evidence Graph.",
            }
        )
    if not paths:
        top = context.ranked_candidates[0] if context.ranked_candidates else ""
        ids = list(context.evidence_ids.get(top, []))[:4] if top else []
        paths.append(
            {
                "path_id": "P1",
                "candidate": top,
                "relation_chain": [f"supports_root(verified_evidence, {top})"] if top else [],
                "evidence_ids": ids,
                "interpretation": "No topology propagation path was emitted; the report falls back to direct verified evidence support.",
            }
        )
    return paths


def build_agent_report(
    *,
    dataset: str,
    case_id: str,
    model: str,
    prediction: NursePrediction,
    context: VerifiedRCLContext,
    incident_prior: Dict[str, Any],
    recovery_advice: Iterable[Dict[str, Any]],
    trace_path: Path,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    report = {
        "case_id": case_id,
        "dataset": dataset,
        "model": model,
        "top_k_ranking": top_k_rows(prediction),
        "evidence_paths": evidence_paths(context),
        "incident_prior": dict(incident_prior),
        "recovery_advice": list(recovery_advice),
        "unsupported_claims": [],
        "agent_trace_path": str(trace_path),
        "artifact_paths": dict(artifact_paths),
    }
    _assert_no_forbidden(report)
    return report


def write_report_files(report: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agent_report.json"
    md_path = output_dir / "agent_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        f"# NURSE Agent Report: {report.get('case_id', '')}",
        "",
        f"Dataset: `{report.get('dataset', '')}`",
        f"Model: `{report.get('model', '')}`",
        "",
        "## Top-K Ranking",
    ]
    for row in report.get("top_k_ranking", []):
        lines.append(
            f"- #{row['rank']} `{row['entity']}` score={row['score']} "
            f"E/P/H/R={row['score_breakdown']}"
        )
    lines.extend(["", "## Evidence Paths"])
    for path in report.get("evidence_paths", []):
        lines.append(
            f"- {path['path_id']}: {', '.join(path.get('relation_chain', []))} "
            f"evidence={path.get('evidence_ids', [])}"
        )
        lines.append(f"  {path.get('interpretation', '')}")
    if report.get("llm_explanation"):
        lines.extend(["", "## LLM Explanation", str(report.get("llm_explanation", ""))])
    lines.extend(["", "## Recovery Advice"])
    for item in report.get("recovery_advice", []):
        lines.append(
            f"- [{item.get('priority', 'medium')}] {item.get('action', '')} "
            f"(verify: {item.get('verification_before_action', '')})"
        )
        lines.append(f"  Rationale: {item.get('rationale', '')}")
    lines.extend(["", f"Trace: `{report.get('agent_trace_path', '')}`", ""])
    return "\n".join(lines)



