"""Generate the Nurse system explanation document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def _load(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_prediction(run_path: Path) -> Dict:
    prediction_dir = run_path / "predictions"
    for path in sorted(prediction_dir.glob("*.json")):
        return _load(path)
    return {}


def generate_system_doc(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    comparison_report: Optional[Dict] = None,
) -> Path:
    run_path = Path(run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = _load(run_path / "summary.json")
    config = _load(run_path / "config.json")
    comparison = comparison_report or _load(run_path / "comparison_report.json")
    prediction = _sample_prediction(run_path)
    metrics = summary.get("metrics", {})
    top_entities = ", ".join(prediction.get("ranked", [])[:5]) if prediction else "not recorded"
    nurse_top1 = comparison.get("nurse", {}).get("top1", metrics.get("acc_at_1", "not recorded"))
    reference_top1 = comparison.get("reference", {}).get("top1", "not recorded")
    gate_status = comparison.get("status", "not recorded")

    text = f"""# NURSE System Explanation

## Overview
Nurse (NeURo-Symbolic Evidence) localizes microservice incident roots with an
agent boundary around a deterministic Evidence Core. The Evidence Core builds a
verified evidence graph and produces the final Rank. The LLM layer packages that
verified context for SRE-facing explanation and advisory next steps, but it does
not directly reorder the final ranking in the default mode.

## System Architecture
1. `raw_adapters` loads RCAEval or AIOps telemetry without answer keys.
2. `alarm_service` constructs the shared alarm artifact and evidence scope.
3. `evidence_extractor` creates metric, alert, event, trace, and topology candidates.
4. `verifier` rechecks every candidate against raw telemetry or topology records.
5. `closure` builds a verified evidence graph with relations such as
   supports_root, propagates_to, symptom_of, and topology_neighbor.
6. `ranker` is the deterministic Evidence Ranker. It produces Top-K services,
   score breakdowns, evidence IDs, and closure paths.
7. `context_packager` converts the structured core output into a verified RCL context.
8. `llm_bridge` calls the User Manager to explain the fixed Rank using only
   verified evidence IDs and to list unsupported ideas separately.

## Run Summary
- Dataset: {summary.get("dataset", "unknown")}
- LLM mode: {summary.get("llm", config.get("llm", "unknown"))}
- Ranked cases: {metrics.get("ranked_cases", "not recorded")}
- Nurse Top-1: {nurse_top1}
- Reference Top-1: {reference_top1}
- Gate status: {gate_status}
- Token usage: {summary.get("token_usage", "not recorded")}
- Sample Top-5: {top_entities}

## Neuro-Symbolic Evidence Closure
The symbolic side owns the final RCL decision. It admits only schema-valid,
provenance-backed evidence into the graph, separates source-like signals from
symptoms, computes counterfactual contribution, and exports score components for
each candidate. The neural side receives this verified context after ranking and
can summarize, explain, or suggest future checks. Same-run LLM proposals are
advisory in the public release configuration.

## Interpretability
Nurse is interpretable because each ranking score is decomposed into explicit
parts: telemetry evidence, evidence-program support, counterfactual drop,
closure and topology consistency, incident-shape priors, and penalties. Every
score can be traced back to a schema-bound evidence id, a verification rule, and
provenance pointing to the raw source.

## Explainability
Nurse is explainable because incident outputs are generated from verified
evidence, closure paths, and score breakdowns. The User Manager may describe why
a candidate looks causal, why another service is likely a downstream symptom,
and which SRE verification action follows, but every claim must cite existing
evidence IDs or be listed as unsupported.

## Iteration Gate
When a run misses the target Acc@K profile on the selected evaluation subset,
`comparison_report.json` marks the run as `iteration-needed` and
`failure_diagnostics.csv` labels low-score cases as missed_alarm, weak_evidence,
wrong_symptom_penalty, closure_missing_edge, or unsupported proposal. Tuning
should change detector parameters, evidence-generation tools, verification
rules, closure weights, or ranker weights for future deterministic runs, not the
same-run final Rank.
"""
    output = out / "NURSE_SYSTEM_EXPLANATION.md"
    output.write_text(text, encoding="utf-8")
    return output
