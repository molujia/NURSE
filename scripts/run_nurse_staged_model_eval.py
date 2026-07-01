"""Run the staged NURSE model evaluation under fixed MJ/W settings.

This script follows the user's current staged protocol:

1. Reuse or verify no-LLM full baselines (Phase A).
2. Probe three requested LLMs on AIOps25 with 5 cases.
3. If the decision is mixed, extend the same AIOps probe to 10 cases.
4. If all surviving models are non-degrading, run the final 40-case sample
   on both datasets and compare against no-LLM on the exact same cases.
5. If all surviving models are degrading, stop and mark the run as needing
   optimization before broader LLM evaluation.

The script never lets the LLM directly rewrite a finished ranking. It uses the
current `full` mode, which routes LLM output only through pre-freeze semantic
facts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from nurse.case_order import load_case_order_manifest, write_case_order_manifest
from nurse.llm_connectivity import failed_probe_results, probe_models
from nurse.sota_reporting import load_run_snapshot


DEFAULT_SEED = 20260628
DEFAULT_MJ = 0
DEFAULT_WINDOW_SECONDS = 480
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_PROBE5_SIZE = 5
DEFAULT_PROBE10_SIZE = 10
DEFAULT_PRIMARY_MODELS = [
    "ep-20260630204353-8j7f2",
    "glm-4.5-air",
    "gpt-5-nano",
]
DEFAULT_FALLBACK_MODELS = [
    "gpt-5-mini",
    "gpt-5",
]
DEFAULT_DATASETS = [
    "aiops25-retained-nonnode",
    "rcaeval-tt",
]
DEFAULT_FULL_BASELINES = {
    "aiops25-retained-nonnode": (
        REPO_ROOT
        / "tmp"
        / "campaign_no_llm_mj_repaired"
        / "20260630_211800"
        / "mj_sensitivity"
        / "runs"
        / "aiops25-retained-nonnode"
        / "mj_0"
    ),
    "rcaeval-tt": (
        REPO_ROOT
        / "tmp"
        / "campaign_no_llm_mj_repaired"
        / "20260630_211800"
        / "mj_sensitivity"
        / "runs"
        / "rcaeval-tt"
        / "mj_0"
    ),
}


def _slug(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    pieces: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            pieces.append(char)
        else:
            pieces.append("_")
    return "".join(pieces)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _manifest_case_ids(manifest_path: Path, dataset: str) -> list[str]:
    payload = load_case_order_manifest(manifest_path)
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), dict) else {}
    raw_case_ids = datasets.get(dataset, [])
    if not isinstance(raw_case_ids, list):
        raise ValueError(f"invalid case list for dataset={dataset} in {manifest_path}")
    return [str(case_id) for case_id in raw_case_ids]


def _hit_at(ranking: Sequence[str], labels: Sequence[str], k: int) -> int:
    rank_prefix = {str(item) for item in list(ranking or [])[: max(int(k), 0)]}
    label_set = {str(item) for item in list(labels or [])}
    return int(bool(rank_prefix & label_set))


def _record_map(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = summary.get("records", [])
    if not isinstance(records, list):
        return {}
    by_case: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = str(record.get("case_id") or "")
        if case_id:
            by_case[case_id] = record
    return by_case


def _load_case_metrics_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            case_id = str((row or {}).get("case_id") or "")
            if case_id:
                rows[case_id] = {str(key): "" if value is None else str(value) for key, value in row.items()}
        return rows


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(value)]


def summarize_case_subset(
    *,
    run_dir: str | Path,
    dataset: str,
    case_ids: Sequence[str],
    requested_model: str,
    resolved_model: str,
    sample_name: str,
) -> Dict[str, Any]:
    snapshot = load_run_snapshot(run_dir)
    summary = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    predictions = snapshot.get("predictions", {}) if isinstance(snapshot.get("predictions"), dict) else {}
    labels_by_case = {}
    labels_path = Path(run_dir) / "labels.json"
    case_metrics_by_case = _load_case_metrics_rows(Path(run_dir) / "case_metrics.csv")
    if labels_path.exists():
        labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(labels_payload, dict):
            labels_by_case = labels_payload
    records_by_case = _record_map(summary)

    case_rows: list[Dict[str, Any]] = []
    llm_status_counts: Counter[str] = Counter()
    total_tokens = 0
    total_seconds = 0.0
    non_llm_seconds = 0.0
    llm_seconds = 0.0
    for case_id in [str(case_id) for case_id in case_ids]:
        ranking = _string_list(predictions.get(case_id, []))
        labels = _string_list(labels_by_case.get(case_id, []))
        record = records_by_case.get(case_id, {})
        metric_row = case_metrics_by_case.get(case_id, {})
        llm_payload = record.get("llm") if isinstance(record.get("llm"), dict) else {}
        timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
        llm_status = str(record.get("llm_status") or "")
        if llm_status:
            llm_status_counts[llm_status] += 1
        total_tokens += _safe_int(llm_payload.get("total_tokens", 0))
        total_seconds += _safe_float(timing.get("total_seconds", 0.0))
        non_llm_seconds += _safe_float(timing.get("non_llm_seconds", 0.0))
        llm_seconds += _safe_float(timing.get("llm_seconds", 0.0))
        matched_rank = _safe_int(metric_row.get("matched_rank", 0))
        if matched_rank > 0:
            hit_at_1 = int(matched_rank <= 1)
            hit_at_3 = int(matched_rank <= 3)
            hit_at_5 = int(matched_rank <= 5)
        else:
            hit_at_1 = _hit_at(ranking, labels, 1)
            hit_at_3 = _hit_at(ranking, labels, 3)
            hit_at_5 = _hit_at(ranking, labels, 5)
        case_rows.append(
            {
                "dataset": dataset,
                "sample_name": sample_name,
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "case_id": case_id,
                "labels": json.dumps(labels, ensure_ascii=False),
                "acceptable_labels": str(metric_row.get("acceptable_labels", "")),
                "matched_label": str(metric_row.get("matched_label", "")),
                "matched_rank": matched_rank,
                "top1": ranking[0] if ranking else "",
                "top3": json.dumps(ranking[:3], ensure_ascii=False),
                "top5": json.dumps(ranking[:5], ensure_ascii=False),
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "llm_status": llm_status,
                "llm_total_tokens": _safe_int(llm_payload.get("total_tokens", 0)),
                "total_seconds": round(_safe_float(timing.get("total_seconds", 0.0)), 6),
                "non_llm_seconds": round(_safe_float(timing.get("non_llm_seconds", 0.0)), 6),
                "llm_seconds": round(_safe_float(timing.get("llm_seconds", 0.0)), 6),
                "run_dir": str(run_dir),
            }
        )

    case_count = max(len(case_rows), 1)
    acc_at_1 = sum(row["hit_at_1"] for row in case_rows) / case_count
    acc_at_3 = sum(row["hit_at_3"] for row in case_rows) / case_count
    acc_at_5 = sum(row["hit_at_5"] for row in case_rows) / case_count
    average_score = (acc_at_1 + acc_at_3 + acc_at_5) / 3.0

    return {
        "dataset": dataset,
        "sample_name": sample_name,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "run_dir": str(run_dir),
        "case_ids": [str(case_id) for case_id in case_ids],
        "case_count": len(case_rows),
        "acc_at_1": acc_at_1,
        "acc_at_3": acc_at_3,
        "acc_at_5": acc_at_5,
        "average_score": average_score,
        "token_usage_total": total_tokens,
        "token_usage_per_case": total_tokens / case_count,
        "latency_seconds_per_case": total_seconds / case_count,
        "non_llm_latency_seconds_per_case": non_llm_seconds / case_count,
        "llm_latency_seconds_per_case": llm_seconds / case_count,
        "llm_status_counts": dict(sorted(llm_status_counts.items())),
        "llm_all_ok": set(llm_status_counts) <= {"ok"},
        "rows": case_rows,
    }


def classify_against_baseline(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    deltas = {
        "acc_at_1": _safe_float(candidate_metrics.get("acc_at_1")) - _safe_float(baseline_metrics.get("acc_at_1")),
        "acc_at_3": _safe_float(candidate_metrics.get("acc_at_3")) - _safe_float(baseline_metrics.get("acc_at_3")),
        "acc_at_5": _safe_float(candidate_metrics.get("acc_at_5")) - _safe_float(baseline_metrics.get("acc_at_5")),
        "average_score": _safe_float(candidate_metrics.get("average_score")) - _safe_float(baseline_metrics.get("average_score")),
    }
    rank_deltas = [deltas["acc_at_1"], deltas["acc_at_3"], deltas["acc_at_5"]]
    if all(abs(delta) <= eps for delta in rank_deltas):
        direction = "unchanged"
    elif all(delta >= -eps for delta in rank_deltas) and any(delta > eps for delta in rank_deltas):
        direction = "better"
    elif all(delta <= eps for delta in rank_deltas) and any(delta < -eps for delta in rank_deltas):
        direction = "worse"
    else:
        direction = "mixed"
    return {
        "direction": direction,
        "delta_acc_at_1": deltas["acc_at_1"],
        "delta_acc_at_3": deltas["acc_at_3"],
        "delta_acc_at_5": deltas["acc_at_5"],
        "delta_average_score": deltas["average_score"],
    }


def decide_round(classifications: Sequence[Mapping[str, Any]]) -> str:
    directions = [str(item.get("direction") or "") for item in classifications]
    if not directions:
        return "blocked"
    if any(direction in {"unavailable", "failed"} for direction in directions):
        return "blocked"
    if all(direction in {"better", "unchanged"} for direction in directions):
        return "D"
    if all(direction in {"worse", "unchanged"} for direction in directions):
        return "C"
    return "extend"


def _verify_no_llm_full_baseline(
    *,
    dataset: str,
    run_dir: Path,
    expected_mj: int,
    expected_window_seconds: int,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing no-LLM baseline summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("dataset") or "") != dataset:
        raise ValueError(f"baseline dataset mismatch: expected {dataset}, found {summary.get('dataset')}")
    if str(summary.get("llm") or "") != "none":
        raise ValueError(f"baseline {run_dir} is not a no-LLM run")
    if _safe_int(summary.get("candidate_expansion_rounds")) != int(expected_mj):
        raise ValueError(f"baseline MJ mismatch at {run_dir}")
    if _safe_int(summary.get("window_width_seconds")) != int(expected_window_seconds):
        raise ValueError(f"baseline window mismatch at {run_dir}")
    llm_status_counts = summary.get("llm_status_counts") if isinstance(summary.get("llm_status_counts"), dict) else {}
    if any(str(key) != "disabled" and _safe_int(value) > 0 for key, value in llm_status_counts.items()):
        raise ValueError(f"baseline contains unexpected LLM statuses: {llm_status_counts}")
    return summary


def _existing_report(run_root: Path) -> Dict[str, Any]:
    return _load_json(run_root / "report.json")


def _requested_models_from_report(existing_report: Mapping[str, Any], fallback: Sequence[str]) -> list[str]:
    raw = existing_report.get("requested_models")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return [str(item) for item in fallback]


def _find_existing_run_dir(
    *,
    run_root: Path,
    dataset: str,
    sample_name: str,
    requested_model: str,
    existing_report: Mapping[str, Any],
) -> Path | None:
    for round_report in existing_report.get("rounds", []) if isinstance(existing_report.get("rounds"), list) else []:
        if str(round_report.get("dataset") or "") != dataset:
            continue
        if str(round_report.get("sample_name") or "") != sample_name:
            continue
        model_results = round_report.get("model_results")
        if not isinstance(model_results, list):
            continue
        for result in model_results:
            if not isinstance(result, dict):
                continue
            if str(result.get("requested_model") or "") != requested_model:
                continue
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            run_dir = str(metrics.get("run_dir") or "")
            if run_dir:
                path = Path(run_dir)
                if path.exists():
                    return path

    final_sample = existing_report.get("final_sample") if isinstance(existing_report.get("final_sample"), dict) else {}
    runs = final_sample.get("runs") if isinstance(final_sample.get("runs"), list) else []
    for result in runs:
        if not isinstance(result, dict):
            continue
        if str(result.get("requested_model") or "") != requested_model:
            continue
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        run_dir = str(metrics.get("run_dir") or "")
        if not run_dir:
            continue
        path = Path(run_dir)
        if path.exists() and _slug(dataset) in str(path) and _slug(sample_name) in str(path):
            return path

    default_path = run_root / "runs" / _slug(dataset) / _slug(sample_name) / _slug(requested_model)
    if default_path.exists():
        return default_path
    return None


def rehydrate_existing_run_root(
    *,
    run_root: Path,
    full_baselines: Mapping[str, Path],
    default_requested_models: Sequence[str],
) -> Dict[str, Any]:
    existing_report = _existing_report(run_root)
    requested_models = _requested_models_from_report(existing_report, default_requested_models)

    report: Dict[str, Any] = dict(existing_report)
    report["rehydrated_at"] = datetime.now().isoformat(timespec="seconds")
    report["run_root"] = str(run_root)
    report["requested_models"] = requested_models
    report["baseline_full_runs"] = {dataset: str(path) for dataset, path in full_baselines.items()}

    manifests = {
        "aiops25_probe5": run_root / "manifests" / "aiops25-retained-nonnode_aiops25_probe5_seed20260628_sample5.json",
        "aiops25_probe10": run_root / "manifests" / "aiops25-retained-nonnode_aiops25_probe10_seed20260628_sample10.json",
        "aiops25_sample40": run_root / "manifests" / "aiops25-retained-nonnode_aiops25_sample40_seed20260628_sample40.json",
        "rcaeval_sample40": run_root / "manifests" / "rcaeval-tt_rcaeval_sample40_seed20260628_sample40.json",
    }

    report["baseline_full_metrics"] = {}
    for dataset, manifest_key in [
        ("aiops25-retained-nonnode", "aiops25_sample40"),
        ("rcaeval-tt", "rcaeval_sample40"),
    ]:
        manifest_path = manifests[manifest_key]
        if not manifest_path.exists():
            continue
        case_ids = _manifest_case_ids(manifest_path, dataset)
        baseline_sample = summarize_case_subset(
            run_dir=full_baselines[dataset],
            dataset=dataset,
            case_ids=case_ids,
            requested_model="none",
            resolved_model="none",
            sample_name="sample40",
        )
        report["baseline_full_metrics"][dataset] = {
            "sample40_metrics": {key: value for key, value in baseline_sample.items() if key != "rows"},
        }

    rounds: list[Dict[str, Any]] = []
    for sample_name, manifest_key in [("probe5", "aiops25_probe5"), ("probe10", "aiops25_probe10")]:
        manifest_path = manifests[manifest_key]
        if not manifest_path.exists():
            continue
        case_ids = _manifest_case_ids(manifest_path, "aiops25-retained-nonnode")
        baseline_metrics = summarize_case_subset(
            run_dir=full_baselines["aiops25-retained-nonnode"],
            dataset="aiops25-retained-nonnode",
            case_ids=case_ids,
            requested_model="none",
            resolved_model="none",
            sample_name=sample_name,
        )
        model_results = []
        found_any = False
        for requested_model in requested_models:
            run_dir = _find_existing_run_dir(
                run_root=run_root,
                dataset="aiops25-retained-nonnode",
                sample_name=sample_name,
                requested_model=requested_model,
                existing_report=existing_report,
            )
            if run_dir is None or not (run_dir / "summary.json").exists():
                model_results.append(
                    {
                        "requested_model": requested_model,
                        "resolved_model": "",
                        "status": "unavailable",
                        "attempts": [],
                        "metrics": {},
                    }
                )
                continue
            found_any = True
            model_results.append(
                {
                    "requested_model": requested_model,
                    "resolved_model": run_dir.name,
                    "status": "ok",
                    "attempts": [],
                    "metrics": summarize_case_subset(
                        run_dir=run_dir,
                        dataset="aiops25-retained-nonnode",
                        case_ids=case_ids,
                        requested_model=requested_model,
                        resolved_model=run_dir.name,
                        sample_name=sample_name,
                    ),
                }
            )
        if not found_any:
            continue
        round_report = _round_report(
            dataset="aiops25-retained-nonnode",
            baseline_metrics=baseline_metrics,
            model_results=model_results,
            sample_name=sample_name,
        )
        round_report["baseline_case_rows"] = baseline_metrics["rows"]
        round_report["model_results"] = model_results
        rounds.append(round_report)
    report["rounds"] = rounds
    if rounds:
        report["final_decision_after_probe"] = str(rounds[-1].get("decision") or "")

    final_rows: list[Dict[str, Any]] = []
    final_case_rows: list[Dict[str, Any]] = []
    final_runs: list[Dict[str, Any]] = []
    for dataset, manifest_key in [
        ("aiops25-retained-nonnode", "aiops25_sample40"),
        ("rcaeval-tt", "rcaeval_sample40"),
    ]:
        manifest_path = manifests[manifest_key]
        if not manifest_path.exists():
            continue
        case_ids = _manifest_case_ids(manifest_path, dataset)
        baseline_metrics = summarize_case_subset(
            run_dir=full_baselines[dataset],
            dataset=dataset,
            case_ids=case_ids,
            requested_model="none",
            resolved_model="none",
            sample_name="sample40",
        )
        final_rows.append(
            {
                "dataset": dataset,
                "sample_name": "sample40",
                "requested_model": "none",
                "resolved_model": "none",
                "direction": "baseline",
                "acc_at_1": baseline_metrics["acc_at_1"],
                "acc_at_3": baseline_metrics["acc_at_3"],
                "acc_at_5": baseline_metrics["acc_at_5"],
                "average_score": baseline_metrics["average_score"],
                "token_usage_total": baseline_metrics["token_usage_total"],
                "run_dir": baseline_metrics["run_dir"],
            }
        )
        final_case_rows.extend(baseline_metrics["rows"])
        for requested_model in requested_models:
            run_dir = _find_existing_run_dir(
                run_root=run_root,
                dataset=dataset,
                sample_name="sample40",
                requested_model=requested_model,
                existing_report=existing_report,
            )
            if run_dir is None or not (run_dir / "summary.json").exists():
                final_rows.append(
                    {
                        "dataset": dataset,
                        "sample_name": "sample40",
                        "requested_model": requested_model,
                        "resolved_model": "",
                        "direction": "unavailable",
                    }
                )
                continue
            metrics = summarize_case_subset(
                run_dir=run_dir,
                dataset=dataset,
                case_ids=case_ids,
                requested_model=requested_model,
                resolved_model=run_dir.name,
                sample_name="sample40",
            )
            deltas = classify_against_baseline(
                baseline_metrics=baseline_metrics,
                candidate_metrics=metrics,
            )
            final_rows.append(
                {
                    "dataset": dataset,
                    "sample_name": "sample40",
                    "requested_model": requested_model,
                    "resolved_model": run_dir.name,
                    "acc_at_1": metrics["acc_at_1"],
                    "acc_at_3": metrics["acc_at_3"],
                    "acc_at_5": metrics["acc_at_5"],
                    "average_score": metrics["average_score"],
                    "token_usage_total": metrics["token_usage_total"],
                    "run_dir": metrics["run_dir"],
                    **deltas,
                }
            )
            final_case_rows.extend(metrics["rows"])
            final_runs.append(
                {
                    "requested_model": requested_model,
                    "resolved_model": run_dir.name,
                    "status": "ok" if metrics.get("llm_all_ok") else "non_ok_status",
                    "metrics": metrics,
                }
            )
    report["final_sample"] = {
        "rows": final_rows,
        "runs": final_runs,
        "case_rows_count": len(final_case_rows),
    }

    round_rows: list[Dict[str, Any]] = []
    round_case_rows: list[Dict[str, Any]] = []
    probe_attempt_rows: list[Dict[str, Any]] = []
    for round_report in rounds:
        round_rows.extend(round_report.get("summary_rows", []))
        round_case_rows.extend(round_report.get("baseline_case_rows", []))
        for model_result in round_report.get("model_results", []):
            metrics = model_result.get("metrics") if isinstance(model_result.get("metrics"), dict) else {}
            round_case_rows.extend(metrics.get("rows", []))
            probe_attempt_rows.extend(model_result.get("attempts", []))

    _write_csv(run_root / "probe_round_summary.csv", round_rows)
    _write_csv(run_root / "probe_round_case_rows.csv", round_case_rows)
    _write_csv(run_root / "probe_attempts.csv", probe_attempt_rows)
    _write_csv(run_root / "final_sample_summary.csv", final_rows)
    _write_csv(run_root / "final_sample_case_rows.csv", final_case_rows)
    _write_json(run_root / "report.json", report)
    _write_json(run_root / "report_rehydrated.json", report)
    return report


def _prepare_manifest(
    *,
    manifest_root: Path,
    dataset: str,
    lab_root: Path,
    seed: int,
    sample_size: int,
    name: str,
) -> Path:
    manifest_path = manifest_root / f"{_slug(dataset)}_{name}_seed{int(seed)}_sample{int(sample_size)}.json"
    write_case_order_manifest(
        manifest_path,
        datasets=[dataset],
        lab_root=lab_root,
        seed=seed,
        shuffle=True,
        sample_size=sample_size,
        name=name,
    )
    return manifest_path


def _run_nurse_job(
    *,
    dataset: str,
    model: str,
    run_dir: Path,
    case_manifest: Path,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
    lab_root: Path,
    alarm_cache_root: Path,
) -> Path:
    os.environ["NURSE_LAB_ROOT"] = str(lab_root)
    os.environ["NURSE_ALARM_CACHE_ROOT"] = str(alarm_cache_root)

    import nurse.runner as nurse_runner

    nurse_runner = importlib.reload(nurse_runner)
    return nurse_runner.run_nurse(
        dataset,
        0,
        model,
        str(run_dir.parent),
        generate_doc=False,
        run_dir=str(run_dir),
        resume=False,
        progress=False,
        ranking_engine="evidence_program_v2",
        experiment_mode="full",
        shuffle=True,
        seed=DEFAULT_SEED,
        case_manifest=str(case_manifest),
        allow_no_llm=(str(model).lower() == "none"),
        candidate_expansion_rounds=candidate_expansion_rounds,
        window_width_seconds=window_width_seconds,
    )


def _probe_candidate(model: str) -> Dict[str, Any]:
    report = probe_models(models=[model], openai_model=None, openai_base_url="")
    results = report.get("results", [])
    result = dict(results[0]) if isinstance(results, list) and results else {"model": model, "status": "error"}
    result["probe_created_at"] = report.get("created_at", "")
    return result


def _evaluate_model_with_fallback(
    *,
    dataset: str,
    requested_model: str,
    fallback_models: Sequence[str],
    case_manifest: Path,
    case_ids: Sequence[str],
    run_root: Path,
    lab_root: Path,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
    sample_name: str,
) -> Dict[str, Any]:
    attempts: list[Dict[str, Any]] = []
    for candidate_model in [requested_model, *list(fallback_models)]:
        probe_result = _probe_candidate(candidate_model)
        attempts.append(
            {
                "stage": "probe",
                "requested_model": requested_model,
                "candidate_model": candidate_model,
                **probe_result,
            }
        )
        if str(probe_result.get("status") or "") != "ok":
            continue

        candidate_run_dir = run_root / "runs" / _slug(dataset) / _slug(sample_name) / _slug(candidate_model)
        candidate_cache_root = run_root / "_alarm_cache" / _slug(dataset) / _slug(sample_name) / _slug(candidate_model)
        try:
            _run_nurse_job(
                dataset=dataset,
                model=candidate_model,
                run_dir=candidate_run_dir,
                case_manifest=case_manifest,
                candidate_expansion_rounds=candidate_expansion_rounds,
                window_width_seconds=window_width_seconds,
                lab_root=lab_root,
                alarm_cache_root=candidate_cache_root,
            )
        except Exception as exc:
            attempts.append(
                {
                    "stage": "run",
                    "requested_model": requested_model,
                    "candidate_model": candidate_model,
                    "status": "failed",
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        sample_metrics = summarize_case_subset(
            run_dir=candidate_run_dir,
            dataset=dataset,
            case_ids=case_ids,
            requested_model=requested_model,
            resolved_model=candidate_model,
            sample_name=sample_name,
        )
        attempts.append(
            {
                "stage": "run",
                "requested_model": requested_model,
                "candidate_model": candidate_model,
                "status": "ok" if sample_metrics.get("llm_all_ok") else "non_ok_status",
                "llm_status_counts": sample_metrics.get("llm_status_counts", {}),
                "run_dir": str(candidate_run_dir),
            }
        )
        if sample_metrics.get("llm_all_ok"):
            return {
                "requested_model": requested_model,
                "resolved_model": candidate_model,
                "status": "ok",
                "attempts": attempts,
                "metrics": sample_metrics,
            }

    return {
        "requested_model": requested_model,
        "resolved_model": "",
        "status": "unavailable",
        "attempts": attempts,
        "metrics": {},
    }


def _round_report(
    *,
    dataset: str,
    baseline_metrics: Mapping[str, Any],
    model_results: Sequence[Mapping[str, Any]],
    sample_name: str,
) -> Dict[str, Any]:
    classifications: list[Dict[str, Any]] = []
    summary_rows: list[Dict[str, Any]] = []
    for result in model_results:
        requested_model = str(result.get("requested_model") or "")
        resolved_model = str(result.get("resolved_model") or "")
        if str(result.get("status") or "") != "ok":
            classification = {
                "dataset": dataset,
                "sample_name": sample_name,
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "direction": "unavailable",
            }
            summary_rows.append(classification)
            classifications.append(classification)
            continue
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        deltas = classify_against_baseline(
            baseline_metrics=baseline_metrics,
            candidate_metrics=metrics,
        )
        row = {
            "dataset": dataset,
            "sample_name": sample_name,
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "acc_at_1": metrics.get("acc_at_1"),
            "acc_at_3": metrics.get("acc_at_3"),
            "acc_at_5": metrics.get("acc_at_5"),
            "average_score": metrics.get("average_score"),
            "baseline_acc_at_1": baseline_metrics.get("acc_at_1"),
            "baseline_acc_at_3": baseline_metrics.get("acc_at_3"),
            "baseline_acc_at_5": baseline_metrics.get("acc_at_5"),
            "baseline_average_score": baseline_metrics.get("average_score"),
            "token_usage_total": metrics.get("token_usage_total"),
            "run_dir": metrics.get("run_dir", ""),
            **deltas,
        }
        summary_rows.append(row)
        classifications.append(dict(row))
    return {
        "dataset": dataset,
        "sample_name": sample_name,
        "baseline_metrics": dict(baseline_metrics),
        "summary_rows": summary_rows,
        "classifications": classifications,
        "decision": decide_round(classifications),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged NURSE model evaluation")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "tmp" / "staged_model_eval"))
    parser.add_argument("--lab-root", default=str(REPO_ROOT.parent))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-expansion-rounds", type=int, default=DEFAULT_MJ)
    parser.add_argument("--window-width-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--probe5-size", type=int, default=DEFAULT_PROBE5_SIZE)
    parser.add_argument("--probe10-size", type=int, default=DEFAULT_PROBE10_SIZE)
    parser.add_argument("--models", nargs="+", default=DEFAULT_PRIMARY_MODELS)
    parser.add_argument("--fallback-models", nargs="+", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--aiops-full-run", default=str(DEFAULT_FULL_BASELINES["aiops25-retained-nonnode"]))
    parser.add_argument("--tt-full-run", default=str(DEFAULT_FULL_BASELINES["rcaeval-tt"]))
    parser.add_argument("--skip-final-sample", action="store_true")
    parser.add_argument(
        "--final-sample-only",
        action="store_true",
        help="Skip the probe rounds and directly run the final 40-case sample with the requested models.",
    )
    parser.add_argument(
        "--rehydrate-run-root",
        default="",
        help="Recompute summaries/csv outputs for an existing staged-model-eval run root without rerunning models.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    full_baselines = {
        "aiops25-retained-nonnode": Path(args.aiops_full_run).expanduser().resolve(),
        "rcaeval-tt": Path(args.tt_full_run).expanduser().resolve(),
    }
    if str(args.rehydrate_run_root or "").strip():
        run_root = Path(args.rehydrate_run_root).expanduser().resolve()
        report = rehydrate_existing_run_root(
            run_root=run_root,
            full_baselines=full_baselines,
            default_requested_models=args.models,
        )
        print(f"[NURSE] Rehydrated staged model evaluation root: {run_root}")
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "final_decision_after_probe": report.get("final_decision_after_probe", ""),
                    "final_sample_rows": len((report.get("final_sample") or {}).get("rows", []))
                    if isinstance(report.get("final_sample"), dict)
                    else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_root = run_root / "manifests"
    lab_root = Path(args.lab_root).expanduser().resolve()
    verified_baselines: Dict[str, Dict[str, Any]] = {}
    for dataset, run_dir in full_baselines.items():
        verified_baselines[dataset] = _verify_no_llm_full_baseline(
            dataset=dataset,
            run_dir=run_dir,
            expected_mj=args.candidate_expansion_rounds,
            expected_window_seconds=args.window_width_seconds,
        )

    manifests = {
        "aiops25_probe5": _prepare_manifest(
            manifest_root=manifest_root,
            dataset="aiops25-retained-nonnode",
            lab_root=lab_root,
            seed=args.seed,
            sample_size=args.probe5_size,
            name="aiops25_probe5",
        ),
        "aiops25_probe10": _prepare_manifest(
            manifest_root=manifest_root,
            dataset="aiops25-retained-nonnode",
            lab_root=lab_root,
            seed=args.seed,
            sample_size=args.probe10_size,
            name="aiops25_probe10",
        ),
        "aiops25_sample40": _prepare_manifest(
            manifest_root=manifest_root,
            dataset="aiops25-retained-nonnode",
            lab_root=lab_root,
            seed=args.seed,
            sample_size=args.sample_size,
            name="aiops25_sample40",
        ),
        "rcaeval_sample40": _prepare_manifest(
            manifest_root=manifest_root,
            dataset="rcaeval-tt",
            lab_root=lab_root,
            seed=args.seed,
            sample_size=args.sample_size,
            name="rcaeval_sample40",
        ),
    }

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "lab_root": str(lab_root),
        "seed": int(args.seed),
        "candidate_expansion_rounds": int(args.candidate_expansion_rounds),
        "window_width_seconds": int(args.window_width_seconds),
        "requested_models": [str(model) for model in args.models],
        "fallback_models": [str(model) for model in args.fallback_models],
        "baseline_full_runs": {dataset: str(path) for dataset, path in full_baselines.items()},
        "baseline_full_metrics": {},
        "rounds": [],
        "final_sample": {},
    }

    for dataset, run_dir in full_baselines.items():
        baseline_case_ids = _manifest_case_ids(
            manifests["aiops25_sample40"] if dataset == "aiops25-retained-nonnode" else manifests["rcaeval_sample40"],
            dataset,
        )
        baseline_sample40 = summarize_case_subset(
            run_dir=run_dir,
            dataset=dataset,
            case_ids=baseline_case_ids,
            requested_model="none",
            resolved_model="none",
            sample_name="sample40_no_llm_reference",
        )
        report["baseline_full_metrics"][dataset] = {
            "full_summary_metrics": verified_baselines[dataset].get("paper_metrics", {}),
            "sample40_metrics": {
                key: value
                for key, value in baseline_sample40.items()
                if key != "rows"
            },
        }

    latest_successful_models: Dict[str, str] = {}
    if args.final_sample_only:
        round_decision = "D"
        latest_successful_models = {str(model): str(model) for model in args.models}
        report["final_decision_after_probe"] = round_decision
    else:
        aiops_probe_rounds = [
            ("probe5", manifests["aiops25_probe5"]),
            ("probe10", manifests["aiops25_probe10"]),
        ]
        round_decision = "extend"

        for round_index, (sample_name, manifest_path) in enumerate(aiops_probe_rounds, start=1):
            if round_index > 1 and round_decision != "extend":
                break
            case_ids = _manifest_case_ids(manifest_path, "aiops25-retained-nonnode")
            baseline_metrics = summarize_case_subset(
                run_dir=full_baselines["aiops25-retained-nonnode"],
                dataset="aiops25-retained-nonnode",
                case_ids=case_ids,
                requested_model="none",
                resolved_model="none",
                sample_name=sample_name,
            )
            model_results = [
                _evaluate_model_with_fallback(
                    dataset="aiops25-retained-nonnode",
                    requested_model=str(model),
                    fallback_models=args.fallback_models,
                    case_manifest=manifest_path,
                    case_ids=case_ids,
                    run_root=run_root,
                    lab_root=lab_root,
                    candidate_expansion_rounds=args.candidate_expansion_rounds,
                    window_width_seconds=args.window_width_seconds,
                    sample_name=sample_name,
                )
                for model in args.models
            ]
            round_report = _round_report(
                dataset="aiops25-retained-nonnode",
                baseline_metrics=baseline_metrics,
                model_results=model_results,
                sample_name=sample_name,
            )
            round_report["baseline_case_rows"] = baseline_metrics["rows"]
            round_report["model_results"] = model_results
            report["rounds"].append(round_report)
            round_decision = str(round_report.get("decision") or "blocked")
            for model_result in model_results:
                if str(model_result.get("status") or "") == "ok":
                    latest_successful_models[str(model_result.get("requested_model") or "")] = str(
                        model_result.get("resolved_model") or ""
                    )

        report["final_decision_after_probe"] = round_decision

    if round_decision == "D" and not args.skip_final_sample:
        final_rows: list[Dict[str, Any]] = []
        final_case_rows: list[Dict[str, Any]] = []
        final_runs: list[Dict[str, Any]] = []
        for dataset, manifest_key in [
            ("aiops25-retained-nonnode", "aiops25_sample40"),
            ("rcaeval-tt", "rcaeval_sample40"),
        ]:
            case_ids = _manifest_case_ids(manifests[manifest_key], dataset)
            baseline_metrics = summarize_case_subset(
                run_dir=full_baselines[dataset],
                dataset=dataset,
                case_ids=case_ids,
                requested_model="none",
                resolved_model="none",
                sample_name="sample40",
            )
            final_rows.append(
                {
                    "dataset": dataset,
                    "sample_name": "sample40",
                    "requested_model": "none",
                    "resolved_model": "none",
                    "direction": "baseline",
                    "acc_at_1": baseline_metrics["acc_at_1"],
                    "acc_at_3": baseline_metrics["acc_at_3"],
                    "acc_at_5": baseline_metrics["acc_at_5"],
                    "average_score": baseline_metrics["average_score"],
                    "token_usage_total": baseline_metrics["token_usage_total"],
                    "run_dir": baseline_metrics["run_dir"],
                }
            )
            final_case_rows.extend(baseline_metrics["rows"])
            for requested_model in args.models:
                resolved_model = latest_successful_models.get(str(requested_model)) or str(requested_model)
                result = _evaluate_model_with_fallback(
                    dataset=dataset,
                    requested_model=resolved_model,
                    fallback_models=args.fallback_models,
                    case_manifest=manifests[manifest_key],
                    case_ids=case_ids,
                    run_root=run_root,
                    lab_root=lab_root,
                    candidate_expansion_rounds=args.candidate_expansion_rounds,
                    window_width_seconds=args.window_width_seconds,
                    sample_name="sample40",
                )
                final_runs.append(result)
                if str(result.get("status") or "") != "ok":
                    final_rows.append(
                        {
                            "dataset": dataset,
                            "sample_name": "sample40",
                            "requested_model": requested_model,
                            "resolved_model": result.get("resolved_model", ""),
                            "direction": "unavailable",
                        }
                    )
                    continue
                metrics = result["metrics"]
                deltas = classify_against_baseline(
                    baseline_metrics=baseline_metrics,
                    candidate_metrics=metrics,
                )
                final_rows.append(
                    {
                        "dataset": dataset,
                        "sample_name": "sample40",
                        "requested_model": requested_model,
                        "resolved_model": metrics.get("resolved_model", resolved_model),
                        "acc_at_1": metrics["acc_at_1"],
                        "acc_at_3": metrics["acc_at_3"],
                        "acc_at_5": metrics["acc_at_5"],
                        "average_score": metrics["average_score"],
                        "token_usage_total": metrics["token_usage_total"],
                        "run_dir": metrics["run_dir"],
                        **deltas,
                    }
                )
                final_case_rows.extend(metrics["rows"])

        report["final_sample"] = {
            "rows": final_rows,
            "runs": final_runs,
            "case_rows_count": len(final_case_rows),
        }
        _write_csv(run_root / "final_sample_summary.csv", final_rows)
        _write_csv(run_root / "final_sample_case_rows.csv", final_case_rows)

    round_rows: list[Dict[str, Any]] = []
    round_case_rows: list[Dict[str, Any]] = []
    probe_attempt_rows: list[Dict[str, Any]] = []
    for round_report in report["rounds"]:
        round_rows.extend(round_report.get("summary_rows", []))
        round_case_rows.extend(round_report.get("baseline_case_rows", []))
        for model_result in round_report.get("model_results", []):
            metrics = model_result.get("metrics") if isinstance(model_result.get("metrics"), dict) else {}
            round_case_rows.extend(metrics.get("rows", []))
            probe_attempt_rows.extend(model_result.get("attempts", []))

    _write_csv(run_root / "probe_round_summary.csv", round_rows)
    _write_csv(run_root / "probe_round_case_rows.csv", round_case_rows)
    _write_csv(run_root / "probe_attempts.csv", probe_attempt_rows)
    _write_json(run_root / "report.json", report)

    print(f"[NURSE] Staged model evaluation root: {run_root}")
    print(f"[NURSE] Probe decision: {round_decision}")
    if round_decision == "D":
        print("[NURSE] Final 40-case sample is available." if not args.skip_final_sample else "[NURSE] Final sample was skipped.")
    elif round_decision == "C":
        print("[NURSE] All tested models are non-improving/degrading on the probe; optimization is required before broader LLM evaluation.")
    elif round_decision == "extend":
        print("[NURSE] Probe remained mixed even after the configured rounds.")
    else:
        print("[NURSE] Probe was blocked by model availability or run failures.")


if __name__ == "__main__":
    main()
