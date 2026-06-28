"""Reusable experiment statistics for Nurse and future baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .matching import first_match, hit_at


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _hit_at(ranked: List[str], labels: List[str], k: int) -> bool:
    return hit_at(ranked, labels, k)


def _accuracy(predictions: Dict[str, List[str]], labels: Dict[str, List[str]]) -> Dict[str, float]:
    case_ids = sorted(labels)
    total = max(len(case_ids), 1)
    return {
        "acc_at_1": sum(_hit_at(predictions.get(case_id, []), labels[case_id], 1) for case_id in case_ids) / total,
        "acc_at_3": sum(_hit_at(predictions.get(case_id, []), labels[case_id], 3) for case_id in case_ids) / total,
        "acc_at_5": sum(_hit_at(predictions.get(case_id, []), labels[case_id], 5) for case_id in case_ids) / total,
        "labeled_cases": float(len(case_ids)),
    }


def _float_value(value, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _record_timing(record: Dict) -> Tuple[Optional[float], float, float]:
    timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
    total = _float_value(timing.get("total_seconds", record.get("total_seconds")), default=None)
    llm = _float_value(timing.get("llm_seconds", record.get("llm_seconds")), default=0.0) or 0.0
    non_llm = _float_value(timing.get("non_llm_seconds", record.get("non_llm_seconds")), default=None)
    if non_llm is None and total is not None:
        non_llm = max(total - llm, 0.0)
    return total, llm, non_llm or 0.0


def _record_llm(record: Dict) -> Dict:
    llm = record.get("llm") if isinstance(record.get("llm"), dict) else {}
    usage = llm.get("usage") if isinstance(llm.get("usage"), dict) else {}
    total_tokens = (
        llm.get("total_tokens")
        or usage.get("total_tokens")
        or record.get("total_tokens")
        or record.get("token_usage")
        or 0
    )
    return {
        "enabled": bool(llm.get("enabled", record.get("llm_status") not in {"disabled", None, ""})),
        "status": str(llm.get("status", record.get("llm_status", "")) or ""),
        "model": str(llm.get("model", record.get("llm_model", "")) or ""),
        "calls": _int_value(llm.get("calls", record.get("llm_calls", 0))),
        "input_tokens": _int_value(llm.get("input_tokens", usage.get("input_tokens", 0))),
        "output_tokens": _int_value(llm.get("output_tokens", usage.get("output_tokens", 0))),
        "total_tokens": _int_value(total_tokens),
    }


def _records_from_run(run_dir: Path, summary: Dict) -> List[Dict]:
    records = summary.get("records")
    if isinstance(records, list) and records:
        return [record for record in records if isinstance(record, dict)]
    return _load_jsonl(run_dir / "records.jsonl")


def build_experiment_statistics(
    *,
    run_dir: str | Path,
    method: str,
    dataset: Optional[str] = None,
    llm: Optional[str] = None,
) -> Dict:
    run_path = Path(run_dir)
    summary = _load_json(run_path / "summary.json")
    records = _records_from_run(run_path, summary)
    predictions = _load_json(run_path / "predictions.json")
    labels = _load_json(run_path / "labels.json")
    accuracy = _accuracy(predictions, labels) if labels else summary.get("metrics", {})

    case_rows: List[Dict] = []
    total_sum = 0.0
    llm_sum = 0.0
    non_llm_sum = 0.0
    timing_cases = 0
    token_total = 0

    for record in records:
        case_id = str(record.get("case_id", ""))
        ranking = predictions.get(case_id) or record.get("ranking") or []
        acceptable_labels = [str(item) for item in labels.get(case_id, [])]
        matched_label, matched_rank = first_match(ranking, acceptable_labels, None) if acceptable_labels else ("", 0)
        total_seconds, llm_seconds, non_llm_seconds = _record_timing(record)
        llm_info = _record_llm(record)
        token_total += llm_info["total_tokens"]
        if total_seconds is not None:
            timing_cases += 1
            total_sum += total_seconds
            llm_sum += llm_seconds
            non_llm_sum += non_llm_seconds
        case_rows.append(
            {
                "case_id": case_id,
                "status": record.get("status", ""),
                "top1": record.get("top1", ""),
                "acceptable_labels": ";".join(acceptable_labels),
                "matched_label": matched_label,
                "matched_rank": matched_rank,
                "total_seconds": total_seconds if total_seconds is not None else "",
                "non_llm_seconds": non_llm_seconds if total_seconds is not None else "",
                "llm_seconds": llm_seconds if total_seconds is not None else "",
                "llm_status": llm_info["status"],
                "llm_model": llm_info["model"],
                "llm_calls": llm_info["calls"],
                "input_tokens": llm_info["input_tokens"],
                "output_tokens": llm_info["output_tokens"],
                "total_tokens": llm_info["total_tokens"],
            }
        )

    if token_total == 0:
        token_total = _int_value(summary.get("token_usage", 0))

    processed_cases = len(records)
    if timing_cases == 0 and processed_cases:
        elapsed = _float_value(summary.get("elapsed_seconds"), default=0.0) or 0.0
        timing_cases = processed_cases
        total_sum = elapsed
        non_llm_sum = elapsed
        llm_sum = 0.0

    denominator = max(timing_cases, 1)
    method_name = method
    dataset_name = dataset or str(summary.get("dataset", ""))
    llm_name = llm or str(summary.get("llm", ""))
    ranked_cases = sum(1 for record in records if record.get("status") == "ranked")
    missed_alarm_cases = sum(1 for record in records if record.get("status") == "missed_alarm")
    paper_metrics = {
        "acc_at_1": float(accuracy.get("acc_at_1", accuracy.get("top1", 0.0)) or 0.0),
        "acc_at_3": float(accuracy.get("acc_at_3", accuracy.get("top3", 0.0)) or 0.0),
        "acc_at_5": float(accuracy.get("acc_at_5", accuracy.get("top5", 0.0)) or 0.0),
        "latency_seconds_per_case": round(total_sum / denominator, 6),
        "non_llm_latency_seconds_per_case": round(non_llm_sum / denominator, 6),
        "llm_latency_seconds_per_case": round(llm_sum / denominator, 6),
        "processed_cases": processed_cases,
        "timing_cases": timing_cases,
        "labeled_cases": int(float(accuracy.get("labeled_cases", len(labels)) or 0)),
        "ranked_cases": ranked_cases,
        "missed_alarm_cases": missed_alarm_cases,
        "token_usage_total": token_total,
        "token_usage_per_case": round(token_total / denominator, 6),
    }
    return {
        "method": method_name,
        "dataset": dataset_name,
        "llm": llm_name,
        "run_dir": str(run_path),
        "paper_metrics": paper_metrics,
        "case_metrics": case_rows,
    }


def _write_case_metrics(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "case_id",
        "status",
        "top1",
        "acceptable_labels",
        "matched_label",
        "matched_rank",
        "total_seconds",
        "non_llm_seconds",
        "llm_seconds",
        "llm_status",
        "llm_model",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_paper_row(path: Path, report: Dict) -> None:
    paper = report["paper_metrics"]
    fieldnames = [
        "method",
        "dataset",
        "llm",
        "acc_at_1",
        "acc_at_3",
        "acc_at_5",
        "latency_seconds_per_case",
        "non_llm_seconds_per_case",
        "llm_seconds_per_case",
        "cases",
        "ranked_cases",
        "total_tokens",
    ]
    row = {
        "method": report["method"],
        "dataset": report["dataset"],
        "llm": report["llm"],
        "acc_at_1": f"{paper['acc_at_1']:.6f}",
        "acc_at_3": f"{paper['acc_at_3']:.6f}",
        "acc_at_5": f"{paper['acc_at_5']:.6f}",
        "latency_seconds_per_case": f"{paper['latency_seconds_per_case']:.3f}",
        "non_llm_seconds_per_case": f"{paper['non_llm_latency_seconds_per_case']:.3f}",
        "llm_seconds_per_case": f"{paper['llm_latency_seconds_per_case']:.3f}",
        "cases": str(paper["labeled_cases"] or paper["processed_cases"]),
        "ranked_cases": str(paper["ranked_cases"]),
        "total_tokens": str(paper["token_usage_total"]),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_experiment_statistics(
    *,
    run_dir: str | Path,
    method: str,
    dataset: Optional[str] = None,
    llm: Optional[str] = None,
) -> Dict:
    run_path = Path(run_dir)
    report = build_experiment_statistics(run_dir=run_path, method=method, dataset=dataset, llm=llm)
    (run_path / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_case_metrics(run_path / "case_metrics.csv", report["case_metrics"])
    _write_paper_row(run_path / "paper_table_row.csv", report)

    summary_path = run_path / "summary.json"
    summary = _load_json(summary_path)
    summary["paper_metrics"] = report["paper_metrics"]
    summary["metrics_artifacts"] = {
        "metrics_json": str(run_path / "metrics.json"),
        "case_metrics_csv": str(run_path / "case_metrics.csv"),
        "paper_table_row_csv": str(run_path / "paper_table_row.csv"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Write reusable experiment statistics for an RCL run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", default="Nurse")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--llm", default=None)
    args = parser.parse_args(argv)
    report = write_experiment_statistics(
        run_dir=args.run_dir,
        method=args.method,
        dataset=args.dataset,
        llm=args.llm,
    )
    print(json.dumps(report["paper_metrics"], ensure_ascii=False, indent=2))
    print(f"[Nurse] Experiment metrics: {Path(args.run_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()


