"""Utilities for resuming, merging, and curating split ablation runs."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd

from .experiment_stats import write_experiment_statistics


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_case_ids(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if text and not text.startswith("#"):
            values.append(text)
    return values


def _dedupe_latest_records(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    latest_by_case: Dict[str, Dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "")
        if not case_id:
            continue
        if case_id in latest_by_case:
            latest_by_case.pop(case_id)
        latest_by_case[case_id] = dict(record)
    return list(latest_by_case.values())


def _record_total_tokens(record: Mapping[str, Any]) -> int:
    llm = record.get("llm") if isinstance(record.get("llm"), dict) else {}
    if "total_tokens" in llm:
        try:
            return int(llm.get("total_tokens") or 0)
        except (TypeError, ValueError):
            return 0
    try:
        input_tokens = int(llm.get("input_tokens") or 0)
    except (TypeError, ValueError):
        input_tokens = 0
    try:
        output_tokens = int(llm.get("output_tokens") or 0)
    except (TypeError, ValueError):
        output_tokens = 0
    return input_tokens + output_tokens


def _record_total_seconds(record: Mapping[str, Any]) -> float:
    timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
    value = timing.get("total_seconds", record.get("total_seconds", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _hit_at(ranked: Sequence[str], labels: Sequence[str], k: int) -> bool:
    accepted = {str(label) for label in labels}
    return any(str(candidate) in accepted for candidate in list(ranked or [])[:k])


def _summary_metrics(predictions: Mapping[str, Sequence[str]], labels: Mapping[str, Sequence[str]], records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    case_ids = sorted(str(case_id) for case_id in labels)
    total = max(len(case_ids), 1)
    return {
        "acc_at_1": sum(_hit_at(predictions.get(case_id, []), labels.get(case_id, []), 1) for case_id in case_ids) / total,
        "acc_at_3": sum(_hit_at(predictions.get(case_id, []), labels.get(case_id, []), 3) for case_id in case_ids) / total,
        "acc_at_5": sum(_hit_at(predictions.get(case_id, []), labels.get(case_id, []), 5) for case_id in case_ids) / total,
        "labeled_cases": float(len(case_ids)),
        "ranked_cases": float(sum(1 for record in records if record.get("status") == "ranked")),
    }


def _copy_case_artifacts(*, source_run_dir: Path, target_run_dir: Path, case_id: str) -> None:
    for folder_name in ("predictions", "graphs", "alarms", "llm_artifacts", "contexts"):
        source_path = source_run_dir / folder_name / f"{case_id}.json"
        if not source_path.exists():
            continue
        target_path = target_run_dir / folder_name / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def pending_case_ids(
    *,
    run_dir: str | Path,
    case_ids: Sequence[str],
    success_statuses: Sequence[str] = ("ok",),
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    records = _dedupe_latest_records(_load_jsonl(run_path / "records.jsonl"))
    latest_by_case = {str(record.get("case_id") or ""): record for record in records}
    normalized_statuses = {str(status) for status in success_statuses}
    completed: list[str] = []
    pending: list[str] = []
    for case_id in [str(value) for value in case_ids]:
        record = latest_by_case.get(case_id)
        if record and str(record.get("status") or "") == "ranked" and str(record.get("llm_status") or "") in normalized_statuses:
            completed.append(case_id)
        else:
            pending.append(case_id)
    return {
        "run_dir": str(run_path),
        "requested_cases": [str(value) for value in case_ids],
        "completed_cases": completed,
        "pending_cases": pending,
        "success_statuses": sorted(normalized_statuses),
    }


def write_pending_case_ids(
    *,
    run_dir: str | Path,
    case_ids_file: str | Path,
    output_file: str | Path,
    success_statuses: Sequence[str] = ("ok",),
) -> Dict[str, Any]:
    report = pending_case_ids(
        run_dir=run_dir,
        case_ids=_load_case_ids(case_ids_file),
        success_statuses=success_statuses,
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report["pending_cases"]) + ("\n" if report["pending_cases"] else ""), encoding="utf-8")
    report["output_file"] = str(output_path)
    return report


def merge_run_dirs(
    *,
    run_dirs: Sequence[str | Path],
    output_run_dir: str | Path,
    llm_label: str = "merged",
) -> Dict[str, Any]:
    source_dirs = [Path(run_dir) for run_dir in run_dirs]
    if not source_dirs:
        raise ValueError("merge_run_dirs requires at least one source run directory")
    base_summary = _load_json(source_dirs[0] / "summary.json")
    output_dir = Path(output_run_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ("predictions", "graphs", "alarms", "llm_artifacts", "contexts"):
        (output_dir / folder_name).mkdir(parents=True, exist_ok=True)

    records_merged: list[Dict[str, Any]] = []
    predictions_merged: Dict[str, Any] = {}
    labels_merged: Dict[str, Any] = {}
    chosen_source_by_case: Dict[str, Path] = {}
    source_summaries: list[Dict[str, Any]] = []

    for source_dir in source_dirs:
        summary = _load_json(source_dir / "summary.json")
        source_summaries.append(summary)
        predictions = _load_json(source_dir / "predictions.json")
        labels = _load_json(source_dir / "labels.json")
        records = _dedupe_latest_records(_load_jsonl(source_dir / "records.jsonl"))
        for record in records:
            case_id = str(record.get("case_id") or "")
            if not case_id:
                continue
            if case_id in predictions:
                predictions_merged[case_id] = predictions[case_id]
            if case_id in labels:
                labels_merged[case_id] = labels[case_id]
            chosen_source_by_case[case_id] = source_dir
        records_merged.extend(records)

    merged_records = _dedupe_latest_records(records_merged)
    for case_id, source_dir in chosen_source_by_case.items():
        _copy_case_artifacts(source_run_dir=source_dir, target_run_dir=output_dir, case_id=case_id)

    metrics = _summary_metrics(predictions_merged, labels_merged, merged_records)
    llm_status_counts = Counter(str(record.get("llm_status") or "") for record in merged_records if record.get("llm_status"))
    elapsed_seconds = round(sum(_record_total_seconds(record) for record in merged_records), 3)
    token_usage = sum(_record_total_tokens(record) for record in merged_records)
    merged_summary = {
        "dataset": str(base_summary.get("dataset") or ""),
        "case_limit": int(base_summary.get("case_limit", 0) or 0),
        "llm": llm_label,
        "records": merged_records,
        "elapsed_seconds": elapsed_seconds,
        "token_usage": int(token_usage),
        "candidate_expansion_rounds": int(base_summary.get("candidate_expansion_rounds", 0) or 0),
        "window_width_seconds": int(base_summary.get("window_width_seconds", 0) or 0),
        "candidate_scope": dict(base_summary.get("candidate_scope") or {}),
        "metrics": metrics,
        "nurse_top1": metrics["acc_at_1"],
        "llm_status_counts": dict(sorted(llm_status_counts.items())),
        "merge_metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_run_dirs": [str(path) for path in source_dirs],
            "chosen_source_by_case": {case_id: str(path) for case_id, path in sorted(chosen_source_by_case.items())},
        },
    }
    config = _load_json(source_dirs[0] / "config.json")
    if config:
        config = dict(config)
        config["llm"] = llm_label
        config["merge_metadata"] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_run_dirs": [str(path) for path in source_dirs],
        }
        _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "predictions.json", predictions_merged)
    _write_json(output_dir / "labels.json", labels_merged)
    _write_json(output_dir / "summary.json", merged_summary)
    _write_jsonl(output_dir / "records.jsonl", merged_records)
    _write_json(output_dir / "merge_sources.json", merged_summary["merge_metadata"])
    report = write_experiment_statistics(
        run_dir=output_dir,
        method=str(base_summary.get("method") or "Nurse"),
        dataset=str(base_summary.get("dataset") or ""),
        llm=llm_label,
    )
    return {
        "output_run_dir": str(output_dir),
        "merged_case_count": len(merged_records),
        "source_run_dirs": [str(path) for path in source_dirs],
        "paper_metrics": report["paper_metrics"],
    }


def curate_selected_runs(
    *,
    selection: Mapping[str, Mapping[str, str | Path]],
    output_path: str | Path,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for dataset, mode_map in sorted(selection.items(), key=lambda item: str(item[0])):
        for mode, run_dir in sorted(mode_map.items(), key=lambda item: str(item[0])):
            run_path = Path(run_dir)
            summary = _load_json(run_path / "summary.json")
            paper_metrics = summary.get("paper_metrics") if isinstance(summary.get("paper_metrics"), dict) else {}
            metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
            llm_status_counts = summary.get("llm_status_counts") if isinstance(summary.get("llm_status_counts"), dict) else {}
            acc_at_1 = float(paper_metrics.get("acc_at_1", metrics.get("acc_at_1", 0.0)) or 0.0)
            acc_at_3 = float(paper_metrics.get("acc_at_3", metrics.get("acc_at_3", 0.0)) or 0.0)
            acc_at_5 = float(paper_metrics.get("acc_at_5", metrics.get("acc_at_5", 0.0)) or 0.0)
            processed_cases = int(
                paper_metrics.get(
                    "processed_cases",
                    metrics.get("processed_cases", metrics.get("labeled_cases", len(_load_json(run_path / "predictions.json")))),
                )
                or 0
            )
            ranked_cases = int(paper_metrics.get("ranked_cases", metrics.get("ranked_cases", 0)) or 0)
            labeled_cases = int(paper_metrics.get("labeled_cases", metrics.get("labeled_cases", 0)) or 0)
            token_usage_total = int(paper_metrics.get("token_usage_total", summary.get("token_usage", 0)) or 0)
            bad_statuses = sorted(
                key for key, value in llm_status_counts.items() if value and key not in {"ok", "disabled"}
            )
            rows.append(
                {
                    "dataset": str(dataset),
                    "mode": str(mode),
                    "llm": str(summary.get("llm") or ""),
                    "run_dir": str(run_path),
                    "candidate_expansion_rounds": int(summary.get("candidate_expansion_rounds", 0) or 0),
                    "window_width_seconds": int(summary.get("window_width_seconds", 0) or 0),
                    "acc_at_1": acc_at_1,
                    "acc_at_3": acc_at_3,
                    "acc_at_5": acc_at_5,
                    "processed_cases": processed_cases,
                    "ranked_cases": ranked_cases,
                    "labeled_cases": labeled_cases,
                    "token_usage_total": token_usage_total,
                    "llm_status_counts_json": json.dumps(llm_status_counts, ensure_ascii=False, sort_keys=True),
                    "success": not bad_statuses,
                    "unexpected_llm_statuses": ",".join(bad_statuses),
                }
            )
    rows.sort(key=lambda row: (row["dataset"], row["mode"]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output if output.suffix.lower() == ".csv" else output.with_suffix(".csv")
    xlsx_path = output if output.suffix.lower() == ".xlsx" else output.with_suffix(".xlsx")
    json_path = output.with_suffix(".json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="ablation")
    _write_json(
        json_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selection": {
                str(dataset): {str(mode): str(run_dir) for mode, run_dir in mode_map.items()}
                for dataset, mode_map in selection.items()
            },
            "rows": rows,
        },
    )
    return {
        "rows": rows,
        "csv_path": str(csv_path),
        "xlsx_path": str(xlsx_path),
        "json_path": str(json_path),
    }


def _load_selection_json(path: str | Path) -> Dict[str, Dict[str, str]]:
    payload = _load_json(Path(path))
    return {
        str(dataset): {str(mode): str(run_dir) for mode, run_dir in dict(mode_map).items()}
        for dataset, mode_map in payload.items()
        if isinstance(mode_map, Mapping)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate split/partial NURSE ablation runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pending = subparsers.add_parser("pending", help="Write pending case ids for a partially completed run")
    pending.add_argument("--run-dir", required=True)
    pending.add_argument("--case-ids-file", required=True)
    pending.add_argument("--output", required=True)
    pending.add_argument("--success-status", action="append", default=["ok"])

    merge = subparsers.add_parser("merge", help="Merge split runs into a single curated run directory")
    merge.add_argument("--run-dir", action="append", required=True, help="Provide in precedence order; later runs override earlier runs.")
    merge.add_argument("--output-run-dir", required=True)
    merge.add_argument("--llm-label", default="merged")

    curate = subparsers.add_parser("curate", help="Curate selected final run directories into a single report")
    curate.add_argument("--selection-json", required=True)
    curate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "pending":
        report = write_pending_case_ids(
            run_dir=args.run_dir,
            case_ids_file=args.case_ids_file,
            output_file=args.output,
            success_statuses=args.success_status,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return
    if args.command == "merge":
        report = merge_run_dirs(
            run_dirs=args.run_dir,
            output_run_dir=args.output_run_dir,
            llm_label=args.llm_label,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return
    if args.command == "curate":
        report = curate_selected_runs(
            selection=_load_selection_json(args.selection_json),
            output_path=args.output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
