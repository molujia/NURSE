from __future__ import annotations

import json
from pathlib import Path

from nurse.ablation_curation import curate_selected_runs, merge_run_dirs, pending_case_ids


def _write_run(
    run_dir: Path,
    *,
    dataset: str,
    llm: str,
    records: list[dict],
    predictions: dict,
    labels: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ("predictions", "graphs", "alarms", "llm_artifacts", "contexts"):
        (run_dir / folder_name).mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset,
        "case_limit": 0,
        "llm": llm,
        "records": records,
        "elapsed_seconds": 1.0,
        "token_usage": sum(int((record.get("llm") or {}).get("total_tokens", 0) or 0) for record in records),
        "candidate_expansion_rounds": 2,
        "window_width_seconds": 480,
        "candidate_scope": {
            "candidate_expansion_rounds": 2,
            "window_width_seconds": 480,
        },
        "metrics": {
            "acc_at_1": 0.0,
            "acc_at_3": 0.0,
            "acc_at_5": 0.0,
            "labeled_cases": float(len(labels)),
            "ranked_cases": float(sum(1 for record in records if record.get("status") == "ranked")),
        },
        "llm_status_counts": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    (run_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (run_dir / "config.json").write_text(json.dumps({"dataset": dataset, "llm": llm}), encoding="utf-8")
    for case_id in predictions:
        for folder_name in ("predictions", "graphs", "alarms", "llm_artifacts", "contexts"):
            (run_dir / folder_name / f"{case_id}.json").write_text(json.dumps({"case_id": case_id, "source": llm}), encoding="utf-8")


def test_pending_case_ids_flags_errors_and_missing_cases(tmp_path):
    run_dir = tmp_path / "run"
    records = [
        {
            "case_id": "case-a",
            "status": "ranked",
            "llm_status": "ok",
            "llm": {"total_tokens": 11},
        },
        {
            "case_id": "case-b",
            "status": "ranked",
            "llm_status": "error",
            "llm": {"total_tokens": 0},
        },
    ]
    _write_run(
        run_dir,
        dataset="ds",
        llm="model-a",
        records=records,
        predictions={"case-a": ["svc-a"], "case-b": ["svc-b"]},
        labels={"case-a": ["svc-a"], "case-b": ["svc-b"]},
    )

    report = pending_case_ids(
        run_dir=run_dir,
        case_ids=["case-a", "case-b", "case-c"],
        success_statuses=["ok"],
    )

    assert report["completed_cases"] == ["case-a"]
    assert report["pending_cases"] == ["case-b", "case-c"]


def test_merge_run_dirs_dedupes_cases_and_recomputes_metrics(tmp_path):
    base_run = tmp_path / "base"
    donor_run = tmp_path / "donor"
    output_run = tmp_path / "merged"
    shared_case = {
        "case_id": "case-a",
        "status": "ranked",
        "llm_status": "ok",
        "llm": {"total_tokens": 11},
        "timing": {"total_seconds": 3.0},
        "top1": "svc-a",
    }
    donor_case = {
        "case_id": "case-b",
        "status": "ranked",
        "llm_status": "ok",
        "llm": {"total_tokens": 13},
        "timing": {"total_seconds": 5.0},
        "top1": "svc-b",
    }
    _write_run(
        base_run,
        dataset="ds",
        llm="model-a",
        records=[shared_case],
        predictions={"case-a": ["svc-a"]},
        labels={"case-a": ["svc-a"], "case-b": ["svc-b"]},
    )
    _write_run(
        donor_run,
        dataset="ds",
        llm="model-b",
        records=[shared_case, donor_case],
        predictions={"case-a": ["svc-a"], "case-b": ["svc-b"]},
        labels={"case-a": ["svc-a"], "case-b": ["svc-b"]},
    )

    report = merge_run_dirs(
        run_dirs=[base_run, donor_run],
        output_run_dir=output_run,
        llm_label="merged-model",
    )

    merged_summary = json.loads((output_run / "summary.json").read_text(encoding="utf-8"))
    assert report["merged_case_count"] == 2
    assert merged_summary["llm"] == "merged-model"
    assert merged_summary["token_usage"] == 24
    assert len(merged_summary["records"]) == 2
    assert json.loads((output_run / "predictions.json").read_text(encoding="utf-8"))["case-b"] == ["svc-b"]
    assert (output_run / "metrics.json").exists()


def test_curate_selected_runs_marks_incomplete_statuses(tmp_path):
    ok_run = tmp_path / "ok-run"
    err_run = tmp_path / "err-run"
    _write_run(
        ok_run,
        dataset="ds1",
        llm="model-ok",
        records=[
            {
                "case_id": "case-a",
                "status": "ranked",
                "llm_status": "ok",
                "llm": {"total_tokens": 9},
                "timing": {"total_seconds": 2.0},
                "top1": "svc-a",
            }
        ],
        predictions={"case-a": ["svc-a"]},
        labels={"case-a": ["svc-a"]},
    )
    _write_run(
        err_run,
        dataset="ds2",
        llm="model-err",
        records=[
            {
                "case_id": "case-b",
                "status": "ranked",
                "llm_status": "error",
                "llm": {"total_tokens": 0},
                "timing": {"total_seconds": 2.0},
                "top1": "svc-b",
            }
        ],
        predictions={"case-b": ["svc-b"]},
        labels={"case-b": ["svc-b"]},
    )
    for run_dir, status_counts in (
        (ok_run, {"ok": 1}),
        (err_run, {"error": 1}),
    ):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["paper_metrics"] = {
            "acc_at_1": 1.0,
            "acc_at_3": 1.0,
            "acc_at_5": 1.0,
            "processed_cases": 1,
            "ranked_cases": 1,
            "labeled_cases": 1,
            "token_usage_total": 0,
        }
        summary["llm_status_counts"] = status_counts
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = curate_selected_runs(
        selection={
            "ds1": {"full": ok_run},
            "ds2": {"wo_H": err_run},
        },
        output_path=tmp_path / "final_ablation.xlsx",
    )

    assert Path(report["csv_path"]).exists()
    assert Path(report["xlsx_path"]).exists()
    assert [row["success"] for row in report["rows"]] == [True, False]

