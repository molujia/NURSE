from __future__ import annotations

import json
from pathlib import Path

import pytest

from nurse import sota_reporting
from nurse.sota_reporting import aggregate_run_root, compare_top5_consistency


def test_compare_top5_consistency_flags_case_mismatch():
    report = compare_top5_consistency(
        case_ids_by_dataset={"ds1": ["case-a", "case-b"]},
        snapshots_by_dataset_model={
            "ds1": {
                "model-a": {
                    "summary": {"metrics": {"acc_at_1": 1.0}},
                    "predictions": {
                        "case-a": ["svc-a", "svc-b", "svc-c"],
                        "case-b": ["svc-x", "svc-y", "svc-z"],
                    },
                },
                "model-b": {
                    "summary": {"metrics": {"acc_at_1": 0.5}},
                    "predictions": {
                        "case-a": ["svc-a", "svc-b", "svc-c"],
                        "case-b": ["svc-q", "svc-r", "svc-s"],
                    },
                },
            }
        },
    )

    dataset_report = report["datasets"]["ds1"]
    assert dataset_report["all_models_equal"] is False
    assert dataset_report["mismatch_case_ids"] == ["case-b"]
    assert report["diff_rows"][0]["case_id"] == "case-b"
    assert report["sample_metrics"][0]["top5_mismatch_cases"] == 1


def test_aggregate_run_root_writes_csv_xlsx_and_hashes(tmp_path):
    run_root = tmp_path / "runs"
    run_a = run_root / "ds1" / "model-a"
    run_b = run_root / "ds1" / "model-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    payloads = {
        run_a: {
            "summary": {
                "dataset": "ds1",
                "llm": "model-a",
                "method": "Nurse",
                "paper_metrics": {
                    "acc_at_1": 1.0,
                    "acc_at_3": 1.0,
                    "acc_at_5": 1.0,
                    "labeled_cases": 2,
                    "ranked_cases": 2,
                    "processed_cases": 2,
                    "token_usage_total": 11,
                },
            },
            "predictions": {
                "case-a": ["svc-a", "svc-b", "svc-c"],
                "case-b": ["svc-x", "svc-y", "svc-z"],
            },
        },
        run_b: {
            "summary": {
                "dataset": "ds1",
                "llm": "model-b",
                "method": "Nurse",
                "paper_metrics": {
                    "acc_at_1": 0.5,
                    "acc_at_3": 0.5,
                    "acc_at_5": 0.5,
                    "labeled_cases": 2,
                    "ranked_cases": 2,
                    "processed_cases": 2,
                    "token_usage_total": 17,
                },
            },
            "predictions": {
                "case-a": ["svc-a", "svc-b", "svc-c"],
                "case-b": ["svc-q", "svc-r", "svc-s"],
            },
        },
    }

    for run_dir, payload in payloads.items():
        (run_dir / "summary.json").write_text(json.dumps(payload["summary"]), encoding="utf-8")
        (run_dir / "predictions.json").write_text(json.dumps(payload["predictions"]), encoding="utf-8")

    report = aggregate_run_root(run_root, output_path=tmp_path / "main_results.xlsx")

    assert (tmp_path / "main_results.csv").exists()
    assert (tmp_path / "main_results.xlsx").exists()
    assert (run_root / "top5_hashes.json").exists()
    assert len(report["rows"]) == 2
    assert {row["llm"] for row in report["rows"]} == {"model-a", "model-b"}


def test_run_consistency_sample_writes_outputs_and_exits_on_mismatch(monkeypatch, tmp_path):
    def fake_manifest(*, manifest_dir, dataset, lab_root, seed, sample_size):
        manifest_path = manifest_dir / f"{dataset}.json"
        payload = {
            "datasets": {
                dataset: ["case-a", "case-b"],
            }
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "dataset": dataset,
            "manifest_path": manifest_path,
            "case_ids": ["case-a", "case-b"],
        }

    def fake_run_job(
        *,
        python_exe,
        dataset,
        model,
        mode,
        run_dir,
        case_manifest,
        case_limit,
        seed,
        alarm_cache_root,
        candidate_expansion_rounds=2,
        window_width_seconds=480,
        max_retries=0,
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": dataset,
            "llm": model,
            "method": "Nurse",
            "paper_metrics": {
                "acc_at_1": 1.0 if model == "model-a" else 0.5,
                "acc_at_3": 1.0 if model == "model-a" else 0.5,
                "acc_at_5": 1.0 if model == "model-a" else 0.5,
                "labeled_cases": 2,
                "ranked_cases": 2,
                "processed_cases": 2,
                "token_usage_total": 0,
            },
        }
        predictions = {
            "case-a": ["svc-a", "svc-b", "svc-c"],
            "case-b": ["svc-x", "svc-y", "svc-z"] if model == "model-a" else ["svc-q", "svc-r", "svc-s"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
        return {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "run_dir": str(run_dir),
            "returncode": 0,
        }

    monkeypatch.setattr(sota_reporting, "_manifest_for_dataset", fake_manifest)
    monkeypatch.setattr(sota_reporting, "_run_nurse_job", fake_run_job)

    with pytest.raises(SystemExit) as excinfo:
        sota_reporting.run_consistency_sample(
            output_root=tmp_path / "sample",
            datasets=["ds1"],
            models=["model-a", "model-b"],
            seed=20260628,
            sample_size=2,
            workers=1,
            lab_root=tmp_path,
        )

    assert excinfo.value.code == 3
    sample_root = next((tmp_path / "sample").iterdir())
    assert (sample_root / "top5_by_model.json").exists()
    assert (sample_root / "top5_diff_report.csv").exists()
    assert (sample_root / "sample_metrics.csv").exists()


def test_run_matrix_experiment_writes_matrix_reports(monkeypatch, tmp_path):
    def fake_manifest(*, manifest_dir, dataset, lab_root, seed, sample_size):
        manifest_path = manifest_dir / f"{dataset}.json"
        payload = {"datasets": {dataset: ["case-a", "case-b"]}}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "dataset": dataset,
            "manifest_path": manifest_path,
            "case_ids": ["case-a", "case-b"],
        }

    def fake_run_job(
        *,
        python_exe,
        dataset,
        model,
        mode,
        run_dir,
        case_manifest,
        case_limit,
        seed,
        alarm_cache_root,
        candidate_expansion_rounds=2,
        window_width_seconds=480,
        max_retries=0,
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": dataset,
            "llm": model,
            "method": "Nurse",
            "paper_metrics": {
                "acc_at_1": 1.0,
                "acc_at_3": 1.0,
                "acc_at_5": 1.0,
                "labeled_cases": 2,
                "ranked_cases": 2,
                "processed_cases": 2,
                "token_usage_total": 0,
            },
        }
        predictions = {
            "case-a": ["svc-a", "svc-b", "svc-c"],
            "case-b": ["svc-x", "svc-y", "svc-z"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
        return {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "run_dir": str(run_dir),
            "returncode": 0,
        }

    monkeypatch.setattr(sota_reporting, "_manifest_for_dataset", fake_manifest)
    monkeypatch.setattr(sota_reporting, "_run_nurse_job", fake_run_job)

    report = sota_reporting.run_matrix_experiment(
        output_root=tmp_path / "matrix",
        datasets=["ds1"],
        models=["model-a"],
        modes=["full", "wo_L"],
        seed=20260628,
        case_limit=0,
        workers=1,
        lab_root=tmp_path,
    )

    matrix_root = Path(report["run_root"])
    assert (matrix_root / "matrix_runs.json").exists()
    assert (matrix_root / "matrix_runs.csv").exists()
    assert len(report["rows"]) == 2
    assert [job["mode"] for job in report["run_jobs"]] == ["full", "wo_L"]


def test_run_matrix_experiment_supports_custom_report_prefix(monkeypatch, tmp_path):
    def fake_manifest(*, manifest_dir, dataset, lab_root, seed, sample_size):
        manifest_path = manifest_dir / f"{dataset}.json"
        payload = {"datasets": {dataset: ["case-a"]}}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "dataset": dataset,
            "manifest_path": manifest_path,
            "case_ids": ["case-a"],
        }

    def fake_run_job(
        *,
        python_exe,
        dataset,
        model,
        mode,
        run_dir,
        case_manifest,
        case_limit,
        seed,
        alarm_cache_root,
        candidate_expansion_rounds=2,
        window_width_seconds=480,
        max_retries=0,
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": dataset,
            "llm": model,
            "method": "Nurse",
            "paper_metrics": {
                "acc_at_1": 1.0,
                "acc_at_3": 1.0,
                "acc_at_5": 1.0,
                "labeled_cases": 1,
                "ranked_cases": 1,
                "processed_cases": 1,
                "token_usage_total": 0,
            },
        }
        predictions = {"case-a": ["svc-a", "svc-b", "svc-c"]}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
        return {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "run_dir": str(run_dir),
            "returncode": 0,
        }

    monkeypatch.setattr(sota_reporting, "_manifest_for_dataset", fake_manifest)
    monkeypatch.setattr(sota_reporting, "_run_nurse_job", fake_run_job)

    report = sota_reporting.run_matrix_experiment(
        output_root=tmp_path / "matrix",
        datasets=["ds1"],
        models=["model-a"],
        modes=["full", "wo_E"],
        seed=20260628,
        case_limit=0,
        workers=1,
        lab_root=tmp_path,
        report_name="ablation_results",
    )

    matrix_root = Path(report["run_root"])
    assert (matrix_root / "ablation_results.json").exists()
    assert (matrix_root / "ablation_results.csv").exists()
    assert (matrix_root / "ablation_results.xlsx").exists()
