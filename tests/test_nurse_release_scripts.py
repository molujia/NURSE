from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_text(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_openai_release_quickstart_scripts_are_present():
    full = _script_text("run_nurse_openai_full.ps1")
    consistency = _script_text("run_nurse_openai_consistency_5cases.ps1")

    assert "OPENAI_API_KEY" in full
    assert '"matrix"' in full
    assert '"openai_full_results"' in full
    assert "OPENAI_API_KEY" in consistency
    assert '"sample"' in consistency
    assert '"gpt-5-nano"' in consistency


def test_ablation_release_script_matches_eph_design():
    text = _script_text("run_nurse_sota_ablation_parallel.ps1")

    assert '"wo_E"' in text
    assert '"wo_P"' in text
    assert '"wo_H"' in text
    assert '"wo_L"' not in text
    assert '"gpt-5-nano"' in text


def test_aggregate_script_can_run_by_file_path(tmp_path):
    run_dir = tmp_path / "run_root" / "runs" / "dataset" / "full" / "model"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": "rcaeval-tt",
                "llm": "gpt-5-nano",
                "paper_metrics": {
                    "acc_at_1": 1.0,
                    "acc_at_3": 1.0,
                    "acc_at_5": 1.0,
                    "labeled_cases": 1,
                    "processed_cases": 1,
                    "ranked_cases": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(json.dumps({"case-1": ["svc"]}), encoding="utf-8")
    output = tmp_path / "aggregate.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_nurse_sota_results.py"),
            "aggregate",
            "--run-root",
            str(tmp_path / "run_root"),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.with_suffix(".xlsx").exists()
