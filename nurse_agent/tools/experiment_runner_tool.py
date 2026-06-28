"""Experiment Runner tool for agent-planned benchmark jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from nurse.runner import run_nurse
from nurse.schemas import _assert_no_forbidden


def _rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(payload["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_run = bool(payload.get("dry_run", False))
    config = {
        "dataset": str(payload["dataset"]),
        "metric_score_backend": str(payload.get("metric_score_backend") or "evidence_robust"),
        "methods": list(payload.get("methods") or ["nurse"]),
        "shuffle": bool(payload.get("shuffle", True)),
        "seed": int(payload.get("seed", 20260621)),
        "workers": int(payload.get("workers", 6)),
        "case_limit": int(payload.get("case_limit", 0)),
        "resume": bool(payload.get("resume", True)),
        "dry_run": dry_run,
        "llm_model": str(payload.get("model") or "gpt-5-nano"),
        "launcher": "nurse.runner.run_nurse",
        "status": "dry_run_prepared" if dry_run else "prepared",
    }
    _assert_no_forbidden(config)
    config_path = output_dir / "experiment_config.json"
    summary_path = output_dir / "summary.json"
    paper_row_path = output_dir / "paper_table_row.csv"
    case_metrics_path = output_dir / "case_metrics.csv"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        summary = {
            "status": "dry_run_prepared",
            "message": "Experiment configuration prepared by the NURSE agent.",
            "artifact_paths": {
                "experiment_config": config_path.name,
                "summary": summary_path.name,
                "paper_table_row": paper_row_path.name,
                "case_metrics": case_metrics_path.name,
            },
        }
        _assert_no_forbidden(summary)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if not paper_row_path.exists():
            paper_row_path.write_text("method,dataset,acc1,acc3,acc5,status\n", encoding="utf-8")
        if not case_metrics_path.exists():
            case_metrics_path.write_text("case_id,dataset,top1,status\n", encoding="utf-8")
        artifact_paths = {
            "experiment_config": config_path.name,
            "summary": summary_path.name,
            "paper_table_row": paper_row_path.name,
            "case_metrics": case_metrics_path.name,
        }
    else:
        actual_run_dir = Path(
            run_nurse(
                dataset=config["dataset"],
                case_limit=config["case_limit"],
                llm=config["llm_model"],
                output_root=output_dir,
                run_dir=output_dir,
                resume=config["resume"],
                progress=True,
                shuffle=config["shuffle"],
                seed=config["seed"],
                metric_score_backend=config["metric_score_backend"],
                experiment_mode="full",
            )
        )
        config["status"] = "completed"
        config["actual_run_dir"] = str(actual_run_dir)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact_paths = {
            "experiment_config": _rel_or_abs(config_path, output_dir),
            "summary": _rel_or_abs(actual_run_dir / "summary.json", output_dir),
            "paper_table_row": _rel_or_abs(actual_run_dir / "paper_table_row.csv", output_dir),
            "case_metrics": _rel_or_abs(actual_run_dir / "case_metrics.csv", output_dir),
            "actual_run_dir": _rel_or_abs(actual_run_dir, output_dir),
        }

    return {
        "dataset": config["dataset"],
        "case_id": "",
        "experiment_config": config,
        "artifact_paths": artifact_paths,
    }
