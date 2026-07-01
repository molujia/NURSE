"""Run the requested NURSE parameter campaign.

This script runs four coordinated phases:

1. MJ sensitivity sweep with no LLM.
2. Window-width sensitivity sweep with no LLM.
3. History bootstrap: 20 cold-start corrected cases per dataset.
4. LLM-enabled full-dataset ablations over full / wo_E / wo_P / wo_H.

The non-LLM sweeps run in parallel because they are deterministic. The
history-bootstrap and ablation phases also run in parallel, while rotating
through the supplied Volcano model pool when a model starts returning non-ok
LLM statuses.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nurse.case_order import write_case_order_manifest
from nurse.history_prior import build_corrected_history_record
from nurse.sota_reporting import aggregate_run_root, load_run_snapshot


DEFAULT_DATASETS = ["aiops25-retained-nonnode", "rcaeval-tt"]
DEFAULT_MJ_VALUES = [0, 1, 2, 3, 4, 5]
DEFAULT_WINDOW_VALUES = [60, 120, 240, 480, 960]
DEFAULT_ABLATION_MODES = ["full", "wo_E", "wo_P", "wo_H"]
DEFAULT_ABLATION_MODELS = [
    "ep-20260615114505-247zc",
    "ep-20260623103535-v7z57",
    "ep-20260623103508-6dkw6",
    "ep-m-20260430194631-77b5x",
    "ep-m-20260430105117-46zl7",
    "ep-20260630151544-g59w4",
    "ep-20260630151630-q6q2k",
]
DEFAULT_HISTORY_BOOTSTRAP_CASE_LIMIT = 20


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


def _text_value(value: Any) -> str:
    return "" if value is None else str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _summary_row(job: Dict[str, Any], summary: Dict[str, Any], *, success: bool) -> Dict[str, Any]:
    paper_metrics = summary.get("paper_metrics") if isinstance(summary.get("paper_metrics"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    llm_status_counts = summary.get("llm_status_counts") if isinstance(summary.get("llm_status_counts"), dict) else {}
    status_counts = {
        f"llm_status_{key}": _safe_int(value)
        for key, value in sorted(llm_status_counts.items())
    }
    row = {
        "phase": str(job.get("phase") or ""),
        "dataset": str(job.get("dataset") or ""),
        "setting": str(job.get("setting") or ""),
        "setting_value": _text_value(job.get("setting_value")),
        "mode": str(job.get("mode") or ""),
        "llm": str(summary.get("llm") or job.get("llm") or ""),
        "success": bool(success),
        "case_limit": _safe_int(summary.get("case_limit", job.get("case_limit", 0))),
        "candidate_expansion_rounds": _safe_int(summary.get("candidate_expansion_rounds", job.get("candidate_expansion_rounds", 0))),
        "window_width_seconds": _safe_int(summary.get("window_width_seconds", job.get("window_width_seconds", 0))),
        "acc_at_1": _safe_float(paper_metrics.get("acc_at_1", metrics.get("acc_at_1", 0.0))),
        "acc_at_3": _safe_float(paper_metrics.get("acc_at_3", metrics.get("acc_at_3", 0.0))),
        "acc_at_5": _safe_float(paper_metrics.get("acc_at_5", metrics.get("acc_at_5", 0.0))),
        "processed_cases": _safe_int(paper_metrics.get("processed_cases", 0)),
        "ranked_cases": _safe_int(paper_metrics.get("ranked_cases", 0)),
        "labeled_cases": _safe_int(paper_metrics.get("labeled_cases", 0)),
        "missed_alarm_cases": _safe_int(paper_metrics.get("missed_alarm_cases", 0)),
        "token_usage_total": _safe_int(paper_metrics.get("token_usage_total", summary.get("token_usage", 0))),
        "token_usage_per_case": _safe_float(paper_metrics.get("token_usage_per_case", 0.0)),
        "latency_seconds_per_case": _safe_float(paper_metrics.get("latency_seconds_per_case", 0.0)),
        "non_llm_latency_seconds_per_case": _safe_float(paper_metrics.get("non_llm_latency_seconds_per_case", 0.0)),
        "llm_latency_seconds_per_case": _safe_float(paper_metrics.get("llm_latency_seconds_per_case", 0.0)),
        "elapsed_seconds": _safe_float(summary.get("elapsed_seconds", 0.0)),
        "run_dir": str(job.get("run_dir") or ""),
        "summary_path": str(job.get("summary_path") or ""),
        "top5_hash": str(job.get("top5_hash") or ""),
    }
    row.update(status_counts)
    return row


def _prepare_manifest(
    *,
    manifest_root: Path,
    dataset: str,
    lab_root: Path,
    seed: int,
    sample_size: int,
) -> Path:
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{_slug(dataset)}_sample{int(sample_size)}_seed{int(seed)}.json"
    write_case_order_manifest(
        manifest_path,
        datasets=[dataset],
        lab_root=lab_root,
        seed=seed,
        shuffle=True,
        sample_size=sample_size,
        name="nurse_parameter_campaign",
    )
    return manifest_path


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        os.environ["NURSE_LAB_ROOT"] = str(job["lab_root"])
        os.environ["NURSE_ALARM_CACHE_ROOT"] = str(job["alarm_cache_root"])

        import nurse.runner as nurse_runner

        nurse_runner = importlib.reload(nurse_runner)
        run_dir = nurse_runner.run_nurse(
            job["dataset"],
            int(job["case_limit"]),
            str(job["llm"]),
            str(job["output_root"]),
            generate_doc=False,
            run_dir=str(job["run_dir"]),
            resume=False,
            progress=False,
            ranking_engine="evidence_program_v2",
            experiment_mode=str(job["mode"]),
            shuffle=bool(job.get("shuffle", True)),
            seed=int(job["seed"]),
            case_manifest=str(job["case_manifest"]),
            allow_no_llm=bool(job.get("allow_no_llm", False)),
            candidate_expansion_rounds=int(job["candidate_expansion_rounds"]),
            window_width_seconds=int(job["window_width_seconds"]),
            history_index_in=str(job.get("history_index_in") or ""),
        )
        run_path = Path(run_dir)
        snapshot = load_run_snapshot(run_path)
        summary_path = run_path / "summary.json"
        summary = snapshot["summary"] if isinstance(snapshot.get("summary"), dict) else {}
        llm_status_counts = summary.get("llm_status_counts") if isinstance(summary.get("llm_status_counts"), dict) else {}
        if job.get("allow_no_llm"):
            success = all(key in {"disabled"} for key in llm_status_counts if llm_status_counts.get(key))
        else:
            success = all(key in {"ok"} for key in llm_status_counts if llm_status_counts.get(key))
        return {
            "ok": True,
            "success": success,
            "job": job,
            "run_dir": str(run_path),
            "summary_path": str(summary_path),
            "summary": summary,
            "top5_hash": snapshot.get("top5_hash", ""),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
    except Exception as exc:  # pragma: no cover - surfaced in campaign report
        return {
            "ok": False,
            "success": False,
            "job": job,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }


def _write_phase_report(phase_root: Path, rows: Sequence[Dict[str, Any]], attempts: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    phase_root.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    attempts = list(attempts)
    csv_path = phase_root / "results.csv"
    xlsx_path = phase_root / "results.xlsx"
    json_path = phase_root / "results.json"
    attempts_json = phase_root / "attempts.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="results")
        pd.DataFrame(attempts).to_excel(writer, index=False, sheet_name="attempts")

    _write_json(
        json_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows,
        },
    )
    _write_json(
        attempts_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "attempts": attempts,
        },
    )
    return {
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
        "json": str(json_path),
        "attempts_json": str(attempts_json),
    }


def _run_parallel_jobs(jobs: Sequence[Dict[str, Any]], *, workers: int) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    if not jobs:
        return results
    max_workers = max(1, int(workers))
    if max_workers == 1 or len(jobs) == 1:
        for job in jobs:
            results.append(_run_job(job))
        return results
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: int((item.get("job") or {}).get("job_index", 0)))
    return results


def _dataset_manifest_map(manifest_root: Path, datasets: Sequence[str], lab_root: Path, seed: int, sample_size: int) -> Dict[str, Path]:
    return {
        dataset: _prepare_manifest(
            manifest_root=manifest_root,
            dataset=dataset,
            lab_root=lab_root,
            seed=seed,
            sample_size=sample_size,
        )
        for dataset in datasets
    }


def _no_llm_jobs(
    *,
    phase: str,
    setting: str,
    values: Sequence[int],
    datasets: Sequence[str],
    manifests: Dict[str, Path],
    campaign_root: Path,
    lab_root: Path,
    seed: int,
    output_subdir: str,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
) -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []
    for value in values:
        for dataset in datasets:
            if setting == "mj":
                mj = int(value)
                run_dir = campaign_root / output_subdir / "runs" / _slug(dataset) / f"mj_{mj}"
                jobs.append(
                    {
                        "job_index": len(jobs),
                        "phase": phase,
                        "dataset": dataset,
                        "setting": setting,
                        "setting_value": mj,
                        "mode": "full",
                        "llm": "none",
                        "case_limit": 0,
                        "candidate_expansion_rounds": mj,
                        "window_width_seconds": window_width_seconds,
                        "shuffle": True,
                        "seed": seed,
                        "case_manifest": manifests[dataset],
                        "output_root": campaign_root / output_subdir / "runs",
                        "run_dir": run_dir,
                        "lab_root": lab_root,
                        "alarm_cache_root": campaign_root / output_subdir / "_alarm_cache" / _slug(dataset) / f"mj_{mj}",
                        "allow_no_llm": True,
                    }
                )
            else:
                width = int(value)
                run_dir = campaign_root / output_subdir / "runs" / _slug(dataset) / f"w_{width}"
                jobs.append(
                    {
                        "job_index": len(jobs),
                        "phase": phase,
                        "dataset": dataset,
                        "setting": setting,
                        "setting_value": width,
                        "mode": "full",
                        "llm": "none",
                        "case_limit": 0,
                        "candidate_expansion_rounds": candidate_expansion_rounds,
                        "window_width_seconds": width,
                        "shuffle": True,
                        "seed": seed,
                        "case_manifest": manifests[dataset],
                        "output_root": campaign_root / output_subdir / "runs",
                        "run_dir": run_dir,
                        "lab_root": lab_root,
                        "alarm_cache_root": campaign_root / output_subdir / "_alarm_cache" / _slug(dataset) / f"w_{width}",
                        "allow_no_llm": True,
                    }
                )
    return jobs


def _llm_phase_base_jobs(
    *,
    phase_name: str,
    setting: str,
    datasets: Sequence[str],
    modes: Sequence[str],
    manifests: Dict[str, Path],
    phase_root: Path,
    lab_root: Path,
    seed: int,
    case_limit: int,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
    history_index_by_dataset: Dict[str, Path] | None = None,
) -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []
    for dataset in datasets:
        for mode in modes:
            jobs.append(
                {
                    "job_index": len(jobs),
                    "phase": phase_name,
                    "dataset": dataset,
                    "setting": setting,
                    "setting_value": mode if setting == "ablation" else case_limit,
                    "mode": mode,
                    "case_limit": case_limit,
                    "candidate_expansion_rounds": candidate_expansion_rounds,
                    "window_width_seconds": window_width_seconds,
                    "shuffle": True,
                    "seed": seed,
                    "case_manifest": manifests[dataset],
                    "attempt_root": phase_root / "attempts",
                    "final_run_root": phase_root / "runs",
                    "cache_root": phase_root / "_alarm_cache",
                    "lab_root": lab_root,
                    "allow_no_llm": False,
                    "history_index_in": str((history_index_by_dataset or {}).get(dataset) or ""),
                }
            )
    return jobs


def _finalize_llm_run(
    *,
    attempt_result: Dict[str, Any],
    dataset: str,
    mode: str,
    model: str,
    final_run_root: Path,
) -> Path:
    attempt_run_dir = Path(str(attempt_result["run_dir"]))
    final_run_dir = final_run_root / _slug(dataset) / _slug(mode) / _slug(model)
    final_run_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)
    shutil.move(str(attempt_run_dir), str(final_run_dir))
    return final_run_dir


def _next_model_index(model_pool: Sequence[str], start: int, blocked_models: set[str]) -> int | None:
    for index in range(max(int(start), 0), len(model_pool)):
        if str(model_pool[index]) not in blocked_models:
            return index
    return None


def _materialize_llm_attempt(base_job: Dict[str, Any], *, model: str, attempt_number: int) -> Dict[str, Any]:
    attempt_suffix = f"attempt_{int(attempt_number):02d}"
    dataset_slug = _slug(str(base_job["dataset"]))
    mode_slug = _slug(str(base_job["mode"]))
    model_slug = _slug(model)
    run_dir = Path(base_job["attempt_root"]) / dataset_slug / mode_slug / model_slug / attempt_suffix
    alarm_cache_root = Path(base_job["cache_root"]) / dataset_slug / mode_slug / model_slug / attempt_suffix
    job = dict(base_job)
    job["llm"] = str(model)
    job["run_dir"] = run_dir
    job["output_root"] = Path(base_job["attempt_root"])
    job["alarm_cache_root"] = alarm_cache_root
    return job


def _attempt_row(result: Dict[str, Any]) -> Dict[str, Any]:
    job = result.get("job") if isinstance(result.get("job"), dict) else {}
    return {
        "phase": job.get("phase", ""),
        "dataset": job.get("dataset", ""),
        "setting": job.get("setting", ""),
        "setting_value": job.get("setting_value", ""),
        "mode": job.get("mode", ""),
        "llm": job.get("llm", ""),
        "ok": bool(result.get("ok")),
        "success": bool(result.get("success")),
        "error": result.get("error", ""),
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        "run_dir": result.get("run_dir", ""),
        "summary_path": result.get("summary_path", ""),
        "top5_hash": result.get("top5_hash", ""),
    }


def _run_llm_phase_with_fallback(
    *,
    base_jobs: Sequence[Dict[str, Any]],
    models: Sequence[str],
    workers: int,
    blocked_models: set[str] | None = None,
) -> Dict[str, Any]:
    model_pool = [str(model) for model in models]
    if not model_pool:
        raise ValueError("LLM phase requires at least one model")

    blocked = {str(model) for model in (blocked_models or set())}
    pending = []
    for base_job in base_jobs:
        preferred_start = int(base_job.get("job_index", 0)) % len(model_pool)
        next_index = _next_model_index(model_pool, preferred_start, blocked)
        if next_index is None:
            next_index = _next_model_index(model_pool, 0, blocked)
        if next_index is None:
            raise RuntimeError("no usable LLM models remain for this phase")
        pending.append(
            {
                "base_job": dict(base_job),
                "model_index": next_index,
                "attempt_number": 1,
            }
        )

    final_rows: list[Dict[str, Any]] = []
    attempts: list[Dict[str, Any]] = []
    completed: list[Dict[str, Any]] = []
    exhausted_models: set[str] = set(blocked)

    while pending:
        attempt_jobs = [
            _materialize_llm_attempt(
                item["base_job"],
                model=model_pool[item["model_index"]],
                attempt_number=item["attempt_number"],
            )
            for item in pending
        ]
        results = _run_parallel_jobs(attempt_jobs, workers=workers)
        next_pending = []
        pending_by_index = {
            int(item["base_job"]["job_index"]): item
            for item in pending
        }

        for result in results:
            attempts.append(_attempt_row(result))
            job = result.get("job") if isinstance(result.get("job"), dict) else {}
            job_index = int(job.get("job_index", -1))
            pending_state = pending_by_index[job_index]
            model = str(job.get("llm") or "")

            if result.get("ok") and result.get("success"):
                final_run_dir = _finalize_llm_run(
                    attempt_result=result,
                    dataset=str(job["dataset"]),
                    mode=str(job["mode"]),
                    model=model,
                    final_run_root=Path(pending_state["base_job"]["final_run_root"]),
                )
                attempts[-1]["run_dir"] = str(final_run_dir)
                summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
                final_job = dict(job)
                final_job["run_dir"] = str(final_run_dir)
                final_rows.append(_summary_row(final_job, summary, success=True))
                completed.append(
                    {
                        "job": final_job,
                        "summary": summary,
                        "result": result,
                        "final_run_dir": str(final_run_dir),
                    }
                )
                continue

            exhausted_models.add(model)
            next_index = _next_model_index(model_pool, int(pending_state["model_index"]) + 1, exhausted_models)
            if next_index is None:
                next_index = _next_model_index(model_pool, 0, exhausted_models)
            if next_index is None:
                raise RuntimeError(
                    f"LLM phase failed for dataset={job.get('dataset')} mode={job.get('mode')}: all models exhausted"
                )
            next_pending.append(
                {
                    "base_job": pending_state["base_job"],
                    "model_index": next_index,
                    "attempt_number": int(pending_state["attempt_number"]) + 1,
                }
            )

        pending = next_pending

    return {
        "rows": final_rows,
        "attempts": attempts,
        "completed": completed,
        "exhausted_models": sorted(exhausted_models),
    }


def _build_corrected_history_index(run_dir: Path, *, dataset: str) -> list[Dict[str, Any]]:
    labels_by_case = _load_json(run_dir / "labels.json")
    records: list[Dict[str, Any]] = []
    prediction_dir = run_dir / "predictions"
    for prediction_path in sorted(prediction_dir.glob("*.json")):
        case_id = prediction_path.stem
        prediction = _load_json(prediction_path)
        graph = _load_json(run_dir / "graphs" / f"{case_id}.json")
        metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
        semantic_payload = metadata.get("semantic_rank_payload") if isinstance(metadata.get("semantic_rank_payload"), dict) else {}
        labels = labels_by_case.get(case_id, [])
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            labels = []
        records.append(
            build_corrected_history_record(
                case_id=case_id,
                dataset=dataset,
                semantic_payload=semantic_payload,
                score_summary=prediction.get("scores", {}),
                score_breakdown=prediction.get("score_breakdown", {}),
                ranked_candidates=prediction.get("ranked", []),
                labels=labels,
            )
        )
        records[-1].setdefault("history_correction", {})
        records[-1]["history_correction"]["source_run_dir"] = str(run_dir)
    return records


def _write_history_index_files(
    *,
    phase_root: Path,
    dataset: str,
    history_records: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    history_root = phase_root / "approved_history" / _slug(dataset)
    history_root.mkdir(parents=True, exist_ok=True)
    json_path = history_root / "approved_history_index.json"
    jsonl_path = history_root / "approved_history_index.jsonl"
    metadata_path = history_root / "approved_history_metadata.json"
    records = [dict(record) for record in history_records]
    _write_json(json_path, records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(
        metadata_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "record_count": len(records),
        },
    )
    return {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "metadata_json": str(metadata_path),
    }


def _phase_rows_from_results(results: Sequence[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    rows: list[Dict[str, Any]] = []
    attempts: list[Dict[str, Any]] = []
    for result in results:
        job = result.get("job") if isinstance(result.get("job"), dict) else {}
        attempts.append(
            {
                "phase": job.get("phase", ""),
                "dataset": job.get("dataset", ""),
                "setting": job.get("setting", ""),
                "setting_value": job.get("setting_value", ""),
                "mode": job.get("mode", ""),
                "llm": job.get("llm", ""),
                "ok": bool(result.get("ok")),
                "success": bool(result.get("success")),
                "error": result.get("error", ""),
                "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                "run_dir": result.get("run_dir", ""),
                "summary_path": result.get("summary_path", ""),
                "top5_hash": result.get("top5_hash", ""),
            }
        )
        if result.get("ok"):
            summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
            job_with_artifacts = dict(job)
            job_with_artifacts["summary_path"] = result.get("summary_path", "")
            job_with_artifacts["top5_hash"] = result.get("top5_hash", "")
            rows.append(_summary_row(job_with_artifacts, summary, success=bool(result.get("success"))))
    return rows, attempts


def run_no_llm_phase(
    *,
    phase_name: str,
    setting: str,
    values: Sequence[int],
    datasets: Sequence[str],
    lab_root: Path,
    campaign_root: Path,
    seed: int,
    workers: int,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
) -> Dict[str, Any]:
    phase_root = campaign_root / phase_name
    manifest_root = campaign_root / "manifests"
    manifests = _dataset_manifest_map(
        manifest_root=manifest_root,
        datasets=datasets,
        lab_root=lab_root,
        seed=seed,
        sample_size=0,
    )
    jobs = _no_llm_jobs(
        phase=phase_name,
        setting=setting,
        values=values,
        datasets=datasets,
        manifests=manifests,
        campaign_root=campaign_root,
        lab_root=lab_root,
        seed=seed,
        output_subdir=phase_name,
        candidate_expansion_rounds=candidate_expansion_rounds,
        window_width_seconds=window_width_seconds,
    )
    results = _run_parallel_jobs(jobs, workers=workers)
    rows, attempts = _phase_rows_from_results(results)
    report_files = _write_phase_report(phase_root, rows, attempts)
    aggregate_path = phase_root / "aggregate.csv"
    aggregate_run_root(phase_root / "runs", aggregate_path)
    return {
        "phase": phase_name,
        "phase_root": str(phase_root),
        "manifests": {dataset: str(path) for dataset, path in manifests.items()},
        "jobs": jobs,
        "results": results,
        "rows": rows,
        "attempts": attempts,
        "report_files": report_files,
        "aggregate_csv": str(aggregate_path),
    }


def run_history_bootstrap_phase(
    *,
    datasets: Sequence[str],
    lab_root: Path,
    campaign_root: Path,
    seed: int,
    case_limit: int,
    models: Sequence[str],
    workers: int,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
    blocked_models: set[str] | None = None,
) -> Dict[str, Any]:
    phase_name = "history_bootstrap"
    phase_root = campaign_root / phase_name
    manifest_root = campaign_root / "manifests"
    manifests = _dataset_manifest_map(
        manifest_root=manifest_root,
        datasets=datasets,
        lab_root=lab_root,
        seed=seed,
        sample_size=case_limit,
    )
    base_jobs = _llm_phase_base_jobs(
        phase_name=phase_name,
        setting="history_bootstrap",
        datasets=datasets,
        modes=["full"],
        manifests=manifests,
        phase_root=phase_root,
        lab_root=lab_root,
        seed=seed,
        case_limit=case_limit,
        candidate_expansion_rounds=candidate_expansion_rounds,
        window_width_seconds=window_width_seconds,
    )
    phase_run = _run_llm_phase_with_fallback(
        base_jobs=base_jobs,
        models=models,
        workers=workers,
        blocked_models=blocked_models,
    )
    history_index_files: Dict[str, Dict[str, str]] = {}
    for completed in phase_run["completed"]:
        job = completed["job"]
        dataset = str(job["dataset"])
        run_dir = Path(str(completed["final_run_dir"]))
        history_records = _build_corrected_history_index(run_dir, dataset=dataset)
        history_index_files[dataset] = _write_history_index_files(
            phase_root=phase_root,
            dataset=dataset,
            history_records=history_records,
        )
    report_files = _write_phase_report(phase_root, phase_run["rows"], phase_run["attempts"])
    aggregate_path = phase_root / "aggregate.csv"
    aggregate_run_root(phase_root / "runs", aggregate_path)
    return {
        "phase": phase_name,
        "phase_root": str(phase_root),
        "manifests": {dataset: str(path) for dataset, path in manifests.items()},
        "rows": phase_run["rows"],
        "attempts": phase_run["attempts"],
        "report_files": report_files,
        "aggregate_csv": str(aggregate_path),
        "history_index_files": history_index_files,
        "exhausted_models": phase_run["exhausted_models"],
    }


def run_ablation_phase(
    *,
    datasets: Sequence[str],
    modes: Sequence[str],
    lab_root: Path,
    campaign_root: Path,
    seed: int,
    case_limit: int,
    models: Sequence[str],
    workers: int,
    candidate_expansion_rounds: int,
    window_width_seconds: int,
    history_index_by_dataset: Dict[str, Path],
    blocked_models: set[str] | None = None,
) -> Dict[str, Any]:
    phase_name = "ablation"
    phase_root = campaign_root / phase_name
    manifest_root = campaign_root / "manifests"
    manifests = _dataset_manifest_map(
        manifest_root=manifest_root,
        datasets=datasets,
        lab_root=lab_root,
        seed=seed,
        sample_size=0,
    )
    base_jobs = _llm_phase_base_jobs(
        phase_name=phase_name,
        setting="ablation",
        datasets=datasets,
        modes=modes,
        manifests=manifests,
        phase_root=phase_root,
        lab_root=lab_root,
        seed=seed,
        case_limit=case_limit,
        candidate_expansion_rounds=candidate_expansion_rounds,
        window_width_seconds=window_width_seconds,
        history_index_by_dataset=history_index_by_dataset,
    )
    phase_run = _run_llm_phase_with_fallback(
        base_jobs=base_jobs,
        models=models,
        workers=workers,
        blocked_models=blocked_models,
    )
    report_files = _write_phase_report(phase_root, phase_run["rows"], phase_run["attempts"])
    aggregate_path = phase_root / "aggregate.csv"
    aggregate_run_root(phase_root / "runs", aggregate_path)
    return {
        "phase": phase_name,
        "phase_root": str(phase_root),
        "manifests": {dataset: str(path) for dataset, path in manifests.items()},
        "rows": phase_run["rows"],
        "attempts": phase_run["attempts"],
        "report_files": report_files,
        "aggregate_csv": str(aggregate_path),
        "history_index_by_dataset": {dataset: str(path) for dataset, path in history_index_by_dataset.items()},
        "exhausted_models": phase_run["exhausted_models"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NURSE parameter campaign")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "tmp" / "nurse_parameter_campaign"))
    parser.add_argument("--lab-root", default=str(REPO_ROOT.parent))
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--mj-values", nargs="+", type=int, default=DEFAULT_MJ_VALUES)
    parser.add_argument("--window-values", nargs="+", type=int, default=DEFAULT_WINDOW_VALUES)
    parser.add_argument("--ablation-modes", nargs="+", default=DEFAULT_ABLATION_MODES)
    parser.add_argument("--ablation-models", nargs="+", default=DEFAULT_ABLATION_MODELS)
    parser.add_argument("--no-llm-workers", type=int, default=4)
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--history-bootstrap-case-limit", type=int, default=DEFAULT_HISTORY_BOOTSTRAP_CASE_LIMIT)
    parser.add_argument("--ablation-case-limit", type=int, default=0)
    parser.add_argument("--candidate-expansion-rounds", type=int, default=2)
    parser.add_argument("--window-width-seconds", type=int, default=480)
    parser.add_argument("--skip-mj", action="store_true")
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    campaign_root = Path(args.output_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_root.mkdir(parents=True, exist_ok=True)
    lab_root = Path(args.lab_root).expanduser().resolve()

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_root": str(campaign_root),
        "lab_root": str(lab_root),
        "seed": int(args.seed),
        "datasets": [str(dataset) for dataset in args.datasets],
        "phases": {},
    }

    if not args.skip_mj:
        mj_report = run_no_llm_phase(
            phase_name="mj_sensitivity",
            setting="mj",
            values=args.mj_values,
            datasets=args.datasets,
            lab_root=lab_root,
            campaign_root=campaign_root,
            seed=args.seed,
            workers=args.no_llm_workers,
            candidate_expansion_rounds=0,
            window_width_seconds=args.window_width_seconds,
        )
        report["phases"]["mj_sensitivity"] = mj_report

    if not args.skip_window:
        window_report = run_no_llm_phase(
            phase_name="window_sensitivity",
            setting="window",
            values=args.window_values,
            datasets=args.datasets,
            lab_root=lab_root,
            campaign_root=campaign_root,
            seed=args.seed,
            workers=args.no_llm_workers,
            candidate_expansion_rounds=args.candidate_expansion_rounds,
            window_width_seconds=args.window_width_seconds,
        )
        report["phases"]["window_sensitivity"] = window_report

    if not args.skip_ablation:
        history_bootstrap_report = run_history_bootstrap_phase(
            datasets=args.datasets,
            lab_root=lab_root,
            campaign_root=campaign_root,
            seed=args.seed,
            case_limit=args.history_bootstrap_case_limit,
            models=args.ablation_models,
            workers=args.llm_workers,
            candidate_expansion_rounds=args.candidate_expansion_rounds,
            window_width_seconds=args.window_width_seconds,
        )
        report["phases"]["history_bootstrap"] = history_bootstrap_report
        history_index_by_dataset = {
            dataset: Path(files["json"])
            for dataset, files in history_bootstrap_report.get("history_index_files", {}).items()
        }
        ablation_report = run_ablation_phase(
            datasets=args.datasets,
            modes=args.ablation_modes,
            lab_root=lab_root,
            campaign_root=campaign_root,
            seed=args.seed,
            case_limit=args.ablation_case_limit,
            models=args.ablation_models,
            workers=args.llm_workers,
            candidate_expansion_rounds=args.candidate_expansion_rounds,
            window_width_seconds=args.window_width_seconds,
            history_index_by_dataset=history_index_by_dataset,
            blocked_models=set(history_bootstrap_report.get("exhausted_models", [])),
        )
        report["phases"]["ablation"] = ablation_report

    report_path = campaign_root / "campaign_report.json"
    _write_json(report_path, report)

    # Build a convenience workbook across the final phase rows.
    workbook_path = campaign_root / "campaign_report.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for phase_name, phase_report in report["phases"].items():
            rows = phase_report.get("rows", []) if isinstance(phase_report, dict) else []
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name=_slug(phase_name)[:31])

    print(f"[Nurse] Campaign root: {campaign_root}")
    print(f"[Nurse] Campaign report: {report_path}")
    print(f"[Nurse] Campaign workbook: {workbook_path}")
    print(f"[Nurse] Phases: {', '.join(sorted(report['phases']))}")


if __name__ == "__main__":
    main()
