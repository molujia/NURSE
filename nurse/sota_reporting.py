"""Sampling and matrix helpers for NURSE release runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from .case_order import write_case_order_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAB_ROOT = Path(os.environ.get("NURSE_LAB_ROOT", REPO_ROOT.parent)).expanduser().resolve()
DEFAULT_SAMPLE_MODELS = [
    "ep-20260623103508-6dkw6",
    "Doubao-Seed-2.1-pro",
    "ep-20260623160517-ckhj8",
    "glm-4.5-air",
    "glm-4.6v",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]
DEFAULT_SAMPLE_DATASETS = ["aiops25-retained-nonnode", "rcaeval-tt"]


def _slug(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    allowed = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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


def _canonical_top5(ranking: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in list(ranking or [])[:5])


def _top5_by_case(predictions: Mapping[str, Iterable[Any]]) -> Dict[str, list[str]]:
    return {
        str(case_id): list(_canonical_top5(ranking))
        for case_id, ranking in sorted(predictions.items(), key=lambda item: str(item[0]))
    }


def _top5_hash(top5_by_case: Mapping[str, Sequence[str]]) -> str:
    payload = [(str(case_id), list(top5)) for case_id, top5 in sorted(top5_by_case.items(), key=lambda item: str(item[0]))]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default).encode("utf-8")
    ).hexdigest()
    return digest


def _summary_metrics(summary: Mapping[str, Any], predictions: Mapping[str, Sequence[str]] | None = None) -> Dict[str, Any]:
    paper_metrics = summary.get("paper_metrics") if isinstance(summary.get("paper_metrics"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    source = paper_metrics or metrics
    case_count = (
        source.get("labeled_cases")
        or source.get("processed_cases")
        or len(predictions or {})
        or summary.get("case_limit")
        or 0
    )
    return {
        "acc_at_1": _safe_float(source.get("acc_at_1", 0.0)),
        "acc_at_3": _safe_float(source.get("acc_at_3", 0.0)),
        "acc_at_5": _safe_float(source.get("acc_at_5", 0.0)),
        "cases": _safe_int(case_count),
        "labeled_cases": _safe_int(source.get("labeled_cases", case_count)),
        "processed_cases": _safe_int(source.get("processed_cases", case_count)),
        "ranked_cases": _safe_int(source.get("ranked_cases", 0)),
        "token_usage_total": _safe_int(source.get("token_usage_total", summary.get("token_usage", 0))),
        "candidate_expansion_rounds": _safe_int(summary.get("candidate_expansion_rounds", 0)),
        "window_width_seconds": _safe_int(summary.get("window_width_seconds", 0)),
    }


def load_run_snapshot(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    summary = _load_json(run_path / "summary.json")
    predictions = _load_json(run_path / "predictions.json")
    top5_by_case = _top5_by_case(predictions)
    return {
        "run_dir": str(run_path),
        "summary": summary,
        "predictions": predictions,
        "top5_by_case": top5_by_case,
        "top5_hash": _top5_hash(top5_by_case),
    }


def compare_top5_consistency(
    *,
    case_ids_by_dataset: Mapping[str, Sequence[str]],
    snapshots_by_dataset_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    datasets: Dict[str, Dict[str, Any]] = {}
    diff_rows: list[Dict[str, Any]] = []
    sample_metrics: list[Dict[str, Any]] = []
    mismatch_case_ids_global: list[str] = []

    for dataset in sorted(case_ids_by_dataset):
        case_ids = [str(case_id) for case_id in case_ids_by_dataset.get(dataset, [])]
        model_snapshots = snapshots_by_dataset_model.get(dataset, {})
        model_names = sorted(model_snapshots)
        dataset_models: Dict[str, Dict[str, Any]] = {}
        if not model_names:
            datasets[dataset] = {
                "case_ids": case_ids,
                "model_names": [],
                "all_models_equal": True,
                "mismatch_case_ids": [],
                "models": {},
            }
            continue

        for model in model_names:
            snapshot = dict(model_snapshots[model])
            predictions = snapshot.get("predictions", {})
            top5_by_case = {
                case_id: list(_canonical_top5(predictions.get(case_id, [])))
                for case_id in case_ids
            }
            summary = snapshot.get("summary", {})
            metrics = _summary_metrics(summary, predictions)
            model_hash = _top5_hash(top5_by_case)
            dataset_models[model] = {
                "run_dir": str(snapshot.get("run_dir", "")),
                "summary": summary,
                "metrics": metrics,
                "top5_by_case": top5_by_case,
                "top5_hash": model_hash,
            }

        mismatch_case_ids: list[str] = []
        reference_model = model_names[0]
        for case_id in case_ids:
            per_model = {
                model: tuple(dataset_models[model]["top5_by_case"].get(case_id, ()))
                for model in model_names
            }
            unique_variants = {variant for variant in per_model.values()}
            if len(unique_variants) > 1:
                mismatch_case_ids.append(case_id)
                mismatch_case_ids_global.append(case_id)
                ref_top5 = list(per_model[reference_model])
                diff_rows.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "reference_model": reference_model,
                        "reference_top5": json.dumps(ref_top5, ensure_ascii=False),
                        "unique_top5_count": len(unique_variants),
                        "differing_models": ";".join(
                            model for model, top5 in per_model.items() if list(top5) != ref_top5
                        ),
                        "model_top5": json.dumps(
                            {model: list(top5) for model, top5 in per_model.items()},
                            ensure_ascii=False,
                        ),
                    }
                )

        all_models_equal = not mismatch_case_ids
        for model in model_names:
            metrics = dataset_models[model]["metrics"]
            sample_metrics.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **metrics,
                    "top5_hash": dataset_models[model]["top5_hash"],
                    "top5_mismatch_cases": len(mismatch_case_ids),
                    "top5_mismatch_ratio": round(len(mismatch_case_ids) / max(len(case_ids), 1), 6),
                    "all_models_equal": all_models_equal,
                    "reference_model": reference_model,
                }
            )

        datasets[dataset] = {
            "case_ids": case_ids,
            "model_names": model_names,
            "all_models_equal": all_models_equal,
            "mismatch_case_ids": mismatch_case_ids,
            "models": dataset_models,
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "diff_rows": diff_rows,
        "sample_metrics": sample_metrics,
        "mismatch_case_ids": sorted(set(mismatch_case_ids_global)),
    }


def write_consistency_outputs(report: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    top5_json = output_path / "top5_by_model.json"
    diff_csv = output_path / "top5_diff_report.csv"
    metrics_csv = output_path / "sample_metrics.csv"
    _write_json(top5_json, report)
    pd.DataFrame(report.get("diff_rows", [])).to_csv(diff_csv, index=False)
    pd.DataFrame(report.get("sample_metrics", [])).to_csv(metrics_csv, index=False)
    return {
        "top5_by_model_json": str(top5_json),
        "top5_diff_report_csv": str(diff_csv),
        "sample_metrics_csv": str(metrics_csv),
    }


def _manifest_for_dataset(
    *,
    manifest_dir: Path,
    dataset: str,
    lab_root: str | Path,
    seed: int,
    sample_size: int,
) -> Dict[str, Any]:
    manifest_path = manifest_dir / f"{_slug(dataset)}_seed{int(seed)}_sample{int(sample_size)}.json"
    payload = write_case_order_manifest(
        manifest_path,
        datasets=[dataset],
        lab_root=lab_root,
        seed=seed,
        shuffle=True,
        sample_size=sample_size,
        name="nurse_model_consistency_sample",
    )
    return {
        "dataset": dataset,
        "manifest_path": manifest_path,
        "case_ids": [str(case_id) for case_id in payload.get("datasets", {}).get(dataset, [])],
    }


def _run_nurse_job(
    *,
    python_exe: str,
    dataset: str,
    model: str,
    mode: str,
    run_dir: Path,
    case_manifest: Path,
    case_limit: int,
    seed: int,
    alarm_cache_root: Path,
    candidate_expansion_rounds: int = 2,
    window_width_seconds: int = 480,
    max_retries: int = 0,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    alarm_cache_root.mkdir(parents=True, exist_ok=True)
    command = [
        python_exe,
        "-m",
        "nurse.runner",
        "--dataset",
        dataset,
        "--case-limit",
        str(int(case_limit)),
        "--llm",
        model,
        "--experiment-mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--resume",
        "--shuffle",
        "--seed",
        str(int(seed)),
        "--case-manifest",
        str(case_manifest),
        "--candidate-expansion-rounds",
        str(int(candidate_expansion_rounds)),
        "--window-width-seconds",
        str(int(window_width_seconds)),
        "--no-progress",
    ]
    env = os.environ.copy()
    env["NURSE_ALARM_CACHE_ROOT"] = str(alarm_cache_root)
    attempts = max(1, int(max_retries) + 1)
    result = None
    log_chunks: list[str] = []
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        attempt_log = "\n".join(
            part for part in (result.stdout or "", result.stderr or "") if part.strip()
        )
        if attempt_log:
            log_chunks.append(f"[attempt {attempt}]\n{attempt_log}")
        if result.returncode == 0:
            break
    log_path = run_dir / "run.log"
    if log_chunks:
        log_path.write_text("\n\n".join(log_chunks), encoding="utf-8")
    if result is None or result.returncode != 0:
        raise RuntimeError(
            f"NURSE run failed for dataset={dataset} model={model} with exit code {getattr(result, 'returncode', 'unknown')}"
        )
    return {
        "dataset": dataset,
        "model": model,
        "run_dir": str(run_dir),
        "command": command,
        "returncode": result.returncode,
    }


def run_consistency_sample(
    *,
    output_root: str | Path,
    datasets: Sequence[str] = DEFAULT_SAMPLE_DATASETS,
    models: Sequence[str] = DEFAULT_SAMPLE_MODELS,
    seed: int = 20260628,
    sample_size: int = 25,
    workers: int = 10,
    max_retries: int = 0,
    candidate_expansion_rounds: int = 2,
    window_width_seconds: int = 480,
    python_exe: str = "python",
    lab_root: str | Path = DEFAULT_LAB_ROOT,
) -> Dict[str, Any]:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / timestamp
    manifest_dir = run_root / "manifests"
    run_dir_root = run_root / "runs"
    cache_dir_root = run_root / "_alarm_cache"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_dir_root.mkdir(parents=True, exist_ok=True)
    cache_dir_root.mkdir(parents=True, exist_ok=True)

    manifest_info = {
        dataset: _manifest_for_dataset(
            manifest_dir=manifest_dir,
            dataset=dataset,
            lab_root=lab_root,
            seed=seed,
            sample_size=sample_size,
        )
        for dataset in datasets
    }
    jobs = []
    for dataset in datasets:
        case_manifest = manifest_info[dataset]["manifest_path"]
        for model in models:
            run_dir = run_dir_root / _slug(dataset) / _slug(model)
            alarm_cache_root = cache_dir_root / _slug(dataset) / _slug(model)
            jobs.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "run_dir": run_dir,
                    "case_manifest": case_manifest,
                    "alarm_cache_root": alarm_cache_root,
                }
            )

    job_results: list[Dict[str, Any]] = []
    if workers and int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            future_map = {
                executor.submit(
                    _run_nurse_job,
                    python_exe=python_exe,
                    dataset=job["dataset"],
                    model=job["model"],
                    mode="full",
                    run_dir=job["run_dir"],
                    case_manifest=job["case_manifest"],
                    case_limit=sample_size,
                    seed=seed,
                    alarm_cache_root=job["alarm_cache_root"],
                    candidate_expansion_rounds=candidate_expansion_rounds,
                    window_width_seconds=window_width_seconds,
                    max_retries=max_retries,
                ): job
                for job in jobs
            }
            for future in as_completed(future_map):
                job_results.append(future.result())
    else:
        for job in jobs:
            job_results.append(
                _run_nurse_job(
                    python_exe=python_exe,
                    dataset=job["dataset"],
                    model=job["model"],
                    mode="full",
                    run_dir=job["run_dir"],
                    case_manifest=job["case_manifest"],
                    case_limit=sample_size,
                    seed=seed,
                    alarm_cache_root=job["alarm_cache_root"],
                    candidate_expansion_rounds=candidate_expansion_rounds,
                    window_width_seconds=window_width_seconds,
                    max_retries=max_retries,
                )
            )

    snapshots_by_dataset_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for job in jobs:
        snapshot = load_run_snapshot(job["run_dir"])
        dataset = str(job["dataset"])
        model = str(job["model"])
        snapshots_by_dataset_model.setdefault(dataset, {})[model] = snapshot

    case_ids_by_dataset = {
        dataset: info["case_ids"] for dataset, info in manifest_info.items()
    }
    report = compare_top5_consistency(
        case_ids_by_dataset=case_ids_by_dataset,
        snapshots_by_dataset_model=snapshots_by_dataset_model,
    )
    report["run_root"] = str(run_root)
    report["manifest_dir"] = str(manifest_dir)
    report["run_jobs"] = job_results
    report["output_files"] = write_consistency_outputs(report, run_root)
    report["case_ids_by_dataset"] = case_ids_by_dataset

    if report["mismatch_case_ids"]:
        print(
            "[Nurse] Top-5 mismatches: "
            + ", ".join(report["mismatch_case_ids"]),
            flush=True,
        )
        raise SystemExit(3)

    print(f"[Nurse] Consistency sample complete: {run_root}", flush=True)
    return report


def run_matrix_experiment(
    *,
    output_root: str | Path,
    datasets: Sequence[str] = DEFAULT_SAMPLE_DATASETS,
    models: Sequence[str] = DEFAULT_SAMPLE_MODELS,
    modes: Sequence[str] = ("full",),
    seed: int = 20260628,
    case_limit: int = 0,
    workers: int = 10,
    max_retries: int = 0,
    candidate_expansion_rounds: int = 2,
    window_width_seconds: int = 480,
    report_name: str = "matrix_runs",
    python_exe: str = "python",
    lab_root: str | Path = DEFAULT_LAB_ROOT,
) -> Dict[str, Any]:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / timestamp
    manifest_dir = run_root / "manifests"
    run_dir_root = run_root / "runs"
    cache_dir_root = run_root / "_alarm_cache"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_dir_root.mkdir(parents=True, exist_ok=True)
    cache_dir_root.mkdir(parents=True, exist_ok=True)

    manifest_info = {
        dataset: _manifest_for_dataset(
            manifest_dir=manifest_dir,
            dataset=dataset,
            lab_root=lab_root,
            seed=seed,
            sample_size=case_limit,
        )
        for dataset in datasets
    }
    jobs = []
    for dataset in datasets:
        case_manifest = manifest_info[dataset]["manifest_path"]
        for mode in modes:
            for model in models:
                run_dir = run_dir_root / _slug(dataset) / _slug(mode) / _slug(model)
                alarm_cache_root = cache_dir_root / _slug(dataset) / _slug(mode) / _slug(model)
                jobs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "mode": mode,
                        "run_dir": run_dir,
                        "case_manifest": case_manifest,
                        "alarm_cache_root": alarm_cache_root,
                    }
                )

    job_results: list[Dict[str, Any]] = []
    if workers and int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            future_map = {
                executor.submit(
                    _run_nurse_job,
                    python_exe=python_exe,
                    dataset=job["dataset"],
                    model=job["model"],
                    mode=str(job.get("mode") or "full"),
                    run_dir=job["run_dir"],
                    case_manifest=job["case_manifest"],
                    case_limit=case_limit,
                    seed=seed,
                    alarm_cache_root=job["alarm_cache_root"],
                    candidate_expansion_rounds=candidate_expansion_rounds,
                    window_width_seconds=window_width_seconds,
                    max_retries=max_retries,
                ): job
                for job in jobs
            }
            for future in as_completed(future_map):
                job_results.append(future.result())
    else:
        for job in jobs:
            job_results.append(
                _run_nurse_job(
                    python_exe=python_exe,
                    dataset=job["dataset"],
                    model=job["model"],
                    mode=str(job.get("mode") or "full"),
                    run_dir=job["run_dir"],
                    case_manifest=job["case_manifest"],
                    case_limit=case_limit,
                    seed=seed,
                    alarm_cache_root=job["alarm_cache_root"],
                    candidate_expansion_rounds=candidate_expansion_rounds,
                    window_width_seconds=window_width_seconds,
                    max_retries=max_retries,
                )
            )

    rows: list[Dict[str, Any]] = []
    for job in jobs:
        snapshot = load_run_snapshot(job["run_dir"])
        summary = snapshot["summary"]
        metrics = _summary_metrics(summary, snapshot["predictions"])
        rows.append(
            {
                "dataset": str(job["dataset"]),
                "model": str(job["model"]),
                "mode": str(job["mode"]),
                "run_dir": str(job["run_dir"]),
                **metrics,
                "top5_hash": snapshot["top5_hash"],
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["mode"], row["model"]))

    matrix_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "manifest_dir": str(manifest_dir),
        "datasets": [str(dataset) for dataset in datasets],
        "models": [str(model) for model in models],
        "modes": [str(mode) for mode in modes],
        "report_name": str(report_name),
        "case_limit": int(case_limit),
        "seed": int(seed),
        "run_jobs": job_results,
        "rows": rows,
    }
    prefix = _slug(report_name or "matrix_runs")
    json_path = run_root / f"{prefix}.json"
    csv_path = run_root / f"{prefix}.csv"
    xlsx_path = run_root / f"{prefix}.xlsx"
    _write_json(json_path, matrix_report)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="runs")
    return matrix_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NURSE experiment reporting helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Run the fixed 25-case consistency sample")
    sample.add_argument("--datasets", nargs="+", default=DEFAULT_SAMPLE_DATASETS)
    sample.add_argument("--models", nargs="+", default=DEFAULT_SAMPLE_MODELS)
    sample.add_argument("--seed", type=int, default=20260628)
    sample.add_argument("--sample-size", type=int, default=25)
    sample.add_argument("--workers", type=int, default=10)
    sample.add_argument("--max-retries", type=int, default=0)
    sample.add_argument("--candidate-expansion-rounds", type=int, default=2)
    sample.add_argument("--window-width-seconds", type=int, default=480)
    sample.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "experiments" / "nurse_model_consistency_sample"))
    sample.add_argument("--lab-root", default=str(DEFAULT_LAB_ROOT))
    sample.add_argument("--python", default="python")

    matrix = subparsers.add_parser("matrix", help="Run a model/dataset matrix of experiment jobs")
    matrix.add_argument("--datasets", nargs="+", default=DEFAULT_SAMPLE_DATASETS)
    matrix.add_argument("--models", nargs="+", default=DEFAULT_SAMPLE_MODELS)
    matrix.add_argument("--modes", nargs="+", default=["full"])
    matrix.add_argument("--seed", type=int, default=20260628)
    matrix.add_argument("--case-limit", type=int, default=0)
    matrix.add_argument("--workers", type=int, default=10)
    matrix.add_argument("--max-retries", type=int, default=0)
    matrix.add_argument("--candidate-expansion-rounds", type=int, default=2)
    matrix.add_argument("--window-width-seconds", type=int, default=480)
    matrix.add_argument("--report-name", default="matrix_runs")
    matrix.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "experiments" / "nurse_sota_full_parallel"))
    matrix.add_argument("--lab-root", default=str(DEFAULT_LAB_ROOT))
    matrix.add_argument("--python", default="python")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "sample":
        run_consistency_sample(
            output_root=args.output_root,
            datasets=args.datasets,
            models=args.models,
            seed=args.seed,
            sample_size=args.sample_size,
            workers=args.workers,
            max_retries=args.max_retries,
            candidate_expansion_rounds=args.candidate_expansion_rounds,
            window_width_seconds=args.window_width_seconds,
            python_exe=args.python,
            lab_root=args.lab_root,
        )
        return
    if args.command == "matrix":
        report = run_matrix_experiment(
            output_root=args.output_root,
            datasets=args.datasets,
            models=args.models,
            modes=args.modes,
            seed=args.seed,
            case_limit=args.case_limit,
            workers=args.workers,
            max_retries=args.max_retries,
            candidate_expansion_rounds=args.candidate_expansion_rounds,
            window_width_seconds=args.window_width_seconds,
            report_name=args.report_name,
            python_exe=args.python,
            lab_root=args.lab_root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
