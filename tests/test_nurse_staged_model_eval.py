import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_nurse_staged_model_eval.py"
    spec = importlib.util.spec_from_file_location("run_nurse_staged_model_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_against_baseline_handles_better_worse_and_mixed():
    module = _load_module()
    baseline = {"acc_at_1": 0.2, "acc_at_3": 0.4, "acc_at_5": 0.6, "average_score": 0.4}

    better = module.classify_against_baseline(
        baseline_metrics=baseline,
        candidate_metrics={"acc_at_1": 0.3, "acc_at_3": 0.4, "acc_at_5": 0.7, "average_score": 0.4667},
    )
    worse = module.classify_against_baseline(
        baseline_metrics=baseline,
        candidate_metrics={"acc_at_1": 0.1, "acc_at_3": 0.2, "acc_at_5": 0.5, "average_score": 0.2667},
    )
    mixed = module.classify_against_baseline(
        baseline_metrics=baseline,
        candidate_metrics={"acc_at_1": 0.3, "acc_at_3": 0.3, "acc_at_5": 0.7, "average_score": 0.4333},
    )

    assert better["direction"] == "better"
    assert worse["direction"] == "worse"
    assert mixed["direction"] == "mixed"


def test_decide_round_matches_staged_protocol():
    module = _load_module()

    assert module.decide_round([{"direction": "better"}, {"direction": "unchanged"}]) == "D"
    assert module.decide_round([{"direction": "worse"}, {"direction": "unchanged"}]) == "C"
    assert module.decide_round([{"direction": "better"}, {"direction": "worse"}]) == "extend"
    assert module.decide_round([{"direction": "unavailable"}]) == "blocked"


def test_summarize_case_subset_computes_subset_metrics(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "llm": "none",
                "records": [
                    {
                        "case_id": "case-a",
                        "llm_status": "disabled",
                        "llm": {"total_tokens": 0},
                        "timing": {"total_seconds": 1.0, "non_llm_seconds": 1.0, "llm_seconds": 0.0},
                    },
                    {
                        "case_id": "case-b",
                        "llm_status": "disabled",
                        "llm": {"total_tokens": 0},
                        "timing": {"total_seconds": 2.0, "non_llm_seconds": 2.0, "llm_seconds": 0.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps(
            {
                "case-a": ["svc-a", "svc-b", "svc-c"],
                "case-b": ["svc-x", "svc-b", "svc-y", "svc-z"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "labels.json").write_text(
        json.dumps(
            {
                "case-a": ["svc-a"],
                "case-b": ["svc-b"],
            }
        ),
        encoding="utf-8",
    )

    summary = module.summarize_case_subset(
        run_dir=run_dir,
        dataset="synthetic",
        case_ids=["case-a", "case-b"],
        requested_model="none",
        resolved_model="none",
        sample_name="probe5",
    )

    assert summary["case_count"] == 2
    assert summary["acc_at_1"] == 0.5
    assert summary["acc_at_3"] == 1.0
    assert summary["acc_at_5"] == 1.0
    assert summary["token_usage_total"] == 0
    assert summary["llm_status_counts"] == {"disabled": 2}


def test_summarize_case_subset_prefers_case_metrics_alias_matching(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "llm": "gpt-5-nano",
                "records": [
                    {
                        "case_id": "case-a",
                        "llm_status": "ok",
                        "llm": {"total_tokens": 10},
                        "timing": {"total_seconds": 1.0, "non_llm_seconds": 0.4, "llm_seconds": 0.6},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps({"case-a": ["cart"]}),
        encoding="utf-8",
    )
    (run_dir / "labels.json").write_text(
        json.dumps({"case-a": ["cartservice"]}),
        encoding="utf-8",
    )
    (run_dir / "case_metrics.csv").write_text(
        "\n".join(
            [
                "case_id,status,top1,acceptable_labels,matched_label,matched_rank,total_seconds,non_llm_seconds,llm_seconds",
                "case-a,ranked,cart,cartservice,cartservice,1,1.0,0.4,0.6",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_case_subset(
        run_dir=run_dir,
        dataset="synthetic",
        case_ids=["case-a"],
        requested_model="gpt-5-nano",
        resolved_model="gpt-5-nano",
        sample_name="probe5",
    )

    assert summary["acc_at_1"] == 1.0
    assert summary["acc_at_3"] == 1.0
    assert summary["acc_at_5"] == 1.0
    assert summary["rows"][0]["matched_label"] == "cartservice"
    assert summary["rows"][0]["matched_rank"] == 1
