import json

from nurse.metric_deviation import service_metric_deviation_scores
from nurse.raw_adapters import RawIncident
from nurse.raw_evidence_compiler import METRIC_SCORE_BACKENDS, compile_metric_facts
from nurse.schemas import AlarmArtifact
from nurse_agent.console import run_scripted_console


def _incident():
    return RawIncident(
        dataset="synthetic",
        case_id="case-sliding",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "checkout.cpu": [(idx, 10.0) for idx in range(10)] + [(10, 30.0), (11, 31.0)],
            "frontend.cpu": [(idx, 10.0) for idx in range(12)],
        },
        topology_edges=[],
        metadata={},
    )


def _alarm():
    return AlarmArtifact(
        case_id="case-sliding",
        dataset="synthetic",
        alert_entity="checkout",
        trigger_time=10,
        window={"normal_start": 0, "normal_end": 9, "start": 10, "end": 11},
        detector="unit-test",
        signals=["cpu"],
        provenance={"source": "unit-test"},
    )


def test_sliding_3sigma_backend_is_registered_and_scores_spike():
    assert "sliding_3sigma_window" in METRIC_SCORE_BACKENDS

    scores = service_metric_deviation_scores(
        _incident(),
        _alarm(),
        service_candidates=["checkout", "frontend"],
        score_backend="sliding_3sigma_window",
    )

    assert scores["checkout"]["cpu"] >= 3.0
    assert scores.get("frontend", {}).get("cpu", 0.0) == 0.0


def test_sliding_3sigma_backend_flows_into_metric_candidates():
    facts = compile_metric_facts(
        _incident(),
        _alarm(),
        service_candidates=["checkout", "frontend"],
        score_backend="sliding_3sigma_window",
    )

    assert facts
    assert facts[0].subject == "checkout"
    assert facts[0].attrs["score_backend"] == "sliding_3sigma_window"
    assert facts[0].attrs["score_components"]["threshold_sigma"] == 3.0


def test_agent_request_for_sliding_3sigma_builds_full_experiment_config(tmp_path):
    result = run_scripted_console(
        "Generate and use a sliding-window 3-sigma AD evidence tool for AIOps25.",
        output_root=tmp_path,
        llm_required=False,
        dry_run=True,
    )

    assert result.intent == "full_benchmark"
    assert result.experiment_config["dataset"] == "aiops25-retained-nonnode"
    assert result.experiment_config["metric_score_backend"] == "sliding_3sigma_window"
    config = json.loads(result.output_root.joinpath(result.artifact_paths["experiment_config"]).read_text(encoding="utf-8"))
    assert config["metric_score_backend"] == "sliding_3sigma_window"

