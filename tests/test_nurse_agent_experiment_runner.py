import json

from nurse_agent.console import run_scripted_console


def test_aiops_alarm_fixed_window_request_builds_experiment_config(tmp_path):
    result = run_scripted_console(
        "Run the full AIOps25 benchmark with AD reuse (alarm_fixed_window).",
        output_root=tmp_path,
        llm_required=False,
        dry_run=True,
    )

    assert result.intent == "full_benchmark"
    assert result.experiment_config["dataset"] == "aiops25-retained-nonnode"
    assert result.experiment_config["metric_score_backend"] == "alarm_fixed_window"
    assert result.experiment_config["dry_run"] is True
    config_path = result.artifact_paths["experiment_config"]
    config = json.loads(result.output_root.joinpath(config_path).read_text(encoding="utf-8"))
    assert config["dataset"] == "aiops25-retained-nonnode"
    assert config["metric_score_backend"] == "alarm_fixed_window"