import json

from nurse_agent.session import NurseAgentSession


def test_session_reuses_previous_backend_for_follow_up_dataset_switch(tmp_path):
    session = NurseAgentSession(
        output_root=tmp_path,
        session_id="unit-session",
        llm_required=False,
    )

    first = session.run_command(
        "Run the full AIOps25 benchmark with AD reuse (alarm_fixed_window).",
        dry_run=True,
    )
    second = session.run_command(
        "Run the same configuration on RCAEval-TT.",
        dry_run=True,
    )

    assert first.experiment_config["dataset"] == "aiops25-retained-nonnode"
    assert first.experiment_config["metric_score_backend"] == "alarm_fixed_window"
    assert second.experiment_config["dataset"] == "rcaeval-tt"
    assert second.experiment_config["metric_score_backend"] == "alarm_fixed_window"


def test_session_memory_persists_between_console_instances(tmp_path):
    session = NurseAgentSession(
        output_root=tmp_path,
        session_id="persistent-session",
        llm_required=False,
    )

    result = session.run_command(
        "Generate and use a sliding-window 3-sigma AD evidence tool for AIOps25.",
        dry_run=True,
    )

    reloaded = NurseAgentSession(
        output_root=tmp_path,
        session_id="persistent-session",
        llm_required=False,
    )
    state = reloaded.state.to_dict()

    assert result.experiment_config["metric_score_backend"] == "sliding_3sigma_window"
    assert state["last_dataset"] == "aiops25-retained-nonnode"
    assert state["last_metric_score_backend"] == "sliding_3sigma_window"
    assert state["last_result"]["run_id"] == result.run_id
    assert reloaded.state_path.exists()
    persisted = json.loads(reloaded.state_path.read_text(encoding="utf-8"))
    assert persisted["session_id"] == "persistent-session"


def test_session_understands_english_public_experiment_prompts(tmp_path):
    session = NurseAgentSession(
        output_root=tmp_path,
        session_id="english-prompts",
        llm_required=False,
    )

    default = session.run_command(
        "Run the full RCAEval-TT benchmark with the default evidence backend.",
        dry_run=True,
    )
    ad_reuse = session.run_command(
        "Run the full RCAEval-TT benchmark with AD reuse (alarm_fixed_window).",
        dry_run=True,
    )
    generated = session.run_command(
        "Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT.",
        dry_run=True,
    )

    assert default.experiment_config["metric_score_backend"] == "evidence_robust"
    assert ad_reuse.experiment_config["metric_score_backend"] == "alarm_fixed_window"
    assert generated.experiment_config["metric_score_backend"] == "sliding_3sigma_window"
    assert generated.experiment_config["dataset"] == "rcaeval-tt"
