import json

from nurse_agent.console import interactive_console, run_scripted_console


def test_console_routes_english_single_case_2_command(tmp_path):
    result = run_scripted_console(
        "Run single case 2 for RCAEval-TT.",
        output_root=tmp_path,
        llm_required=False,
    )

    assert result.intent == "single_case"
    assert result.dataset == "rcaeval-tt"
    assert result.report_path.exists()


def test_console_routes_aiops25_single_case_before_numeric_alias(tmp_path):
    result = run_scripted_console(
        "Run single case 1 for AIOps25.",
        output_root=tmp_path,
        llm_required=False,
    )

    assert result.intent == "single_case"
    assert result.dataset == "aiops25-retained-nonnode"
    assert result.report_path.exists()


def test_interactive_console_persists_follow_up_memory_in_dry_run(monkeypatch, tmp_path):
    commands = iter(
        [
            "Run the full AIOps25 benchmark with AD reuse (alarm_fixed_window).",
            "Run the same configuration on RCAEval-TT.",
            "exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands))

    result = interactive_console(
        output_root=tmp_path,
        session_id="interactive-test",
        llm_required=False,
        dry_run=True,
    )

    assert result is not None
    assert result.experiment_config["dataset"] == "rcaeval-tt"
    assert result.experiment_config["metric_score_backend"] == "alarm_fixed_window"
    state_path = tmp_path / "agent_sessions" / "interactive-test" / "session_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_dataset"] == "rcaeval-tt"
    assert state["last_metric_score_backend"] == "alarm_fixed_window"