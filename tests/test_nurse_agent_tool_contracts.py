import pytest

from nurse_agent.state import AgentTrace
from nurse_agent.tool_registry import build_default_registry


def test_tool_registry_rejects_forbidden_inference_fields():
    registry = build_default_registry()
    tool = registry.get("evidence_admission")

    with pytest.raises(ValueError, match="forbidden inference field"):
        tool.validate_input({"case_id": "case-1", "answer_key": "checkout"})


def test_tool_result_rejects_forbidden_inference_fields():
    registry = build_default_registry()
    tool = registry.get("report_writer")

    with pytest.raises(ValueError, match="forbidden inference field"):
        tool.validate_output({"artifact_paths": {}, "root_cause": "checkout"})


def test_tool_call_trace_records_tool_name_and_artifact_paths():
    trace = AgentTrace(run_id="run-1", user_request="Run single case 1 for AIOps25.")

    trace.record_tool_call(
        "evidence_ranker",
        {"case_id": "case-1"},
        {"artifact_paths": {"prediction": "predictions/case-1.json"}},
    )

    payload = trace.to_dict()
    assert payload["tool_calls"][0]["tool_name"] == "evidence_ranker"
    assert (
        payload["tool_calls"][0]["output_summary"]["artifact_paths"]["prediction"]
        == "predictions/case-1.json"
    )


def test_default_registry_exposes_core_nurse_modules():
    registry = build_default_registry()

    assert registry.names()[:5] == [
        "alarm_context_adapter",
        "evidence_candidates_builder",
        "evidence_admission",
        "evidence_graph_builder",
        "evidence_ranker",
    ]
    assert "incident_prior" in registry.names()
    assert "recovery_advice" in registry.names()
    assert "experiment_runner" in registry.names()