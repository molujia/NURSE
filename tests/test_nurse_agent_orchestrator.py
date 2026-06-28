import json

from nurse_agent.orchestrator import NurseAgentOrchestrator


def test_single_case_request_calls_rcl_tools_and_writes_report(tmp_path):
    orchestrator = NurseAgentOrchestrator(output_root=tmp_path, llm_required=False)

    result = orchestrator.run_script("Run single case 2 for RCAEval-TT.")

    tool_names = [call.tool_name for call in result.trace.tool_calls]
    assert tool_names[:7] == [
        "alarm_context_adapter",
        "evidence_candidates_builder",
        "evidence_admission",
        "evidence_graph_builder",
        "evidence_ranker",
        "incident_prior",
        "rcl_context_packager",
    ]
    assert result.report_path is not None and result.report_path.exists()
    assert result.report_json_path is not None and result.report_json_path.exists()

    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert report["top_k_ranking"]
    assert "evidence_paths" in report
    assert report["recovery_advice"]
    assert report["incident_prior"]["active"] is False
    assert result.trace.to_dict()["tool_calls"][0]["tool_name"] == "alarm_context_adapter"