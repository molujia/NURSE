import json

from nurse_agent.orchestrator import NurseAgentOrchestrator


def test_single_case_report_has_fixed_sections(tmp_path):
    orchestrator = NurseAgentOrchestrator(output_root=tmp_path, llm_required=False)

    result = orchestrator.run_script("Run single case 1 for AIOps25.")

    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    markdown = result.report_path.read_text(encoding="utf-8")

    assert report["top_k_ranking"]
    assert report["evidence_paths"]
    assert report["recovery_advice"]
    assert report["incident_prior"]["active"] is False
    assert report["agent_trace_path"].endswith("agent_trace.json")
    assert "## Top-K Ranking" in markdown
    assert "## Evidence Paths" in markdown
    assert "## Recovery Advice" in markdown