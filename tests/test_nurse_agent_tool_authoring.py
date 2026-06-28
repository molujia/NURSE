import pytest

from nurse_agent.tools.tool_authoring_tool import run


def test_unapproved_generated_tool_is_advisory_not_ranked(tmp_path):
    result = run(
        {
            "action": "draft",
            "tool_id": "metric_selector_v1",
            "draft_dir": str(tmp_path),
            "candidate_schema_valid": True,
            "smoke_test_passed": False,
            "source": "emit EvidenceCandidate records from metric names",
        }
    )

    assert result["tool_id"] == "metric_selector_v1"
    assert result["approved"] is False
    assert result["ranking_status"] == "advisory_not_ranked"


def test_generated_tool_with_forbidden_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="forbidden inference field"):
        run(
            {
                "action": "draft",
                "tool_id": "bad_tool",
                "draft_dir": str(tmp_path),
                "source": "read answer_key and emit candidates",
            }
        )


def test_tool_approval_requires_contract_and_smoke_checks(tmp_path):
    denied = run(
        {
            "action": "approve",
            "tool_id": "metric_selector_v1",
            "draft_dir": str(tmp_path),
            "contract_check_passed": True,
            "smoke_test_passed": False,
        }
    )
    assert denied["approved"] is False

    approved = run(
        {
            "action": "approve",
            "tool_id": "metric_selector_v1",
            "draft_dir": str(tmp_path),
            "contract_check_passed": True,
            "smoke_test_passed": True,
        }
    )
    assert approved["approved"] is True
    assert approved["ranking_status"] == "approved_for_ranking"


def test_console_tool_approval_uses_existing_draft_checks(tmp_path):
    from nurse_agent.console import handle_tool_authoring_command

    run(
        {
            "action": "draft",
            "tool_id": "metric_selector_v2",
            "draft_dir": str(tmp_path / "agent_tool_drafts"),
            "candidate_schema_valid": True,
            "smoke_test_passed": True,
            "source": "emit safe EvidenceCandidate records",
        }
    )

    approved = handle_tool_authoring_command(
        "approve tool metric_selector_v2",
        output_root=tmp_path,
    )
    assert approved["approved"] is True
    assert approved["ranking_status"] == "approved_for_ranking"

    rejected = handle_tool_authoring_command(
        "reject tool metric_selector_v2",
        output_root=tmp_path,
    )
    assert rejected["approved"] is False
    assert rejected["state"] == "rejected"
