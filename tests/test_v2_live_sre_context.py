"""Public intent is not evidence; steps are projections of actual saved work."""

# ruff: noqa: F811 -- imported pytest fixtures
# ruff: noqa: RUF001 -- intentional Chinese public statements

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreInvestigationLedger
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_ledger import action, setup


def api():
    name = "nurse_rcl.v2.live.ops_sre_context"
    assert importlib.util.find_spec(name), "SRE public memory/step projection not implemented"
    return importlib.import_module(name)


def decision(actions=(), *, summary="先核对边界", unresolved=("仍需检查机制",), responds=()):
    return canonical_json_bytes(dict(actions=list(actions),
                                     stop_reason=None if actions else "证据不足",
                                     public_summary=summary, unresolved=list(unresolved),
                                     responds_to=list(responds)))


def project(ledger, *, decisions=(), index=1):
    result = api().sre_investigation_input(
        capabilities=ledger.capabilities(), records=ledger.records, decisions=decisions,
        decision_index=index, make_wire=lambda view: {"context": view})
    assert result.status == "ready"
    return json.loads(result.view_json)


def record(view, response, index=1, action_index=1):
    return OpsDecisionRecord(index=index, context_json=canonical_json_bytes(view).decode(),
                             response_json=response.decode(), action_index=action_index)


def test_public_memory_keeps_question_and_unknown_distinct_from_facts(complete, tmp_path):
    options, _ = setup(complete, tmp_path, core=True)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        proposed = action(cursor=None)
        proposed[0]["question"] = "为何不能由顶层支持就确认机制？" + "待核验" * 3000
        response = decision(proposed, summary="只是工作假设，不是已证实根因")
        ledger.dispatch(proposed)
        view = project(ledger, decisions=(record(before, response),), index=2)
        memory = view["public_working_memory"][0]
        assert memory["role"] == "agent_statement"
        assert memory["public_summary"]["text"] == "只是工作假设，不是已证实根因"
        assert memory["questions"][0]["text"].startswith("为何不能")
        assert memory["questions"][0]["omitted_characters"] > 0
        assert view["previous_feedback"] == json.loads(ledger.records[-1].model_view_json)
        assert not view["critical_evidence"], "candidate metadata is not delivered evidence"
        assert "只是工作假设" not in json.dumps(view["critical_evidence"], ensure_ascii=False)
        assert view["unexpanded_critical_overviews"]
        from nurse_rcl.v2.live.ops_sre_encoding import expand

        saved = expand(view)["unexpanded_critical_overviews"][0]
        assert saved["candidate_overviews"]
        assert saved["candidate_overviews"][0]["mechanisms"]["states"]


def test_response_can_only_link_prior_visible_action_cards(complete, tmp_path):
    options, _ = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        proposed = action(cursor=None)
        ledger.dispatch(proposed)
        after = project(ledger, decisions=(record(before, decision(proposed)),), index=2)
        key = after["actions"][0]["visible_card_ids"][0]
        valid = decision(responds=[dict(action_index=1, card_ids=[key])])
        assert api().parse_sre_decision(valid, context=after)["responds_to"]
        for link in (dict(action_index=2, card_ids=[]),
                     dict(action_index=1, card_ids=["not-visible"]),
                     dict(action_index=True, card_ids=[])):
            with pytest.raises(ValueError):
                api().parse_sre_decision(decision(responds=[link]), context=after)


def test_steps_separate_plan_execution_response_and_final_synthesis(complete, tmp_path):
    options, _ = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        first = action(cursor=None)
        r1 = record(before, decision(first))
        ledger.dispatch(first)
        after = project(ledger, decisions=(r1,), index=2)
        ref = after["actions"][0]["visible_card_ids"][0]
        second = action("inspect_candidate", candidate_service_key="not-admitted", cursor=None)
        r2 = record(after, decision(second, summary="根据边界继续调查",
                    responds=[dict(action_index=1, card_ids=[ref])]), index=2, action_index=2)
        ledger.dispatch(second)
        steps = api().investigation_steps(decisions=(r1, r2), records=ledger.records)
        assert [s["step_ref"] for s in steps] == ["s1", "s2"]
        assert steps[0]["planned"]["role"] == "agent_statement"
        assert steps[0]["executed"]["status"] == "completed"
        assert steps[0]["responded_to"][0]["decision_index"] == 2
        assert steps[1]["executed"]["status"] == "rejected"
        assert steps[1]["responded_to"] == []
        assert steps[1]["response_state"] == "not_recorded"
        assert steps[1]["report_synthesis"] == "later_interpretation_requires_fact_refs"
        assert steps[0]["known_at_decision"]["visible_card_ids"] == []


def test_interrupted_and_not_dispatched_are_not_completed_steps(complete, tmp_path, monkeypatch):
    options, tool = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        proposed = action(cursor=None)
        row = record(before, decision(proposed))
        planned = api().investigation_steps(decisions=(row,), records=())
        assert planned[0]["executed"]["status"] == "not_dispatched"
        assert planned[0]["planned"]["requested_action"] == {
            "tool": "inspect_case", "skill_id": "sre-boundary/1", "parameters": {"cursor": None},
            "authority": "unverified_proposal_not_execution",
        }

        def interrupt(*_):
            raise KeyboardInterrupt

        monkeypatch.setattr(tool, "execute", interrupt)
        with pytest.raises(KeyboardInterrupt):
            ledger.dispatch(proposed)
    options, _ = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        steps = api().investigation_steps(decisions=(row,), records=ledger.records)
        assert steps[0]["executed"]["status"] == "interrupted"
        assert steps[0]["executed"]["visible_card_ids"] == []
        assert steps[0]["responded_to"] == []


def test_private_model_memory_is_still_subject_to_egress_scanning(complete, tmp_path):
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    options, tool = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        proposed = action(cursor=None)
        row = record(before, decision(proposed, summary="ts0-mysql-container-kill-9t6n24"))
        ledger.dispatch(proposed)
        view = project(ledger, decisions=(row,), index=2)
        with pytest.raises(ValueError, match="private"):
            _require_outbound(view, case=complete[2].opaque_case_id, snapshot=tool.case_snapshot_id)


def test_directory_card_ids_can_be_acknowledged_without_becoming_health_facts(complete, tmp_path):
    options, _ = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        proposed = action("browse_index", kind="nodes", filters={}, cursor=None)
        ledger.dispatch(proposed)
        view = project(ledger, decisions=(record(before, decision(proposed)),), index=2)
        key = view["previous_feedback"]["page"]["items"][0]["card_id"]
        assert key in view["actions"][0]["visible_card_ids"]
        assert api().parse_sre_decision(decision(responds=[
            dict(action_index=1, card_ids=[key])]), context=view)["responds_to"]
        assert not view["critical_evidence"]


def test_late_core_publication_never_retrofits_early_step_knowledge(complete, tmp_path):
    from nurse_rcl.v2.live.ops_nodes import terminal_nodes

    options, tool = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        first = action(cursor=None)
        r1 = record(before, decision(first))
        ledger.dispatch(first)
        tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=())
        after = project(ledger, decisions=(r1,), index=2)
        r2 = record(after, decision(), index=2, action_index=None)
        steps = api().investigation_steps(decisions=(r1, r2), records=ledger.records)
        assert steps[0]["known_at_decision"]["core_publication_summaries"] == []
        assert steps[1]["known_at_decision"]["core_publication_summaries"] == (
            after["capabilities"]["published_node_summaries"])
        assert steps[0]["core_actions"] == "independent_not_agent_scheduled"
        assert steps[0]["responded_to"] == []


def test_partial_section_unknown_remains_after_a_later_unrelated_action(complete, tmp_path):
    options, tool = setup(complete, tmp_path, core=True)
    key = next(k for k, c in tool.cards.items()
               if c.node == "mechanisms" and c.truth_value == "unknown")
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        first = action("read_node", card_id=key, section="limitations", cursor=None)
        r1 = record(before, decision(first))
        ledger.dispatch(first)
        actual = json.loads(ledger.records[-1].model_view_json)
        after = project(ledger, decisions=(r1,), index=2)
        second = action("browse_index", kind="candidates", filters={}, cursor=None)
        r2 = record(after, decision(second), index=2, action_index=2)
        ledger.dispatch(second)
        latest = project(ledger, decisions=(r1, r2), index=3)
        retained = next((r for r in latest["critical_evidence"] if r["card_id"] == key), None)
        assert retained is not None, "already delivered unknown must remain visible"
        assert retained["record"]["representation"] == "delivered_section_not_full_judgment"
        assert retained["record"]["payload"]["delivered_view"] == actual
        assert "conditions" not in retained["record"]["payload"]
