"""Guard lifecycle over real module aliases, without executing any guarded work."""

import pytest

from tests import sre_feedback_fact_fixtures as guard_helpers


def test_report_guard_covers_stage_alias_and_restores_both_entrypoints():
    from nurse_rcl.v2.live import ops_agent, ops_sre_directed_stage

    original_agent = ops_agent.run_ops_agent
    original_stage_alias = ops_sre_directed_stage.run_ops_agent
    assert original_stage_alias is original_agent
    with pytest.MonkeyPatch.context() as guard:
        fixture = guard_helpers.deny_effects.__wrapped__(guard)
        counters = next(fixture)
        try:
            agent_guarded = ops_agent.run_ops_agent is not original_agent
            stage_alias_guarded = ops_sre_directed_stage.run_ops_agent is not original_stage_alias
            assert all(value == 0 for value in counters.values())
        finally:
            with pytest.raises(StopIteration):
                next(fixture)
    assert ops_agent.run_ops_agent is original_agent
    assert ops_sre_directed_stage.run_ops_agent is original_stage_alias
    assert agent_guarded
    assert stage_alias_guarded
