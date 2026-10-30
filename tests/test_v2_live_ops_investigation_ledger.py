"""Every attempted tool round spends its own durable slot, including refusals."""

# ruff: noqa: F811 -- pytest injects the imported complete fixture by name.

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_tools import OpsToolbox
from tests.test_v2_live_normal_queries import prepared_cases  # noqa: F401
from tests.test_v2_live_ops_tools import complete  # noqa: F401


def api():
    name = "nurse_rcl.v2.live.ops_investigation_ledger"
    assert importlib.util.find_spec(name), "missing independent investigation ledger"
    return importlib.import_module(name)


def setup(complete, tmp_path, *, core=False):
    data, blocks, evidence, _ = complete
    tool = OpsToolbox(data=data, block_root=blocks)
    if core:
        tool.admit_core(nodes=terminal_nodes(data, evidence), steps=evidence.session.steps)
    args = dict(root=tmp_path, toolbox=tool, code_sha256="1" * 64)
    return args, tool


def action(tool, **parameters):
    return [{"skill_id": "ops-symptoms/1" if tool == "measure_signal" else "ops-boundary/1",
             "tool": tool, "parameters": parameters, "question": "检查已注册观测"}]


def test_each_rejection_or_read_spends_one_of_eight_slots(complete, tmp_path):
    args, tool = setup(complete, tmp_path)
    key = next(k for k, c in tool.cards.items() if c.node == "boundary")
    with api().OpsInvestigationLedger(**args) as ledger:
        first = ledger.dispatch(action("read_node", card_id=key))
        assert first.origin == "read_card"
        for _ in range(7):
            rejected = ledger.dispatch(action("execute", path="/etc/passwd"))
            assert rejected.origin == "rejected"
        assert ledger.used == 8
        with pytest.raises(ValueError, match="budget"):
            ledger.dispatch(action("read_node", card_id=key))
    state = json.loads((tmp_path / "investigation/state.json").read_bytes())
    assert len(state["records"]) == 8
    assert not (tmp_path / "core").exists()


def test_reservation_is_durable_before_execution_and_reuse_costs(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path, core=True)
    step = complete[2].session.steps[0]
    original = tool.execute

    def observe(*a, **kw):
        state = json.loads((tmp_path / "investigation/state.json").read_bytes())
        assert state["records"][-1]["status"] == "reserved"
        return original(*a, **kw)

    monkeypatch.setattr(tool, "execute", observe)
    with api().OpsInvestigationLedger(**args) as ledger:
        result = ledger.dispatch(action("measure_signal", request_key=step.feedback.request_key))
        assert result.origin == "core_reuse"
        assert result.role == "supplementary"
        assert ledger.used == 1


def test_multi_action_round_is_wholly_refused_and_charged_once(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path)

    def forbidden(*a, **kw):
        pytest.fail("multiple actions must not execute the first or any others")

    monkeypatch.setattr(tool, "execute", forbidden)
    with api().OpsInvestigationLedger(**args) as ledger:
        result = ledger.dispatch(action("read_node", card_id="a") * 2)
        assert result.origin == "rejected"
        assert ledger.used == 1


def test_restart_retains_unfinished_slot_and_does_not_rerun_it(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path)
    key = next(iter(tool.cards))

    def interrupted(*a, **kw):
        raise KeyboardInterrupt("simulated interruption after reservation")

    monkeypatch.setattr(tool, "execute", interrupted)
    with api().OpsInvestigationLedger(**args) as ledger, pytest.raises(KeyboardInterrupt):
        ledger.dispatch(action("read_node", card_id=key))
    resumed, _ = setup(complete, tmp_path)
    with api().OpsInvestigationLedger(**resumed) as ledger:
        assert ledger.used == 1
        assert ledger.records[0].status == "interrupted"
        result = ledger.dispatch(action("read_node", card_id=key))
        assert result.action_index == 2


def test_saved_supplementary_feedback_recovers_without_measurement(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path)
    request = complete[2].session.steps[0].feedback.request_key
    with api().OpsInvestigationLedger(**args) as ledger:
        saved = ledger.dispatch(action("measure_signal", request_key=request))
    resumed, restored = setup(complete, tmp_path)

    def forbidden(*a, **kw):
        pytest.fail("recovery must not remeasure completed investigation")

    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._measure_values", forbidden)
    with api().OpsInvestigationLedger(**resumed) as ledger:
        assert ledger.records[0].result == saved
        assert restored.capabilities()["onset_ids"]
    changed = {**resumed, "code_sha256": "2" * 64}
    with pytest.raises(ValueError, match="identity"), api().OpsInvestigationLedger(**changed):
        pass


def test_investigation_first_cannot_fill_core_cache(complete, tmp_path, monkeypatch):
    from nurse_rcl.v2.live import normal_queries

    args, _ = setup(complete, tmp_path)
    data, blocks, expected, _ = complete
    request = expected.session.steps[0].feedback.request_key
    with api().OpsInvestigationLedger(**args) as ledger:
        ledger.dispatch(action("measure_signal", request_key=request))
    count = []
    original = normal_queries._measure_values

    def measured(source):
        count.append(source.source_sha256)
        return original(source)

    monkeypatch.setattr(normal_queries, "_measure_values", measured)
    session = normal_queries.run_normal_queries(
        data=data, ledger_root=tmp_path / "core-queries", block_root=blocks,
    )
    assert len(count) == 8
    assert session == expected.session
