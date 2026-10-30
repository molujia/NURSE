"""SRE receipts use the existing eight-action budget, with isolated recovery."""

# ruff: noqa: F811 -- imported pytest fixtures

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedInvestigationLedger, OpsIndexedToolbox
from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_checkpoint import pack_navigation, unpack_navigation
from nurse_rcl.v2.live.ops_sre_encoding import expand
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_tools import toolbox


def api():
    name = "nurse_rcl.v2.live.ops_sre_ledger"
    assert importlib.util.find_spec(name), "SRE durable action ledger is missing"
    return importlib.import_module(name)


def action(tool="inspect_case", skill="boundary", **params):
    return [dict(skill_id=f"sre-{skill}/1", tool=tool, parameters=params,
                 question="核对实际证据和未知; 不修改正式定位")]


def setup(complete, tmp_path, *, core=False):
    tool = toolbox(complete, published=core)
    return dict(root=tmp_path, toolbox=tool, code_sha256="1" * 64), tool


def test_sre_page_cursor_and_roles_survive_restart(complete, tmp_path):
    args, tool = setup(complete, tmp_path, core=True)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.capabilities()
        ledger.dispatch(action(cursor=None))
        view = json.loads(ledger.records[-1].model_view_json)
        cursor = view["page"]["next_cursor"]
        expected = tool.execute("inspect_case", {"cursor": cursor}).model_view_json
        saved = json.loads((tmp_path / "investigation/state.json").read_bytes())
        assert saved["investigation_profile_id"] == "hermes-sre-tools/4"
        assert cursor in unpack_navigation(saved["navigation"])["inspection_cursors"]
        assert ledger.records[-1].elapsed_seconds > 0
    args, _ = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        assert ledger.used == 1
        ledger.dispatch(action(cursor=cursor))
        resumed = json.loads(ledger.records[-1].model_view_json)
        for key in ("page", "critical_record_ids", "publication_prefix", "sections_state"):
            assert resumed[key] == json.loads(expected)[key]
        assert resumed["action_index"] == 2


def test_navigation_refusal_and_reuse_all_consume_actions(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path, core=True)
    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._measure_values",
                        lambda *_: pytest.fail("core reuse or navigation must not remeasure"))
    request = complete[2].session.steps[0].feedback.request_key
    with api().OpsSreInvestigationLedger(**args) as ledger:
        assert ledger.dispatch(action("measure_signal", "symptoms",
                                      request_key=request)).origin == "core_reuse"
        for _ in range(7):
            assert ledger.dispatch(action(cursor="forged")).origin == "rejected"
        view = json.loads(ledger.records[-1].model_view_json)
        assert view["parameter_feedback"]["contract"]["cursor"]
        assert "forged" not in ledger.records[-1].model_view_json
        assert ledger.used == 8
        with pytest.raises(ValueError, match="budget"):
            ledger.dispatch(action(cursor=None))
        assert all(r.elapsed_seconds > 0 for r in ledger.records)
    assert not (tmp_path / "core").exists()
    assert tool._core_steps[request] == complete[2].session.steps[0]


def test_sre_and_legacy_profiles_cannot_resume_each_other(complete, tmp_path):
    args, _ = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args):
        pass
    old = OpsIndexedToolbox(data=complete[0], block_root=complete[1], soft_targets=True)
    with pytest.raises(ValueError), OpsIndexedInvestigationLedger(**{**args, "toolbox": old}):
        pass
    path = tmp_path / "old"
    path.mkdir()
    with OpsIndexedInvestigationLedger(**{**args, "root": path, "toolbox": old}):
        pass
    with pytest.raises(ValueError), api().OpsSreInvestigationLedger(**{**args, "root": path}):
        pass


def test_interruption_does_not_refund_action(complete, tmp_path, monkeypatch):
    args, tool = setup(complete, tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tool, "execute", interrupt)
    with api().OpsSreInvestigationLedger(**args) as ledger, pytest.raises(KeyboardInterrupt):
        ledger.dispatch(action(cursor=None))
    args, _ = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        assert ledger.used == 1
        assert ledger.records[0].status == "interrupted"
        ledger.dispatch(action(cursor=None))
        assert ledger.used == 2


def test_resume_rejects_changed_cursor_binding_even_with_new_outer_digest(complete, tmp_path):
    from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

    args, _ = setup(complete, tmp_path, core=True)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.dispatch(action(cursor=None))
    path = tmp_path / "investigation/state.json"
    saved = json.loads(path.read_bytes())
    navigation = unpack_navigation(saved["navigation"])
    token = next(iter(navigation["inspection_cursors"]))
    navigation["inspection_cursors"][token]["offset"] += 1
    saved["navigation"] = pack_navigation(navigation)
    saved["state_sha256"] = sha256_hex(canonical_json_bytes({
        k: v for k, v in saved.items() if k != "state_sha256"}))
    path.write_bytes(canonical_json_bytes(saved))
    args, _ = setup(complete, tmp_path)
    with pytest.raises(ValueError, match="cursor"), api().OpsSreInvestigationLedger(**args):
        pass


def test_supplementary_measurement_cannot_replace_core_query(complete, tmp_path, monkeypatch):
    from nurse_rcl.v2.live import normal_queries

    args, tool = setup(complete, tmp_path)
    data, blocks, formal, _ = complete
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.dispatch(action("measure_signal", "symptoms",
                               request_key=formal.session.steps[0].feedback.request_key))
    measured = []
    original = normal_queries._measure_values

    def observe(source):
        measured.append(source.source_sha256)
        return original(source)

    monkeypatch.setattr(normal_queries, "_measure_values", observe)
    session = normal_queries.run_normal_queries(
        data=data, block_root=blocks, ledger_root=tmp_path / "core-queries")
    assert len(measured) == 8
    assert session == formal.session
    tool.admit_core(nodes=terminal_nodes(data, formal), steps=session.steps)


def test_semantic_expansion_and_directory_bounds_coexist(complete, tmp_path):
    args, tool = setup(complete, tmp_path, core=True)
    key = next(k for k, card in tool.cards.items() if card.node == "mechanisms")
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.dispatch(action("browse_index", kind="nodes", filters={}, cursor=None))
        assert len(ledger.records[-1].model_view_json.encode()) <= 4096
        ledger.dispatch(action("read_node", card_id=key, section="payload", cursor=None))
        view = json.loads(ledger.records[-1].model_view_json)
        assert expand(view)["semantic_record"]["payload"] == json.loads(
            tool.cards[key].payload_json)
        assert view["score_authority"] is False and ledger.used == 2


def test_browse_then_restart_preserves_navigation_without_fact_origin(complete, tmp_path):
    args, _ = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.dispatch(action("browse_index", kind="candidates", filters={}, cursor=None))
    args, _ = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        assert ledger.used == 1
        ledger.dispatch(action(cursor=None))
        assert ledger.records[-1].status == "completed"


def test_read_after_measurement_retains_origin_on_resume(complete, tmp_path):
    from nurse_rcl.v2.live.ops_tools import card_id

    args, tool = setup(complete, tmp_path)
    request = complete[2].session.steps[0].feedback.request_key
    with api().OpsSreInvestigationLedger(**args) as ledger:
        outcome = ledger.dispatch(action("measure_signal", "symptoms", request_key=request))
        key = card_id(outcome.evidence)
        ledger.dispatch(action("read_node", card_id=key, section="payload", cursor=None))
        view = json.loads(ledger.records[-1].model_view_json)
        assert view["semantic_record"]["source_role"] == "investigation_measured"
    args, tool = setup(complete, tmp_path)
    with api().OpsSreInvestigationLedger(**args) as ledger:
        ledger.dispatch(action("read_node", card_id=key, section="payload", cursor=None))
        view = json.loads(ledger.records[-1].model_view_json)
        assert view["semantic_record"]["source_role"] == "investigation_measured"


def test_measurement_view_keeps_positive_episodes_and_predicates(complete, tmp_path):
    args, _ = setup(complete, tmp_path, core=True)
    request = complete[2].session.steps[0].feedback.request_key
    with api().OpsSreInvestigationLedger(**args) as ledger:
        result = ledger.dispatch(action("measure_signal", "symptoms", request_key=request))
        view = json.loads(ledger.records[-1].model_view_json)
        assert expand(view)["semantic_record"]["payload"] == json.loads(
            result.evidence.payload_json)
        assert view["semantic_record"]["source_role"] == "investigation_core_reuse"
        assert view["reused_core_evidence_ids"] == list(result.reused_core_evidence_ids)
