"""Directed source selection is a tool receipt, not an ARK or whole-owner run."""

# ruff: noqa: F811 -- imported pytest fixtures.

import importlib
import importlib.util
import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_tools import card_id
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_d15_rejections import bind_visible


def api():
    name = "nurse_rcl.v2.live.ops_sre_directed_tools"
    assert importlib.util.find_spec(name), "directed source toolbox is missing"
    return importlib.import_module(name)


def opened(complete):
    tool = api().OpsSreDirectedToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]),
                    steps=complete[2].session.steps)
    bind_visible(tool)
    return tool


def sources(tool):
    tool._refresh_semantic()
    return [row for row in tool._semantic.values() if row["node"] == "symptoms"
            and row["payload"].get("kind") == "prepared_summary"]


def selection(tool, count=2):
    rows = sources(tool)
    left = rows[0]
    right = next(r for r in rows if r["targets"] != left["targets"])
    chosen = [left, right][:count]
    return dict(card_ids=[r["card_id"] for r in chosen],
                candidate_service_keys=[r["targets"][0] for r in chosen]), chosen


def test_registered_directed_entry_is_new_profile_only(complete):
    tool = opened(complete)
    capabilities = tool.capabilities()
    assert tool.profile_id == "hermes-sre-tools/18"
    assert capabilities["tools"]["inspect_sources"] == ["card_ids", "candidate_service_keys"]
    assert capabilities["parameter_contracts"]["inspect_sources"]
    assert "inspect_sources" in tool.skill_tools["sre-symptoms/14"]
    navigation = capabilities["directed_sources"]
    assert navigation["authority"] == "navigation_not_delivered_evidence"
    assert len(navigation["entries"]) <= 16
    assert all(row["node"] == "symptoms" for row in navigation["entries"])
    from nurse_rcl.v2.live.ops_sre_action_tools import OpsSreActionToolbox

    old = OpsSreActionToolbox(data=complete[0], block_root=complete[1])
    assert "inspect_sources" not in old.tool_parameters
    assert "directed_sources" not in old.capabilities()


def test_navigation_balances_families_without_using_scores_or_fault_values(complete):
    from nurse_rcl.v2.live.ops_sre_directed_sources import directed_source_navigation

    tool = opened(complete)
    params, originals = selection(tool)
    rows = []
    for original in originals:
        for family, count in (("cpu", 20), ("lifecycle", 1)):
            for index in range(count):
                row = deepcopy(original)
                row["card_id"] = f"fixture_{original['targets'][0]}_{family}_{index}"
                row["payload"].update(signal_family=family, metric_name=f"fixture_{index:02}")
                rows.append(row)
    navigation = directed_source_navigation(rows, params["candidate_service_keys"], limit=4)
    assert len(navigation["entries"]) == 4
    for candidate in params["candidate_service_keys"]:
        assert {row["signal_family"] for row in navigation["entries"]
                if row["targets"] == [candidate]} == {"cpu", "lifecycle"}
    assert navigation["omitted"] == len(rows) - 4


@pytest.mark.parametrize("count", [1, 2])
def test_selected_receipt_preserves_values_windows_and_coverage_without_measurement(
    complete, count,
):
    tool = opened(complete)
    params, expected = selection(tool, count)
    before = (canonical_json_bytes(complete[2].model_dump(mode="json")),
              dict(tool._feedback), dict(tool._core_steps))
    result = tool.execute("inspect_sources", params)
    view = expand(json.loads(result.model_view_json))
    rows = view["page"]["records"]
    assert [r["card_id"] for r in rows] == params["card_ids"]
    assert view["status"] == "completed" and view["page"]["next_cursor"] is None
    assert view["page"]["count"] == count
    for actual, original in zip(rows, expected, strict=True):
        assert canonical_json_bytes(actual) == canonical_json_bytes(original)
        assert actual["payload"]["baseline"] == original["payload"]["baseline"]
        assert actual["payload"]["fault_window"] == original["payload"]["fault_window"]
        assert actual["coverage"] == original["coverage"]
        assert actual["score_authority"] is False
    assert view["source_relation"]["authority"] == "source_relationship_not_new_evidence"
    assert result.origin == "read_card"
    assert before == (canonical_json_bytes(complete[2].model_dump(mode="json")),
                      dict(tool._feedback), dict(tool._core_steps))


def test_same_metric_pair_and_missing_side_are_not_invented(complete):
    tool = opened(complete)
    params, rows = selection(tool)
    left = rows[0]
    matching = next(r for r in sources(tool) if r["targets"] != left["targets"]
                    and r["payload"]["metric_name"] == left["payload"]["metric_name"]
                    and r["payload"]["signal_family"] == left["payload"]["signal_family"])
    params["card_ids"] = [left["card_id"], matching["card_id"]]
    params["candidate_service_keys"] = [left["targets"][0], matching["targets"][0]]
    view = expand(json.loads(tool.execute("inspect_sources", params).model_view_json))
    assert view["source_relation"]["status"] == "same_metric_family_windows"
    params["card_ids"] = [left["card_id"]]
    view = expand(json.loads(tool.execute("inspect_sources", params).model_view_json))
    assert view["source_relation"]["status"] == "missing_candidate_source"
    assert view["source_relation"]["missing_candidate_service_keys"] == matching["targets"]
    assert len(view["page"]["records"]) == 1


def test_different_metric_is_delivered_but_not_a_matched_control(complete):
    tool = opened(complete)
    params, rows = selection(tool)
    left = rows[0]
    # The small eight-service fixture has one metric; publish a separate, explicit
    # fixture source instead of pretending the real material contained a second metric.
    original = tool.cards[rows[1]["card_id"]]
    body = json.loads(original.payload_json)
    body["metric_name"] = "different_fixture_metric"
    changed = original.model_copy(update={"payload_json": canonical_json_bytes(body).decode()})
    tool.admit_core(nodes=(changed,), steps=())
    bind_visible(tool)
    tool._refresh_semantic()
    mismatch = tool._semantic[card_id(changed)]
    params["card_ids"] = [left["card_id"], mismatch["card_id"]]
    params["candidate_service_keys"] = [left["targets"][0], mismatch["targets"][0]]
    view = expand(json.loads(tool.execute("inspect_sources", params).model_view_json))
    assert view["source_relation"]["status"] == "metric_family_or_window_mismatch"
    assert len(view["page"]["records"]) == 2


@pytest.mark.parametrize("change", ["unissued", "future", "foreign", "target", "kind"])
def test_hidden_future_foreign_wrong_target_and_wrong_node_stay_rejected(complete, change):
    tool = opened(complete)
    params, rows = selection(tool, 1)
    if change in {"unissued", "future"}:
        bind_visible(tool, hidden=(params["card_ids"][0],))
    elif change == "foreign":
        params["card_ids"] = ["ops_card_" + "f" * 64]
    elif change == "target":
        params["candidate_service_keys"] = [next(c for c in tool._by_candidate
                                                if c not in rows[0]["targets"])]
    else:
        params["card_ids"] = [next(k for k, c in tool.cards.items() if c.node == "scoring")]
    with pytest.raises(ValueError) as caught:
        tool.execute("inspect_sources", params)
    assert caught.value.diagnostic.reason_code in {
        "reference_not_available", "invalid_parameters", "card_kind_mismatch",
    }


@pytest.mark.parametrize("params", [None, {}, dict(card_ids=[], candidate_service_keys=[]),
    dict(card_ids=["x"] * 3, candidate_service_keys=["y"]),
    dict(card_ids=["x"], candidate_service_keys=["y"] * 3),
    dict(card_ids="x", candidate_service_keys=["y"]),
    dict(card_ids=["x"], candidate_service_keys=["y"], cursor=None)])
def test_parameter_bounds_do_not_accept_paths_queries_or_capacity_expansion(complete, params):
    with pytest.raises(ValueError) as caught:
        opened(complete).execute("inspect_sources", params)
    assert caught.value.diagnostic.reason_code == "invalid_parameters"


def test_duplicate_sources_are_not_double_evidence(complete):
    tool = opened(complete)
    params, _ = selection(tool, 1)
    params["card_ids"] *= 2
    with pytest.raises(ValueError) as caught:
        tool.execute("inspect_sources", params)
    assert caught.value.diagnostic.reason_code == "invalid_parameters"


def test_navigation_retains_all_candidates_access_and_measured_request_links(complete):
    tool = opened(complete)
    params, _ = selection(tool, 1)
    candidate = params["candidate_service_keys"][0]
    view = expand(json.loads(tool.execute("inspect_candidate", dict(
        candidate_service_key=candidate, focus="symptoms", cursor=None,
    )).model_view_json))
    navigation = view["directed_sources"]
    assert all(r["targets"] == [candidate] for r in navigation["entries"])
    assert navigation["entries"]
    assert all(r["entry"]["tool"] == "inspect_sources" for r in navigation["entries"])
    assert any(r.get("request_key") for r in navigation["entries"])
    assert {r["candidate_service_key"] for r in tool.capabilities()[
        "investigation_situation"]["candidates"]} == set(complete[0].catalog.candidate_service_keys)


def test_local_real_ledger_dispatch_saves_and_fresh_reader_restores(complete, tmp_path):
    tool = opened(complete)
    params, _ = selection(tool)
    before = complete[2].model_dump_json()
    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64,
    ) as ledger:
        ledger.dispatch([dict(skill_id="sre-symptoms/14", tool="inspect_sources",
                              parameters=params, question="Which supplied symptom differs?")])
        row = ledger.records[-1]
        assert ledger.used == 1 and row.status == "completed"
        view = expand(json.loads(row.model_view_json))
        assert view["page"]["count"] == 2
        saved = row.model_view_json
    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=opened(complete), code_sha256="2" * 64,
    ) as restored:
        assert restored.used == 1
        assert restored.records[-1].model_view_json == saved
    assert complete[2].model_dump_json() == before


def test_source_view_is_not_an_independent_observation(complete):
    tool = opened(complete)
    params, _ = selection(tool, 1)
    view = expand(json.loads(tool.execute("inspect_sources", params).model_view_json))
    assert view["source_role"] == "aggregation_not_independent_evidence"
    assert "summary_cannot_prove_episode_or_normality" in view["limitations"]
    assert view["page"]["records"][0]["payload"]["episode_status"] == (
        "not_measured_by_summary")
    assert view["score_authority"] is False


def test_same_source_navigation_restores_without_implicit_delivery(complete):
    first = opened(complete)
    state = deepcopy(first.export_navigation())
    restored = opened(complete)
    restored.restore_navigation(state)
    assert restored.capabilities()["directed_sources"] == first.capabilities()["directed_sources"]
    assert restored._feedback == first._feedback


def test_capacity_receipt_cannot_claim_the_full_source_was_delivered(complete, monkeypatch):
    from nurse_rcl.v2.exchange_json import ExchangeLimits
    from nurse_rcl.v2.live import ops_sre_encoding

    tool = opened(complete)
    params, _ = selection(tool)
    monkeypatch.setattr(ops_sre_encoding, "LIMITS", ExchangeLimits(max_bytes=1024))
    result = tool.execute("inspect_sources", params)
    view = json.loads(result.model_view_json)
    assert view["status"] == "resource_limit" and "page" not in view
    assert json.loads(result.evidence.payload_json)["record_ids"] == []


def test_capacity_including_receipt_identity_does_not_grant_source_delivery(complete, monkeypatch):
    from nurse_rcl.v2.exchange_json import ExchangeLimits
    from nurse_rcl.v2.live import ops_sre_encoding

    tool = opened(complete)
    params, _ = selection(tool)
    full = expand(json.loads(tool.execute("inspect_sources", params).model_view_json))
    without_identity = deepcopy(full)
    without_identity.pop("card_id")
    lower = len(canonical_json_bytes(without_identity))
    upper = len(canonical_json_bytes(full))
    assert lower < upper
    monkeypatch.setattr(ops_sre_encoding, "LIMITS", ExchangeLimits(
        max_bytes=(lower + upper) // 2,
    ))
    result = tool.execute("inspect_sources", params)
    assert json.loads(result.model_view_json)["status"] == "resource_limit"
    assert json.loads(result.evidence.payload_json)["record_ids"] == []


def test_full_pair_above_soft_target_has_the_same_delivered_ids_in_receipt(complete):
    tool = opened(complete)
    params, _ = selection(tool)
    result = tool.execute("inspect_sources", params)
    view = expand(json.loads(result.model_view_json))
    assert view["status"] == "completed"
    assert len(canonical_json_bytes(view)) > 4096
    assert json.loads(result.evidence.payload_json)["record_ids"] == params["card_ids"]


def test_completed_receipt_write_failure_preserves_reserved_slot_and_previous_state(
    complete, tmp_path, monkeypatch,
):
    from nurse_rcl.v2.live import ops_sre_checkpoint_ledger

    tool = opened(complete)
    params, _ = selection(tool)
    proposal = [dict(skill_id="sre-symptoms/14", tool="inspect_sources",
                     parameters=params, question="Read the supplied symptom pair.")]
    original_write = ops_sre_checkpoint_ledger._write_checkpoint

    def fail_completed_write(path, raw):
        if json.loads(raw)["records"][-1]["status"] == "completed":
            raise ops_sre_checkpoint_ledger.CheckpointWriteError(OSError("fixture_write_failed"))
        original_write(path, raw)

    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64,
    ) as ledger:
        monkeypatch.setattr(ops_sre_checkpoint_ledger, "_write_checkpoint", fail_completed_write)
        with pytest.raises(ops_sre_checkpoint_ledger.CheckpointWriteError):
            ledger.dispatch(proposal)
        assert ledger.used == 1 and ledger.records[-1].status == "reserved"
        saved = (ledger._root / "state.json").read_bytes()
        assert json.loads(saved)["records"][-1]["status"] == "reserved"
        with pytest.raises(ops_sre_checkpoint_ledger.CheckpointWriteError):
            ledger.dispatch(proposal)
        assert (ledger._root / "state.json").read_bytes() == saved
    monkeypatch.setattr(ops_sre_checkpoint_ledger, "_write_checkpoint", original_write)
    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=opened(complete), code_sha256="2" * 64,
    ) as restored:
        assert restored.used == 1 and restored.records[-1].status == "interrupted"
        assert restored.records[-1].failure_type == "completion_unknown_no_refund"
        assert "page" not in json.loads(restored.records[-1].model_view_json or "{}")


def test_refusal_is_charged_and_survives_fresh_ledger_resume(complete, tmp_path):
    tool = opened(complete)
    params, _ = selection(tool, 1)
    params["card_ids"] = ["ops_card_" + "f" * 64]
    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64,
    ) as ledger:
        ledger.dispatch([dict(skill_id="sre-symptoms/14", tool="inspect_sources",
                              parameters=params, question="Read the requested source.")])
        assert ledger.used == 1 and ledger.records[-1].status == "rejected"
        saved = ledger.records[-1].model_view_json
    with api().OpsSreDirectedInvestigationLedger(
        root=tmp_path, toolbox=opened(complete), code_sha256="2" * 64,
    ) as restored:
        assert restored.used == 1
        assert restored.records[-1].model_view_json == saved
