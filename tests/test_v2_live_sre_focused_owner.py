"""Real synthetic source publication -> browse -> next frozen action -> resume."""

# ruff: noqa: F811 -- imported generated fixtures.

import json
from types import SimpleNamespace

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision


@pytest.fixture
def owner_errors(monkeypatch):
    """Expose a swallowed synthetic owner failure without changing its handling."""
    import traceback

    from nurse_rcl.v2.live import ops_agent, ops_sre_context

    captured = []
    original = ops_agent.failure_detail

    def retain(stage, error):
        captured.append(
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
        return original(stage, error)

    monkeypatch.setattr(ops_agent, "failure_detail", retain)
    monkeypatch.setattr(ops_sre_context, "failure_detail", retain)
    return captured


def opened(complete):
    from nurse_rcl.v2.live.ops_nodes import terminal_nodes
    from nurse_rcl.v2.live.ops_sre_focused_tools import OpsSreFocusedToolbox

    tool = OpsSreFocusedToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=complete[2].session.steps)
    return tool


def test_catalog_next_frozen_read_real_delivery_and_same_ceiling_resume(
    complete, tmp_path, monkeypatch, owner_errors
):
    from nurse_rcl.v2.live.ops_agent import run_ops_agent
    from nurse_rcl.v2.live.ops_agent_state import load_conversation
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
    from nurse_rcl.v2.live.ops_sre_encoding import expand
    from nurse_rcl.v2.live.ops_sre_focused_tools import (
        OpsSreFocusedInvestigationLedger,
        OpsSreFocusedLedgerState,
    )
    from nurse_rcl.v2.live.ops_sre_stage_handoff import retained_sources

    probe = opened(complete)
    probe._refresh_semantic()
    assert len(probe._core_cards) > 16
    visible = {r["card_id"] for r in probe.capabilities()["directed_sources"]["entries"]}
    target = next(
        row
        for row in probe._semantic.values()
        if row["node"] == "symptoms"
        and row["card_id"] not in visible
        and row["payload"].get("kind") == "prepared_summary"
    )
    candidate, metric = target["targets"][0], target["payload"]["metric_name"]
    before = canonical_json_bytes(complete[2].model_dump(mode="json"))
    sent, contexts = [], []

    def forbidden(*_args, **_kwargs):
        pytest.fail("focused source navigation/read replayed a measurement")

    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._measure_values", forbidden)
    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._consume_blocks", forbidden)

    def send(index, wire):
        sent.append(index)
        context = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=probe.case_snapshot_id,
            decision_index=index,
        )
        contexts.append(context)
        value = decision(first=index == 1)
        value["investigation_note"] = None
        if index == 1:
            # The original-name order is public metadata. It identifies an issued
            # candidate only; no source read occurs in this deterministic provider.
            originals = sorted(probe._by_candidate)
            short = context["capabilities"]["candidate_entry"]["items"][originals.index(candidate)][
                "candidate_service_key"
            ]
            value["actions"] = [
                dict(
                    navigation=dict(
                        kind="catalog",
                        catalog_kind="evidence",
                        filters=dict(
                            candidate_service_keys=[short],
                            metric_name=metric,
                            record_kind="prepared_summary",
                        ),
                        cursor=None,
                    ),
                    question="Find exact source.",
                )
            ]
        else:
            targets = context["capabilities"]["evidence_catalog_targets"]
            assert len(targets) == 1
            chosen = next(
                e
                for e in context["action_entries"]["entries"]
                if e["action"]["tool"] == "read_node"
                and e["action"]["parameters"]["card_id"] == targets[0]["card_id"]
            )
            assert context["action_entries"]["entries"][0] == chosen
            value["actions"] = [chosen["entry_ref"]]
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            latest = context["investigation_progress"]["latest_actual_result"]
            value["latest_result_update"] = dict(
                action_index=latest["action_index"],
                content_relation=latest["content_relation"],
                summary="Catalog navigation listed source identities.",
            )
        return canonical_json_bytes(value)

    def run(limit, callback):
        return run_ops_agent(
            root=tmp_path,
            toolbox=opened(complete),
            code_sha256="b" * 64,
            complete=callback,
            empirical=True,
            sre=True,
            sre_profile="hermes-sre-tools/25",
            opaque_case_id=complete[0].catalog.opaque_case_id,
            stage_decision_limit=limit,
        )

    first = run(1, send)
    assert first.termination == "stage_pause" and first.action_count == 1, owner_errors
    second = run(2, send)
    assert second.termination == "stage_pause" and second.action_count == 2, owner_errors
    state_path = tmp_path / "investigation" / "state.json"
    saved_state = state_path.read_bytes()
    ledger = OpsSreFocusedLedgerState.model_validate_json(saved_state)
    assert set(ledger.navigation) == {"storage_encoding", "source", "base"}
    assert {"source_registry", "evidence_catalog"} <= set(ledger.navigation["base"])
    # Use the real owner's pooled checkpoint at the pure report boundary. The
    # exact prepared-summary read needs no retained brief, source load or owner.
    with monkeypatch.context() as pure_report:
        pure_report.setattr("nurse_rcl.v2.live.ops_sre_score_lineage.load_score_lineage", forbidden)
        pure_report.setattr("nurse_rcl.v2.live.ops_sre_source_checkpoint.admit_source", forbidden)
        pure_report.setattr(OpsSreFocusedInvestigationLedger, "_restore_tool_state", forbidden)
        assert (
            retained_sources(
                stage=None,
                ledger=ledger,
                core=SimpleNamespace(
                    identity=SimpleNamespace(case_snapshot_id=probe.case_snapshot_id)
                ),
                focused=True,
            )
            == ()
        )
    assert state_path.read_bytes() == saved_state
    record = json.loads(saved_state)["records"][-1]
    view = expand(json.loads(record["model_view_json"]))
    assert view["semantic_record"]["card_id"] == target["card_id"]
    assert view["semantic_record"]["source_attribution"]["binding"]["status"] == "bound"
    assert view["semantic_record"]["payload"] == target["payload"]
    conversation = load_conversation(
        tmp_path / "investigation" / "conversation.json",
        snapshot=probe.case_snapshot_id,
        code="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/25",
        opaque_case_id=complete[0].catalog.opaque_case_id,
    )
    assert conversation.schema_version == "nurse.v2.sre-conversation/22"
    saved = state_path.read_bytes()
    assert run(2, forbidden).action_count == 2
    assert state_path.read_bytes() == saved
    assert sent == [1, 2]
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == before


def test_new_direct_provider_cannot_enter_bounded_focused_profile(tmp_path):
    from nurse_rcl.v2.live.ops_provider import OpsLiveProvider

    with pytest.raises(ValueError, match="focused|bounded"):
        OpsLiveProvider(
            root=tmp_path,
            code_sha256="1" * 64,
            execution_sha256="2" * 64,
            approved_execution_sha256="2" * 64,
            opaque_case_id="opq_" + "1" * 32,
            case_snapshot_id="3" * 64,
            empirical=True,
            sre=True,
            sre_profile="hermes-sre-tools/25",
        )


def test_frozen_inspection_attribution_binds_the_actual_supplied_publication_role(complete):
    from copy import deepcopy

    from nurse_rcl.artifacts import sha256_hex
    from nurse_rcl.v2.live.ops_sre_encoding import expand

    tool = opened(complete)
    tool._refresh_semantic()
    row = deepcopy(
        next(r for r in tool._semantic.values() if r["payload"].get("kind") == "prepared_summary")
    )
    original = canonical_json_bytes(row["payload"])
    # Inspection prefixes freeze publication metadata independently from later
    # cache refreshes; their attribution must describe the row actually supplied.
    row.update(source_role="core_published", published_by_core=True)
    expected = {k: v for k, v in row.items() if k != "source_attribution"}
    delivered = expand(tool._inspection_record(row))
    assert delivered["source_attribution"]["binding"]["record_sha256"] == sha256_hex(
        canonical_json_bytes(expected)
    )
    assert canonical_json_bytes(delivered["payload"]) == original


def test_exact_repeat_guard_preserves_actual_enriched_focused_read(
    complete, tmp_path, monkeypatch, owner_errors
):
    from nurse_rcl.artifacts import sha256_hex
    from nurse_rcl.v2.live.ops_agent import run_ops_agent
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
    from nurse_rcl.v2.live.ops_sre_d21_reuse import exact_delivered_ledger_read
    from nurse_rcl.v2.live.ops_sre_encoding import expand
    from nurse_rcl.v2.live.ops_sre_focused_tools import (
        OpsSreFocusedInvestigationLedger,
        OpsSreFocusedLedgerState,
    )

    tool = opened(complete)
    core_before = canonical_json_bytes(complete[2].model_dump(mode="json"))
    sent = []

    def no_measurement(*_args, **_kwargs):
        pytest.fail("exact saved read guard performed another measurement")

    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._measure_values", no_measurement)
    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._consume_blocks", no_measurement)

    def send(index, wire):
        sent.append(index)
        context = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=tool.case_snapshot_id,
            decision_index=index,
        )
        entry = next(
            e for e in context["action_entries"]["entries"] if e["action"]["tool"] == "read_node"
        )
        response = decision(first=True)
        response.update(actions=[entry["entry_ref"]], investigation_note=None)
        return canonical_json_bytes(response)

    result = run_ops_agent(
        root=tmp_path,
        toolbox=tool,
        code_sha256="b" * 64,
        complete=send,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/25",
        opaque_case_id=complete[0].catalog.opaque_case_id,
        stage_decision_limit=1,
    )
    assert result.termination == "stage_pause" and result.action_count == 1 and sent == [1], (
        owner_errors
    )
    path = tmp_path / "investigation/state.json"
    original = path.read_bytes()
    ledger = OpsSreFocusedLedgerState.model_validate_json(original)
    record = ledger.records[0]
    delivered = expand(json.loads(record.model_view_json))["semantic_record"]
    assert "source_attribution" in delivered
    parameters = json.loads(record.action.parameters_json)
    proposal = dict(
        actions=[
            dict(
                tool="read_node",
                parameters=parameters,
                skill_id=record.action.skill_id,
                question="Check the identical delivered record.",
            )
        ]
    )

    # The guard authenticates the actual enriched delivery without dispatch or restoration.
    proof = exact_delivered_ledger_read(proposal, ledger.records, toolbox=tool)
    assert proof is not None, "focused attribution made the exact delivered read look new"
    assert proof["prior_action_index"] == 1
    assert proof["record_sha256"] == sha256_hex(canonical_json_bytes(delivered))
    assert proof["source_refs"] == delivered["source_refs"]
    owner = OpsSreFocusedInvestigationLedger(root=tmp_path, toolbox=tool, code_sha256="b" * 64)
    assert canonical_json_bytes(owner._current(parameters["card_id"])) == canonical_json_bytes(
        delivered
    )
    forged_view = expand(json.loads(record.model_view_json))
    forged_record = forged_view["semantic_record"]
    forged_record["source_attribution"]["fields"]["modality"]["value"] = "logs"
    # Recompute derived delivery bookkeeping so this specifically tests actual
    # source attribution admission, rather than only detecting an old digest.
    forged_view["delivery_feedback"]["records"][0]["record_sha256"] = sha256_hex(
        canonical_json_bytes(forged_record)
    )
    forged = record.model_copy(update={"model_view_json": tool._serialize_view(forged_view)})
    with pytest.raises(ValueError, match="attribution|source"):
        owner._restore_tool_state(ledger.model_copy(update={"records": (forged,)}))
    assert path.read_bytes() == original
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core_before
