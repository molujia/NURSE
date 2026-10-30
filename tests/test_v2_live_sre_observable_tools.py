"""New scope identity over the real ledger, with no provider or paid input."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import importlib
import importlib.util
import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries
from nurse_rcl.v2.live.ops_sre_references import encode_references
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import context, decision, note
from tests.test_v2_live_sre_observable_purpose import api, declaration, selected


def tools():
    name = "nurse_rcl.v2.live.ops_sre_observable_tools"
    assert importlib.util.find_spec(name), "new observable toolbox/ledger missing"
    return importlib.import_module(name)


def opened(complete):
    tool = tools().OpsSreObservableToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=complete[2].session.steps)
    book = SreActionReferenceBook(snapshot=tool.case_snapshot_id, run_id="3" * 64)
    case = complete[0].catalog.opaque_case_id
    caps = tool.capabilities()
    encode_references(caps, book=book, visible_at=1, case_id=case)
    table = api().compile_observable_entries(caps, book=book, request_index=1, case_id=case)
    return tool, book, table, case


def test_new_skills_contract_and_strict_old_new_entry_rejection(complete):
    tool, book, table, _ = opened(complete)
    assert tool.profile_id == "hermes-sre-tools/20" and tool.view_schema.endswith("/15")
    assert all(e["action"]["skill_id"].endswith("/16") for e in table["entries"])
    tools().validate_observable_entries(table, book=book, request_index=1)
    with pytest.raises(ValueError, match="binding"):
        validate_action_entries(table, book=book, request_index=1)
    changed = deepcopy(table)
    selected(changed)["observable_contract"]["metric_name"] = "unsupplied_metric"
    with pytest.raises(ValueError, match="binding"):
        tools().validate_observable_entries(changed, book=book, request_index=1)


def test_rejected_typed_purpose_does_not_cancel_real_action_and_fresh_restore(complete, tmp_path):
    tool, book, table, case = opened(complete)
    entry = selected(table)
    ctx = context(first=True)
    ctx.update(
        schema_version="nurse.v2.sre-investigation-input/21",
        action_entries=table,
        question_state=api().project_observable_questions([]),
    )
    ctx["capabilities"]["investigation_situation"]["candidates"] = [
        dict(candidate_service_key=s) for s in entry["observable_contract"]["binding"]["targets"]
    ]
    proposal = decision(first=True)
    proposal["actions"] = [dict(entry_ref=entry["entry_ref"], question="Read the supplied metric.")]
    claim = declaration(entry)
    claim["metric_name"] = "wrong_metric"
    proposal["investigation_note"] = dict(
        note(),
        observable_test=claim,
        basis_refs=[],
        target_service_keys=entry["observable_contract"]["binding"]["targets"],
    )
    before = canonical_json_bytes(proposal)
    assert api().admit_observable_note(proposal, context=ctx)["code"] == "objective_not_supplied"
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))
    with tools().OpsSreObservableInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        ledger.dispatch(proposal["actions"])
        assert (
            json.loads((tmp_path / "investigation/state.json").read_bytes())["schema_version"]
            == "nurse.v2.sre-investigation-ledger/21"
        )
        record = ledger.records[0]
        assert record.status == "completed" and record.action.tool == "inspect_sources"
        assert json.loads(record.selection_json) == proposal["actions"][0]
        view = expand(json.loads(record.model_view_json))
        assert view["page"]["records"][0]["card_id"] == entry["source_binding"]["card_id"]
    fresh = opened(complete)[0]
    with tools().OpsSreObservableInvestigationLedger(
        root=tmp_path, toolbox=fresh, code_sha256="2" * 64
    ) as restored:
        assert restored.used == 1 and restored.records[0] == record
    assert canonical_json_bytes(proposal) == before
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core
