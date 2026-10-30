"""Atomic selector pure metadata tests; no dataset fixture or external network."""

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from tests.test_v2_live_sre_action_decision import context, decision, note
from tests.test_v2_live_sre_action_entries import issued, metadata
from tests.test_v2_live_sre_observable_purpose import declaration, selected


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        pytest.fail("atomic pure test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def api(part="decision"):
    name = f"nurse_rcl.v2.live.ops_sre_atomic_selector_{part}"
    assert util.find_spec(name), "atomic selector capability is absent"
    return import_module(name)


def new_table():
    caps = metadata()
    caps["directed_sources"]["entries"][0].update(
        record_kind="fixed_summary", record_version="ops-symptoms/1"
    )
    caps, book = issued(caps)
    table = api().compile_atomic_entries(caps, book=book, request_index=1, case_id="opaque-case")
    return table, book


@pytest.fixture
def new_setup():
    table, book = new_table()
    entry = selected(table)
    ctx = context(first=True)
    ctx.update(
        schema_version="nurse.v2.sre-investigation-input/24",
        action_entries=table,
        question_state=api("questions").project_atomic_questions([]),
    )
    ctx["capabilities"]["investigation_situation"]["candidates"] = [
        dict(candidate_service_key="candidate")
    ]
    value = decision(first=True)
    value["actions"] = [entry["entry_ref"]]
    annotation = note()
    annotation.pop("discriminating_value")
    annotation.update(
        basis_refs=[], target_service_keys=["candidate"], observable_test=declaration(entry)
    )
    value["investigation_note"] = annotation
    return value, ctx, book


def test_atomic_native_string_and_context_identity():
    ctx = context(first=True)
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/24"
    value = decision(first=True)
    value.update(actions=["A26"], investigation_note=None)
    before = deepcopy((value, ctx))
    assert api().parse_atomic_decision(canonical_json_bytes(value), context=ctx) == value
    assert (value, ctx) == before
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/23"
    with pytest.raises(ExchangeBoundaryError):
        api().parse_atomic_decision(canonical_json_bytes(value), context=ctx)


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        True,
        "",
        [],
        {"entry_ref": "A26"},
        {"entry_ref": "A26", "action": {}},
        {"entry_ref": "A26", "action_index": 2},
        {"entry_ref": "A26", "investigation_note": None},
        {"entry_ref": "A26", "navigation": None},
        {"entry_ref": "A26", "observable_test": {}},
    ],
)
def test_atomic_shape_rejects_objects_without_repair(value):
    before = deepcopy(value)
    with pytest.raises(ExchangeBoundaryError):
        api().selection_shape(value)
    assert value == before


def test_navigation_shape_keeps_original_question():
    nav = {
        "navigation": {"kind": "case", "cursor": None},
        "question": "Inspect only the published case scope.",
    }
    before = deepcopy(nav)
    api().selection_shape(nav)
    assert nav == before


@pytest.mark.parametrize("actions", [None, ["A1", "A2"], [""]])
def test_atomic_actions_structure_rejected(actions):
    ctx = context(first=True)
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/24"
    value = decision(first=True)
    value["actions"] = actions
    with pytest.raises(ExchangeBoundaryError):
        api().parse_atomic_decision(canonical_json_bytes(value), context=ctx)


def test_atomic_stop_and_old_shape_domain():
    from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import selection_shape

    ctx = context(first=True)
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/24"
    value = decision(first=True, stop=True)
    assert api().parse_atomic_decision(canonical_json_bytes(value), context=ctx) == value
    with pytest.raises(ExchangeBoundaryError):
        selection_shape("A1")


def test_atomic_entries_identity_and_canonical_action():
    table, book = new_table()
    assert table["schema_version"] == "nurse.v2.sre-action-entries/5"
    ref = selected(table)["entry_ref"]
    result = api().resolve_atomic_selection(ref, table=table, book=book, request_index=1)
    assert result["status"] == "accepted" and result["action"]["question"] is None
    assert result["action"]["skill_id"] == "sre-symptoms/19"
    nav = dict(navigation=dict(kind="case", cursor=None), question="Original navigation question.")
    resolved = api().resolve_atomic_selection(nav, table=table, book=book, request_index=1)
    assert resolved["action"]["question"] == nav["question"]
    legacy = deepcopy(table)
    legacy["schema_version"] = "nurse.v2.sre-action-entries/4"
    with pytest.raises(ValueError):
        api().validate_atomic_entries(legacy, book=book, request_index=1)


@pytest.mark.parametrize("reference", [" ", " A1", "S1", "A999999", '{"entry_ref":"A1"}'])
def test_atomic_unavailable_reference_is_not_normalized(new_setup, reference):
    _, ctx, book = new_setup
    before = book.to_state()
    api().selection_shape(reference)
    result = api().resolve_atomic_selection(
        reference, table=ctx["action_entries"], book=book, request_index=1
    )
    assert result["status"] == "rejected" and result["action"] is None
    assert book.to_state() == before


def test_atomic_note_and_projection(new_setup):
    value, ctx, book = new_setup
    entry_ref = value["actions"][0]
    before = deepcopy(value)
    result = api("questions").admit_atomic_note(value, context=ctx, book=book)
    assert result["status"] == "accepted", result
    state = api("questions").apply_atomic_plan(
        ctx["question_state"], result["plan"], decision_index=1
    )
    row = state["questions"][0]
    assert row["state"] == "open"
    assert row["question_origin"] == "program_from_admitted_observable_scope"
    assert row["observable_scope"]["entry_ref"] == entry_ref
    assert value == before
    value["investigation_note"]["discriminating_value"] = "Forbidden second purpose."
    refused = api("questions").admit_atomic_note(value, context=ctx, book=book)
    assert refused["status"] == "rejected" and refused["plan"] is None


def test_atomic_note_capability_before_fixture():
    assert api("questions").project_atomic_questions([])["questions"] == []


@pytest.mark.parametrize("state_name", ["open", "answered", "blocked", "deferred"])
def test_atomic_lifecycle_explicit_only(new_setup, state_name):
    value, ctx, book = new_setup
    plan = api("questions").admit_atomic_note(value, context=ctx, book=book)["plan"]
    saved = api("questions").apply_atomic_plan(ctx["question_state"], plan, decision_index=1)
    before = deepcopy(saved)
    no_update = dict(updates=[], current_question_id=None, action_purpose=None)
    unchanged = api("questions").apply_atomic_plan(saved, no_update, decision_index=2)
    assert unchanged["questions"] == before["questions"]
    row = deepcopy(plan["updates"][0])
    row["state"] = state_name
    if state_name == "answered":
        row.update(remaining_gap="", remaining_facets=[])
    updated = api("questions").apply_atomic_plan(
        saved, dict(updates=[row], current_question_id=None, action_purpose=None), decision_index=2
    )
    assert updated["questions"][0]["state"] == state_name
    assert saved == before


@pytest.mark.parametrize(
    "field", ["question", "question_origin", "observable_scope", "target_service_keys"]
)
def test_atomic_question_identity_cannot_change(new_setup, field):
    value, ctx, book = new_setup
    plan = api("questions").admit_atomic_note(value, context=ctx, book=book)["plan"]
    saved = api("questions").apply_atomic_plan(ctx["question_state"], plan, decision_index=1)
    altered = deepcopy(plan)
    altered.update(current_question_id=None, action_purpose=None)
    altered["updates"][0][field] = (
        None
        if field == "observable_scope"
        else ["other"]
        if field == "target_service_keys"
        else "changed"
    )
    with pytest.raises(ExchangeBoundaryError):
        api("questions").apply_atomic_plan(saved, altered, decision_index=2)


def test_atomic_prose_not_certified_and_null_note_absent(new_setup):
    value, ctx, book = new_setup
    value["investigation_note"]["why_now"] = "This read determines CPU saturation support."
    result = api("questions").admit_atomic_note(value, context=ctx, book=book)
    assert result["status"] == "accepted"
    assert result["plan"]["action_purpose"]["why_now"] == value["investigation_note"]["why_now"]
    assert "CPU saturation" not in result["plan"]["updates"][0]["question"]
    value["investigation_note"] = None
    assert api("questions").admit_atomic_note(value, context=ctx, book=book) == dict(
        status="absent", code=None, plan=None
    )


def test_atomic_ledger_nullable_action_keeps_old_type():
    from pydantic import ValidationError

    from nurse_rcl.v2.live.ops_contracts import OpsEmpiricalInvestigationAction

    props = dict(
        action_index=1,
        skill_id="sre-symptoms/19",
        tool="read_node",
        parameters_json="{}",
        question=None,
    )
    assert api("tools").OpsAtomicSelectorAction(**props).question is None
    with pytest.raises(ValidationError):
        OpsEmpiricalInvestigationAction(**props)


def test_selector_choices_are_derived_and_readonly(new_setup):
    _, ctx, _ = new_setup
    table = ctx["action_entries"]
    before = deepcopy(table)
    choices = api("model").selector_choices(table)
    assert choices == [e["entry_ref"] for e in table["entries"]]
    api("model").validate_selector_choices(choices, table)
    choices.append("A999999")
    assert table == before


def test_atomic_restore_unavailable_string_is_a_binding_rejection(new_setup, monkeypatch):
    """Pure frozen-selection seam; full tool-state restoration is separately data-backed."""
    from types import SimpleNamespace

    from nurse_rcl.v2.live.ops_sre_directed_tools import OpsSreDirectedInvestigationLedger

    module = api("tools")
    _, context_value, book = new_setup
    # Isolate the selection layer, not a mocked successful data/source admission.
    monkeypatch.setattr(OpsSreDirectedInvestigationLedger, "_restore_tool_state", lambda *_: None)
    owner = object.__new__(module.OpsSreAtomicSelectorInvestigationLedger)
    owner._indexed = SimpleNamespace(
        _data=SimpleNamespace(catalog=SimpleNamespace(opaque_case_id="opaque-case"))
    )
    selection = "A999999"
    frozen = dict(
        table=context_value["action_entries"],
        references=book.to_state(),
        request_index=1,
        case_id="opaque-case",
    )
    row = SimpleNamespace(
        index=1,
        selection_json=canonical_json_bytes(selection).decode(),
        frozen_entry_context_json=canonical_json_bytes(frozen).decode(),
        input_sha256=sha256_hex(canonical_json_bytes([selection])),
        action=module.OpsAtomicSelectorAction(
            action_index=1,
            skill_id="sre-symptoms/19",
            tool="read_node",
            parameters_json="{}",
            question=None,
        ),
    )
    state = module.OpsSreAtomicSelectorLedgerState.model_construct(
        records=(row,), case_snapshot_id=book.snapshot
    )
    with pytest.raises(ValueError, match="actual action binding differs"):
        owner._restore_tool_state(state)


@pytest.mark.parametrize("damage", ["reverse", "duplicate", "missing", "extra", "type"])
def test_selector_choices_mismatch_is_rejected(new_setup, damage):
    _, ctx, _ = new_setup
    table = ctx["action_entries"]
    expected = [e["entry_ref"] for e in table["entries"]]
    value = list(expected)
    if damage == "reverse":
        assert len(value) >= 2, "pure menu fixture must issue two entries"
        value.reverse()
    elif damage == "duplicate":
        value.append(value[0])
    elif damage == "missing":
        value.pop()
    elif damage == "extra":
        value.append("A999999")
    else:
        value[0] = 1
    before = deepcopy(table)
    with pytest.raises(ValueError):
        api("model").validate_selector_choices(value, table)
    assert table == before
