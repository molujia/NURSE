"""Typed selections use the real action debit and source receipt; no provider."""

# ruff: noqa: F811 -- real small fixtures.

import importlib
import importlib.util
import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_references import encode_references
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import context, decision


def api(name="tools"):
    module = "nurse_rcl.v2.live.ops_sre_entry_" + name
    assert importlib.util.find_spec(module), "typed selection admission is missing"
    return importlib.import_module(module)


def opened(complete):
    tool = api().OpsSreEntryToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=complete[2].session.steps)
    book = SreActionReferenceBook(snapshot=tool.case_snapshot_id, run_id="3" * 64)
    case = complete[0].catalog.opaque_case_id
    capabilities = tool.capabilities()
    encode_references(capabilities, book=book, visible_at=1, case_id=case)
    table = compile_action_entries(capabilities, book=book, request_index=1, case_id=case)
    tool.bind_visible_references(book=book, request_index=1)
    return tool, book, table, case


def symptom(table):
    return next(e for e in table["entries"] if e["action"]["tool"] == "inspect_sources")


def select(entry):
    return dict(entry_ref=entry["entry_ref"], question="Which supplied symptom changed?")


def resolve(value, table, book):
    return api("decision").resolve_entry_selection(value, table=table, book=book, request_index=1)


@pytest.mark.parametrize("reference", ["A999", "S1", "future-action"])
def test_structured_unknown_or_wrong_type_reference_is_semantic_not_parse_retry(reference):
    c, value = context(first=True), decision(first=True)
    c["schema_version"] = "nurse.v2.sre-investigation-input/20"
    value["actions"] = [dict(entry_ref=reference, question="Inspect a symptom.")]
    value["investigation_note"] = "bad annotation must not cancel action"
    assert api("decision").parse_entry_decision(canonical_json_bytes(value), context=c) == value
    assert api("decision").admit_entry_note(value, context=c)["status"] == "rejected"


@pytest.mark.parametrize(
    "action",
    [
        dict(entry_ref=1, question="x"),
        dict(entry_ref="A1", navigation={}, question="x"),
        dict(tool="inspect_sources", question="x"),
        dict(entry_ref="A1", question=""),
        dict(navigation=dict(kind="candidate", cursor=None), question="x"),
    ],
)
def test_malformed_branch_is_format_error(action):
    c, value = context(first=True), decision(first=True)
    c["schema_version"] = "nurse.v2.sre-investigation-input/20"
    value["actions"] = [action]
    with pytest.raises(ExchangeBoundaryError):
        api("decision").parse_entry_decision(canonical_json_bytes(value), context=c)


def test_entry_resolves_uniquely_from_frozen_source_without_mutation(complete):
    _, book, table, _ = opened(complete)
    entry = symptom(table)
    before = deepcopy((table, book.to_state()))
    result = resolve(select(entry), table, book)
    assert result == dict(
        status="accepted",
        reason_code=None,
        action=dict(entry["action"], question=select(entry)["question"]),
    )
    assert (table, book.to_state()) == before
    assert (
        resolve(dict(entry_ref="A999", question="x"), table, book)["reason_code"]
        == "entry_not_available"
    )
    changed = deepcopy(table)
    symptom(changed)["action"]["parameters"]["card_ids"] = ["other"]
    with pytest.raises(ValueError, match="binding"):
        resolve(select(entry), changed, book)


def test_real_dispatch_and_fresh_restore_keep_selection_and_actual_values(complete, tmp_path):
    tool, book, table, case = opened(complete)
    entry = symptom(table)
    before = complete[2].model_dump_json()
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        ledger.dispatch([select(entry)])
        row = ledger.records[-1]
        assert ledger.used == 1 and row.status == "completed"
        assert json.loads(row.selection_json) == select(entry)
        assert row.action.tool == entry["action"]["tool"]
        assert json.loads(row.action.parameters_json) == entry["action"]["parameters"]
        view = expand(json.loads(row.model_view_json))
        original = tool._semantic[entry["action"]["parameters"]["card_ids"][0]]
        assert canonical_json_bytes(view["page"]["records"]) == canonical_json_bytes([original])
        for field in ("baseline", "fault", "baseline_window", "fault_window", "episode_status"):
            if field in original["payload"]:
                assert view["page"]["records"][0]["payload"][field] == original["payload"][field]
        assert view["page"]["records"][0]["coverage"] == original["coverage"]
        saved = row.model_dump(mode="json")
    fresh, _, _, _ = opened(complete)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=fresh, code_sha256="2" * 64
    ) as restored:
        assert restored.records[-1].model_dump(mode="json") == saved
        assert restored.used == 1
    assert complete[2].model_dump_json() == before


def test_unknown_entry_is_charged_without_dispatch_or_automatic_repair(
    complete, tmp_path, monkeypatch
):
    tool, book, table, case = opened(complete)

    def forbidden(*args, **kwargs):
        pytest.fail("semantic rejection must not execute a tool")

    monkeypatch.setattr(tool, "execute", forbidden)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        result = ledger.dispatch([dict(entry_ref="A999", question="Inspect a symptom.")])
        assert ledger.used == 1 and ledger.records[-1].status == "rejected"
        assert ledger.records[-1].selection_reason_code == "entry_not_available"
        assert ledger.records[-1].action is None
        assert result.evidence.availability == "unavailable"
    fresh, _, _, _ = opened(complete)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=fresh, code_sha256="2" * 64
    ) as restored:
        assert restored.used == 1 and restored.records[-1].status == "rejected"


def test_navigation_keeps_arbitrary_candidates_and_pair_endpoints(complete):
    _, book, table, _ = opened(complete)
    services = [r[1] for r in book._rows if r[0] == "service"]
    assert len(services) >= 3
    for nav, expected in (
        (
            dict(
                kind="candidate", candidate_service_key=services[-1], focus="symptoms", cursor=None
            ),
            "inspect_candidate",
        ),
        (
            dict(
                kind="candidate_pair",
                left_candidate_service_key=services[-1],
                right_candidate_service_key=services[-2],
                focus="mechanisms",
                cursor=None,
            ),
            "compare_candidates",
        ),
        (
            dict(
                kind="dependency",
                caller_service_key=services[-1],
                callee_service_key=services[-2],
                cursor=None,
            ),
            "inspect_dependency",
        ),
        (dict(kind="case", cursor=None), "inspect_case"),
        (dict(kind="catalog", catalog_kind="candidates", filters={}, cursor=None), "browse_index"),
    ):
        result = resolve(dict(navigation=nav, question="Check this scope."), table, book)
        assert result["status"] == "accepted" and result["action"]["tool"] == expected
    nav = dict(
        kind="candidate_pair",
        left_candidate_service_key=services[-1],
        right_candidate_service_key=services[-1],
        focus="symptoms",
        cursor=None,
    )
    assert resolve(dict(navigation=nav, question="x"), table, book)["status"] == "rejected"


def test_completed_save_failure_recovers_reserved_without_refund(complete, tmp_path, monkeypatch):
    from nurse_rcl.v2.live import ops_sre_checkpoint_ledger as storage

    tool, book, table, case = opened(complete)
    original = storage._write_checkpoint

    def fail_completed(path, raw):
        if json.loads(raw)["records"][-1]["status"] == "completed":
            raise storage.CheckpointWriteError(OSError("explicit_fixture_failure"))
        original(path, raw)

    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        monkeypatch.setattr(storage, "_write_checkpoint", fail_completed)
        with pytest.raises(storage.CheckpointWriteError):
            ledger.dispatch([select(symptom(table))])
        assert ledger.used == 1 and ledger.records[-1].status == "reserved"
    monkeypatch.setattr(storage, "_write_checkpoint", original)
    fresh, _, _, _ = opened(complete)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=fresh, code_sha256="2" * 64
    ) as restored:
        assert restored.used == 1 and restored.records[-1].status == "interrupted"
        assert restored.records[-1].failure_type == "completion_unknown_no_refund"


def test_exact_read_uses_actual_source_not_action_name_and_gate_does_not_dispatch(
    complete, tmp_path, monkeypatch
):
    tool, book, table, case = opened(complete)
    entry = next(e for e in table["entries"] if e["action"]["tool"] == "read_node")
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        ledger.dispatch([select(entry)])
        assert ledger.records[-1].status == "completed"
        book.issue("digest", "f" * 64, visible_at=2)
        # Isolate the same real supplied source; a 16-item menu may legitimately
        # omit a previous action, so the gate fixture must not assume it is shown.
        supplied = tool.capabilities(ledger.records)
        supplied["directed_sources"]["entries"] = [
            r
            for r in supplied["directed_sources"]["entries"]
            if r["card_id"] == entry["action"]["parameters"]["card_id"]
        ]
        supplied["investigation_routes"]["entries"] = []
        later = compile_action_entries(supplied, book=book, request_index=2, case_id=case)
        matching = next(e for e in later["entries"] if e["action"] == entry["action"])
        assert matching["entry_ref"] != entry["entry_ref"]
        before = (ledger.used, canonical_json_bytes(tool.export_navigation()))

        def forbidden(*args, **kwargs):
            pytest.fail("exact precheck must not reserve, dispatch or replay")

        monkeypatch.setattr(tool, "execute", forbidden)
        gate = getattr(api("decision"), "exact_entry_delivered_read", None)
        assert callable(gate), "canonical action duplicate gate is missing"
        proof = gate(
            dict(actions=[select(matching)]),
            table=later,
            book=book,
            request_index=2,
            records=ledger.records,
            toolbox=tool,
        )
        assert proof["prior_action_index"] == 1
        assert proof["card_id"] == entry["action"]["parameters"]["card_id"]
        assert before == (ledger.used, canonical_json_bytes(tool.export_navigation()))
        assert (
            gate(
                dict(actions=[dict(entry_ref="A999", question="x")]),
                table=later,
                book=book,
                request_index=2,
                records=ledger.records,
                toolbox=tool,
            )
            is None
        )


def test_stop_handoff_uses_entry_names_without_granting_evidence(complete):
    _, _, table, _ = opened(complete)
    from tests.test_v2_live_sre_action_decision import note

    c, value = context(), decision(stop=True)
    c["schema_version"] = "nurse.v2.sre-investigation-input/20"
    c["action_entries"] = table
    value["investigation_note"] = dict(
        note(),
        basis_refs=[],
        target_service_keys=[],
        stop_assessment=dict(
            completed_scope="Only the existing score was read.",
            unchecked_scope=["Symptoms and mechanisms remain uninvestigated."],
            remaining_questions=[
                dict(
                    question_id="iq_fault",
                    why_stop="Handoff for further evidence.",
                    entry_refs=[dict(entry_ref=table["entries"][0]["entry_ref"])],
                )
            ],
        ),
    )
    result = api("decision").admit_entry_note(value, context=c)
    assert result["status"] == "accepted", result
    assert result["plan"]["updates"] == []  # A is never a basis fact or a question answer.
    value["investigation_note"]["stop_assessment"]["remaining_questions"][0]["entry_refs"] = [
        dict(entry_ref="A999")
    ]
    assert api("decision").admit_entry_note(value, context=c)["status"] == "rejected"


def test_invalid_navigation_cursor_keeps_original_guard_and_debit(complete, tmp_path, monkeypatch):
    tool, book, table, case = opened(complete)
    candidate = next(r[1] for r in book._rows if r[0] == "service")

    def forbidden(*args, **kwargs):
        pytest.fail("unregistered cursor must fail before execution")

    monkeypatch.setattr(tool, "execute", forbidden)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        ledger.dispatch(
            [
                dict(
                    navigation=dict(
                        kind="candidate",
                        candidate_service_key=candidate,
                        focus="symptoms",
                        cursor="unregistered_cursor",
                    ),
                    question="Inspect this scope.",
                )
            ]
        )
        assert ledger.used == 1 and ledger.records[-1].status == "rejected"
        assert ledger.records[-1].rejection.reason_code == "request_binding_mismatch"


@pytest.mark.parametrize(
    "field", ["selection", "table", "snapshot", "run_id", "case_id", "request_index"]
)
def test_fresh_reader_rejects_changed_frozen_binding_even_with_new_outer_digest(
    complete, tmp_path, field
):
    tool, book, table, case = opened(complete)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        ledger.dispatch([select(symptom(table))])
    path = tmp_path / "investigation/state.json"
    state = json.loads(path.read_bytes())
    row = state["records"][0]
    frozen = json.loads(row["frozen_entry_context_json"])
    if field == "selection":
        row["selection_json"] = canonical_json_bytes(dict(entry_ref="A999", question="x")).decode()
    elif field == "table":
        frozen["table"]["entries"][0]["source_prefix"] = "f" * 64
    elif field in {"snapshot", "run_id"}:
        frozen["references"][field] = "f" * 64
        frozen["references"]["state_sha256"] = sha256_hex(
            canonical_json_bytes(
                {k: v for k, v in frozen["references"].items() if k != "state_sha256"}
            )
        )
    else:
        frozen[field] = "opq_" + "f" * 32 if field == "case_id" else 2
    row["frozen_entry_context_json"] = canonical_json_bytes(frozen).decode()
    state["state_sha256"] = sha256_hex(
        canonical_json_bytes({k: v for k, v in state.items() if k != "state_sha256"})
    )
    path.write_bytes(canonical_json_bytes(state))
    fresh, _, _, _ = opened(complete)
    with (
        pytest.raises(ValueError, match="binding"),
        api().OpsSreEntryInvestigationLedger(root=tmp_path, toolbox=fresh, code_sha256="2" * 64),
    ):
        pytest.fail("changed frozen binding must not be restored")


def test_interruption_before_selection_admission_keeps_reserved_debit(
    complete, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_sre_checkpoint_ledger as storage

    tool, book, table, case = opened(complete)
    original = storage._write_checkpoint

    def fail_first_admission(path, raw):
        rows = json.loads(raw)["records"]
        if rows and rows[-1]["selection_json"] is not None:
            raise storage.CheckpointWriteError(OSError("fixture_admission_interruption"))
        original(path, raw)

    def forbidden(*args, **kwargs):
        pytest.fail("interrupted admission must never execute a tool")

    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.freeze_entries(table=table, book=book, request_index=1, case_id=case)
        monkeypatch.setattr(tool, "execute", forbidden)
        monkeypatch.setattr(storage, "_write_checkpoint", fail_first_admission)
        with pytest.raises(storage.CheckpointWriteError):
            ledger.dispatch([select(symptom(table))])
        assert ledger.used == 1
        assert ledger.records[-1].selection_json is None
    monkeypatch.setattr(storage, "_write_checkpoint", original)
    fresh, _, _, _ = opened(complete)
    monkeypatch.setattr(fresh, "execute", forbidden)
    with api().OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=fresh, code_sha256="2" * 64
    ) as restored:
        assert restored.used == 1 and restored.records[-1].status == "interrupted"
        assert restored.records[-1].failure_type == "completion_unknown_no_refund"
        assert restored.records[-1].action is None
