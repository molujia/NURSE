"""Entry owner uses real tools and durable history, with explicit local decisions."""

# ruff: noqa: F811 -- real small fixtures.

import json

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_agent_state import load_conversation
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_encoding import expand
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision
from tests.test_v2_live_sre_entry_admission import opened


def run(complete, root, ceiling, send):
    tool, _, _, case = opened(complete)
    return run_ops_agent(
        root=root,
        toolbox=tool,
        code_sha256="a" * 64,
        complete=send,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/19",
        opaque_case_id=case,
        stage_decision_limit=ceiling,
    )


def context(complete, index, wire):
    return brief_context_from_wire(
        wire,
        opaque_case_id=complete[0].catalog.opaque_case_id,
        case_snapshot_id=opened(complete)[0].case_snapshot_id,
        decision_index=index,
    )


def response(ctx, action, *, stop=False):
    value = decision(first=ctx["decision_index"] == 1, stop=stop)
    value["investigation_note"] = None
    value["actions"] = [] if stop else [action]
    if ctx["decision_index"] > 1:
        latest = ctx["investigation_progress"]["latest_actual_result"]
        value["responds_to"] = [dict(action_index=latest["action_index"], card_ids=[])]
        value["latest_result_update"].update(
            action_index=latest["action_index"], content_relation=latest["content_relation"]
        )
    return canonical_json_bytes(value)


def saved(complete, root):
    return load_conversation(
        root / "investigation/conversation.json",
        snapshot=opened(complete)[0].case_snapshot_id,
        code="a" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/19",
        opaque_case_id=complete[0].catalog.opaque_case_id,
    )


def test_two_symptom_actions_keep_native_selection_actual_receipts_and_fresh_restore(
    complete, tmp_path
):
    sent, seen, selected = [], [], []
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        ctx = context(complete, index, wire)
        sent.append(index)
        seen.append(ctx)
        available = [
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["action"]["parameters"] not in selected
        ]
        assert available, "current frozen page needs a distinct legal symptom check"
        entry = available[0]
        selected.append(entry["action"]["parameters"])
        return response(
            ctx, dict(entry_ref=entry["entry_ref"], question="Read the actual symptom windows.")
        )

    first = run(complete, tmp_path, 1, send)
    assert first.termination == "stage_pause", first
    assert first.action_count == first.decision_count == 1
    state = saved(complete, tmp_path)
    assert state.schema_version == "nurse.v2.sre-conversation/17"
    assert state.reference_state["schema_version"] == "nurse.v2.sre-reference-book/2"
    assert state.result is None
    prefix = canonical_json_bytes(state.decisions[0].model_dump(mode="json"))
    again = run(complete, tmp_path, 1, lambda *_: pytest.fail("paused prefix cannot send"))
    assert again.termination == "stage_pause" and sent == [1]
    second = run(complete, tmp_path, 2, send)
    assert second.termination == "stage_pause" and second.action_count == 2, second
    assert sent == [1, 2]
    assert seen[1]["investigation_progress"]["latest_actual_result"]["action_index"] == 1
    assert "baseline" in canonical_json_bytes(seen[1]).decode()
    state = saved(complete, tmp_path)
    assert canonical_json_bytes(state.decisions[0].model_dump(mode="json")) == prefix
    stored = json.loads((tmp_path / "investigation/state.json").read_bytes())
    assert stored["schema_version"] == "nurse.v2.sre-investigation-ledger/20"
    assert len(stored["records"]) == 2
    for row in stored["records"]:
        assert row["status"] == "completed" and row["action"]["tool"] == "inspect_sources"
        assert set(json.loads(row["selection_json"])) == {"entry_ref", "question"}
        view = expand(json.loads(row["model_view_json"]))
        assert view["page"]["records"]
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core


def test_read_then_aggregate_same_record_has_delivery_awareness_in_real_wire_and_restore(
    complete, tmp_path, monkeypatch
):
    from copy import deepcopy

    from nurse_rcl.v2.live import ops_sre_action_entries
    from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
    from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries

    compile_entries = ops_sre_action_entries.compile_action_entries
    source_card = []

    def single_visible_source(capabilities, **kwargs):
        # Isolate the delivery seam, not the finite menu's identity-based ordering.
        # Both entries still compile from one real, issued source each turn.
        subset = deepcopy(capabilities)
        rows = subset["directed_sources"]["entries"]
        if not source_card:
            source_card.append(rows[0]["card_id"])
        subset["directed_sources"]["entries"] = [r for r in rows if r["card_id"] == source_card[0]]
        subset["investigation_routes"]["entries"] = []
        return compile_entries(subset, **kwargs)

    monkeypatch.setattr(ops_sre_action_entries, "compile_action_entries", single_visible_source)
    chosen, sent = [], []

    def send(index, wire):
        ctx = context(complete, index, wire)
        sent.append(index)
        table = ctx["action_entries"]
        if index == 1:
            entry = next(
                e
                for e in table["entries"]
                if e["action"]["tool"] == "read_node" and "record_sha256" in e["source_binding"]
            )
            assert entry["source_context"]["delivery"]["state"] == "unread"
            chosen.append(entry["source_binding"])
        else:
            entry = next(
                e
                for e in table["entries"]
                if e["action"]["tool"] == "inspect_sources" and e["source_binding"] == chosen[0]
            )
            assert entry["source_context"]["delivery"]["state"] == "complete_same_record"
            assert entry["source_context"]["delivery"]["prior_action_indices"] == [1]
            assert (
                entry["source_context"]["delivery"]["expected_record_relation"] == "no_new_record"
            )
            assert (
                "cannot_fill_unsupplied_coverage_or_windows"
                in entry["source_context"]["supply_limits"]
            )
        return response(
            ctx, dict(entry_ref=entry["entry_ref"], question="Inspect this actual scope.")
        )

    first = run(complete, tmp_path, 1, send)
    assert first.termination == "stage_pause" and sent == [1]
    second = run(complete, tmp_path, 2, send)
    assert second.termination == "stage_pause" and second.action_count == 2 and sent == [1, 2]
    # This is legal aggregation, not a new cross-tool correction or hard block.
    state = saved(complete, tmp_path)
    assert all(d.correction_wire_json is None for d in state.decisions)
    records = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"]
    assert [r["action"]["tool"] for r in records] == ["read_node", "inspect_sources"]
    assert (
        run(complete, tmp_path, 2, lambda *_: pytest.fail("saved prefix must not resend")) == second
    )
    frozen = json.loads(records[1]["frozen_entry_context_json"])
    book = SreActionReferenceBook.from_state(
        frozen["references"],
        snapshot=frozen["references"]["snapshot"],
        run_id=frozen["references"]["run_id"],
    )
    selected_ref = json.loads(records[1]["selection_json"])["entry_ref"]
    entry = next(e for e in frozen["table"]["entries"] if e["entry_ref"] == selected_ref)
    entry["source_context"]["delivery"]["state"] = "unread"
    with pytest.raises(ValueError, match="binding"):
        validate_action_entries(frozen["table"], book=book, request_index=2)


def test_contradictory_note_keeps_legal_dispatch_and_rejected_status_in_fresh_restore(
    complete, tmp_path
):
    from nurse_rcl.v2.live.ops_sre_context import _memory
    from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state

    sent = []
    selected = []
    original_core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        ctx = context(complete, index, wire)
        sent.append(index)
        if index == 3:
            memory = next(
                row for row in ctx["public_working_memory"]["recent"] if row["decision_index"] == 2
            )
            assert memory["annotation_admission"] == dict(
                status="rejected", code="sre_note_latest_question_status"
            )
            assert "investigation_plan" not in memory
            assert ctx["question_state"]["questions"][0]["state"] == "open"
            assert ctx["question_state"]["latest_action_question_id"] is None
            assert ctx["investigation_progress"]["latest_actual_result"]["action_index"] == 2
            return response(ctx, {}, stop=True)
        entry = next(
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["action"]["parameters"] not in selected
        )
        selected.append(entry["action"]["parameters"])
        value = json.loads(
            response(ctx, dict(entry_ref=entry["entry_ref"], question="Read this record."))
        )
        refs = []
        updates = []
        if index == 2:
            source = ctx["delivered_record_index"]["records"][0]
            refs = [dict(action_index=1, card_id=source["card_id"])]
            latest_id = ctx["question_state"]["latest_action_question_id"]
            assert latest_id == "iq_d1"
            updates = [
                dict(question_id=latest_id, state="answered", remaining_gap="", basis_refs=refs)
            ]
            value["latest_result_update"].update(
                answer_status="partial", remaining_question="The window remains missing."
            )
        value["investigation_note"] = dict(
            role="symptoms",
            target_service_keys=entry["source_context"]["targets"][:1],
            why_now="Read actual available symptoms.",
            discriminating_value="Unknown remains a gap, not refutation.",
            basis_refs=refs,
            question_updates=updates,
            stop_assessment=None,
        )
        return canonical_json_bytes(value)

    second = run(complete, tmp_path, 2, send)
    assert second.termination == "stage_pause" and second.action_count == 2
    state = saved(complete, tmp_path)
    native = json.loads(state.decisions[1].native_response_json)
    assert native["investigation_note"]["question_updates"][0]["state"] == "answered"
    assert all(d.correction_wire_json is None for d in state.decisions)
    rows = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"]
    assert [r["status"] for r in rows] == ["completed", "completed"]
    assert (
        run(complete, tmp_path, 2, lambda *_: pytest.fail("saved prefix must not resend")) == second
    )
    final = run(complete, tmp_path, 3, send)
    assert final.termination == "model_stop" and final.action_count == 2
    final_state = saved(complete, tmp_path)
    # The stopped owner's original memory remains a reproducible projection.
    memory = [_memory(d, json.loads(d.response_json)) for d in final_state.decisions]
    assert (
        project_question_state(memory, entry_field="entry_ref")["questions"][0]["state"] == "open"
    )
    assert sent == [1, 2, 3]
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == original_core


def test_scoped_questions_survive_two_legal_reads_and_no_send_restore(complete, tmp_path):
    from nurse_rcl.v2.live.ops_sre_context import _memory
    from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state

    questions = [
        "Which values, missing fields and limits does the selected record supply?",
        "Following iq_d1, what does this other record supply for its own symptom hypothesis?",
    ]
    sent, selected = [], []
    original_core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        ctx = context(complete, index, wire)
        sent.append(index)
        entry = next(
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["action"]["parameters"] not in selected
        )
        selected.append(entry["action"]["parameters"])
        value = json.loads(
            response(ctx, dict(entry_ref=entry["entry_ref"], question=questions[index - 1]))
        )
        refs, updates = [], []
        if index == 2:
            assert ctx["question_state"]["questions"][0]["question"] == questions[0]
            source = ctx["delivered_record_index"]["records"][0]
            refs = [dict(action_index=1, card_id=source["card_id"])]
            updates = [
                dict(question_id="iq_d1", state="answered", remaining_gap="", basis_refs=refs)
            ]
            value["latest_result_update"].update(
                answer_status="answered",
                remaining_question=None,
                summary="The supplied and missing fields are identified; fault status unresolved.",
            )
        value["unresolved"] = ["Fault identity requires unread mechanism and direction evidence."]
        value["investigation_note"] = dict(
            role="symptoms",
            target_service_keys=entry["source_context"]["targets"][:1],
            why_now="Separate the read answer from the wider diagnostic gap.",
            discriminating_value="Test only this record's symptom hypothesis, not source identity.",
            basis_refs=refs,
            question_updates=updates,
            stop_assessment=None,
        )
        return canonical_json_bytes(value)

    outcome = run(complete, tmp_path, 2, send)
    assert outcome.termination == "stage_pause" and outcome.action_count == 2
    again = run(complete, tmp_path, 2, lambda *_: pytest.fail("saved prefix must not resend"))
    assert again == outcome
    restored = saved(complete, tmp_path)
    memory = [_memory(d, json.loads(d.response_json)) for d in restored.decisions]
    state = project_question_state(memory, entry_field="entry_ref")
    rows = {row["question_id"]: row for row in state["questions"]}
    assert rows["iq_d1"]["question"] == questions[0] and rows["iq_d1"]["state"] == "answered"
    assert rows["iq_d2"]["question"] == questions[1] and rows["iq_d2"]["state"] == "open"
    assert rows["iq_d2"]["parent_question_id"] is None  # No fabricated structured hierarchy.
    assert all(row["annotation_admission"]["status"] == "accepted" for row in memory)
    records = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"]
    assert [row["status"] for row in records] == ["completed", "completed"]
    assert all(d.correction_wire_json is None for d in restored.decisions)
    assert sent == [1, 2]  # No third model turn to inspect the second receipt.
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == original_core


def test_unknown_selection_is_debited_refusal_with_no_correction_or_dispatch(complete, tmp_path):
    sent = []

    def send(index, wire):
        sent.append(index)
        return response(
            context(complete, index, wire), dict(entry_ref="A999", question="Unknown selection.")
        )

    result = run(complete, tmp_path, 1, send)
    assert result.termination == "stage_pause" and result.action_count == 1, result
    row = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"][0]
    assert row["status"] == "rejected" and row["action"] is None
    assert row["selection_reason_code"] == "entry_not_available" and sent == [1]
    state = saved(complete, tmp_path)
    assert state.decisions[0].correction_wire_json is None
    assert (
        run(
            complete, tmp_path, 1, lambda *_: pytest.fail("rejected prefix cannot send")
        ).action_count
        == 1
    )


def test_unknown_owner_outcome_is_terminal_and_cannot_resend(complete, tmp_path):
    sent = []

    def interrupted(index, wire):
        sent.append(index)
        raise KeyboardInterrupt("synthetic unknown owner outcome")

    with pytest.raises(KeyboardInterrupt):
        run(complete, tmp_path, 1, interrupted)
    restored = run(complete, tmp_path, 2, lambda *_: pytest.fail("unknown owner cannot resend"))
    assert restored.termination == "unavailable" and restored.decision_count == 1
    assert restored.action_count == 0 and sent == [1]


def test_native_navigation_history_does_not_issue_short_names_as_new_service_objects(
    complete, tmp_path
):
    selected, seen = [], []

    def send(index, wire):
        ctx = context(complete, index, wire)
        seen.append(ctx)
        if index == 1:
            candidate = ctx["capabilities"]["directed_sources"]["entries"][0]["targets"][0]
            action = dict(
                navigation=dict(
                    kind="candidate", candidate_service_key=candidate, focus="symptoms", cursor=None
                ),
                question="Navigate to this real candidate.",
            )
            selected.append(action)
            return response(ctx, action)
        return response(ctx, None, stop=True)

    result = run(complete, tmp_path, 2, send)
    assert result.termination == "model_stop" and result.action_count == 1, result
    state = saved(complete, tmp_path)
    services = [r[2] for r in state.reference_state["entries"] if r[0] == "service"]
    assert selected[0]["navigation"]["candidate_service_key"] not in services
    memory = seen[1]["public_working_memory"]["recent"][0]["requested_action"]
    assert json.loads(memory["native_selection_json"]) == selected[0]
    actual = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"][0][
        "action"
    ]
    assert json.loads(actual["parameters_json"])["candidate_service_key"] in services


def test_exact_read_correction_is_unique_and_blocked_exit_does_not_debit_or_replay(
    complete, tmp_path, monkeypatch
):
    from copy import deepcopy

    from nurse_rcl.v2.live import ops_sre_action_entries

    compile_entries = ops_sre_action_entries.compile_action_entries
    source_card = []

    def single_visible_source(capabilities, **kwargs):
        # This test isolates the duplicate gate, not the 16-item page selection.
        # Use one real, issued source; never force a prior A into a live menu.
        subset = deepcopy(capabilities)
        rows = subset["directed_sources"]["entries"]
        if not source_card:
            source_card.append(rows[0]["card_id"])
        subset["directed_sources"]["entries"] = [r for r in rows if r["card_id"] == source_card[0]]
        subset["investigation_routes"]["entries"] = []
        return compile_entries(subset, **kwargs)

    monkeypatch.setattr(ops_sre_action_entries, "compile_action_entries", single_visible_source)
    sent, params = [], []

    def send(index, wire):
        ctx = context(complete, index, wire)
        sent.append(index)
        entries = [
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "read_node"
            and e["action"]["parameters"]["section"] == "payload"
            and e["action"]["parameters"]["cursor"] is None
        ]
        entry = next(e for e in entries if not params or e["action"]["parameters"] == params[0])
        if not params:
            params.append(entry["action"]["parameters"])
        return response(
            ctx, dict(entry_ref=entry["entry_ref"], question="Read the same complete source.")
        )

    result = run(complete, tmp_path, 2, send)
    assert result.termination == "no_progress_after_correction", result
    assert result.action_count == 1 and sent == [1, 2, 2]
    ledger = json.loads((tmp_path / "investigation/state.json").read_bytes())
    assert len(ledger["records"]) == 1 and ledger["records"][0]["status"] == "completed"
    state = saved(complete, tmp_path)
    assert state.decisions[-1].blocked_read_proof is not None
    assert state.decisions[-1].action_index is None
    assert (
        run(complete, tmp_path, 2, lambda *_: pytest.fail("blocked terminal cannot send")) == result
    )


@pytest.mark.parametrize("ceiling", [None, True, 0, 9, 1.0])
def test_entry_owner_requires_explicit_exact_ceiling_before_storage(complete, tmp_path, ceiling):
    with pytest.raises(ValueError, match="stage decision"):
        run(complete, tmp_path, ceiling, lambda *_: pytest.fail("invalid ceiling cannot send"))
    assert not (tmp_path / "investigation").exists()
