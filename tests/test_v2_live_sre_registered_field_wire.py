"""R2 actual wire/owner/ledger: atomic note refusal is not action refusal."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_context import _memory
from nurse_rcl.v2.live.ops_sre_observable_scope import project_observable_questions
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision, note
from tests.test_v2_live_sre_facet_reading_wire import error_tool
from tests.test_v2_live_sre_observable_owner import saved
from tests.test_v2_live_sre_observable_purpose import declaration


def run(complete, root, send, ceiling=2):
    return run_ops_agent(
        root=root,
        toolbox=error_tool(complete),
        code_sha256="a" * 64,
        complete=send,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
        opaque_case_id=complete[0].catalog.opaque_case_id,
        stage_decision_limit=ceiling,
    )


@pytest.mark.parametrize("mode", ["contradiction", "open", "answered", "absent", "past"])
def test_wire_note_atomicity_native_preservation_and_fresh_restore(
    complete, tmp_path, mode, monkeypatch
):
    import traceback

    from nurse_rcl.v2.live import ops_agent

    original_failure = ops_agent.failure_detail

    def diagnostic(stage, exc):
        print(
            "R2 boundary",
            stage,
            type(exc).__name__,
            str(exc),
            [(r.name, r.lineno) for r in traceback.extract_tb(exc.__traceback__)],
        )
        return original_failure(stage, exc)

    monkeypatch.setattr(ops_agent, "failure_detail", diagnostic)
    native, seen, cards = [], [], []
    snapshot = error_tool(complete).case_snapshot_id
    case = complete[0].catalog.opaque_case_id
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        ctx = brief_context_from_wire(
            wire, opaque_case_id=case, case_snapshot_id=snapshot, decision_index=index
        )
        assert ctx["delivered_record_index"]["schema_version"].endswith("/2")
        aid = ctx["factual_reading_aid"]
        assert aid["schema_version"] == "nurse.v2.sre-factual-reading-aid/4"
        rows = aid["question_delivery_fields"]
        if index == 1:
            assert rows == []
            entry = next(
                e
                for e in ctx["action_entries"]["entries"]
                if e["observable_contract"]["objective_kind"] == "error_summary"
            )
        else:
            assert rows[0]["question_id"] == "iq_d1"
            assert rows[0]["action_indices"] == [1]
            assert len(rows[0]["fields"]) == 7
            assert all("value" not in f for f in rows[0]["fields"])
            if index == 3:
                assert aid["latest_question_fields"] == []
                entry = next(
                    e
                    for e in ctx["action_entries"]["entries"]
                    if e["action"]["tool"] == "inspect_candidate"
                )
            else:
                entry = next(
                    e
                    for e in ctx["action_entries"]["entries"]
                    if e["action"]["tool"] == "inspect_sources"
                    and e["source_binding"]["card_id"] not in cards
                )
                broken = json.loads(wire)
                body = json.loads(broken["messages"][-1]["content"])
                block = body["context"]["workbench"]
                block["factual_reading_aid"]["question_delivery_fields"][0]["fields"][-1][
                    "state"
                ] = "not_supplied"
                broken["messages"][-1]["content"] = canonical_json_bytes(body).decode()
                with pytest.raises(ValueError, match="factual reading aid"):
                    brief_context_from_wire(
                        canonical_json_bytes(broken),
                        opaque_case_id=case,
                        case_snapshot_id=snapshot,
                        decision_index=index,
                    )
        cards.append(entry["source_binding"].get("card_id"))
        value = decision(first=index == 1)
        value["actions"] = [dict(entry_ref=entry["entry_ref"], question="Read registered fields.")]
        value["investigation_note"] = dict(
            note(),
            observable_test=declaration(entry),
            basis_refs=[],
            target_service_keys=entry["observable_contract"]["binding"]["targets"],
        )
        if index >= 2:
            value["responds_to"] = [dict(action_index=index - 1, card_ids=[])]
            answered = mode == "answered"
            value["latest_result_update"].update(
                answer_status="answered" if answered else "partial",
                action_index=index - 1,
                remaining_question=None if answered else "Diagnostic adequacy unresolved.",
                content_relation=ctx["investigation_progress"]["latest_actual_result"][
                    "content_relation"
                ],
            )
            value["investigation_note"]["question_updates"] = [
                dict(
                    question_id="iq_d1",
                    state="answered" if answered else "open",
                    remaining_gap="" if answered else "Diagnostic adequacy unresolved.",
                    remaining_facets=["coverage"] if mode == "contradiction" or index == 3 else [],
                    basis_refs=[dict(action_index=1, card_id=cards[0])],
                )
            ]
            if mode == "absent":
                value["investigation_note"] = None
        seen.append(ctx)
        native.append(canonical_json_bytes(value))
        return native[-1]

    ceiling = 3 if mode == "past" else 2
    result = run(complete, tmp_path, send, ceiling)
    assert result.termination == "stage_pause" and result.action_count == ceiling, result
    conversation = saved(complete, tmp_path)
    assert [r.native_response_json.encode() for r in conversation.decisions] == native
    assert all(r.correction_wire_json is None for r in conversation.decisions)
    memory = [_memory(r, json.loads(r.response_json)) for r in conversation.decisions]
    admission = memory[1]["annotation_admission"]
    expected = (
        "rejected" if mode == "contradiction" else "absent" if mode == "absent" else "accepted"
    )
    assert admission["status"] == expected
    if mode == "contradiction":
        assert admission["code"] == "remaining_facets_already_supplied"
        assert "investigation_plan" not in memory[1]  # no partial new iq_d2 either
    if mode == "past":
        assert memory[2]["annotation_admission"]["code"] == "remaining_facets_already_supplied"
        assert "investigation_plan" not in memory[2]
    questions = project_observable_questions(memory)["questions"]
    assert questions[0]["state"] == ("answered" if mode == "answered" else "open")
    if mode in {"contradiction", "absent"}:
        assert len(questions) == 1
        assert questions[0]["remaining_facets"] == list(
            questions[0]["observable_scope"]["requested_facets"]
        )
    else:
        assert len(questions) == 2 and questions[0]["remaining_facets"] == []
    second = conversation.decisions[1]
    original = brief_context_from_wire(
        second.wire_json.encode(),
        opaque_case_id=case,
        case_snapshot_id=snapshot,
        decision_index=2,
        references=conversation.reference_state,
    )
    assert original["factual_reading_aid"] == json.loads(second.context_json)["factual_reading_aid"]
    assert run(complete, tmp_path, lambda *_: pytest.fail("must not resend"), ceiling) == result
    assert (
        len(seen) == ceiling and canonical_json_bytes(complete[2].model_dump(mode="json")) == core
    )
    if mode == "past":
        # Consistent historical displays are not original-content authentication.
        # Re-seal this synthetic checkpoint only; fresh owner must replay the ledger.
        from nurse_rcl.v2.live.ops_agent_state import save_conversation
        from nurse_rcl.v2.live.ops_sre_brief_exchange import prepare_brief_decision
        from nurse_rcl.v2.live.ops_sre_encoding import expand

        last = conversation.decisions[-1]
        changed = expand(json.loads(last.context_json))
        enriched = next(
            r
            for r in changed["delivered_record_index"]["records"]
            if "registered_field_delivery" in r
        )
        enriched["registered_field_delivery"]["fields"][-1]["state"] = "not_supplied"
        changed["factual_reading_aid"]["question_delivery_fields"][0]["fields"][-1]["state"] = (
            "not_supplied"
        )
        refs = deepcopy(conversation.reference_state)
        changed_wire = prepare_brief_decision(
            opaque_case_id=case,
            case_snapshot_id=snapshot,
            decision_index=3,
            context=changed,
            references=refs,
        )
        assert refs == conversation.reference_state
        tampered = last.model_copy(
            update=dict(
                context_json=canonical_json_bytes(changed).decode(),
                wire_json=changed_wire.decode(),
                wire_sha256=sha256_hex(changed_wire),
            )
        )
        save_conversation(
            tmp_path / "investigation/conversation.json",
            conversation.model_copy(
                update=dict(decisions=(*conversation.decisions[:-1], tampered))
            ),
        )
        saved(complete, tmp_path)  # transport equality alone is intentionally insufficient
        with pytest.raises(ValueError, match="saved workbench source differs"):
            run(complete, tmp_path, lambda *_: pytest.fail("no resend on tamper"), ceiling)
