"""Local seams: field presence never grants an automatic answer or quality retry."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live import ops_sre_factual_reading as reading
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_observable_exchange import observable_decision_policy
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision, note
from tests.test_v2_live_sre_facet_reading import context
from tests.test_v2_live_sre_observable_owner import saved
from tests.test_v2_live_sre_observable_provider import frozen
from tests.test_v2_live_sre_observable_purpose import declaration
from tests.test_v2_live_sre_observable_tools import opened, tools


def error_tool(complete):
    """Explicit synthetic error card; the scalar-only eight fixture has no errors.

    Use real node admission, ledger and source guards, not a patched runtime. The
    deliberately synthetic values never enter any approved case or paid run.
    """
    data = complete[0]
    tool = tools().OpsSreObservableToolbox(data=data, block_root=complete[1])
    source = data.sources[0]
    candidate = source.candidate_service_key
    facts = deepcopy(context()["previous_feedback"]["semantic_record"]["payload"]["facts"])
    facts["candidate_service_key"] = candidate
    card = node_card(
        snapshot=tool.case_snapshot_id,
        node="symptoms",
        parameters=dict(candidate_service_key=candidate),
        payload=dict(kind="error_summary", facts=facts),
        evidence_ids=(source.source_sha256,),
        source_refs=(source.source_sha256,),
        interval=data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
        unit="observed_record_fraction",
        coverage=None,
        truth="unknown",
        limitations=("synthetic_fixture_not_real_case_evidence",),
    )
    tool.admit_core(nodes=(card,), steps=())
    # Deterministic tiny synthetic supply: retain this registered error card and
    # one real prepared metric, not a hash-order lottery in the menu16. This is
    # fixture material selection before publication, not a runtime/guard patch.
    from nurse_rcl.v2.live.ops_tools import card_id

    error_id = card_id(card)
    ordinary = next(key for key, item in sorted(tool.cards.items())
                    if item.node == "symptoms" and key != error_id)
    for key, item in list(tool.cards.items()):
        if item.node == "symptoms" and key not in {error_id, ordinary}:
            tool._cards.pop(key)
    return tool


def error_run(complete, root, send):
    return run_ops_agent(
        root=root,
        toolbox=error_tool(complete),
        code_sha256="a" * 64,
        complete=send,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
        opaque_case_id=complete[0].catalog.opaque_case_id,
        stage_decision_limit=2,
    )


def test_first_real_wire_declares_empty_latest_fields_without_inventing_a_question(
    complete, tmp_path
):
    _, wire, _, _ = frozen(complete, tmp_path / "tools")
    body = json.loads(json.loads(wire)["messages"][-1]["content"])
    aid = body["context"]["workbench"]["factual_reading_aid"]
    assert "latest_question_fields" in aid, "field supply projection must be present in real wire"
    assert aid["latest_question_fields"] == []


def test_transport_validation_rebuilds_field_state_from_supplied_record_not_aid():
    ctx = context()
    ctx["capabilities"] = {}
    aid = reading.factual_reading_aid(ctx)
    assert "latest_question_fields" in aid, "field states must join the existing aid"
    assert aid["latest_question_fields"][0]["fields"][0]["state"] == "supplied_unknown"
    ctx["factual_reading_aid"] = aid
    reading.validate_factual_reading_aid(ctx)
    changed = deepcopy(ctx)
    changed["factual_reading_aid"]["latest_question_fields"][0]["fields"][0]["state"] = (
        "not_supplied"
    )
    with pytest.raises(ValueError, match="factual reading aid"):
        reading.validate_factual_reading_aid(changed)
    assert ctx["question_state"]["questions"][0]["state"] == "open"


def test_prompt_keeps_record_delivery_and_diagnostic_adequacy_as_separate_dimensions():
    system = observable_decision_policy()["system"]
    assert "latest_question_fields" in system
    assert "supplied_unknown is not not_supplied" in system
    assert "no automatic question update" in system
    assert "diagnostic adequacy remains a separate unresolved question" in system


@pytest.mark.parametrize("answered", [False, True])
def test_actual_two_round_wire_preserves_native_partial_or_explicit_answer_and_never_resends(
    complete, tmp_path, answered
):
    seen, native = [], []
    selected_card = []
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        from nurse_rcl.v2.live.ops_sre_context import _delivered_records

        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=opened(complete)[0].case_snapshot_id,
            decision_index=index,
        )
        aid = ctx["factual_reading_aid"]
        assert "latest_question_fields" in aid, "latest fields must survive readable wire"
        if index == 1:
            assert aid["latest_question_fields"] == []
        else:
            original = next(
                r["payload"]["facts"]
                for r in _delivered_records(ctx["previous_feedback"])
                if r["payload"].get("kind") == "error_summary" and r["card_id"] == selected_card[0]
            )
            coverage = next(
                f for f in aid["latest_question_fields"][0]["fields"] if f["facet"] == "coverage"
            )
            assert coverage["state"] == "supplied_unknown"
            assert coverage["value"] == original["sampling_coverage"] == "unknown"
            assert ctx["question_state"]["questions"][0]["state"] == "open"
            # Readability and a matching field path never turn an old intent into an answer.
            for field, changed in (("state", "not_supplied"), ("value", "known")):
                envelope = json.loads(wire)
                body = json.loads(envelope["messages"][-1]["content"])
                body["context"]["workbench"]["factual_reading_aid"]["latest_question_fields"][0][
                    "fields"
                ][-1][field] = changed
                envelope["messages"][-1]["content"] = canonical_json_bytes(body).decode()
                with pytest.raises(ValueError, match="factual reading aid"):
                    brief_context_from_wire(
                        canonical_json_bytes(envelope),
                        opaque_case_id=complete[0].catalog.opaque_case_id,
                        case_snapshot_id=opened(complete)[0].case_snapshot_id,
                        decision_index=index,
                    )
        entries = ctx["action_entries"]["entries"]
        if index == 1:
            # A bounded menu may offer this exact card via inspect_sources only.
            # Both original tools supply the complete record; never force a menu.
            choices = [
                e for e in entries if e["observable_contract"]["objective_kind"] == "error_summary"
            ]
            assert choices, "synthetic fixture must supply its registered error card"
            entry = next((e for e in choices if e["action"]["tool"] == "read_node"), choices[0])
        else:
            # A distinct lawful next read avoids buying an exact-repeat correction.
            entry = next(
                e
                for e in entries
                if e["action"]["tool"] == "inspect_sources"
                and e["source_binding"]["card_id"] != selected_card[0]
            )
        if index == 1:
            selected_card.append(entry["source_binding"]["card_id"])
        value = decision(first=index == 1)
        value["actions"] = [
            dict(
                entry_ref=entry["entry_ref"], question="Read the selected record's supplied fields."
            )
        ]
        value["investigation_note"] = dict(
            note(),
            observable_test=declaration(entry),
            basis_refs=[],
            target_service_keys=entry["observable_contract"]["binding"]["targets"],
        )
        if index == 2:
            status = "answered" if answered else "partial"
            state = "answered" if answered else "open"
            gap = (
                ""
                if answered
                else "Coverage is supplied as unknown but the record question is not answered."
            )
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            value["latest_result_update"].update(
                answer_status=status,
                remaining_question=None if answered else gap,
                summary="Sampling coverage is supplied as unknown.",
                content_relation=ctx["investigation_progress"]["latest_actual_result"][
                    "content_relation"
                ],
            )
            value["investigation_note"]["question_updates"] = [
                dict(
                    question_id="iq_d1",
                    state=state,
                    remaining_gap=gap,
                    remaining_facets=[] if answered else ["coverage"],
                    basis_refs=[],
                )
            ]
            value["unresolved"] = [
                "Diagnostic coverage adequacy remains unknown, not a missing record field."
            ]
        seen.append(index)
        native.append(canonical_json_bytes(value))
        return native[-1]

    result = error_run(complete, tmp_path, send)
    assert result.termination == "stage_pause" and result.action_count == 2, result
    conversation = saved(complete, tmp_path)
    assert [r.native_response_json.encode() for r in conversation.decisions] == native
    assert all(r.correction_wire_json is None for r in conversation.decisions)
    from nurse_rcl.v2.live.ops_sre_context import _memory
    from nurse_rcl.v2.live.ops_sre_observable_scope import project_observable_questions

    memory = [_memory(r, json.loads(r.response_json)) for r in conversation.decisions]
    assert memory[0]["annotation_admission"]["status"] == "accepted"
    assert memory[1]["annotation_admission"]["status"] == ("accepted" if answered else "rejected")
    if not answered:
        assert memory[1]["annotation_admission"]["code"] == "remaining_facets_already_supplied"
    questions = project_observable_questions(memory)["questions"]
    assert questions[0]["state"] == ("answered" if answered else "open")
    assert questions[0]["remaining_facets"] == (
        [] if answered else questions[0]["observable_scope"]["requested_facets"]
    )
    assert len(questions) == (2 if answered else 1)
    if answered:
        assert questions[1]["question_id"] == "iq_d2"
    second = json.loads(conversation.decisions[1].context_json)
    assert second["factual_reading_aid"]["latest_question_fields"][0]["question_id"] == "iq_d1"
    original = brief_context_from_wire(
        conversation.decisions[1].wire_json.encode(),
        opaque_case_id=complete[0].catalog.opaque_case_id,
        case_snapshot_id=opened(complete)[0].case_snapshot_id,
        decision_index=2,
        references=conversation.reference_state,
    )
    assert original["factual_reading_aid"] == reading.factual_reading_aid(
        original, schema_version="nurse.v2.sre-factual-reading-aid/4"
    )
    assert original["factual_reading_aid"] == second["factual_reading_aid"]
    assert (
        error_run(complete, tmp_path, lambda *_: pytest.fail("no model resend on restore"))
        == result
    )
    assert seen == [1, 2] and canonical_json_bytes(complete[2].model_dump(mode="json")) == core
    # False now tests new note rejection, never a green model semantic repair.
