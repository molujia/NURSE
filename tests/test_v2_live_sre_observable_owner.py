"""Two local decisions exercise new wire, original receipts and fresh recovery."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import json
import traceback
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_agent_state import load_conversation
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision, note
from tests.test_v2_live_sre_observable_purpose import declaration
from tests.test_v2_live_sre_observable_tools import opened


def run(complete, root, ceiling, send):
    tool, _, _, case = opened(complete)
    return run_ops_agent(
        root=root,
        toolbox=tool,
        code_sha256="a" * 64,
        complete=send,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
        opaque_case_id=case,
        stage_decision_limit=ceiling,
    )


def saved(complete, root, profile="hermes-sre-tools/20"):
    return load_conversation(
        root / "investigation/conversation.json",
        snapshot=opened(complete)[0].case_snapshot_id,
        code="a" * 64,
        empirical=True,
        sre=True,
        sre_profile=profile,
        opaque_case_id=complete[0].catalog.opaque_case_id,
    )


def test_new_wire_scope_history_and_fresh_resume_never_resend(complete, tmp_path, monkeypatch):
    from nurse_rcl.v2.live import ops_agent

    real_failure = ops_agent.failure_detail

    def diagnostic(stage, exc):
        print(
            "fixture-boundary",
            stage,
            type(exc).__name__,
            str(exc),
            [(r.name, r.lineno) for r in traceback.extract_tb(exc.__traceback__)],
        )
        return real_failure(stage, exc)

    monkeypatch.setattr(ops_agent, "failure_detail", diagnostic)
    seen, wire_schemas, selected = [], [], []
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        body = json.loads(json.loads(wire)["messages"][-1]["content"])
        assert body["protocol"] == "nurse.v2.sre-decision-exchange/21"
        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=opened(complete)[0].case_snapshot_id,
            decision_index=index,
        )
        assert ctx["schema_version"].endswith("/21")
        assert ctx["question_state"]["schema_version"].endswith("/2")
        assert "observable_test" in canonical_json_bytes(body).decode()
        assert all(s["skill_id"].endswith("/16") for s in ctx["skills"])
        entries = [
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["source_binding"]["record_sha256"] not in selected
        ]
        entry = entries[0]
        selected.append(entry["source_binding"]["record_sha256"])
        value = decision(first=index == 1)
        value["actions"] = [
            dict(entry_ref=entry["entry_ref"], question="Read supplied summary facets.")
        ]
        value["investigation_note"] = dict(
            note(),
            observable_test=declaration(entry),
            basis_refs=[],
            target_service_keys=entry["observable_contract"]["binding"]["targets"],
        )
        if index == 2:
            assert ctx["question_state"]["questions"][0]["observable_scope"]["requested_facets"]
            latest = ctx["investigation_progress"]["latest_actual_result"]
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            value["latest_result_update"].update(
                action_index=1,
                answer_status="answered",
                remaining_question=None,
                content_relation=latest["content_relation"],
            )
            value["investigation_note"]["question_updates"] = [
                dict(
                    question_id="iq_d1",
                    state="answered",
                    remaining_gap="",
                    remaining_facets=[],
                    basis_refs=[],
                )
            ]
        seen.append(ctx)
        wire_schemas.append(body["context"]["schema_version"])
        return canonical_json_bytes(value)

    first = run(complete, tmp_path, 1, send)
    assert first.termination == "stage_pause" and first.action_count == 1, first
    state = saved(complete, tmp_path)
    assert state.schema_version == "nurse.v2.sre-conversation/18"
    prefix = canonical_json_bytes(state.decisions[0].model_dump(mode="json"))
    assert (
        run(complete, tmp_path, 1, lambda *_: pytest.fail("no resend")).termination == "stage_pause"
    )
    second = run(complete, tmp_path, 2, send)
    assert second.termination == "stage_pause" and second.action_count == 2, second
    state = saved(complete, tmp_path)
    assert canonical_json_bytes(state.decisions[0].model_dump(mode="json")) == prefix
    assert len(seen) == 2 and wire_schemas == ["nurse.v2.sre-model-view/17"] * 2
    assert json.loads((tmp_path / "investigation/state.json").read_bytes())[
        "schema_version"
    ].endswith("/21")
    with pytest.raises(ValueError):
        saved(complete, tmp_path, profile="hermes-sre-tools/19")
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core


def test_rejected_scope_keeps_real_action_and_rejection_visible_after_restore(complete, tmp_path):
    from nurse_rcl.v2.live.ops_sre_context import _memory
    from nurse_rcl.v2.live.ops_sre_observable_scope import project_observable_questions

    sent = []

    def send(index, wire):
        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=opened(complete)[0].case_snapshot_id,
            decision_index=index,
        )
        sent.append(index)
        entry = next(
            e for e in ctx["action_entries"]["entries"] if e["action"]["tool"] == "inspect_sources"
        )
        value = decision(first=index == 1)
        value["actions"] = [
            dict(entry_ref=entry["entry_ref"], question="Read actual supplied data.")
        ]
        value["investigation_note"] = None
        if index == 1:
            claim = dict(declaration(entry), metric_name="unsupplied_metric")
            value["investigation_note"] = dict(
                note(),
                observable_test=claim,
                basis_refs=[],
                target_service_keys=entry["observable_contract"]["binding"]["targets"],
            )
        else:
            memory = ctx["public_working_memory"]["recent"][0]
            assert memory["annotation_admission"] == dict(
                status="rejected", code="objective_not_supplied"
            )
            assert ctx["question_state"]["questions"] == []
            latest = ctx["investigation_progress"]["latest_actual_result"]
            assert latest["action_index"] == 1
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            value["latest_result_update"].update(content_relation=latest["content_relation"])
        return canonical_json_bytes(value)

    first = run(complete, tmp_path, 1, send)
    assert first.termination == "stage_pause" and first.action_count == 1
    assert run(complete, tmp_path, 1, lambda *_: pytest.fail("no rejected-note resend")) == first
    state = saved(complete, tmp_path)
    memory = [_memory(d, json.loads(d.response_json)) for d in state.decisions]
    assert project_observable_questions(memory)["questions"] == []
    result = run(complete, tmp_path, 2, send)
    assert result.termination == "stage_pause" and result.action_count == 2
    assert sent == [1, 2]
    records = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"]
    assert all(
        r["status"] == "completed" and r["action"]["tool"] == "inspect_sources" for r in records
    )
    assert all(d.correction_wire_json is None for d in saved(complete, tmp_path).decisions)


@pytest.mark.parametrize("question_state", ["answered", "partial"])
def test_explicit_narrow_update_or_illegal_partial_never_rewrites_native_or_cancels_action(
    complete, tmp_path, question_state
):
    from nurse_rcl.v2.live.ops_sre_context import _memory
    from nurse_rcl.v2.live.ops_sre_observable_scope import project_observable_questions

    sent, selected, native, original = [], [], [], []
    core = canonical_json_bytes(complete[2].model_dump(mode="json"))
    diagnostic_gap = "Causal mechanism, capacity thresholds and direction remain unchecked."

    def send(index, wire):
        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=opened(complete)[0].case_snapshot_id,
            decision_index=index,
        )
        entry = next(
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["source_binding"]["record_sha256"] not in selected
        )
        selected.append(entry["source_binding"]["record_sha256"])
        value = decision(first=index == 1)
        value["actions"] = [
            dict(
                entry_ref=entry["entry_ref"],
                question="What values and markers does this record supply?",
            )
        ]
        value["unresolved"] = [diagnostic_gap]
        value["investigation_note"] = dict(
            note(),
            observable_test=declaration(entry),
            basis_refs=[],
            target_service_keys=entry["observable_contract"]["binding"]["targets"],
            discriminating_value=(
                "Read the selected record; no threshold or causal test is supplied."
            ),
        )
        if index == 2:
            original.append(deepcopy(ctx["question_state"]["questions"][0]))
            latest = ctx["investigation_progress"]["latest_actual_result"]
            assert latest["action_index"] == 1 and ctx["actions"][0]["status"] == "completed"
            answered = question_state == "answered"
            remaining = [] if answered else original[0]["observable_scope"]["requested_facets"][:1]
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            value["latest_result_update"].update(
                action_index=1,
                answer_status="answered" if answered else "partial",
                remaining_question=(
                    None if answered else "One requested observation remains unresolved."
                ),
                summary=(
                    "The prior receipt supplies record values and availability markers, "
                    "not a causal answer."
                ),
                content_relation=latest["content_relation"],
            )
            value["investigation_note"]["why_now"] = (
                "The prior narrow read concerns iq_d1; this distinct record has its "
                "own observation scope."
            )
            value["investigation_note"]["question_updates"] = [
                dict(
                    question_id="iq_d1",
                    state=question_state,
                    remaining_gap="" if answered else "One original observation is unresolved.",
                    remaining_facets=remaining,
                    basis_refs=[],
                )
            ]
        sent.append(index)
        native.append(canonical_json_bytes(value))
        return native[-1]

    first = run(complete, tmp_path, 1, send)
    assert first.termination == "stage_pause" and first.action_count == 1, first
    initial = saved(complete, tmp_path)
    first_prefix = canonical_json_bytes(initial.decisions[0].model_dump(mode="json"))
    initial_memory = [_memory(d, json.loads(d.response_json)) for d in initial.decisions]
    initial_question = deepcopy(project_observable_questions(initial_memory)["questions"][0])
    second = run(complete, tmp_path, 2, send)
    assert second.termination == "stage_pause" and second.action_count == 2, second
    state = saved(complete, tmp_path)
    assert canonical_json_bytes(state.decisions[0].model_dump(mode="json")) == first_prefix
    assert [d.native_response_json.encode() for d in state.decisions] == native
    assert all(d.correction_wire_json is None for d in state.decisions)
    memory = [_memory(d, json.loads(d.response_json)) for d in state.decisions]
    admission = memory[-1]["annotation_admission"]
    questions = project_observable_questions(memory)["questions"]
    old = questions[0]
    assert old["question"] == initial_question["question"]
    assert old["observable_scope"] == initial_question["observable_scope"]
    assert (
        original[0]["observable_scope"]["requested_facets"]
        == (initial_question["observable_scope"]["requested_facets"])
    )
    if question_state == "answered":
        assert admission["status"] == "accepted"
        assert old["state"] == "answered" and old["remaining_facets"] == []
        assert len(questions) == 2 and questions[1]["question_id"] == "iq_d2"
    else:
        assert admission == dict(status="rejected", code="sre_note_update_shape")
        assert old == initial_question and len(questions) == 1
    assert json.loads(state.decisions[1].native_response_json)["unresolved"] == [diagnostic_gap]
    records = json.loads((tmp_path / "investigation/state.json").read_bytes())["records"]
    assert len(records) == 2 and all(r["status"] == "completed" for r in records)
    assert (
        run(complete, tmp_path, 2, lambda *_: pytest.fail("fresh restore must not resend"))
        == second
    )
    assert sent == [1, 2]
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core


def test_new_exact_repeat_correction_does_not_reserve_dispatch_or_replay(
    complete, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_sre_action_entries

    compile_entries = ops_sre_action_entries.compile_action_entries
    source_card, sent, params = [], [], []

    def single_source(capabilities, **kwargs):
        subset = deepcopy(capabilities)
        rows = subset["directed_sources"]["entries"]
        if not source_card:
            source_card.append(rows[0]["card_id"])
        subset["directed_sources"]["entries"] = [r for r in rows if r["card_id"] == source_card[0]]
        subset["investigation_routes"]["entries"] = []
        return compile_entries(subset, **kwargs)

    monkeypatch.setattr(ops_sre_action_entries, "compile_action_entries", single_source)

    def send(index, wire):
        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=opened(complete)[0].case_snapshot_id,
            decision_index=index,
        )
        sent.append(index)
        entry = next(
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "read_node"
            and e["action"]["parameters"]["section"] == "payload"
            and e["action"]["parameters"]["cursor"] is None
        )
        if not params:
            params.append(entry["action"]["parameters"])
        assert entry["action"]["parameters"] == params[0]
        value = decision(first=index == 1)
        value["investigation_note"] = None
        value["actions"] = [
            dict(entry_ref=entry["entry_ref"], question="Read same complete source.")
        ]
        if index > 1:
            latest = ctx["investigation_progress"]["latest_actual_result"]
            value["responds_to"] = [dict(action_index=1, card_ids=[])]
            value["latest_result_update"].update(content_relation=latest["content_relation"])
        return canonical_json_bytes(value)

    result = run(complete, tmp_path, 2, send)
    assert result.termination == "no_progress_after_correction" and sent == [1, 2, 2], result
    assert result.action_count == 1
    ledger = json.loads((tmp_path / "investigation/state.json").read_bytes())
    assert len(ledger["records"]) == 1 and ledger["records"][0]["status"] == "completed"
    state = saved(complete, tmp_path)
    assert state.decisions[-1].blocked_read_proof is not None
    assert state.decisions[-1].action_index is None
    assert sum(d.correction_wire_json is not None for d in state.decisions) == 1
    assert (
        run(complete, tmp_path, 2, lambda *_: pytest.fail("blocked terminal cannot resend"))
        == result
    )
