"""D26 public questions are an original-conversation projection, not facts."""

# ruff: noqa: RUF001 -- Chinese public question punctuation.

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError


def api():
    name = "nurse_rcl.v2.live.ops_sre_d26_questions"
    assert util.find_spec(name) is not None, "D26 stable question closure is missing"
    return import_module(name)


def question(key="iq_fault", **changes):
    value = dict(
        question_id=key,
        parent_question_id=None,
        role="mechanisms",
        question="候选故障解释有哪些支持、反证与未查边界？",
        target_service_keys=["candidate_a"],
        state="open",
        basis_refs=[],
        remaining_gap="尚未读取故障证据",
        superseded_by=None,
    )
    return dict(value, **changes)


def context():
    return dict(
        decision_index=2,
        capabilities=dict(
            investigation_situation=dict(
                candidates=[
                    dict(
                        candidate_service_key="candidate_a",
                        scoring=dict(records=[]),
                    )
                ]
            ),
            published_node_summaries=[dict(card_id="entry_mechanism")],
        ),
        actions=[dict(action_index=1, status="completed", visible_card_ids=["card_family"])],
        delivered_record_index=dict(
            records=[
                dict(
                    card_id="card_family",
                    record_sha256="a" * 64,
                    delivery_kind="complete",
                    action_indices=[1],
                )
            ]
        ),
        investigation_routes=dict(entries=[]),
        delivered_navigation=dict(schema_version="nurse.v2.sre-delivered-navigation/1", sources=[]),
    )


def plan(*rows, current="iq_fault", stop=False):
    return dict(
        updates=list(rows),
        current_question_id=None if stop else current,
        action_purpose=None
        if stop
        else dict(
            card_ids=["entry_mechanism"],
            discriminating_value="区分机制与症状",
            why_now="评分身份核对不足以解释故障",
        ),
    )


def initial():
    return api().apply_question_plan(
        api().project_question_state([]), plan(question()), context=context()
    )


def test_local_score_answer_does_not_erase_fault_question():
    before = initial()
    local = question(
        "iq_score",
        parent_question_id="iq_fault",
        role="scoring",
        question="cpu是否已参与正式评分？",
        state="answered",
        remaining_gap="",
        basis_refs=[
            dict(
                action_index=1,
                card_id="card_family",
                record_sha256="a" * 64,
                delivery_kind="complete",
            )
        ],
    )
    state = api().apply_question_plan(before, plan(local, current="iq_score"), context=context())
    assert [q["state"] for q in state["questions"]] == ["open", "answered"]
    assert before == initial()
    assert state["authority"] == "agent_statement_not_diagnostic_evidence"
    assert state["questions"][1]["last_update_decision_index"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", "改成另一个问题"),
        ("role", "scoring"),
        ("target_service_keys", []),
        ("parent_question_id", "iq_score"),
    ],
)
def test_stable_id_cannot_silently_replace_question(field, value):
    with pytest.raises(ExchangeBoundaryError, match="sre_question_identity_changed"):
        api().apply_question_plan(initial(), plan(question(**{field: value})), context=context())


def test_explicit_replacement_keeps_original_question_and_parent_chain():
    old = question(
        state="superseded", remaining_gap="替换为可核验的局部问题", superseded_by="iq_new"
    )
    new = question("iq_new", question="先核对关键症状覆盖", role="symptoms")
    state = api().apply_question_plan(
        initial(), plan(old, new, current="iq_new"), context=context()
    )
    assert [q["question_id"] for q in state["questions"]] == ["iq_fault", "iq_new"]
    assert state["questions"][0]["superseded_by"] == "iq_new"


@pytest.mark.parametrize("kind", ["parent", "replacement", "missing_replacement"])
def test_cycles_and_missing_replacements_rejected(kind):
    a, b = question("iq_a"), question("iq_b")
    if kind == "parent":
        a["parent_question_id"], b["parent_question_id"] = "iq_b", "iq_a"
    else:
        a.update(state="superseded", superseded_by="iq_b")
        if kind == "replacement":
            b.update(state="superseded", superseded_by="iq_a")
        else:
            b = None
    rows = [r for r in (a, b) if r is not None]
    with pytest.raises(ExchangeBoundaryError, match="sre_question_relation"):
        api().apply_question_plan(
            api().project_question_state([]), plan(*rows, current="iq_b"), context=context()
        )


@pytest.mark.parametrize("tamper", ["future", "digest", "card", "representation", "failed"])
def test_basis_requires_exact_delivered_source_at_this_turn(tamper):
    ctx = context()
    ref = dict(
        action_index=1, card_id="card_family", record_sha256="a" * 64, delivery_kind="complete"
    )
    if tamper == "future":
        ref["action_index"] = 2
    elif tamper == "digest":
        ref["record_sha256"] = "b" * 64
    elif tamper == "card":
        ref["card_id"] = "entry_mechanism"
    elif tamper == "representation":
        ref["delivery_kind"] = "brief"
    else:
        ctx["actions"][0]["status"] = "failed"
    with pytest.raises(ExchangeBoundaryError, match="sre_question_basis_not_delivered"):
        api().apply_question_plan(initial(), plan(question(basis_refs=[ref])), context=ctx)


def test_ninth_stable_question_is_rejected_and_no_unresolved_question_is_dropped():
    empty = api().project_question_state([])
    rows = [question(f"iq_{i}", question=f"问题{i}") for i in range(8)]
    state = api().apply_question_plan(empty, plan(*rows, current="iq_0"), context=context())
    assert len(state["questions"]) == 8
    with pytest.raises(ExchangeBoundaryError, match="sre_question_limit"):
        api().apply_question_plan(state, plan(question("iq_8"), current="iq_8"), context=context())
    assert (
        len(api().apply_question_plan(state, plan(stop=True), context=context())["questions"]) == 8
    )


def assessment(**changes):
    return dict(
        dict(
            completed_scope="仅完成评分身份核对",
            remaining_questions=[
                dict(
                    question_id="iq_fault",
                    why_stop="自主判断本轮先交接未查边界",
                    entry_refs=[dict(card_id="entry_mechanism")],
                )
            ],
            unchecked_scope=["症状与机制未展开"],
        ),
        **changes,
    )


def test_stop_keeps_unresolved_questions_and_allows_honest_autonomous_stop():
    state = api().apply_question_plan(initial(), plan(stop=True), context=context())
    assert state["current_question_id"] is None
    assert state["questions"][0]["state"] == "open"
    assert api().validate_stop_assessment(state, assessment(), context=context()) is None


@pytest.mark.parametrize("tamper", ["omit", "unknown", "duplicate", "unseen_entry", "empty_reason"])
def test_stop_handoff_cannot_hide_question_or_invent_entry(tamper):
    value = assessment()
    if tamper == "omit":
        value["remaining_questions"] = []
    elif tamper == "unknown":
        value["remaining_questions"][0]["question_id"] = "iq_unknown"
    elif tamper == "duplicate":
        value["remaining_questions"] *= 2
    elif tamper == "unseen_entry":
        value["remaining_questions"][0]["entry_refs"] = [dict(card_id="hidden")]
    else:
        value["remaining_questions"][0]["why_stop"] = ""
    with pytest.raises(ExchangeBoundaryError, match="sre_stop_assessment"):
        api().validate_stop_assessment(initial(), value, context=context())


def test_true_no_entry_may_be_empty_but_does_not_claim_program_exhaustion():
    ctx = context()
    ctx["capabilities"]["published_node_summaries"] = []
    value = assessment()
    value["remaining_questions"][0]["entry_refs"] = []
    assert api().validate_stop_assessment(initial(), value, context=ctx) is None


def test_projection_uses_only_original_public_updates_and_keeps_stop():
    rows = [
        dict(
            decision_index=1,
            action_index=1,
            role="agent_statement",
            investigation_plan=plan(question()),
            stop_assessment=None,
        ),
        dict(
            decision_index=2,
            action_index=None,
            role="agent_statement",
            investigation_plan=plan(stop=True),
            stop_assessment=assessment(),
        ),
    ]
    before = deepcopy(rows)
    state = api().project_question_state(rows)
    assert state["questions"][0]["question_id"] == "iq_fault"
    assert state["latest_action_question_id"] == "iq_fault"
    assert state["stop_assessment"] == assessment()
    assert rows == before
    assert (
        api().project_question_state([dict(decision_index=1, role="agent_statement")])["questions"]
        == []
    )  # no backfilled D25 plan
