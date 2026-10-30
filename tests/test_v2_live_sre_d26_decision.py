"""D26 shape/source checks do not grade stop reasons or choose tools."""

# ruff: noqa: RUF001 -- Chinese public task statements.

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from tests.test_v2_live_sre_d26_questions import assessment, context, initial, plan, question


def api():
    name = "nurse_rcl.v2.live.ops_sre_d26_decision"
    assert util.find_spec(name) is not None, "D26 question-led parser is missing"
    return import_module(name)


def ctx(first=False):
    value = context()
    value["schema_version"] = "nurse.v2.sre-investigation-input/17"
    value["question_state"] = dict(initial(), latest_action_question_id="iq_fault")
    value["budget"] = dict(decision_index=2, actions_available_now=7, decisions_after_this=6)
    value["investigation_progress"] = dict(
        latest_actual_result=dict(
            action_index=1,
            content_relation="no_new_content",
        )
    )
    if first:
        from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state

        value["question_state"] = project_question_state([])
        value["actions"] = []
        value["decision_index"] = 1
        value["budget"] = dict(decision_index=1, actions_available_now=8, decisions_after_this=7)
        value["investigation_progress"]["latest_actual_result"] = None
    return value


def decision(first=False, stop=False):
    return dict(
        actions=[]
        if stop
        else [
            dict(
                tool="read_node",
                skill_id="sre-mechanisms/13",
                question="核对故障证据",
                parameters=dict(card_id="entry_mechanism", section="payload", cursor=None),
            )
        ],
        stop_reason="本轮交接未决范围" if stop else None,
        public_summary="评分身份不是故障解释，保留未查机制边界",
        unresolved=["机制未查"],
        responds_to=[] if first else [dict(action_index=1, card_ids=["card_family"])],
        investigation_plan=plan(*([question()] if first else []), stop=stop),
        stop_basis="no_useful_check" if stop else None,
        stop_assessment=assessment() if stop else None,
        latest_result_update=None
        if first
        else dict(
            action_index=1,
            answer_status="partial",
            summary="同源重复无新增，机制仍待查",
            remaining_question="故障证据未展开",
            content_relation="no_new_content",
            question_id="iq_fault",
        ),
    )


@pytest.mark.parametrize("first,stop", [(True, False), (False, False), (False, True)])
def test_new_shape_accepts_public_question_plan_and_honest_stop(first, stop):
    value = decision(first=first, stop=stop)
    assert api().parse_question_decision(canonical_json_bytes(value), context=ctx(first)) == value


@pytest.mark.parametrize(
    "tamper",
    [
        "old_field",
        "missing_assessment",
        "empty_first",
        "wrong_latest_question",
        "uncovered_stop",
        "false_budget",
        "continue_with_assessment",
        "stop_with_current",
    ],
)
def test_invalid_shape_feedback_or_stop_is_rejected_before_native_success(tamper):
    first = tamper == "empty_first"
    stop = tamper in {"uncovered_stop", "false_budget", "stop_with_current"}
    value = decision(first=first, stop=stop)
    if tamper == "old_field":
        value["investigation_question"] = {}
    elif tamper == "missing_assessment":
        value.pop("stop_assessment")
    elif tamper == "empty_first":
        value["investigation_plan"]["updates"] = []
    elif tamper == "wrong_latest_question":
        value["latest_result_update"]["question_id"] = "iq_other"
    elif tamper == "uncovered_stop":
        value["stop_assessment"]["remaining_questions"] = []
    elif tamper == "false_budget":
        value["stop_basis"] = "budget_exhausted"
    elif tamper == "continue_with_assessment":
        value["stop_assessment"] = assessment()
    else:
        value["investigation_plan"] = plan()
    before = deepcopy(value)
    with pytest.raises(ExchangeBoundaryError):
        api().parse_question_decision(canonical_json_bytes(value), context=ctx(first))
    assert value == before


def test_same_saved_old_balanced_response_remains_old_and_not_auto_migrated():
    from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

    value = decision()
    value.pop("investigation_plan")
    value.pop("stop_assessment")
    value["latest_result_update"].pop("question_id")
    value["investigation_question"] = dict(
        target_service_keys=[], card_ids=[], discriminating_value="比较", why_now="回执"
    )
    assert parse_balanced_decision(canonical_json_bytes(value), context=ctx()) == value
    with pytest.raises(ExchangeBoundaryError, match="sre_question_decision_shape"):
        api().parse_question_decision(canonical_json_bytes(value), context=ctx())
