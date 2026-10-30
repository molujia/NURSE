"""Guide delivery only; native prose remains independently admitted, not graded."""

# ruff: noqa: F811 -- imported pytest fixture names intentionally match arguments.

import json

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live import ops_sre_observable_exchange as exchange
from nurse_rcl.v2.live.ops_sre_entry_decision import parse_observable_decision
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_observable_provider import frozen
from tests.test_v2_live_sre_observable_questions import api, first

SURFACES = [
    "system",
    "common",
    "boundary",
    "symptoms",
    "mechanisms",
    "dependency",
    "timing",
    "scoring",
]
REQUIREMENTS = {
    "count_unit": (
        "dependency-node card count is not topology degree",
        "unique neighbors, callees or edges",
        "not a measured failure count",
    ),
    "comparability": (
        "An unread record is not yet a matched symptom observation",
        "modality, statistic, observation role, unit",
        "named windows, denominators and coverage",
        "same record kind or unit alone does not establish comparability",
        "Registered caller/callee navigation does not establish symptom propagation",
    ),
}


@pytest.mark.parametrize("topic", REQUIREMENTS)
@pytest.mark.parametrize("surface", SURFACES)
def test_count_and_matching_limits_are_supplied_separately(topic, surface):
    policy = exchange.observable_decision_policy()
    text = (
        policy["system"]
        if surface == "system"
        else policy["guidance"]["guidance"]
        if surface == "common"
        # Skills reference the actual common supply; all original limits remain tested.
        else exchange.OBSERVABLE_SKILL_GUIDANCE[surface] + " " + policy["guidance"]["guidance"]
    )
    for required in REQUIREMENTS[topic]:
        assert required in text


def test_real_fixture_frozen_wire_has_both_narrow_limits(complete, tmp_path):
    _, raw, _, _ = frozen(complete, tmp_path / "tools")
    wire = json.loads(raw)
    for requirements in REQUIREMENTS.values():
        for required in requirements:
            assert required in wire["messages"][0]["content"]
    assert wire["max_tokens"] == 65536
    assert wire["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "weak_prose",
    [
        "The dependency card count proves this service has that topology degree.",
        "The unread same-kind card is already a matched symptom observation.",
    ],
)
def test_weak_prose_is_not_rewritten_or_promoted_by_program(weak_prose):
    value, context = first()
    value["investigation_note"]["why_now"] = weak_prose
    raw = canonical_json_bytes(value)
    assert parse_observable_decision(raw, context=context) == value
    note = api().admit_observable_note(value, context=context)
    assert note["status"] == "accepted"
    assert note["plan"]["updates"][0]["state"] == "open"
    assert canonical_json_bytes(value) == raw
    # A human semantic review still fails these claims; acceptance is not correctness.


def test_older_entry_policy_does_not_receive_new_guidance():
    from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_GUIDANCE, entry_decision_policy

    for requirements in REQUIREMENTS.values():
        assert requirements[0] not in ENTRY_GUIDANCE
    assert entry_decision_policy()["protocol"] == "nurse.v2.sre-decision-exchange/20"
    assert exchange.observable_decision_policy()["protocol"] == "nurse.v2.sre-decision-exchange/21"
