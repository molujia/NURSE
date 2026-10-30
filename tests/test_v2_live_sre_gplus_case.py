"""Composition authorization does not change ordinary investigation profiles."""

from importlib import import_module, util

import pytest


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_case"
    assert util.find_spec(name), "missing G+ one-case composition entry"
    return import_module(name)


def plan(limit):
    return api().GplusCompositionPlan(
        core_plan_path="/isolated/core-plan.json",
        core_plan_sha256="a" * 64,
        run_root="/isolated/run",
        investigation_decision_limit=limit,
    )


@pytest.mark.parametrize("limit", [0, -1, 9, True, 2.0, "2"])
def test_composition_requires_explicit_strict_legal_stage_ceiling(limit):
    api()
    with pytest.raises(ValueError):
        plan(limit)


def test_two_round_grant_distinguishes_actions_correction_attempts_and_report():
    module = api()
    bound = module.composition_grant(plan(2))
    assert bound["investigation_decisions"] == 2
    assert bound["investigation_actions"] == 2
    assert bound["strict_duplicate_correction_limit"] == 1
    assert bound["report_logical_limit"] == 1
    assert bound["ordinary_logical_limit"] == 3
    assert bound["logical_limit_with_correction"] == 4
    assert bound["attempt_limit_with_correction"] == 20


def test_experiment_ceiling_does_not_become_product_or_quickstart_two_round_limit():
    module = api()
    from nurse_rcl.v2.live.ops_contracts import OpsSreFeedbackFactProfile

    assert OpsSreFeedbackFactProfile().decision_limit == 8
    assert module.composition_grant(plan(8))["ordinary_logical_limit"] == 9
    assert plan(2).investigation_decision_limit == 2


def test_composition_round_ceiling_and_core_pin_are_in_execution_identity():
    module = api()
    first, second = plan(2), plan(3)
    assert module.composition_sha256(first) != module.composition_sha256(second)
    changed = first.model_copy(update=dict(core_plan_sha256="b" * 64))
    assert module.composition_sha256(first) != module.composition_sha256(changed)


def test_no_plan_approval_cannot_start_deterministic_or_paid_child(tmp_path):
    module = api()
    from nurse_rcl.artifacts import canonical_json_bytes

    value = plan(2)
    path = tmp_path / "composition.json"
    path.write_bytes(canonical_json_bytes(value.model_dump(mode="json")))
    with pytest.raises(ValueError, match="approval"):
        module.run_gplus_case(
            path,
            module.composition_sha256(value),
            approved_composition_sha256=None,
            progress=lambda _: pytest.fail("unapproved child started"),
        )


def test_retained_decoder_does_not_open_navigation_when_no_brief_was_delivered():
    module = api()
    from types import SimpleNamespace

    # The decoder must return before accessing stage, core or navigation.
    assert module._retained_sources(None, SimpleNamespace(records=()), None) == ()


def test_original_atomic_full_case_guard_remains_closed():
    api()
    from types import SimpleNamespace

    from nurse_rcl.v2.live.ops_case import run_ops_case
    from nurse_rcl.v2.live.ops_contracts import OpsSreFeedbackFactProfile

    admitted = SimpleNamespace(plan=SimpleNamespace(profile=OpsSreFeedbackFactProfile()))
    with pytest.raises(ValueError, match="bounded decision stage"):
        run_ops_case(admitted, approved_plan_sha256="a" * 64)
