"""Exact reading aids, not semantic grading or a rewrite of model text."""

import importlib
import importlib.util
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes


def api():
    name = "nurse_rcl.v2.live.ops_sre_factual_reading"
    assert importlib.util.find_spec(name), "exact source-window reading aid is missing"
    return importlib.import_module(name)


def context():
    row = dict(
        card_id="card-log",
        node="symptoms",
        tool_version="ops-symptoms/1",
        targets=["svc-log"],
        source_role="core_published",
        source_refs=["a" * 64],
        window_ns=[1754983484000000000, 1754983724000000000],
        payload=dict(kind="error_summary", sampling_coverage="unknown"),
    )
    return dict(
        actions=[dict(action_index=1, status="completed")],
        previous_feedback=dict(
            action_index=1, semantic_record=row, critical_record_ids=[], origin="read_card"
        ),
    )


def test_actual_ns_and_seconds_are_separate_exact_quantities():
    ctx = context()
    before = canonical_json_bytes(ctx)
    aid = api().latest_window_reading_aid(ctx)
    assert len(aid) == 1
    assert aid[0] == dict(
        action_index=1,
        card_id="card-log",
        source_field="window_ns",
        source_unit="Unix ns",
        exact_window_ns=[1754983484000000000, 1754983724000000000],
        exact_unix_seconds=["1754983484", "1754983724"],
        duration_ns=240000000000,
        duration_seconds="240",
    )
    assert aid[0]["exact_window_ns"] != [1754983484, 1754983724]
    assert canonical_json_bytes(ctx) == before


@pytest.mark.parametrize(
    "ns,seconds",
    [
        (1754983484000000001, "1754983484.000000001"),
        (1000000100, "1.0000001"),
        (-1, "-0.000000001"),
        (-1000000001, "-1.000000001"),
        (0, "0"),
    ],
)
def test_seconds_conversion_does_not_round_via_float(ns, seconds):
    assert api().exact_unix_seconds(ns) == seconds


@pytest.mark.parametrize("bad", [True, 1.0, "1754983484000000000", None])
def test_conversion_does_not_infer_numeric_type_or_unit(bad):
    with pytest.raises(ValueError, match="integer"):
        api().exact_unix_seconds(bad)


@pytest.mark.parametrize("reason", ["failed", "unknown_version", "partial", "missing", "null"])
def test_unavailable_or_unregistered_windows_never_borrow_hidden_values(reason):
    ctx = context()
    row = ctx["previous_feedback"]["semantic_record"]
    if reason == "failed":
        ctx["actions"][-1]["status"] = "rejected"
    elif reason == "unknown_version":
        row["tool_version"] = "unknown/99"
    elif reason == "partial":
        row["representation"] = "overview_not_full_judgment"
    elif reason == "missing":
        row.pop("window_ns")
    else:
        row["window_ns"] = None
    assert api().latest_window_reading_aid(ctx) == []


def test_registered_named_windows_are_not_synthesized_for_error_summary():
    ctx = context()
    row = ctx["previous_feedback"]["semantic_record"]
    row["payload"]["baseline_window"] = dict(start_unix_ns=0, end_unix_ns=1000000001)
    assert [r["source_field"] for r in api().latest_window_reading_aid(ctx)] == ["window_ns"]
    row["payload"]["kind"] = "fixed_summary"
    aid = api().latest_window_reading_aid(ctx)
    assert aid[1]["source_field"] == "payload.baseline_window"
    assert aid[1]["exact_unix_seconds"] == ["0", "1.000000001"]
    assert not any(r["source_field"] == "payload.fault_window" for r in aid)


@pytest.mark.parametrize("window", [[True, 2], [1.0, 2], [2, 1], [1], [1, 2, 3], [1, 1], [-1, 2]])
def test_bad_window_is_not_repaired_or_magnitude_guessed(window):
    ctx = context()
    ctx["previous_feedback"]["semantic_record"]["window_ns"] = window
    before = deepcopy(ctx)
    assert api().latest_window_reading_aid(ctx) == []
    assert ctx == before


def test_no_actual_action_has_no_window_and_mismatched_latest_identity_is_rejected():
    assert api().latest_window_reading_aid(dict(actions=[], previous_feedback={})) == []
    ctx = context()
    ctx["previous_feedback"]["action_index"] = 2
    with pytest.raises(ValueError, match="latest"):
        api().latest_window_reading_aid(ctx)


def test_resource_limit_feedback_does_not_make_undelivered_payload_authoritative():
    ctx = context()
    ctx["previous_feedback"]["status"] = "resource_limit"
    assert api().latest_window_reading_aid(ctx) == []


def test_unregistered_episode_shape_is_not_silently_added_to_summary_time_registry():
    ctx = context()
    ctx["previous_feedback"]["semantic_record"]["payload"]["kind"] = "paid_episode"
    assert api().latest_window_reading_aid(ctx) == []
