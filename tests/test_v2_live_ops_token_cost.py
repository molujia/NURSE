"""Fixed-denominator empirical cost, including failures and unknown spend."""

from copy import deepcopy

import pytest

from tests.test_v2_live_ops_usage import api, attempt, usage


def case(key="one", count=20000, *, status="completed", raw=None):
    return dict(case_id=key, dataset="rcabench", version="fixture-v2",
        case_snapshot_id="a" * 64, execution_sha256=("1" if key == "one" else "2") * 64,
        status=status, channels=dict(
            core="completed", investigation="completed",
            report="completed" if status == "completed" else "unavailable"),
        attempts=[attempt(raw=usage(count) if raw is None else raw)],
        confirmed_no_send=False, incremental_attempt_ids=[("investigation", 1, 1)])


def summary(cases, ids=("one", "two")):
    return api().summarize_cost(cases, planned_case_ids=ids, dataset="rcabench",
                                version="fixture-v2")


def test_high_cost_failed_case_stays_in_fixed_denominator():
    result = summary([case(), case("two", 220000, status="failed")])
    assert result["input_mean"] == 120000
    assert result["input_target_met"] is False
    assert result["failed_cases"] == 1 and result["completed_cases"] == 2
    assert result["successful_input_mean"] == 20000
    assert result["failure_rate"] == 0.5
    assert result["input_distribution"] == dict(
        count=2, median=120000, p95=220000, maximum=220000, p95_method="nearest_rank")


def test_unknown_input_cannot_make_target_pass():
    unknown = case("two")
    unknown["attempts"].append(attempt(2))
    result = summary([case(), unknown])
    assert result["input_mean"] is None and result["input_target_met"] is None
    assert result["unknown_input_attempts"] == 1
    assert result["input_complete_cases"] == 1
    assert result["reported_input_tokens"] == 40000
    assert result["known_input_lower_bound_per_planned_case"] == 20000


def test_unrun_and_running_cases_are_not_zero_cost_or_final_cost():
    missing = summary([case()])
    assert missing["planned_cases"] == 2 and missing["started_cases"] == 1
    assert missing["input_mean"] is None and missing["input_target_met"] is None
    running = summary([case(), case("two", status="running")])
    assert running["completed_cases"] == 1 and running["failed_cases"] == 0
    assert running["input_mean"] is None and running["failure_rate"] is None
    assert running["input_distribution"]["count"] == 1


def test_output_unknown_does_not_hide_complete_input_or_cache_subset():
    raw = dict(prompt_tokens=40000, completion_tokens="unknown",
               prompt_tokens_details=dict(cached_tokens=30000))
    result = summary([case(raw=raw)], ids=("one",))
    assert result["input_mean"] == 40000 and result["input_target_met"] is True
    assert result["unknown_output_attempts"] == 1
    assert result["reported_cached_tokens"] == 30000
    assert result["input_complete_cases"] == 1 and result["usage_complete_cases"] == 0


def test_empty_attempts_need_explicit_no_send_evidence():
    row = case()
    row["attempts"] = []
    row["incremental_attempt_ids"] = []
    assert summary([row], ids=("one",))["input_mean"] is None
    row["confirmed_no_send"] = True
    assert summary([row], ids=("one",))["input_mean"] == 0


def test_reusing_paid_responses_is_zero_incremental_not_zero_from_scratch():
    row = case(count=120000)
    row["incremental_attempt_ids"] = []
    row["response_reuses"] = 2
    result = summary([row], ids=("one",))
    assert result["input_mean"] == 120000 and not result["input_target_met"]
    assert result["incremental_input_tokens"] == 0
    assert result["paid_execution_sources"] == [row["execution_sha256"]]
    assert result["response_reuses"] == 2
    assert summary([deepcopy(row)], ids=("one",)) == result


def test_correction_usage_is_included_in_lifetime_and_incremental_cost():
    row = case(count=100)
    row["attempts"].append(
        attempt(phase="investigation_correction", raw=usage(200, 3))
    )
    row["incremental_attempt_ids"].append(("investigation_correction", 1, 1))
    result = summary([row], ids=("one",))
    assert result["attempts"] == 2
    assert result["reported_input_tokens"] == result["incremental_input_tokens"] == 300
    assert result["input_mean"] == 300


def test_missing_incremental_baseline_is_unknown_not_zero():
    row = case()
    del row["incremental_attempt_ids"]
    assert summary([row], ids=("one",))["incremental_input_tokens"] is None


@pytest.mark.parametrize("damage", ["dataset", "version", "outside", "duplicate_case",
    "duplicate_attempt", "duplicate_planned", "empty_planned", "snapshot", "unknown_status",
    "duplicate_incremental", "foreign_incremental", "duplicate_execution"])
def test_reject_mixed_or_duplicate_scope(damage):
    cases, ids = [case(), case("two")], ("one", "two")
    if damage in {"dataset", "version"}:
        cases[1][damage] = "wrong"
    elif damage == "outside":
        cases[1]["case_id"] = "foreign"
    elif damage == "duplicate_case":
        cases.append(deepcopy(cases[0]))
    elif damage == "duplicate_attempt":
        cases[0]["attempts"] *= 2
    elif damage == "duplicate_planned":
        ids = ("one", "one")
    elif damage == "empty_planned":
        ids = ()
    elif damage == "snapshot":
        cases[0]["case_snapshot_id"] = "not-pinned"
    elif damage == "unknown_status":
        cases[0]["status"] = "mystery"
    elif damage == "duplicate_incremental":
        cases[0]["incremental_attempt_ids"] *= 2
    elif damage == "foreign_incremental":
        cases[0]["incremental_attempt_ids"] = [("report", 1, 1)]
    else:
        cases[1]["execution_sha256"] = cases[0]["execution_sha256"]
    with pytest.raises(ValueError):
        summary(cases, ids)


def test_p95_uses_nearest_rank_with_visible_sample_count():
    cases = []
    for index in range(20):
        row = case(str(index), index + 1)
        row["execution_sha256"] = f"{index:064x}"
        cases.append(row)
    result = summary(cases, tuple(str(i) for i in range(20)))
    assert result["input_distribution"]["p95"] == 19
    assert result["input_distribution"]["count"] == 20
