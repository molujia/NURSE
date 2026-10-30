"""Empirical usage: independent reported parts, never token admission."""

import importlib
import importlib.util

import pytest


def api():
    name = "nurse_rcl.v2.live.ops_usage"
    assert importlib.util.find_spec(name), "missing empirical usage component"
    return importlib.import_module(name)


def usage(prompt=1000, output=25):
    return dict(prompt_tokens=prompt, completion_tokens=output, total_tokens=prompt + output)


def attempt(number=1, *, slot=1, phase="investigation", raw=None):
    return dict(phase=phase, slot=slot, attempt=number, usage=raw)


def test_input_survives_invalid_output_and_inconsistent_total():
    parts = api().usage_parts(dict(prompt_tokens=1000, completion_tokens="bad", total_tokens=2))
    assert parts["input_tokens"] == 1000
    assert parts["output_tokens"] is None
    assert parts["total_tokens"] is None
    assert "output_invalid" in parts["usage_issues"]
    assert "total_below_input" in parts["usage_issues"]


def test_output_is_not_debited_to_input():
    parts = api().usage_parts(usage(1000, 120000))
    assert parts["input_tokens"] == 1000
    assert parts["output_tokens"] == 120000
    assert parts["total_tokens"] == 121000


@pytest.mark.parametrize("key", ["input", "output"])
def test_agreeing_aliases_are_accepted(key):
    raw = usage()
    raw[key + "_tokens"] = 1000 if key == "input" else 25
    assert api().usage_parts(raw)[key + "_tokens"] == raw[key + "_tokens"]


@pytest.mark.parametrize("key,alias", [("input", "input_tokens"), ("output", "output_tokens")])
def test_conflicting_aliases_are_unknown(key, alias):
    raw = {**usage(), alias: 3}
    parts = api().usage_parts(raw)
    assert parts[key + "_tokens"] is None
    assert key + "_conflict" in parts["usage_issues"]
    assert parts["raw_usage"] == raw


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "100", None])
def test_invalid_input_is_not_coerced_or_inferred(value):
    parts = api().usage_parts({**usage(), "prompt_tokens": value})
    assert parts["input_tokens"] is None
    assert parts["output_tokens"] == 25
    assert "input_invalid" in parts["usage_issues"]


@pytest.mark.parametrize("raw", [None, {}, [], "usage unavailable", {"total_tokens": 8}])
def test_missing_usage_is_unknown_not_zero(raw):
    parts = api().usage_parts(raw)
    assert parts["input_tokens"] is None
    assert parts["output_tokens"] is None
    assert parts["raw_usage"] == raw


def test_inconsistent_total_does_not_poison_valid_parts():
    parts = api().usage_parts({**usage(), "total_tokens": 9999})
    assert parts["input_tokens"] == 1000 and parts["output_tokens"] == 25
    assert parts["total_tokens"] is None
    assert "total_inconsistent" in parts["usage_issues"]


def test_total_cannot_be_less_than_known_output():
    parts = api().usage_parts(dict(completion_tokens=25, total_tokens=20))
    assert parts["output_tokens"] == 25 and parts["input_tokens"] is None
    assert parts["total_tokens"] is None
    assert "total_below_output" in parts["usage_issues"]


def test_cache_is_subset_not_extra_or_discounted_input():
    raw = {**usage(), "prompt_tokens_details": {"cached_tokens": 800},
           "input_tokens_details": {"cached_tokens": 800}}
    parts = api().usage_parts(raw)
    assert parts["input_tokens"] == 1000 and parts["cached_tokens"] == 800
    raw["prompt_tokens_details"]["cached_tokens"] = 0
    assert parts["raw_usage"]["prompt_tokens_details"]["cached_tokens"] == 800


@pytest.mark.parametrize("details", [
    {"prompt_tokens_details": {"cached_tokens": 1001}},
    {"prompt_tokens_details": {"cached_tokens": True}},
    {"prompt_tokens_details": {"cached_tokens": 800},
     "input_tokens_details": {"cached_tokens": 500}},
    {"prompt_tokens_details": []},
])
def test_invalid_or_conflicting_cache_is_not_claimed_as_known_input(details):
    parts = api().usage_parts({**usage(), **details})
    assert parts["input_tokens"] is None
    assert parts["cached_tokens"] is None
    assert parts["output_tokens"] == 25


def test_missing_cache_does_not_poison_input():
    parts = api().usage_parts(usage())
    assert parts["input_tokens"] == 1000 and parts["cached_tokens"] is None


def test_summary_includes_failed_and_unknown_attempts_across_phases():
    rows = [attempt(raw=usage(120001, 5)), attempt(2),
            attempt(3, raw=usage(2000, 7)),
            attempt(phase="report", raw=usage(3000, 120000))]
    result = api().summarize_attempts(rows)
    assert result["attempts"] == 4 and result["logical_requests"] == 2
    assert result["reported_input_tokens"] == 125001
    assert result["unknown_input_attempts"] == 1
    assert result["input_tokens"] is None
    assert result["input_usage_complete"] is False
    assert result["reported_output_tokens"] == 120012
    assert not {"remaining", "balance", "input_upper_bound", "denied"} & result.keys()


def test_known_input_missing_output_has_complete_input_only():
    result = api().summarize_attempts([attempt(raw={"input_tokens": 2})])
    assert result["input_tokens"] == result["reported_input_tokens"] == 2
    assert result["input_usage_complete"] is True
    assert result["usage_complete"] is False
    assert result["unknown_output_attempts"] == 1


def test_summary_uses_fresh_derivation_without_double_count_on_reread():
    rows = [attempt(raw=usage())]
    first = api().summarize_attempts(rows)
    assert api().summarize_attempts(rows) == first
    assert first["input_tokens"] == 1000 and first["usage_complete"] is True


def test_duplicate_attempt_identity_is_rejected_not_silently_summed():
    row = attempt(raw=usage())
    with pytest.raises(ValueError, match="duplicate"):
        api().summarize_attempts([row, row])


def test_one_correction_is_counted_as_distinct_paid_request():
    rows = [
        attempt(raw=usage(100, 2)),
        attempt(phase="investigation_correction", raw=usage(200, 3)),
        attempt(phase="report", raw=usage(300, 4)),
    ]
    result = api().summarize_attempts(rows)
    assert result["attempts"] == result["logical_requests"] == 3
    assert result["input_tokens"] == 600
    assert result["output_tokens"] == 9
    assert result["total_tokens"] == 609
    assert result["usage_complete"] is True
    with pytest.raises(ValueError, match="duplicate"):
        api().summarize_attempts([*rows, rows[1]])


@pytest.mark.parametrize("change", [
    dict(phase="core"), dict(slot=0), dict(slot=9), dict(slot=True),
    dict(phase="investigation_correction", slot=2),
    dict(phase="report", slot=2), dict(attempt=0), dict(attempt=6), dict(attempt=True),
])
def test_invalid_attempt_identity_is_rejected(change):
    with pytest.raises(ValueError, match="identity"):
        api().summarize_attempts([{**attempt(raw=usage()), **change}])


def test_empty_inventory_requires_explicit_no_send_confirmation():
    unconfirmed = api().summarize_attempts([])
    assert unconfirmed["input_tokens"] is None
    assert unconfirmed["input_usage_complete"] is False
    confirmed = api().summarize_attempts([], confirmed_no_send=True)
    assert confirmed["input_tokens"] == 0
    assert confirmed["usage_complete"] is True
    with pytest.raises(ValueError, match="no.send"):
        api().summarize_attempts([attempt()], confirmed_no_send=True)
