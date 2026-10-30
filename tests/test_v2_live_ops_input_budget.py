"""Durable per-attempt INPUT accounting, without credentials or a live model."""

import importlib
import importlib.util
import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

CASE = "opq_" + "1" * 32


def api():
    name = "nurse_rcl.v2.live.ops_input_budget"
    assert importlib.util.find_spec(name), "missing persistent per-case input budget"
    return importlib.import_module(name)


def ledger(root, **changes):
    kwargs = dict(root=root, opaque_case_id=CASE, case_snapshot_id="2" * 64,
                  execution_sha256="3" * 64, code_sha256="4" * 64, counter_sha256="5" * 64)
    kwargs.update(changes)
    return api().OpsInputBudget(**kwargs)


def request(index=1, *, report=False, payload=b"offline"):
    return api().OpsInputRequest(
        phase="report" if report else "investigation", decision_index=None if report else index,
        protocol="nurse.v2.ops-report-exchange/2" if report else "nurse.v2.ops-decision-exchange/2",
        payload_sha256=sha256_hex(payload),
    )


def settle(budget, req, number=1, usage=None):
    budget.settle(req, number=number, usage=usage,
                  response_sha256=sha256_hex(canonical_json_bytes(dict(usage=usage))))


def test_reservation_is_durable_before_sending_and_unknown_after_reopen(tmp_path):
    with ledger(tmp_path) as budget:
        budget.reserve(request(), number=1, input_upper_bound=20000)
        saved = json.loads((tmp_path / "input-budget/state.json").read_bytes())
        assert saved["attempts"][0]["input_upper_bound"] == 20000
        assert saved["attempts"][0]["state"] == "reserved"
        assert budget.summary()["in_flight_input_reserved_tokens"] == 20000
    with ledger(tmp_path) as resumed:
        summary = resumed.summary()
        assert summary["unknown_input_reserved_tokens"] == 20000
        assert summary["in_flight_input_reserved_tokens"] == 0
        assert summary["available_input_tokens"] == 100000 - 36864 - 20000
        with pytest.raises(ValueError, match="already reserved"):
            resumed.reserve(request(), number=1, input_upper_bound=20000)


def test_report_reserve_stops_retries_before_five_without_transport(tmp_path):
    from nurse_rcl.agents.retry import CircuitBreaker, execute_llm_request

    sent = []
    with ledger(tmp_path) as budget:
        req = request()

        def send(http_request):
            state = json.loads((tmp_path / "input-budget/state.json").read_bytes())
            assert state["attempts"][-1]["state"] == "reserved"
            sent.append(http_request)
            return httpx.Response(503)

        with httpx.Client(transport=httpx.MockTransport(send)) as client:
            def attempt(number):
                return client.post("https://offline.invalid/test").raise_for_status()

            with pytest.raises(api().InputBudgetError, match="token_budget"):
                execute_llm_request(
                    attempt, sleep=lambda _: None, checkpoint=lambda _: None,
                    circuit_breaker=CircuitBreaker(),
                    reserve_attempt=lambda n: budget.reserve(
                        req, number=n, input_upper_bound=20000,
                    ),
                    record_attempt=lambda r, _: budget.mark_unknown(req, number=r.attempt_number),
                )
        assert len(sent) == 3
        assert budget.summary()["attempts"] == 3
        assert budget.summary()["report_input_reserve"] == 36864
        assert budget.summary()["last_denial"] == "token_budget"
        assert budget.summary()["scope"] == "input"
        budget.begin_report()
        budget.reserve(request(report=True), number=1, input_upper_bound=36864)
        assert budget.summary()["occupied_input_tokens"] == 96864
        assert budget.summary()["report_input_reserve"] == 0
        with pytest.raises(ValueError, match="investigation closed"):
            budget.reserve(request(2), number=1, input_upper_bound=1)


def test_only_actual_input_is_charged_even_with_huge_output(tmp_path):
    with ledger(tmp_path) as budget:
        budget.reserve(request(), number=1, input_upper_bound=20000)
        settle(budget, request(), usage=dict(prompt_tokens=1000, completion_tokens=120000,
                                             total_tokens=121000,
                                             prompt_tokens_details=dict(cached_tokens=800)))
        summary = budget.summary()
        assert summary["reported_input_tokens"] == 1000
        assert summary["occupied_input_tokens"] == 1000
        assert summary["reported_output_tokens"] == 120000
        assert summary["reported_total_tokens"] == 121000
        assert summary["input_counter_contract_failed"] is False
        budget.reserve(request(2), number=1, input_upper_bound=20000)


@pytest.mark.parametrize("usage,known,reason", [
    ({"prompt_tokens": 1000}, 1000, "output_missing"),
    ({"input_tokens": 1000, "output_tokens": 2}, 1000, "total_missing"),
    ({"prompt_tokens": 1000, "completion_tokens": "invalid"}, 1000, "output_invalid"),
    ({"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 3}, 1000,
     "total_inconsistent"),
    ({"total_tokens": 1000}, 0, "input_missing"),
    ({"prompt_tokens": True}, 0, "input_invalid"),
    ({"prompt_tokens": -1}, 0, "input_invalid"),
    ({"prompt_tokens": "1000"}, 0, "input_invalid"),
    ({"prompt_tokens": 1000, "input_tokens": 900}, 0, "input_conflict"),
    ({"prompt_tokens": 1000, "input_tokens": 1000}, 1000, "output_missing"),
    ({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 1001}}, 0,
     "cached_exceeds_input"),
    (None, 0, "input_missing"),
])
def test_usage_parts_are_independent_and_raw_values_are_retained(tmp_path, usage, known, reason):
    with ledger(tmp_path) as budget:
        req = request()
        budget.reserve(req, number=1, input_upper_bound=2000)
        settle(budget, req, usage=usage)
        summary = budget.summary()
        assert summary["reported_input_tokens"] == known
        assert summary["occupied_input_tokens"] == (known or 2000)
        row = budget.attempts[0]
        assert json.loads(row.usage_json) == usage
        assert reason in row.usage_issues
        if not known:
            assert summary["unknown_input_reserved_tokens"] == 2000


def test_known_zero_input_releases_reservation_but_not_attempt_slot(tmp_path):
    with ledger(tmp_path) as budget:
        req = request()
        budget.reserve(req, number=1, input_upper_bound=2000)
        settle(budget, req, usage={"prompt_tokens": 0})
        assert budget.summary()["occupied_input_tokens"] == 0
        assert budget.summary()["attempts"] == 1


def test_saved_receipt_can_settle_once_after_crash_without_new_reservation(tmp_path):
    req = request()
    usage = dict(prompt_tokens=1000)
    with ledger(tmp_path) as budget:
        budget.reserve(req, number=1, input_upper_bound=20000)
    with ledger(tmp_path) as resumed:
        settle(resumed, req, usage=usage)
        before = (tmp_path / "input-budget/state.json").read_bytes()
        settle(resumed, req, usage=usage)
        assert (tmp_path / "input-budget/state.json").read_bytes() == before
        assert resumed.summary()["attempts"] == 1
        assert resumed.summary()["occupied_input_tokens"] == 1000
        with pytest.raises(ValueError, match="settlement differs"):
            settle(resumed, req, usage={"prompt_tokens": 999})


def test_actual_input_over_reservation_is_saved_and_blocks_future_requests(tmp_path):
    with ledger(tmp_path) as budget:
        req = request()
        budget.reserve(req, number=1, input_upper_bound=1000)
        settle(budget, req, usage=dict(prompt_tokens=1001, completion_tokens=2))
        assert budget.summary()["reported_input_tokens"] == 1001
        assert budget.summary()["input_counter_contract_failed"] is True
    with ledger(tmp_path) as resumed:
        resumed.begin_report()
        with pytest.raises(api().InputBudgetError, match="input_counter_contract"):
            resumed.reserve(request(report=True), number=1, input_upper_bound=2000)
        assert resumed.summary()["attempts"] == 1


def test_prior_generated_content_in_new_input_has_a_new_reservation(tmp_path):
    with ledger(tmp_path) as budget:
        req = request(payload=b"first context")
        budget.reserve(req, number=1, input_upper_bound=2000)
        settle(budget, req, usage=dict(prompt_tokens=1000, completion_tokens=4000))
        following = request(2, payload=b"previous output included in next input")
        budget.reserve(following, number=1, input_upper_bound=5000)
        settle(budget, following, usage=dict(prompt_tokens=4500, completion_tokens=10))
        assert budget.summary()["reported_input_tokens"] == 5500
        assert budget.summary()["reported_output_tokens"] == 4010


@pytest.mark.parametrize("field", ["opaque_case_id", "case_snapshot_id", "execution_sha256",
                                   "code_sha256", "counter_sha256"])
def test_restore_cannot_change_case_execution_or_counter_identity(tmp_path, field):
    with ledger(tmp_path):
        pass
    replacement = "opq_" + "a" * 32 if field == "opaque_case_id" else "a" * 64
    with pytest.raises(ValueError, match="identity"), ledger(tmp_path, **{field: replacement}):
        pass


def test_same_slot_cannot_change_payload_or_use_old_protocol(tmp_path):
    with ledger(tmp_path) as budget:
        budget.reserve(request(), number=1, input_upper_bound=2000)
        budget.mark_unknown(request(), number=1)
        with pytest.raises(ValueError, match="request binding"):
            budget.reserve(request(payload=b"changed"), number=2, input_upper_bound=2000)
        with pytest.raises(ValueError):
            api().OpsInputRequest(phase="investigation", decision_index=1,
                                 protocol="nurse.v2.ops-decision-exchange/1",
                                 payload_sha256="a" * 64)


def test_attempt_numbers_are_contiguous_and_cannot_exceed_five(tmp_path):
    with ledger(tmp_path) as budget:
        req = request()
        with pytest.raises(ValueError, match="sequence"):
            budget.reserve(req, number=2, input_upper_bound=10)
        for n in range(1, 6):
            budget.reserve(req, number=n, input_upper_bound=10)
            settle(budget, req, number=n, usage={"prompt_tokens": 1})
        with pytest.raises(ValueError, match="attempt"):
            budget.reserve(req, number=6, input_upper_bound=10)
        assert budget.summary()["attempts"] == 5


def test_case_lock_prevents_two_balance_writers_and_symlinks_are_refused(tmp_path):
    with ledger(tmp_path), pytest.raises(BlockingIOError), ledger(tmp_path):
        pass
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "input-budget", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"), ledger(alias):
        pass


def test_corrupt_state_is_not_a_fresh_budget(tmp_path):
    with ledger(tmp_path) as budget:
        budget.reserve(request(), number=1, input_upper_bound=2000)
    path = tmp_path / "input-budget/state.json"
    value = json.loads(path.read_bytes())
    value["attempts"][0]["input_upper_bound"] = 1
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="digest"), ledger(tmp_path):
        pass


def test_reserve_requires_positive_exact_input_bound_and_report_transition(tmp_path):
    with ledger(tmp_path) as budget:
        for bound in (0, -1, True, 1.5, "1000"):
            with pytest.raises(ValueError, match="input bound"):
                budget.reserve(request(), number=1, input_upper_bound=bound)
        with pytest.raises(ValueError, match="report not opened"):
            budget.reserve(request(report=True), number=1, input_upper_bound=2000)
        assert budget.summary()["attempts"] == 0


@pytest.mark.parametrize("failure_point", ["atomic_write_bytes", "_fsync"])
def test_uncertain_write_invalidates_owner_and_cannot_erase_overage(
    tmp_path, monkeypatch, failure_point,
):
    module = api()
    req = request()
    with ledger(tmp_path) as budget:
        budget.reserve(req, number=1, input_upper_bound=1000)
        original = getattr(module, failure_point)

        def fail_after_write(*args):
            original(*args)
            raise OSError("injected persistence failure")

        with monkeypatch.context() as patch:
            patch.setattr(module, failure_point, fail_after_write)
            with pytest.raises(OSError, match="injected persistence"):
                settle(budget, req, usage={"prompt_tokens": 1001})
        path = tmp_path / "input-budget/state.json"
        persisted = path.read_bytes()
        assert json.loads(persisted)["attempts"][0]["input_tokens"] == 1001
        with pytest.raises(ValueError, match="closed"):
            budget.mark_unknown(req, number=1)
        with pytest.raises(ValueError, match="closed"):
            budget.reserve(req, number=2, input_upper_bound=1000)
        assert path.read_bytes() == persisted
    with ledger(tmp_path) as resumed:
        assert resumed.summary()["input_counter_contract_failed"] is True
        with pytest.raises(module.InputBudgetError, match="input_counter_contract"):
            resumed.reserve(req, number=2, input_upper_bound=1000)


def test_new_budget_directory_is_durable_before_first_send_permission(tmp_path, monkeypatch):
    module = api()
    synced = []
    original = module._fsync

    def observe(path):
        original(path)
        synced.append(path)

    monkeypatch.setattr(module, "_fsync", observe)
    with ledger(tmp_path) as budget:
        budget.reserve(request(), number=1, input_upper_bound=1000)
        assert tmp_path in synced, "new budget directory entry must be fsynced in parent"
        assert synced.index(tmp_path) < synced.index(tmp_path / "input-budget")
        assert synced[-1] == tmp_path / "input-budget"
