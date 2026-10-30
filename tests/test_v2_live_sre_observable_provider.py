"""Synthetic HTTP only: authorization, native debit and fixed request identity."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import importlib
import importlib.util
import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision, prepare_brief_decision
from nurse_rcl.v2.live.ops_sre_context import sre_investigation_input
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_provider import mocked
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision
from tests.test_v2_live_sre_entry_provider import options
from tests.test_v2_live_sre_observable_tools import opened, tools


def frozen(complete, root):
    root.mkdir()
    tool, book, _, case = opened(complete)
    refs = book.to_state()
    with tools().OpsSreObservableInvestigationLedger(
        root=root, toolbox=tool, code_sha256="a" * 64
    ) as ledger:
        projected = sre_investigation_input(
            capabilities=ledger.capabilities(),
            records=(),
            decisions=(),
            decision_index=1,
            entry_references=refs,
            entry_case_id=case,
            make_wire=lambda view: dict(context=view),
        )
    assert projected.status == "ready", projected
    view = json.loads(projected.view_json)
    wire = prepare_brief_decision(
        opaque_case_id=case,
        case_snapshot_id=tool.case_snapshot_id,
        decision_index=1,
        context=view,
        references=refs,
    )
    binding = bind_brief_decision(
        wire, opaque_case_id=case, case_snapshot_id=tool.case_snapshot_id, decision_index=1
    )
    value = decision(first=True)
    value["actions"] = [dict(entry_ref="A999", question="Unknown entry remains one paid refusal.")]
    value["investigation_note"] = "bad typed purpose must not create a quality retry"
    return case, wire, binding, value


def api():
    name = "nurse_rcl.v2.live.ops_sre_observable_provider"
    assert importlib.util.find_spec(name), "observable bounded provider missing"
    return importlib.import_module(name)


def test_ordinary_provider_rejects_new_profile_before_any_send(complete, tmp_path):
    from nurse_rcl.v2.live.ops_provider import OpsLiveProvider

    case, _, binding, _ = frozen(complete, tmp_path / "tools")
    with pytest.raises(ValueError, match="bounded stage"):
        OpsLiveProvider(
            root=tmp_path / "ordinary",
            code_sha256="a" * 64,
            execution_sha256="b" * 64,
            approved_execution_sha256="b" * 64,
            opaque_case_id=case,
            case_snapshot_id=binding.input_sha256,
            empirical=True,
            sre=True,
            sre_profile="hermes-sre-tools/20",
        )


def test_new_provider_native_success_saved_reuse_usage_and_report_disabled(
    complete, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_provider

    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    calls = []

    def http(request):
        calls.append(request)
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    opts = options(case, binding, tmp_path / "provider")
    owner = api().OpsSreObservableDecisionProvider(**opts)
    assert json.loads(owner.decision(1, wire)) == value
    assert (
        json.loads(api().OpsSreObservableDecisionProvider(**opts).saved_decision(1, wire)) == value
    )
    assert json.loads(api().OpsSreObservableDecisionProvider(**opts).decision(1, wire)) == value
    assert len(calls) == 1
    usage = ops_provider.usage_summary(
        tmp_path / "provider",
        execution_sha256="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
    )
    assert usage["logical_requests"] == usage["attempts"] == 1
    assert usage["reported_input_tokens"] == 10 and usage["reported_output_tokens"] == 2
    with pytest.raises(ValueError, match="report"):
        owner.report(wire)
    with pytest.raises(ValueError, match="stage decision"):
        owner.decision(2, wire)
    with pytest.raises(ValueError, match="stage approval"):
        api().OpsSreObservableDecisionProvider(**dict(opts, stage_decision_limit=2))
    assert len(calls) == 1


def test_new_slots_distinguish_decisions_and_keep_one_global_correction(complete, tmp_path):
    _, _, first, _ = frozen(complete, tmp_path / "tools")
    assert first.variant == "v2-ops-sre/10"
    second = type(first)(**dict(first.model_dump(), decision_index=2))
    assert second.slot != first.slot
    correction = type(first)(
        **dict(first.model_dump(), phase="investigation_correction", correction_index=1)
    )
    later = type(first)(**dict(correction.model_dump(), decision_index=2))
    assert correction.slot == later.slot


@pytest.mark.parametrize("unknown", [False, True])
def test_failed_or_unknown_native_cost_is_never_refunded_or_hidden(
    complete, tmp_path, monkeypatch, unknown
):
    from nurse_rcl.v2.live import ops_provider
    from nurse_rcl.v2.live.ark_bridge import LiveTransportError, read_attempt_ledger

    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    value["actions"] = [dict(entry_ref="A1", tool="forbidden", question="Bad shape.")]
    calls = []

    def http(request):
        calls.append(request)
        if unknown:
            raise KeyboardInterrupt("synthetic network outcome unknown")
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    opts = options(case, binding, tmp_path / "provider")
    with pytest.raises(KeyboardInterrupt if unknown else LiveTransportError):
        api().OpsSreObservableDecisionProvider(**opts).decision(1, wire)
    if unknown:
        assert api().OpsSreObservableDecisionProvider(**opts).saved_decision(1, wire) is None
    else:
        with pytest.raises(LiveTransportError):
            api().OpsSreObservableDecisionProvider(**opts).saved_decision(1, wire)
    usage = ops_provider.usage_summary(
        tmp_path / "provider",
        execution_sha256="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
    )
    if unknown:
        assert usage["attempts"] == usage["unknown_input_attempts"] == len(calls) == 1
        assert usage["input_tokens"] is None
    else:
        with pytest.raises(LiveTransportError):
            api().OpsSreObservableDecisionProvider(**opts).decision(1, wire)
        ledger = read_attempt_ledger(
            tmp_path / "provider/ark-investigation" / f"request-{binding.slot}.json", empirical=True
        )
        assert len(ledger.attempts) == len(calls) == usage["attempts"] == 5
        assert usage["reported_input_tokens"] == 50 and usage["reported_output_tokens"] == 10


def test_missing_only_stop_basis_exhausts_five_attempts_and_saved_failure_never_resends(
    complete, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_provider
    from nurse_rcl.v2.live.ark_bridge import LiveTransportError, read_attempt_ledger
    from nurse_rcl.v2.live.ops_sre_action_decision import ACTION_FIELDS

    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    value.pop("stop_basis")
    assert set(value) == ACTION_FIELDS - {"stop_basis"}
    raw = canonical_json_bytes(value).decode()
    calls = []

    def http(request):
        calls.append(request)
        return httpx.Response(200, json=response(raw))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    opts = options(case, binding, tmp_path / "provider")
    with pytest.raises(LiveTransportError):
        api().OpsSreObservableDecisionProvider(**opts).decision(1, wire)
    ledger_path = tmp_path / "provider/ark-investigation" / f"request-{binding.slot}.json"
    ledger = read_attempt_ledger(ledger_path, empirical=True)
    assert len(calls) == len(ledger.attempts) == 5
    for row in ledger.attempts:
        assert row.response_json is not None and row.record is not None
        assert json.loads(row.response_json)["choices"][0]["message"]["content"] == raw
        assert row.record.usage.prompt_tokens == 10
        assert row.record.usage.completion_tokens == 2
    usage = ops_provider.usage_summary(
        tmp_path / "provider",
        execution_sha256="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/20",
    )
    assert usage["logical_requests"] == 1 and usage["attempts"] == 5
    assert usage["reported_input_tokens"] == 50 and usage["reported_output_tokens"] == 10
    assert usage["unknown_input_attempts"] == 0
    before = ledger_path.read_bytes()
    for method in ("saved_decision", "decision"):
        owner = api().OpsSreObservableDecisionProvider(**opts)
        with pytest.raises(LiveTransportError):
            getattr(owner, method)(1, wire)
    assert len(calls) == 5 and ledger_path.read_bytes() == before
