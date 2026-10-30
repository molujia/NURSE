"""Entry native and bounded provider boundaries: synthetic HTTP and settings only."""

# ruff: noqa: F811 -- real small fixtures.

import importlib
import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision, prepare_brief_decision
from nurse_rcl.v2.live.ops_sre_context import sre_investigation_input
from nurse_rcl.v2.live.ops_sre_entry_tools import OpsSreEntryInvestigationLedger
from tests.test_v2_live_ark import bridge, module, response
from tests.test_v2_live_ops_provider import mocked
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision
from tests.test_v2_live_sre_entry_admission import opened


def frozen(complete, root):
    root.mkdir()
    tool, book, _, case = opened(complete)
    refs = book.to_state()
    with OpsSreEntryInvestigationLedger(root=root, toolbox=tool, code_sha256="a" * 64) as ledger:
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
    value["actions"] = [
        dict(
            entry_ref=view["action_entries"]["entries"][0]["entry_ref"],
            question="Check the supplied engineering evidence.",
        )
    ]
    return case, wire, binding, value


def provider_api():
    name = "nurse_rcl.v2.live.ops_sre_entry_provider"
    assert importlib.util.find_spec(name), "bounded entry provider is missing"
    return importlib.import_module(name)


def options(case, binding, root):
    from nurse_rcl.v2.live.ops_sre_directed_provider import stage_grant

    return dict(
        root=root,
        code_sha256="a" * 64,
        execution_sha256="b" * 64,
        approved_execution_sha256="b" * 64,
        opaque_case_id=case,
        case_snapshot_id=binding.input_sha256,
        stage_decision_limit=1,
        approved_stage_sha256=sha256_hex(canonical_json_bytes(stage_grant("b" * 64, 1))),
    )


def test_unknown_entry_is_one_native_success_not_five_parse_attempts(complete, tmp_path):
    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    value["actions"] = [dict(entry_ref="A999", question="Unknown but well-shaped selection.")]
    value["investigation_note"] = "invalid note cannot cancel the action"
    calls = []

    def http(request):
        calls.append(request)
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    opts = dict(request_protocol="nurse.v2.sre-decision-exchange/20", authorized_case_ids=(case,))
    with bridge(tmp_path, http, **opts) as client:
        native = client.complete(binding, wire)
        assert client.receipt(binding)["attempt_count"] == 1
    with bridge(
        tmp_path, lambda _: pytest.fail("saved native success cannot send"), **opts
    ) as client:
        assert client.complete(binding, wire) == native
    assert len(calls) == 1


def test_entry_structure_errors_keep_five_attempt_cap_and_usage(complete, tmp_path):
    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    value["actions"] = [dict(entry_ref="A1", tool="inspect_sources", question="Forbidden escape.")]
    calls = []

    def http(request):
        calls.append(request)
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    opts = dict(request_protocol="nurse.v2.sre-decision-exchange/20", authorized_case_ids=(case,))
    for _ in range(2):
        with bridge(tmp_path, http, **opts) as client, pytest.raises(module().LiveTransportError):
            client.complete(binding, wire)
    ledger = module().read_attempt_ledger(
        tmp_path / "run" / f"request-{binding.slot}.json", empirical=True
    )
    assert len(calls) == len(ledger.attempts) == 5
    assert all(row.response_json is not None and row.record is not None for row in ledger.attempts)
    assert sum(row.record.usage.prompt_tokens for row in ledger.attempts) == 50


def test_entry_stage_provider_grant_report_saved_success_and_usage(complete, tmp_path, monkeypatch):
    case, wire, binding, value = frozen(complete, tmp_path / "tools")
    from nurse_rcl.v2.live import ops_provider

    calls = []

    def http(request):
        calls.append(request)
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    api = provider_api()
    opts = options(case, binding, tmp_path / "provider")
    owner = api.OpsSreEntryDecisionProvider(**opts)
    assert json.loads(owner.decision(1, wire)) == value
    assert json.loads(api.OpsSreEntryDecisionProvider(**opts).saved_decision(1, wire)) == value
    assert json.loads(api.OpsSreEntryDecisionProvider(**opts).decision(1, wire)) == value
    assert len(calls) == 1
    usage = ops_provider.usage_summary(
        tmp_path / "provider",
        execution_sha256="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/19",
    )
    assert usage["logical_requests"] == usage["attempts"] == 1
    assert usage["reported_input_tokens"] == 10 and usage["reported_output_tokens"] == 2
    with pytest.raises(ValueError, match="stage decision"):
        owner.decision(2, wire)
    with pytest.raises(ValueError, match="report"):
        owner.report(wire)
    with pytest.raises(ValueError, match="approval"):
        api.OpsSreEntryDecisionProvider(**dict(opts, approved_execution_sha256=None))
    with pytest.raises(ValueError, match="stage approval"):
        api.OpsSreEntryDecisionProvider(**dict(opts, stage_decision_limit=2))
    assert len(calls) == 1


def test_entry_unknown_native_retains_cost_and_saved_read_never_sends(
    complete, tmp_path, monkeypatch
):
    case, wire, binding, _ = frozen(complete, tmp_path / "tools")
    from nurse_rcl.v2.live import ops_provider

    calls = []

    def interrupted(request):
        calls.append(request)
        raise KeyboardInterrupt("synthetic unknown network outcome")

    mocked(ops_provider, tmp_path, monkeypatch, interrupted)
    api = provider_api()
    opts = options(case, binding, tmp_path / "provider")
    with pytest.raises(KeyboardInterrupt):
        api.OpsSreEntryDecisionProvider(**opts).decision(1, wire)
    assert api.OpsSreEntryDecisionProvider(**opts).saved_decision(1, wire) is None
    usage = ops_provider.usage_summary(
        tmp_path / "provider",
        execution_sha256="b" * 64,
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/19",
    )
    assert usage["attempts"] == usage["unknown_input_attempts"] == 1
    assert usage["input_tokens"] is None and len(calls) == 1
