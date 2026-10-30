"""Explicit output capacity through real exchange/owner code, with synthetic HTTP."""

# ruff: noqa: F811 -- imported pytest fixtures

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_contracts import OpsEmpiricalProfile
from nurse_rcl.v2.live.ops_exchange import (
    bind_empirical_decision,
    prepare_ops_decision,
    validate_ops_decision_payload,
)
from nurse_rcl.v2.live.ops_report import run_ops_report
from nurse_rcl.v2.live.ops_report_exchange import (
    bind_empirical_report,
    prepare_ops_report,
    validate_ops_report_payload,
)
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_empirical_provider import (
    complete,  # noqa: F401
    inputs,  # noqa: F401
    owner,
    prepared_cases,  # noqa: F401
    report_inputs,  # noqa: F401
    report_text,
)
from tests.test_v2_live_ops_provider import mocked


def wire(inputs, channel):
    ctx, view = inputs
    if channel == "report":
        return prepare_ops_report(ctx, empirical=True)
    return prepare_ops_decision(opaque_case_id=ctx.opaque_case_id,
        case_snapshot_id=ctx.case_snapshot_id, decision_index=1, context=view, empirical=True)


@pytest.mark.parametrize("channel", ["decision", "report"])
def test_explicit_65536_is_bound_to_new_protocol_and_prompt(inputs, channel):
    from nurse_rcl.v2.live import ops_exchange, ops_report_exchange

    raw, binding = wire(inputs, channel)
    request = json.loads(raw)
    assert request.get("max_tokens") == 65536
    assert "max_completion_tokens" not in request
    body = json.loads(request["messages"][1]["content"])
    assert body["protocol"] == f"nurse.v2.ops-{channel}-exchange/3"
    assert binding.request_sha256 == sha256_hex(raw)
    if channel == "report":
        rebound = bind_empirical_report(raw)
        old_prompt = dict(protocol="nurse.v2.ops-report-exchange/2",
            system=ops_report_exchange.EMPIRICAL_SYSTEM,
            output_schema=ops_report_exchange.EMPIRICAL_OUTPUT_SCHEMA,
            output_policy="deployment_default_omitted_max_tokens_unverified")
    else:
        rebound = bind_empirical_decision(raw, opaque_case_id=inputs[0].opaque_case_id,
            case_snapshot_id=inputs[0].case_snapshot_id, decision_index=1)
        old_prompt = dict(protocol="nurse.v2.ops-decision-exchange/2",
            system=ops_exchange.SYSTEM,
            output_policy="deployment_default_omitted_max_tokens_unverified")
    assert rebound == binding
    assert binding.prompt_schema_sha256 != sha256_hex(canonical_json_bytes(old_prompt))


@pytest.mark.parametrize("channel", ["decision", "report"])
@pytest.mark.parametrize("value", [None, 4096, 65535, 65537, True, 65536.0, "65536"])
def test_omission_or_capacity_drift_is_rejected(inputs, channel, value):
    raw, _ = wire(inputs, channel)
    request = json.loads(raw)
    if value is None:
        request.pop("max_tokens", None)
    else:
        request["max_tokens"] = value
    validate = validate_ops_report_payload if channel == "report" else validate_ops_decision_payload
    with pytest.raises(ExchangeBoundaryError):
        validate(request, empirical=True)


@pytest.mark.parametrize("channel", ["decision", "report"])
@pytest.mark.parametrize("damage", ["old_protocol", "alternate_field"])
def test_old_wire_cannot_be_rebound_as_new_request(inputs, channel, damage):
    raw, _ = wire(inputs, channel)
    request = json.loads(raw)
    if damage == "old_protocol":
        body = json.loads(request["messages"][1]["content"])
        body["protocol"] = f"nurse.v2.ops-{channel}-exchange/2"
        request["messages"][1]["content"] = canonical_json_bytes(body).decode()
    else:
        request.pop("max_tokens", None)
        request["max_completion_tokens"] = 65536
    raw = canonical_json_bytes(request)
    with pytest.raises(ExchangeBoundaryError):
        if channel == "report":
            bind_empirical_report(raw)
        else:
            bind_empirical_decision(raw, opaque_case_id=inputs[0].opaque_case_id,
                case_snapshot_id=inputs[0].case_snapshot_id, decision_index=1)


def test_registered_profile_pins_capacity_policy_and_rejects_old_plan():
    profile = OpsEmpiricalProfile()
    assert profile.output_policy == "explicit_max_tokens_65536/1"
    assert profile.decision_protocol == "nurse.v2.ops-decision-exchange/3"
    assert profile.report_protocol == "nurse.v2.ops-report-exchange/3"
    path = Path(__file__).resolve().parents[1] / "configs/v2/ops_isolated_empirical_profile.json"
    assert OpsEmpiricalProfile.model_validate_json(path.read_bytes()) == profile
    old = profile.model_dump(mode="json")
    old.update(output_policy="deployment_default_omitted_max_tokens_unverified",
        decision_protocol="nurse.v2.ops-decision-exchange/2",
        report_protocol="nurse.v2.ops-report-exchange/2")
    with pytest.raises(ValidationError):
        OpsEmpiricalProfile.model_validate(old)


def test_capacity_reaches_http_and_long_report_is_checked_saved_and_reused(
    tmp_path, inputs, monkeypatch,
):
    from nurse_rcl.v2.live import ops_provider as module

    context = inputs[0]
    before = context.model_dump_json()
    calls = []

    def http(request):
        calls.append(request)
        assert json.loads(request.content)["max_tokens"] == 65536
        result = response(report_text(context, 2, long=True))
        result["usage"] = dict(prompt_tokens=19000, completion_tokens=9000, total_tokens=28000)
        return httpx.Response(200, json=result)

    mocked(module, tmp_path, monkeypatch, http)
    provider = owner(module, tmp_path / "provider", context)
    options = dict(root=tmp_path, context=context, code_sha256="a" * 64, empirical=True)
    report = run_ops_report(**options, complete=provider.report)
    assert report.status == "completed" and report.model_delivered_count == 2
    assert context.model_dump_json() == before
    assert report.core_result_digest == context.core_result_digest
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
        lambda _: pytest.fail("saved response cannot access KEY2"))
    assert run_ops_report(**options, complete=provider.report) == report
    provider.report(report.input_wire_json.encode())
    usage = module.usage_summary(tmp_path / "provider", execution_sha256="b" * 64, empirical=True)
    assert len(calls) == usage["attempts"] == 1
    assert usage["input_tokens"] == 19000 and usage["output_tokens"] == 9000
