"""Actual owner/bridge boundary, with synthetic credentials and HTTP only."""

# ruff: noqa: F811

import json

import httpx
import pytest

from nurse_rcl.agents.ark import load_ark_settings
from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_agent_decision import context_json, initial_messages
from nurse_rcl.v2.live.ops_tools import OpsToolbox
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_execution import api
from tests.test_v2_live_ops_report import (
    complete,  # noqa: F401
    context,
    prepared_cases,  # noqa: F401
    proposal,
    report_inputs,  # noqa: F401
)


def provider(tmp_path, ctx, *, approved=True):
    return api("ops_provider").OpsLiveProvider(
        root=tmp_path,
        code_sha256="a" * 64,
        execution_sha256="b" * 64,
        approved_execution_sha256="b" * 64 if approved else None,
        opaque_case_id=ctx.opaque_case_id,
        case_snapshot_id=ctx.case_snapshot_id,
    )


@pytest.fixture(scope="module")
def inputs(report_inputs):
    data, _, _ = report_inputs
    ctx = context(report_inputs)
    toolbox = OpsToolbox(data=data, block_root=None)
    decision = json.loads(
        context_json(
            messages=initial_messages(),
            capabilities=toolbox.capabilities(),
            decision_index=1,
            actions_used=0,
        )
    )
    return ctx, decision


def mocked(module, tmp_path, monkeypatch, handler):
    from nurse_rcl.v2.live.ark_bridge import LiveArkBridge

    monkeypatch.setattr(module, "KEY_PATH", tmp_path / "key.env")
    monkeypatch.setattr(module, "LOCK_PATH", tmp_path / "shared.lock")
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: load_ark_settings({"ARK_API_KEY2": "synthetic-key-two"}),
    )
    monkeypatch.setattr(
        module,
        "LiveArkBridge",
        lambda **kw: LiveArkBridge(
            **kw,
            transport=httpx.MockTransport(handler),
            sleep=lambda _: None,
        ),
    )


def test_unapproved_owner_never_reads_key_or_constructs_http(tmp_path, inputs, monkeypatch):
    module = api("ops_provider")
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("unapproved key access"),
    )
    with pytest.raises(ValueError, match="approval"):
        provider(tmp_path, inputs[0], approved=False)


def test_two_protocols_share_lock_and_keep_independent_slots(tmp_path, inputs, monkeypatch):
    module = api("ops_provider")
    ctx, decision = inputs
    calls = []

    def http(req):
        calls.append(req)
        payload = json.loads(req.content)
        assert req.headers["Authorization"] == "Bearer synthetic-key-two"
        body = json.loads(payload["messages"][1]["content"])
        output = (
            proposal(ctx).model_dump_json()
            if "report" in body["protocol"]
            else (
                canonical_json_bytes(
                    dict(actions=[], stop_reason="done", public_summary="unknown", unresolved=[])
                ).decode()
            )
        )
        return httpx.Response(200, json=response(output))

    mocked(module, tmp_path, monkeypatch, http)
    owner = provider(tmp_path, ctx)
    assert json.loads(owner.decision(1, decision))["stop_reason"] == "done"
    assert json.loads(owner.report(ctx))["core_result_digest"] == ctx.core_result_digest
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("saved response must not require credentials"),
    )
    owner.decision(1, decision)
    owner.report(ctx)
    assert len(calls) == 2
    assert (tmp_path / "ark-investigation/run-identity.json").is_file()
    assert (tmp_path / "ark-report/run-identity.json").is_file()
    assert len(tuple((tmp_path / "outbound").glob("*.json"))) == 2
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64)
    assert usage["logical_requests"] == usage["attempts"] == 2
    assert usage["prompt_tokens"] == 20 and usage["completion_tokens"] == 4
    assert usage["cost"] is None


def test_report_bridge_receipt_before_owner_save_can_recover_offline(tmp_path, inputs, monkeypatch):
    module = api("ops_provider")
    ctx, _ = inputs
    mocked(
        module,
        tmp_path,
        monkeypatch,
        lambda _: httpx.Response(200, json=response(proposal(ctx).model_dump_json())),
    )
    expected = provider(tmp_path, ctx).report(ctx)
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("offline recovery read key"),
    )
    assert (
        module.saved_ops_report(tmp_path, ctx, code_sha256="a" * 64, execution_sha256="b" * 64)
        == expected
    )


def test_response_before_attempt_record_recovers_usage_without_key(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ark_bridge

    module = api("ops_provider")
    ctx, _ = inputs
    mocked(
        module,
        tmp_path,
        monkeypatch,
        lambda _: httpx.Response(200, json=response(proposal(ctx).model_dump_json())),
    )
    original = ark_bridge._write

    def crash_after_response(path, kind, value):
        original(path, kind, value)
        attempts = value.get("attempts", [])
        if attempts and attempts[-1].get("response_json") and attempts[-1].get("record") is None:
            raise KeyboardInterrupt("synthetic response/record crash")

    monkeypatch.setattr(ark_bridge, "_write", crash_after_response)
    with pytest.raises(KeyboardInterrupt):
        provider(tmp_path, ctx).report(ctx)
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("offline recovery read key"),
    )
    assert (
        module.saved_ops_report(tmp_path, ctx, code_sha256="a" * 64, execution_sha256="b" * 64)
        is not None
    )
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64)
    assert usage["attempts"] == 1
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 2
    assert usage["unknown_latency_attempts"] == 1


@pytest.mark.parametrize(
    "contamination",
    [
        {"ground_truth": "answer"},
        {"source_root": "/home/private"},
        {"opaque_case_id": "opq_" + "f" * 32},
        {"case_snapshot_id": "f" * 64},
        {"payload_json": '{"fault_type":"kill"}'},
        {"text": "ts0-mysql-container-kill-9t6n24"},
        {"is_root_cause_probability": True},
    ],
)
def test_actual_outbound_rejects_private_nested_and_cross_case_data(
    tmp_path,
    inputs,
    monkeypatch,
    contamination,
):
    module = api("ops_provider")
    ctx, decision = inputs
    decision = json.loads(json.dumps(decision))
    decision["messages"].append({"role": "tool", "content": json.dumps(contamination)})
    monkeypatch.setattr(
        module.ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("must block before credential access"),
    )
    with pytest.raises(ValueError, match="outbound"):
        provider(tmp_path, ctx).decision(1, decision)


def test_real_scoring_card_false_causality_flag_can_be_explained(
    tmp_path,
    inputs,
    report_inputs,
    monkeypatch,
):
    from nurse_rcl.v2.live.ops_nodes import terminal_nodes

    module = api("ops_provider")
    ctx, decision = inputs
    data, core, _ = report_inputs
    card = next(c for c in terminal_nodes(data, core.evidence) if c.node == "scoring")
    decision = json.loads(json.dumps(decision))
    decision["messages"].append({"role": "tool", "content": card.model_dump_json()})
    mocked(
        module,
        tmp_path,
        monkeypatch,
        lambda req: httpx.Response(
            200,
            json=response(
                canonical_json_bytes(
                    dict(actions=[], stop_reason="done", public_summary="unknown", unresolved=[])
                ).decode()
            ),
        ),
    )
    assert json.loads(provider(tmp_path, ctx).decision(1, decision))["stop_reason"] == "done"


@pytest.mark.parametrize("summary", ["[待核查] CPU 升高", "尚未排除\uff02并发原因\uff02"])
def test_ordinary_model_text_is_not_reparsed_or_nfkc_corrupted_json(
    tmp_path,
    inputs,
    monkeypatch,
    summary,
):
    module = api("ops_provider")
    ctx, decision = inputs
    decision = json.loads(json.dumps(decision))
    decision["messages"].append(
        {
            "role": "assistant",
            "content": json.dumps(
                dict(actions=[], stop_reason="done", public_summary=summary, unresolved=[]),
                ensure_ascii=False,
            ),
        }
    )
    mocked(
        module,
        tmp_path,
        monkeypatch,
        lambda req: httpx.Response(
            200,
            json=response(
                canonical_json_bytes(
                    dict(actions=[], stop_reason="done", public_summary="unknown", unresolved=[])
                ).decode()
            ),
        ),
    )
    assert json.loads(provider(tmp_path, ctx).decision(1, decision))["stop_reason"] == "done"


def test_attempts_stay_spent_and_unknown_usage_is_not_zero(tmp_path, inputs, monkeypatch):
    module = api("ops_provider")
    calls = []

    def http(_):
        calls.append(1)
        raise KeyboardInterrupt("synthetic in-flight interruption")

    mocked(module, tmp_path, monkeypatch, http)
    ctx, decision = inputs
    for _ in range(5):
        with pytest.raises(KeyboardInterrupt):
            provider(tmp_path, ctx).decision(1, decision)
    with pytest.raises(RuntimeError, match="exhausted"):
        provider(tmp_path, ctx).decision(1, decision)
    assert len(calls) == 5
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64)
    assert usage["attempts"] == 5 and usage["logical_requests"] == 1
    assert usage["prompt_tokens"] is None and usage["completion_tokens"] is None
    assert usage["unknown_attempts"] == 5


def test_key1_only_is_not_a_fallback(tmp_path, inputs, monkeypatch):
    module = api("ops_provider")
    key = tmp_path / "ark.env"
    key.write_text("ARK_API_KEY=synthetic-key-one\n")
    key.chmod(0o600)
    monkeypatch.setattr(module, "KEY_PATH", key)
    monkeypatch.setattr(module, "LiveArkBridge", lambda **_: pytest.fail("KEY1 fallback"))
    with pytest.raises(ValueError):
        provider(tmp_path, inputs[0]).decision(1, inputs[1])
