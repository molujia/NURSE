"""Real original client/retry; only the external HTTP socket is substituted."""

import json
import uuid
from importlib import import_module, util
from pathlib import Path

import httpx
import pytest

from nurse_rcl.agents.ark import (
    ARK_CHAT_COMPLETIONS_URL,
    ARK_MODEL,
    ArkChatCompletionsClient,
    ArkCredential,
    ArkSettings,
)
from nurse_rcl.artifacts import sha256_hex

PROTOCOL = "nurse.v2.sre-typed-report-exchange/1"
USAGE = dict(
    prompt_tokens=11,
    completion_tokens=3,
    total_tokens=14,
    prompt_tokens_details=dict(cached_tokens=0),
)


def api():
    name = "nurse_rcl.v2.live.ops_sre_typed_report_provider"
    assert util.find_spec(name), "missing explicit typed durable-receive strategy"
    return import_module(name)


def root():
    path = Path(__file__).resolve().parents[1] / (
        "docs/artifacts/sre-mc03-normal-entry-implementation-20261016/delivery-fixtures/"
        + uuid.uuid4().hex
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def outer(text=" \n{broken native json", *, usage=USAGE, finish="stop", **message):
    return dict(
        choices=[
            dict(
                index=0,
                finish_reason=finish,
                message=dict(role="assistant", content=text, **message),
            )
        ],
        usage=usage,
    )


def body(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode()


def settings():
    # Opaque synthetic fixture, never key-file/environment loading.
    return ArkSettings(ArkCredential("fixture-only-no-network-secret"))


@pytest.mark.parametrize("text", [" \n{broken native json", ' {"old":true}  \n', "", "不是JSON\n"])
def test_receive_keeps_exact_native_without_contract_quality_retry(text):
    module = api()
    parsed = module.parse_typed_receipt(body(outer(text)))
    assert parsed.output_json == text.encode()
    assert parsed.normalized.content == text
    assert parsed.normalized.tool_calls == ()


@pytest.mark.parametrize(
    "change",
    [
        dict(finish="length"),
        dict(refusal="no"),
        dict(tool_calls=[{}]),
        dict(function_call={}),
        dict(content={}),
        dict(role="user"),
    ],
)
def test_outer_refusal_partial_or_tools_are_fatal_not_retryable(change):
    module = api()
    value = outer()
    if "finish" in change:
        value["choices"][0]["finish_reason"] = change["finish"]
    else:
        value["choices"][0]["message"].update(change)
    with pytest.raises(module.TypedReportReceiptError):
        module.parse_typed_receipt(body(value))


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        dict(prompt_tokens=11, completion_tokens=3, total_tokens=14),
        dict(USAGE, input_tokens=12),
        dict(USAGE, total_tokens=15),
        dict(USAGE, prompt_tokens_details=dict(cached_tokens=12)),
    ],
)
def test_accounting_rejects_missing_conflicting_inconsistent_without_zero(usage):
    module = api()
    raw = body(outer(usage=usage))
    with pytest.raises(module.TypedReportAccountingError):
        module.require_typed_accounting(raw)
    assert module.typed_attempt_usage(raw).status != "reported"


def test_accounting_reported_cache_is_subset_not_extra_input():
    module = api()
    raw = body(outer())
    module.require_typed_accounting(raw)
    usage = module.typed_attempt_usage(raw)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens) == (11, 3, 0)


def recorder(module, path, wire, handler):
    return module.TypedReportTransport(
        root=path / "capture",
        expected_wire=wire,
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        transport=httpx.MockTransport(handler),
    )


def test_actual_original_client_bytes_and_unredacted_native_are_durable():
    module = api()
    path = root()
    request = dict(model=ARK_MODEL, messages=[dict(role="user", content="独立装置")])
    wire = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    native = '  {"text":"原样  双空格"}\n'
    raw = body(outer(native))
    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(200, content=raw)

    transport = recorder(module, path, wire, respond)
    with ArkChatCompletionsClient(
        settings(), transport=transport, response_limit_bytes=4194304
    ) as client:
        assert client.complete(json.loads(wire)) == outer(native)
    assert seen == [wire]
    folder = path / "capture/attempt-1"
    assert (folder / "request.bin").read_bytes() == wire
    assert (folder / "http-body.bin").read_bytes() == raw
    assert (folder / "decoded-body.bin").read_bytes() == raw
    assert (folder / "native.bin").read_bytes() == native.encode()
    proof = json.loads((folder / "receive.json").read_bytes())
    assert proof["http_body_sha256"] == sha256_hex(raw)
    assert "headers" not in proof
    assert b"fixture-only-no-network-secret" not in (folder / "receive.json").read_bytes()


def test_transport_rejects_byte_drift_before_dispatch_not_dict_equivalence():
    module = api()
    path = root()
    wire = b'{"model":"' + ARK_MODEL.encode() + b'"}'
    sent = []
    transport = recorder(module, path, wire, lambda r: sent.append(r))
    with pytest.raises(ValueError, match="wire"):
        transport.handle_request(
            httpx.Request("POST", ARK_CHAT_COMPLETIONS_URL, content=b" " + wire)
        )
    assert not sent


def test_replay_capture_directory_is_rejected():
    module = api()
    path = root()
    wire = json.dumps(dict(model=ARK_MODEL)).encode()
    recorder(module, path, wire, lambda _: None).close()
    with pytest.raises(FileExistsError):
        recorder(module, path, wire, lambda _: None)


def bridge_fixture(handler, *, change_protocol=None):
    module = api()
    from nurse_rcl.v2.live.ark_bridge import LiveArkBridge
    from tests.test_v2_live_sre_typed_report_wire import projected

    projection_module, _, _, projection = projected()
    wire = projection.wire_bytes
    binding = projection_module.bind_typed_report(wire)
    path = root()
    capture = recorder(module, path, wire, handler)
    bridge = LiveArkBridge(
        root=path / "ledger",
        lock_path=path / "in-flight.lock",
        settings=settings(),
        code_sha256="c" * 64,
        config_sha256="e" * 64,
        authorized_case_ids=(binding.opaque_case_id,),
        request_protocol=change_protocol or PROTOCOL,
        transport=capture,
        sleep=lambda _: None,
    )
    return module, path, bridge, binding, wire


@pytest.mark.parametrize(
    "text", ["not-json", '{"old_schema":true}', '{"expression_ref":"e999999"}']
)
def test_original_bridge_receives_complete_bad_native_once_without_quality_resampling(text):
    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(200, content=body(outer(text)))

    module, _, bridge, binding, wire = bridge_fixture(respond)
    with bridge:
        raw = bridge.complete(binding, wire)
        assert module.parse_typed_receipt(raw).output_json == text.encode()
        assert bridge.receipt(binding)["attempt_count"] == 1
        assert bridge.complete(binding, wire) == raw  # saved known receipt, no send
    assert seen == [wire]


@pytest.mark.parametrize(
    "kind", ["missing_usage", "inconsistent", "http_unknown", "timeout", "bad_outer"]
)
def test_original_retry_callback_persists_then_halts_before_next_attempt_or_reopen(kind):
    seen = []

    def respond(request):
        seen.append(request.content)
        if kind == "timeout":
            raise httpx.ReadTimeout("synthetic", request=request)
        if kind == "bad_outer":
            return httpx.Response(200, content=b"not outer JSON")
        usage = None if kind in ("missing_usage", "http_unknown") else dict(USAGE, total_tokens=99)
        return httpx.Response(
            500 if kind == "http_unknown" else 200, content=body(outer(usage=usage))
        )

    module, path, bridge, binding, wire = bridge_fixture(respond)
    with bridge:
        with pytest.raises(module.TypedReportAccountingError):
            bridge.complete(binding, wire)
        receipt = bridge.receipt(binding)
        assert receipt["attempt_count"] == 1
        assert receipt["ledger"]["attempts"][0]["record"] is not None
        with pytest.raises(module.TypedReportAccountingError):
            bridge.complete(binding, wire)
    assert seen == [wire]
    assert (path / "capture/attempt-1/request.bin").read_bytes() == wire


def test_original_five_known_http_failures_still_use_shared_budget():
    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(500, content=body(outer()))

    _, _, bridge, binding, wire = bridge_fixture(respond)
    from nurse_rcl.v2.live.ark_bridge import LiveTransportError

    with bridge:
        with pytest.raises(LiveTransportError, match="exhausted"):
            bridge.complete(binding, wire)
        receipt = bridge.receipt(binding)
    assert len(seen) == receipt["attempt_count"] == 5
    inputs = [a["record"]["usage"]["prompt_tokens"] for a in receipt["ledger"]["attempts"]]
    assert inputs == [11] * 5


def test_real_filesystem_ledger_write_failure_cannot_trigger_retry():
    seen = []
    bound = {}

    def respond(request):
        seen.append(request.content)
        path = bound["path"] / "ledger" / ("request-" + bound["binding"].slot + ".json")
        path.rename(path.with_suffix(".before-write-failure"))
        path.mkdir()  # real atomic replacement failure, no callback monkeypatch
        return httpx.Response(200, content=body(outer()))

    _, path, bridge, binding, wire = bridge_fixture(respond)
    bound.update(path=path, binding=binding)
    with bridge, pytest.raises(OSError):
        bridge.complete(binding, wire)
    assert seen == [wire]
    assert (path / "capture/attempt-1/http-body.bin").exists()


def test_receive_origin_recomputes_actual_body_native_and_ledger_binding():
    module, path, bridge, binding, wire = bridge_fixture(
        lambda _: httpx.Response(200, content=body(outer("native exact \n")))
    )
    with bridge:
        raw = bridge.complete(binding, wire)
        chain = module.typed_delivery_origin(
            capture_root=path / "capture",
            receipt=bridge.receipt(binding),
            raw=raw,
            binding=binding,
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
        )
        assert chain["native_sha256"] == sha256_hex(b"native exact \n")
        assert chain["request_sha256"] == sha256_hex(wire)
        assert chain["semantic_certified"] is False
        (path / "capture/attempt-1/native.bin").write_bytes(b"changed")
        with pytest.raises(ValueError):
            module.typed_delivery_origin(
                capture_root=path / "capture",
                receipt=bridge.receipt(binding),
                raw=raw,
                binding=binding,
                code_sha256="c" * 64,
                execution_sha256="e" * 64,
            )
