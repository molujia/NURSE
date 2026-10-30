"""Empirical protocol through the real persistent bridge, never a real endpoint."""

# ruff: noqa: F811 -- imported pytest fixtures

import fcntl
import gzip
import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_context_view import investigation_input
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolbox
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_provider import mocked
from tests.test_v2_live_ops_report import (
    complete,  # noqa: F401
    prepared_cases,  # noqa: F401
    proposal,
    report_inputs,  # noqa: F401
)


@pytest.fixture(scope="module")
def inputs(report_inputs):
    from nurse_rcl.v2.live.ops_report_context import build_report_context

    data, core, finding = report_inputs
    ctx = build_report_context(core=core, sources=data.sources, investigation=(finding,),
                               soft_targets=True)
    tools = OpsIndexedToolbox(data=data, block_root=None, soft_targets=True)
    projected = investigation_input(capabilities=tools.capabilities(), records=(),
        decision_index=1, make_wire=lambda view: {"context": view}, soft_targets=True)
    return ctx, json.loads(projected.view_json)


def owner(module, root, ctx):
    return module.OpsLiveProvider(root=root, code_sha256="a" * 64,
        execution_sha256="b" * 64, approved_execution_sha256="b" * 64,
        opaque_case_id=ctx.opaque_case_id, case_snapshot_id=ctx.case_snapshot_id,
        empirical=True)


def stopped(**changes):
    return canonical_json_bytes(dict(actions=[], stop_reason="done",
        public_summary="unknown", unresolved=[], **changes)).decode()


def report_text(ctx, count=1, *, long=False):
    old = proposal(ctx).model_dump(mode="json")
    claims = []
    for index in range(count):
        item = json.loads(json.dumps(old["claims"][0]))
        item["claim"]["claim_id"] = f"claim{index}"
        if long:
            item["claim"]["surface_text"] += " Evidence remains uncertain." * 500
        claims.append(item)
    return canonical_json_bytes(dict(schema_version="nurse.v2.ops-layered-report-response/2",
        core_result_digest=ctx.core_result_digest, claims=claims)).decode()


def test_new_wire_sets_capacity_and_preserves_legacy_rejection(inputs):
    from nurse_rcl.v2.hermes_exchange import _validate_payload
    from nurse_rcl.v2.live.ops_exchange import prepare_ops_decision, validate_ops_decision_payload
    from nurse_rcl.v2.live.ops_report_exchange import prepare_ops_report

    ctx, view = inputs
    raw, binding = prepare_ops_decision(opaque_case_id=ctx.opaque_case_id,
        case_snapshot_id=ctx.case_snapshot_id, decision_index=1, context=view, empirical=True)
    request = json.loads(raw)
    assert request["max_tokens"] == 65536 and "max_completion_tokens" not in request
    assert binding.variant == "v2-ops-isolated/2"
    assert validate_ops_decision_payload(request, empirical=True)["context"] == view
    assert _validate_payload({k: v for k, v in request.items() if k != "thinking"}) is None
    with pytest.raises(ExchangeBoundaryError):
        validate_ops_decision_payload(request)
    report, report_binding = prepare_ops_report(ctx, empirical=True)
    request = json.loads(report)
    assert request["max_tokens"] == 65536 and report_binding.variant == "v2-ops-isolated/2"
    body = json.loads(request["messages"][1]["content"])
    assert "fact_tables" in body["context"] and "facts" not in body["context"]


def test_input_above_target_and_missing_output_do_not_block_next_request(
    tmp_path, inputs, monkeypatch,
):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    sent = []

    def http(req):
        sent.append(req)
        value = response(stopped())
        value["usage"] = {"prompt_tokens": 120001, "completion_tokens": "bad"}
        return httpx.Response(200, json=value)

    mocked(module, tmp_path, monkeypatch, http)
    instance = owner(module, tmp_path, ctx)
    instance.decision(1, view)
    second = dict(view, decision_index=2, remaining_decisions=7)
    instance.decision(2, second)
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert len(sent) == usage["attempts"] == 2
    assert usage["reported_input_tokens"] == 240002
    assert usage["input_usage_complete"] is True and usage["usage_complete"] is False
    assert usage["output_tokens"] is None and usage["unknown_input_attempts"] == 0


def test_long_output_survives_bridge_storage_and_keyless_restore(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, _ = inputs
    text = report_text(ctx, 30, long=True)
    assert len(text.encode()) > 262144
    calls = []

    def http(req):
        calls.append(req)
        value = response(text)
        value["usage"] = {"input_tokens": 1000, "output_tokens": 120000, "total_tokens": 121000}
        return httpx.Response(200, json=value)

    mocked(module, tmp_path, monkeypatch, http)
    instance = owner(module, tmp_path, ctx)
    first = instance.report(ctx)
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("paid response must not read KEY2"))
    assert instance.report(ctx) == first
    assert len(json.loads(first)["claims"]) == 30 and len(calls) == 1
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["input_tokens"] == 1000 and usage["output_tokens"] == 120000
    assert usage["attempts"] == 1


def test_long_decision_keeps_structure_without_text_quota():
    from nurse_rcl.v2.live.ops_agent_decision import parse_decision

    value = dict(actions=[], stop_reason="s" * 3000, public_summary="x" * 80000,
                 unresolved=["q" * 2000] * 20)
    raw = canonical_json_bytes(value)
    assert parse_decision(raw, empirical=True) == value
    with pytest.raises(ExchangeBoundaryError):
        parse_decision(raw)
    with pytest.raises(ExchangeBoundaryError):
        parse_decision(canonical_json_bytes(dict(value, extra="forbidden")), empirical=True)


def test_nested_report_fields_have_no_legacy_length_quotas(inputs):
    from nurse_rcl.v2.live.ops_report_schema import parse_ops_report

    raw = json.loads(report_text(inputs[0], 7))
    for item in raw["claims"]:
        claim = item["claim"]
        claim.update(fact_refs=[f"f{i}" for i in range(1, 12)],
            target_service_keys=["service" + str(i) + "x" * 300 for i in range(4)],
            assertions=[dict(fact_ref=f"f{i}", field="value", value="x" * 300)
                        for i in range(1, 12)],
            recommendation=dict(action_class="read_only_diagnostic_check",
                precondition_refs=[f"f{i}" for i in range(1, 12)], status="proposed",
                possible_outcomes=[dict(outcome=v, interpretation="x" * 300)
                                   for v in ("true", "false", "unknown")],
                safety_limitations=["x" * 300] * 8))
    parsed = parse_ops_report(canonical_json_bytes(raw), empirical=True)
    assert len(parsed.claims) == 7 and len(parsed.claims[0].claim.assertions) == 11
    with pytest.raises(ExchangeBoundaryError):
        parse_ops_report(canonical_json_bytes(raw))


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_native_partial_is_saved_and_never_retried(tmp_path, inputs, monkeypatch, finish):
    from nurse_rcl.v2.live import ops_provider as module
    from nurse_rcl.v2.live.ark_bridge import _read

    ctx, view = inputs
    sent = []

    def http(req):
        sent.append(req)
        value = response('{"actions":[')
        value["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=value)

    mocked(module, tmp_path, monkeypatch, http)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="native_partial"):
            owner(module, tmp_path, ctx).decision(1, view)
    assert len(sent) == 1
    path = next((tmp_path / "ark-investigation").glob("request-*.json"))
    stored = _read(path, "live_ark_request", empirical=True)
    raw = json.loads(stored["attempts"][0]["response_json"])
    assert raw["choices"][0]["finish_reason"] == finish
    assert raw["choices"][0]["message"]["content"] == '{"actions":['
    assert module.usage_summary(tmp_path, execution_sha256="b" * 64,
                                empirical=True)["input_tokens"] == 10


def test_structural_failures_have_five_lifetime_attempts_and_all_input_cost(
    tmp_path, inputs, monkeypatch,
):
    from nurse_rcl.v2.live import ops_provider as module

    calls = []

    def http(req):
        calls.append(req)
        return httpx.Response(200, json=response('{"not":"decision"}'))

    mocked(module, tmp_path, monkeypatch, http)
    ctx, view = inputs
    for _ in range(2):
        with pytest.raises(RuntimeError, match="exhausted"):
            owner(module, tmp_path, ctx).decision(1, view)
    assert len(calls) == 5
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["attempts"] == 5 and usage["input_tokens"] == 50


def test_response_record_gap_is_repaired_once_without_key(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ark_bridge
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    mocked(module, tmp_path, monkeypatch,
           lambda _: httpx.Response(200, json=response(stopped())))
    original = ark_bridge._write

    def crash(path, kind, value):
        original(path, kind, value)
        attempts = value.get("attempts", [])
        if attempts and attempts[-1].get("response_json") and attempts[-1].get("record") is None:
            raise KeyboardInterrupt("response before accounting")

    monkeypatch.setattr(ark_bridge, "_write", crash)
    with pytest.raises(KeyboardInterrupt):
        owner(module, tmp_path, ctx).decision(1, view)
    monkeypatch.setattr(ark_bridge, "_write", original)
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("recovery read KEY2"))
    owner(module, tmp_path, ctx).decision(1, view)
    owner(module, tmp_path, ctx).decision(1, view)
    path = next((tmp_path / "ark-investigation").glob("request-*.json"))
    stored = ark_bridge._read(path, "live_ark_request", empirical=True)
    assert len(stored["attempts"]) == 1 and stored["attempts"][0]["record"] is not None
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["input_tokens"] == 10 and usage["unknown_latency_attempts"] == 1


def test_unknown_send_keeps_slots_and_unknown_cost(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    calls = []

    def http(req):
        calls.append(req)
        raise KeyboardInterrupt("unknown network outcome")

    mocked(module, tmp_path, monkeypatch, http)
    ctx, view = inputs
    for _ in range(5):
        with pytest.raises(KeyboardInterrupt):
            owner(module, tmp_path, ctx).decision(1, view)
    with pytest.raises(RuntimeError, match="exhausted"):
        owner(module, tmp_path, ctx).decision(1, view)
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert len(calls) == usage["unknown_input_attempts"] == 5
    assert usage["input_tokens"] is None


def test_saved_recovery_and_usage_share_send_lock(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    mocked(module, tmp_path, monkeypatch,
           lambda _: httpx.Response(200, json=response(stopped())))
    instance = owner(module, tmp_path, ctx)
    instance.decision(1, view)
    with (tmp_path / "shared.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="llm_in_flight"):
            instance.decision(1, view)
        with pytest.raises(RuntimeError, match="llm_in_flight"):
            module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)


@pytest.mark.parametrize("field,value", [("case_snapshot_id", "f" * 64),
    ("private", {"ground_truth": "answer"}), ("private", "ts0-mysql-corrupt-jkgn5j")])
def test_empirical_outbound_still_blocks_contamination(tmp_path, inputs, monkeypatch, field, value):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("private outbound read credentials"))
    with pytest.raises(ValueError):
        owner(module, tmp_path, ctx).decision(1, dict(view, **{field: value}))


def test_http_failure_usage_is_not_lost_before_success(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    calls = []

    def http(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "busy", "usage": {"prompt_tokens": 45}})
        return httpx.Response(200, json=response(stopped()))

    mocked(module, tmp_path, monkeypatch, http)
    owner(module, tmp_path, ctx).decision(1, view)
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["attempts"] == 2 and usage["input_tokens"] == 55
    assert usage["unknown_input_attempts"] == 0 and usage["unknown_output_attempts"] == 1


def test_empirical_transport_bounds_stream_before_full_allocation(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    yielded = []

    class LargeBody(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(128):
                yielded.append(i)
                yield b" " * (1024 * 1024)

    mocked(module, tmp_path, monkeypatch, lambda _: httpx.Response(200, stream=LargeBody()))
    ctx, view = inputs
    with pytest.raises(RuntimeError, match="response_resource_limit"):
        owner(module, tmp_path, ctx).decision(1, view)
    assert len(yielded) <= 5
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("saved resource failure read KEY2"))
    with pytest.raises(RuntimeError, match="response_resource_limit"):
        owner(module, tmp_path, ctx).decision(1, view)
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["attempts"] == usage["unknown_input_attempts"] == 1


@pytest.mark.parametrize("damage", ["column", "common", "duplicate", "short_row",
    "reserved_fact", "reserved_values", "reserved_column_fact", "reserved_column_values",
    "empty_private_table", "empty_table"])
def test_report_table_codec_cannot_hide_fields_or_alias_drift(inputs, damage):
    from nurse_rcl.v2.live.ops_context_view import expand_report_facts
    from nurse_rcl.v2.live.ops_report_exchange import prepare_ops_report

    raw, _ = prepare_ops_report(inputs[0], empirical=True)
    view = json.loads(json.loads(raw)["messages"][1]["content"])["context"]
    table = view["fact_tables"][0]
    if damage in {"empty_private_table", "empty_table"}:
        common = {"fact": {"ground_truth": "answer"}} if damage == "empty_private_table" else {}
        view["fact_tables"].append(dict(common=common, columns=[], rows=[]))
    elif damage == "column":
        table["columns"].append("ground_truth")
        for row in table["rows"]:
            row.append("answer")
    elif damage == "common":
        table["common"]["fact.root_cause"] = "answer"
    elif damage == "duplicate":
        table["rows"].append(table["rows"][0])
    elif damage.startswith("reserved_column"):
        table["columns"].append("fact" if damage.endswith("fact") else "fact.values")
        for row in table["rows"]:
            row.append({"ground_truth": "answer"})
    elif damage.startswith("reserved"):
        table["common"]["fact" if damage == "reserved_fact" else "fact.values"] = {
            "ground_truth": "answer"}
    else:
        table["rows"][0].pop()
    with pytest.raises(ValueError):
        expand_report_facts(view)


def test_saved_empirical_payload_drift_is_rejected_without_key(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    ctx, view = inputs
    mocked(module, tmp_path, monkeypatch,
           lambda _: httpx.Response(200, json=response(stopped())))
    instance = owner(module, tmp_path, ctx)
    instance.decision(1, view)
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("drift accessed KEY2"))
    with pytest.raises(ValueError, match="binding"):
        instance.decision(1, dict(view, omitted_detail=["changed payload"]))
    instance.code = "c" * 64
    with pytest.raises(ValueError, match="identity"):
        instance.decision(1, view)


def test_compressed_native_http_response_is_decoded_once(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ops_provider as module

    calls = []

    def http(req):
        calls.append(req)
        return httpx.Response(200, headers={"content-encoding": "gzip"},
            stream=httpx.ByteStream(gzip.compress(canonical_json_bytes(response(stopped())))))

    mocked(module, tmp_path, monkeypatch, http)
    ctx, view = inputs
    assert json.loads(owner(module, tmp_path, ctx).decision(1, view))["stop_reason"] == "done"
    assert len(calls) == 1


def test_fatal_http_response_record_gap_cannot_become_retry(tmp_path, inputs, monkeypatch):
    from nurse_rcl.v2.live import ark_bridge
    from nurse_rcl.v2.live import ops_provider as module

    calls = []

    def http(req):
        calls.append(req)
        return httpx.Response(400, json={"error": "configuration", "usage": {"prompt_tokens": 99}})

    mocked(module, tmp_path, monkeypatch, http)
    original = ark_bridge._write

    def crash(path, kind, value):
        original(path, kind, value)
        attempts = value.get("attempts", [])
        if attempts and attempts[-1].get("response_json") and attempts[-1].get("record") is None:
            raise KeyboardInterrupt("http receipt before record")

    monkeypatch.setattr(ark_bridge, "_write", crash)
    ctx, view = inputs
    with pytest.raises(KeyboardInterrupt):
        owner(module, tmp_path, ctx).decision(1, view)
    monkeypatch.setattr(ark_bridge, "_write", original)
    monkeypatch.setattr(module.ark, "load_ark_settings_from_key_file",
                        lambda _: pytest.fail("saved fatal failure must not send again"))
    with pytest.raises(RuntimeError, match="fatal"):
        owner(module, tmp_path, ctx).decision(1, view)
    assert len(calls) == 1
    usage = module.usage_summary(tmp_path, execution_sha256="b" * 64, empirical=True)
    assert usage["input_tokens"] == 99 and usage["attempts"] == 1
