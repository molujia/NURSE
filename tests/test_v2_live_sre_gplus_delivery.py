"""New G+ route uses original client/retry and one durable accounting ledger."""

from importlib import import_module, util

import pytest


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_provider"
    assert util.find_spec(name), "missing explicit G+ durable provider"
    return import_module(name)


def test_gplus_protocol_requires_explicit_registration():
    from nurse_rcl.v2.live.ops_empirical_policy import SRE_REPORT_PROTOCOLS

    assert "nurse.v2.sre-gplus-report-exchange/1" in SRE_REPORT_PROTOCOLS


def fixture(tmp_path, handler):
    module = api()
    import httpx

    from nurse_rcl.agents.ark import ArkCredential, ArkSettings
    from tests.test_v2_live_sre_gplus_wire import projected

    _, context, references, projection = projected()
    provider = module.GplusReportProvider(
        root=tmp_path / "provider",
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        approved_execution_sha256="e" * 64,
        expected_context_sha256=context.context_sha256,
        wire=projection.wire_bytes,
        references=references,
        settings=ArkSettings(ArkCredential("fixture-only-no-network-secret")),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        lock_path=tmp_path / "in-flight.lock",
    )
    return module, provider, projection.wire_bytes


def test_complete_bad_native_is_preserved_once_not_quality_resampled(tmp_path):
    api()
    import httpx

    from tests.test_v2_live_sre_typed_report_delivery import body, outer

    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(200, content=body(outer("not JSON \n")))

    module, provider, wire = fixture(tmp_path, respond)
    delivered = provider.report(wire)
    assert delivered.native == b"not JSON \n"
    assert delivered.origin["protocol"] == "nurse.v2.sre-gplus-report-exchange/1"
    assert delivered.receipt["attempt_count"] == 1
    recovered = module.read_gplus_delivery(
        root=tmp_path / "provider", wire=wire, code_sha256="c" * 64, execution_sha256="e" * 64
    )
    assert recovered == delivered
    with pytest.raises(ValueError):
        provider.report(wire)
    assert seen == [wire]


@pytest.mark.parametrize("kind", ["timeout", "missing_usage", "inconsistent", "bad_outer"])
def test_unknown_send_or_accounting_halts_after_preserving_attempt(tmp_path, kind):
    api()
    import httpx

    from nurse_rcl.v2.live.ops_sre_typed_report_provider import TypedReportAccountingError
    from tests.test_v2_live_sre_typed_report_delivery import USAGE, body, outer

    seen = []

    def respond(request):
        seen.append(request.content)
        if kind == "timeout":
            raise httpx.ReadTimeout("synthetic", request=request)
        if kind == "bad_outer":
            return httpx.Response(200, content=b"bad outer")
        return httpx.Response(
            200,
            content=body(
                outer(usage=None if kind == "missing_usage" else dict(USAGE, total_tokens=99))
            ),
        )

    _, provider, wire = fixture(tmp_path, respond)
    with pytest.raises(TypedReportAccountingError):
        provider.report(wire)
    assert seen == [wire]
    assert (tmp_path / "provider/capture/attempt-1/request.bin").read_bytes() == wire
    assert (tmp_path / "provider/receipt.json").is_file()
    with pytest.raises(ValueError):
        provider.report(wire)


def test_known_http_failures_preserve_original_five_attempt_budget(tmp_path):
    api()
    import httpx

    from nurse_rcl.v2.live.ark_bridge import LiveTransportError
    from tests.test_v2_live_sre_typed_report_delivery import body, outer

    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(500, content=body(outer()))

    _, provider, wire = fixture(tmp_path, respond)
    with pytest.raises(LiveTransportError):
        provider.report(wire)
    assert len(seen) == 5


def test_mismatched_owner_context_refused_before_http(tmp_path):
    api()
    from tests.test_v2_live_sre_gplus_wire import projected

    module = api()
    _, _, refs, projection = projected()
    with pytest.raises(ValueError):
        module.GplusReportProvider(
            root=tmp_path / "provider",
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
            approved_execution_sha256="e" * 64,
            expected_context_sha256="f" * 64,
            wire=projection.wire_bytes,
            references=refs,
        )
    assert not (tmp_path / "provider").exists()
