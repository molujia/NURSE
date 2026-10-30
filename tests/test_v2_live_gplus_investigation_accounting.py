"""The adopted investigation route must halt, not re-send, after unknown spend."""

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes


def bridge(tmp_path, handler):
    from nurse_rcl.agents.ark import ArkCredential, ArkSettings
    from nurse_rcl.v2.live.ark_bridge import LiveArkBridge
    from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import (
        bind_entry_decision,
        prepare_entry_decision,
    )
    from tests.sre_m3_entry_fixtures import normal_context

    context, refs, options = normal_context()
    wire = prepare_entry_decision(context, references=refs, **options)
    binding = bind_entry_decision(wire, **options)
    instance = LiveArkBridge(
        root=tmp_path / "ledger",
        lock_path=tmp_path / "in-flight.lock",
        settings=ArkSettings(ArkCredential("fixture-only-no-network-secret")),
        code_sha256="c" * 64,
        config_sha256="e" * 64,
        authorized_case_ids=(binding.opaque_case_id,),
        request_protocol="nurse.v2.sre-decision-exchange/25",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    return instance, binding, wire


def outer(usage):
    value = dict(
        actions=[],
        stop_reason="No supplied useful check",
        stop_basis="no_useful_check",
        public_summary="The synthetic fixture exposes no diagnostic action.",
        unresolved=["No observations supplied."],
        responds_to=[],
        latest_result_update=None,
        investigation_note=None,
    )
    return canonical_json_bytes(
        dict(
            choices=[
                dict(
                    index=0,
                    finish_reason="stop",
                    message=dict(role="assistant", content=canonical_json_bytes(value).decode()),
                )
            ],
            usage=usage,
        )
    )


@pytest.mark.parametrize("kind", ["timeout", "missing_usage", "inconsistent"])
def test_current_investigation_persists_and_halts_unknown_before_next_attempt_or_reopen(
    tmp_path, kind
):
    from nurse_rcl.v2.live.ops_sre_typed_report_provider import TypedReportAccountingError
    from tests.test_v2_live_sre_typed_report_delivery import USAGE

    seen = []

    def respond(request):
        seen.append(request.content)
        if kind == "timeout":
            raise httpx.ReadTimeout("synthetic", request=request)
        usage = None if kind == "missing_usage" else dict(USAGE, total_tokens=99)
        return httpx.Response(200, content=outer(usage))

    instance, binding, wire = bridge(tmp_path, respond)
    with instance:
        with pytest.raises(TypedReportAccountingError):
            instance.complete(binding, wire)
        receipt = instance.receipt(binding)
        assert receipt["attempt_count"] == 1
        assert receipt["ledger"]["attempts"][0]["record"] is not None
        with pytest.raises(TypedReportAccountingError):
            instance.complete(binding, wire)
    assert seen == [wire]


def test_known_investigation_http_failures_keep_original_five_attempts(tmp_path):
    from nurse_rcl.v2.live.ark_bridge import LiveTransportError
    from tests.test_v2_live_sre_typed_report_delivery import USAGE

    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(500, content=outer(USAGE))

    instance, binding, wire = bridge(tmp_path, respond)
    with instance:
        with pytest.raises(LiveTransportError):
            instance.complete(binding, wire)
        assert instance.receipt(binding)["attempt_count"] == 5
    assert len(seen) == 5
