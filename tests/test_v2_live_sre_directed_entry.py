"""Bounded stage entry integrates real core reuse, tool receipts and synthetic native HTTP."""

# ruff: noqa: F811 -- imported pytest fixtures.

import importlib
import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_case import run_ops_case
from nurse_rcl.v2.live.ops_contracts import OpsSreDirectedProfile
from nurse_rcl.v2.live.ops_plan import load_ops_plan
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_directed_provider import stage_grant
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_execution import ops_plan, prepared_case, save  # noqa: F401
from tests.test_v2_live_ops_provider import mocked
from tests.test_v2_live_sre_action_decision import decision


def api():
    name = "nurse_rcl.v2.live.ops_sre_directed_stage"
    assert importlib.util.find_spec(name), "bounded stage entry is missing"
    return importlib.import_module(name)


def stage_plan(tmp_path, prepared, *, source_profile=None):
    path, pin, value = ops_plan(tmp_path, prepared)
    if source_profile is not None:
        value["profile"] = source_profile.model_dump(mode="json")
        pin = save(path, value)
    original = load_ops_plan(path, pin)
    core_result = run_ops_case(original, approved_plan_sha256=None)
    value.update(
        mode="with-agent", profile=OpsSreDirectedProfile().model_dump(mode="json"),
        output_root=str(tmp_path / "directed-stage"),
        source_core=dict(
            plan_path=str(path), plan_sha256=pin,
            case_result_sha256=sha256_hex((original.state_root / "case-result.json").read_bytes()),
        ),
    )
    out = tmp_path / "directed-stage"
    out.mkdir()
    stage = tmp_path / "stage-plan.json"
    admitted = load_ops_plan(stage, save(stage, value))
    return stage, admitted, core_result


def approve(admitted, ceiling):
    return dict(
        stage_decision_limit=ceiling,
        approved_plan_sha256=admitted.plan_sha256,
        approved_stage_sha256=sha256_hex(canonical_json_bytes(
            stage_grant(admitted.plan_sha256, ceiling),
        )),
    )


def native_mock(tmp_path, monkeypatch, *, fail=None):
    from nurse_rcl.v2.live import ops_provider

    calls = []

    def http(request):
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        index = body["context"]["reference_prefix"]["visible_at"]
        ctx = brief_context_from_wire(
            request.content, opaque_case_id=body["opaque_case_id"],
            case_snapshot_id=body["case_snapshot_id"], decision_index=index,
        )
        calls.append(index)
        if fail is not None:
            return httpx.Response(fail, json={"error": "synthetic native rejection"})
        value = decision(first=index == 1)
        value["investigation_note"] = None
        value["actions"] = [dict(
            skill_id="sre-symptoms/14", tool="inspect_sources",
            parameters=ctx["capabilities"]["directed_sources"]["entries"][index - 1][
                "entry"]["parameters"],
            question="Which supplied symptom distinguishes these candidates?",
        )]
        if index > 1:
            latest = ctx["investigation_progress"]["latest_actual_result"]
            value["responds_to"] = [dict(action_index=latest["action_index"], card_ids=[])]
            value["latest_result_update"].update(
                action_index=latest["action_index"], content_relation="new_content",
            )
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    return calls


@pytest.mark.parametrize("missing", ["plan", "grant", "ceiling"])
def test_entry_requires_both_approvals_before_preparation(
    prepared_case, tmp_path, monkeypatch, missing,
):
    module = api()
    _, admitted, _ = stage_plan(tmp_path, prepared_case)
    options = approve(admitted, 1)
    key = dict(plan="approved_plan_sha256", grant="approved_stage_sha256",
               ceiling="stage_decision_limit")[missing]
    options[key] = None
    monkeypatch.setattr(module, "_preparation", lambda *_: pytest.fail("unapproved preparation"))
    with pytest.raises(ValueError, match="approval|ceiling"):
        module.run_directed_stage(admitted, **options)
    assert not admitted.state_root.exists()


def test_entry_reuses_fixed_core_and_paid_prefix_without_report(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, original = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch)
    from nurse_rcl.v2.live import ops_case

    monkeypatch.setattr(ops_case, "run_ops_core", lambda **_: pytest.fail("recomputed core"))
    first = module.run_directed_stage(admitted, **approve(admitted, 1))
    assert first["termination"] == "stage_pause" and first["decision_count"] == 1
    assert first["channels"] == dict(core="completed", investigation="partial", report="disabled")
    assert first["core_result_digest"] == original["core_result_digest"]
    assert calls == [1]
    assert not (admitted.state_root / "report").exists()
    assert not (admitted.state_root / "case-result.json").exists()
    core_bytes = (admitted.state_root / "core/result.json").read_bytes()
    conversation = json.loads(
        (admitted.state_root / "investigation/conversation.json").read_bytes(),
    )
    assert conversation["result"] is None
    prefix = conversation["decisions"][0]
    with monkeypatch.context() as local:
        local.setattr(module, "_preparation", lambda *_: pytest.fail("reloaded paused stage"))
        again = module.run_directed_stage(admitted, **approve(admitted, 1))
        assert again == first and calls == [1]
    second = module.run_directed_stage(admitted, **approve(admitted, 2))
    assert second["decision_count"] == second["action_count"] == 2 and calls == [1, 2]
    restored = json.loads((admitted.state_root / "investigation/conversation.json").read_bytes())
    assert restored["decisions"][0] == prefix and restored["result"] is None
    assert (admitted.state_root / "core/result.json").read_bytes() == core_bytes
    assert module.stage_status(path, admitted.plan_sha256)["decision_count"] == 2
    with pytest.raises(ValueError, match="prefix"):
        module.run_directed_stage(admitted, **approve(admitted, 1))
    assert calls == [1, 2]


def test_stage_status_rejects_modified_members_without_sending(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch)
    module.run_directed_stage(admitted, **approve(admitted, 1))
    member = admitted.state_root / "investigation/conversation.json"
    member.write_bytes(member.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        module.stage_status(path, admitted.plan_sha256)
    with pytest.raises(ValueError, match="integrity|digest"):
        module.run_directed_stage(admitted, **approve(admitted, 2))
    assert calls == [1]


def test_stage_extension_interruption_cannot_strand_or_refund_successful_prefix(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    _, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch)
    module.run_directed_stage(admitted, **approve(admitted, 1))
    provider = module.OpsSreDirectedDecisionProvider
    original_send = provider.decision

    def interrupted(self, index, wire):
        if index == 2:
            raise KeyboardInterrupt("test interruption after decision owner reservation")
        return original_send(self, index, wire)

    with monkeypatch.context() as local:
        local.setattr(provider, "decision", interrupted)
        with pytest.raises(KeyboardInterrupt):
            module.run_directed_stage(admitted, **approve(admitted, 2))
    restored = module.run_directed_stage(admitted, **approve(admitted, 2))
    assert restored["termination"] == "unavailable" and restored["decision_count"] == 2
    assert calls == [1] and restored["action_count"] == 1


def test_supervised_entry_forwards_only_bounded_scope_and_retains_usage(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch)
    commands = []

    def worker(command, **options):
        commands.append(command)
        assert "--worker" in command
        assert command[command.index("--through-decision-index") + 1] == "1"
        module.run_directed_stage(admitted, **approve(admitted, 1))
        return 0

    monkeypatch.setattr(module, "supervise_phases", worker)
    events = []
    result = module.run_stage_round(path, admitted.plan_sha256, progress=events.append,
                                    **approve(admitted, 1))
    assert result["status"] == "stage_paused" and result["channels"]["report"] == "disabled"
    assert result["usage"]["attempts"] == 1
    assert result["usage"]["reported_input_tokens"] == 10
    assert result["usage"]["reported_output_tokens"] == 2
    assert result["usage"]["incremental_reported_input_tokens"] == 10
    assert events[-1]["event"] == "sre_stage_round_saved"
    again = module.run_stage_round(path, admitted.plan_sha256, progress=events.append,
                                   **approve(admitted, 1))
    assert again["usage"]["incremental_reported_input_tokens"] == 0
    assert again["usage"]["attempts"] == 1 and calls == [1]
    assert len(commands) == 1


@pytest.mark.parametrize("http_status,expected_attempts", [(400, 1), (503, 5)])
def test_supervised_failure_keeps_actual_worker_cause_and_all_attempts(
    prepared_case, tmp_path, monkeypatch, http_status, expected_attempts,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch, fail=http_status)

    def worker(command, **options):
        result = module.run_directed_stage(admitted, **approve(admitted, 1))
        assert result["termination"] == "unavailable"
        return 1

    monkeypatch.setattr(module, "supervise_phases", worker)
    result = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None,
                                    **approve(admitted, 1))
    assert result["status"] == "failed"
    assert result["channels"]["investigation"] == "unavailable"
    assert result["primary_failure"]["stage"] == "investigation_provider"
    assert result["primary_failure"]["failure_type"] == "LiveTransportError"
    assert result["failure_type"] == "LiveTransportError"
    assert result["worker_exit_code"] == 1
    assert result["usage"]["attempts"] == len(calls) == expected_attempts
    assert result["usage"]["unknown_input_attempts"] == expected_attempts


def test_supervised_extension_cannot_present_old_outcome_as_new_success(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch)
    first = module.run_directed_stage(admitted, **approve(admitted, 1))
    monkeypatch.setattr(module, "supervise_phases", lambda *_, **__: 0)
    result = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None,
                                    **approve(admitted, 2))
    assert result["status"] == "failed"
    assert result["failure_type"] == "ValueError"
    assert "approved_grant" not in result
    assert calls == [1] and result["usage"]["attempts"] == 1
    assert result["usage"]["incremental_reported_input_tokens"] == 0
    assert module.stage_status(path, admitted.plan_sha256) == first


def test_failed_stage_reuse_keeps_failure_and_never_sends(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    calls = native_mock(tmp_path, monkeypatch, fail=400)
    first = module.run_directed_stage(admitted, **approve(admitted, 1))
    assert first["termination"] == "unavailable"
    monkeypatch.setattr(module, "supervise_phases",
                        lambda *_, **__: pytest.fail("failed stage cannot restart"))
    result = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None,
                                    **approve(admitted, 1))
    assert result["status"] == "failed"
    assert result["primary_failure"] == first["primary_failure"]
    assert result["invocation_kind"] == "saved_stage_reuse"
    assert result["usage"]["attempts"] == len(calls) == 1
    assert result["usage"]["incremental_unknown_input_attempts"] == 0


def test_action_first_completed_core_is_usable_without_importing_old_agent_history(
    prepared_case, tmp_path, monkeypatch,
):
    from nurse_rcl.v2.live.ops_contracts import OpsSreActionProfile

    module = api()
    _, admitted, original = stage_plan(
        tmp_path, prepared_case, source_profile=OpsSreActionProfile(),
    )
    calls = native_mock(tmp_path, monkeypatch)
    result = module.run_directed_stage(admitted, **approve(admitted, 1))
    assert result["termination"] == "stage_pause"
    assert result["core_result_digest"] == original["core_result_digest"]
    assert result["decision_count"] == result["action_count"] == 1 and calls == [1]
    assert result["channels"]["report"] == "disabled"


def test_prefix_core_reader_uses_existing_core_capacity_not_ledger_capacity(
    prepared_case, tmp_path, monkeypatch,
):
    from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES

    module = api()
    _, admitted, _ = stage_plan(tmp_path, prepared_case)
    native_mock(tmp_path, monkeypatch)
    result = module.run_directed_stage(admitted, **approve(admitted, 1))
    core_member = admitted.state_root / "core/result.json"
    raw = core_member.read_bytes()
    # A synthetically padded, valid core publication crosses only the mistaken
    # 64-MiB ledger boundary, never the existing 256-MiB core reader boundary.
    assert len(raw) < 64 * 1024**2 < REPAIR_MAX_JSON_BYTES
    core_member.write_bytes(raw + b" " * (64 * 1024**2 + 1 - len(raw)))
    assert module._prefix_pins(admitted.state_root, module.case_identity(admitted)) == (
        result["prefix_pins"]
    )


def test_supervised_early_failure_is_authenticated_without_stage_outcome_or_ark(
    prepared_case, tmp_path, monkeypatch,
):
    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    actual = module.supervise_phases
    script = (
        "from nurse_rcl.v2.live import ops_sre_directed_stage as entry; "
        "from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity; "
        "entry.import_source_core=lambda *args: "
        "OpsCoreIdentity.model_validate({'private_test_payload':'MUST_NOT_SERIALIZE'}); "
        "raise SystemExit(entry.main())"
    )
    monkeypatch.setattr(module, "supervise_phases", lambda command, **kw: actual(
        [command[0], "-c", script, *command[3:]], **kw,
    ))
    result = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None,
                                    **approve(admitted, 1))
    assert result["status"] == "failed" and result["worker_exit_code"] == 1
    assert result["failure_type"] == "ValidationError"
    assert result["primary_failure"]["failure_type"] == "ValidationError"
    assert result["channels"] == dict(core="failed", investigation="unavailable", report="disabled")
    assert "MUST_NOT_SERIALIZE" not in canonical_json_bytes(result).decode()
    assert any(row["failure_type"] == "FileNotFoundError" for row in result["secondary_failures"])
    assert result["usage"]["attempts"] == 0
    assert not (admitted.state_root / "provider").exists()
    assert not (admitted.state_root / "stage-outcome.json").exists()


def test_saved_early_failure_reuse_never_launches_another_worker(
    prepared_case, tmp_path, monkeypatch,
):
    from nurse_rcl.v2.live.ops_failure import OpsFailureDetail
    from nurse_rcl.v2.live.ops_worker_outcome import write_outcome

    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    options = approve(admitted, 1)
    calls = []

    def worker(command, **kwargs):
        invocation = command[command.index("--worker-invocation-id") + 1]
        write_outcome(admitted.state_root, dict(
            module.case_identity(admitted), stage_grant_sha256=options["approved_stage_sha256"],
            worker_invocation_id=invocation,
        ), OpsFailureDetail(stage="case_worker", failure_type="ValidationError"))
        calls.append(invocation)
        return 1

    monkeypatch.setattr(module, "supervise_phases", worker)
    first = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None, **options)
    assert first["status"] == "failed" and first["failure_type"] == "ValidationError"
    again = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None, **options)
    assert len(calls) == 1, "a persisted early failure must not be restarted"
    assert again["invocation_kind"] == "saved_stage_failure_reuse"
    assert again["status"] == "failed" and again["primary_failure"] == first["primary_failure"]
    assert again["worker_receipt_sha256"] == first["worker_receipt_sha256"]
    assert again["usage"]["attempts"] == 0
    with pytest.raises(ValueError, match="grant"):
        module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None,
                               **approve(admitted, 2))
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["grant", "invocation", "digest"])
def test_supervised_early_failure_rejects_stale_or_modified_worker_receipt(
    prepared_case, tmp_path, monkeypatch, damage,
):
    from nurse_rcl.v2.live.ops_failure import OpsFailureDetail
    from nurse_rcl.v2.live.ops_worker_outcome import write_outcome

    module = api()
    path, admitted, _ = stage_plan(tmp_path, prepared_case)
    options = approve(admitted, 1)
    published = []

    def worker(command, **kwargs):
        invocation = command[command.index("--worker-invocation-id") + 1]
        binding = dict(module.case_identity(admitted),
                       stage_grant_sha256=options["approved_stage_sha256"],
                       worker_invocation_id=invocation)
        if damage == "grant":
            binding["stage_grant_sha256"] = "a" * 64
        elif damage == "invocation":
            binding["worker_invocation_id"] = "b" * 32
        write_outcome(admitted.state_root, binding,
                      OpsFailureDetail(stage="case_worker", failure_type="PrivateOldError"))
        if damage == "digest":
            receipt = admitted.state_root / "worker-outcome.json"
            value = json.loads(receipt.read_bytes())
            value["primary_failure"]["failure_type"] = "PrivateForgedError"
            receipt.write_bytes(canonical_json_bytes(value))
        published.append(invocation)
        return 1

    monkeypatch.setattr(module, "supervise_phases", worker)
    result = module.run_stage_round(path, admitted.plan_sha256, progress=lambda _: None, **options)
    assert len(published) == 1, "the test must actually publish the mismatched worker receipt"
    assert result["status"] == "failed" and result["failure_type"] == "ValueError"
    assert result.get("primary_failure") is None
    assert result["usage"]["attempts"] == 0
