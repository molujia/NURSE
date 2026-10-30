"""Server entry admission: one complete case, frozen code and no raw rescan."""

# ruff: noqa: F811 -- shared pytest fixture names.

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.test_v2_live_normal_execution import (
    freeze_plan,
    prepared_case,  # noqa: F401
    save,
)


def api(name="ops_plan"):
    qualified = "nurse_rcl.v2.live." + name
    assert importlib.util.find_spec(qualified), "missing isolated server " + name
    return importlib.import_module(qualified)


def ops_plan(tmp_path, prepared, *, mode="core-only"):
    module = api()
    _, _, previous = freeze_plan(tmp_path, prepared)
    code = Path(previous["code_root"])
    value = dict(
        schema_version="nurse.v2.ops-single-plan/1",
        code_root=str(code),
        code_files=module.code_inventory(code),
        environment=module.runtime_environment(),
        profile=module.OpsProfile().model_dump(mode="json"),
        mode=mode,
        case=previous["case"],
        preparation=previous["preparation"],
        identity_manifest_path=previous["identity_manifest_path"],
        identity_artifact_sha256=previous["identity_artifact_sha256"],
        parent_plan_path=previous["parent_plan_path"],
        parent_plan_sha256=previous["parent_plan_sha256"],
        output_root=previous["output_root"],
    )
    path = tmp_path / "ops-plan.json"
    return path, save(path, value), value


def test_new_plan_is_single_case_metadata_only(prepared_case, tmp_path, monkeypatch):
    module = api()
    path, pin, value = ops_plan(tmp_path, prepared_case)
    from nurse_rcl.v2.live import normal_preparation, repair_stage

    monkeypatch.setattr(repair_stage, "validate_input", lambda **_: pytest.fail("raw scan"))
    monkeypatch.setattr(
        normal_preparation,
        "load_normal_preparation",
        lambda **_: pytest.fail("metadata admission loaded telemetry"),
    )
    admitted = module.load_ops_plan(path, pin)
    assert admitted.plan.case.opaque_case_id == admitted.plan.preparation.key.opaque_case_id
    assert admitted.raw_arguments["dataset"] == value["case"]["dataset"]
    assert admitted.plan.profile.profile_id == "v2-ops-isolated/1"
    assert admitted.plan.code_sha256 == sha256_hex(canonical_json_bytes(value["code_files"]))
    assert any(n.endswith("ops_skills/timing.md") for n in value["code_files"])


def test_code_inventory_freezes_public_wrapper():
    root = Path(__file__).resolve().parents[1]
    assert "scripts/dev/run_ops_isolated.sh" in api().code_inventory(root)


@pytest.mark.parametrize("damage", ["extra_case", "scope", "environment", "code", "skill"])
def test_plan_rejects_scope_or_execution_drift(prepared_case, tmp_path, damage):
    module = api()
    path, _, value = ops_plan(tmp_path, prepared_case)
    if damage == "extra_case":
        value["cases"] = [value["case"], value["case"]]
    elif damage == "scope":
        value["case"]["opaque_case_id"] = "opq_" + "f" * 32
    elif damage == "environment":
        value["environment"]["python_version"] = "0.0"
    else:
        name = next(
            n
            for n in value["code_files"]
            if n.endswith("ops_skills/timing.md" if damage == "skill" else "ops_core.py")
        )
        value["code_files"][name] = "0" * 64
    with pytest.raises(ValueError):
        module.load_ops_plan(path, save(path, value))


def test_new_output_cannot_alias_old_output_or_preparation(prepared_case, tmp_path):
    module = api()
    path, _, value = ops_plan(tmp_path, prepared_case)
    value["output_root"] = value["preparation"]["archive_root"]
    with pytest.raises(ValueError, match="output"):
        module.load_ops_plan(path, save(path, value))


def test_same_core_identity_configuration_ignores_investigation_mode(prepared_case, tmp_path):
    module = api()
    path, pin, value = ops_plan(tmp_path, prepared_case)
    first = module.load_ops_plan(path, pin)
    value["mode"] = "with-agent"
    second = module.load_ops_plan(path, save(path, value))
    assert first.plan.core_configuration_sha256 == second.plan.core_configuration_sha256
    assert first.plan.environment_sha256 == second.plan.environment_sha256


@pytest.mark.parametrize("mode", ["core-only", "with-agent"])
def test_case_keeps_core_without_key_and_recovers_without_preparation(
    prepared_case,
    tmp_path,
    monkeypatch,
    mode,
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case, mode=mode)
    admitted = api().load_ops_plan(path, pin)
    from nurse_rcl.agents import ark

    def no_key(_):
        if mode == "core-only":
            pytest.fail("core-only read credentials")
        raise ValueError("missing KEY2")

    monkeypatch.setattr(ark, "load_ark_settings_from_key_file", no_key)
    result = module.run_ops_case(admitted, approved_plan_sha256=pin)
    assert result["channels"]["core"] == "completed"
    assert result["channels"]["investigation"] == (
        "disabled" if mode == "core-only" else "unavailable"
    )
    assert result["channels"]["report"] == ("disabled" if mode == "core-only" else "unavailable")
    core = admitted.state_root / "core/result.json"
    core_bytes = core.read_bytes()
    assert result["core_result_digest"] == json.loads(core_bytes)["core_result_digest"]
    assert result["candidate_count"] > 0
    monkeypatch.setattr(
        module, "load_normal_preparation", lambda **_: pytest.fail("reloaded completed preparation")
    )
    monkeypatch.setattr(module, "run_ops_core", lambda **_: pytest.fail("reran core"))
    recovered = module.run_ops_case(admitted, approved_plan_sha256=None)
    assert recovered["core_result_digest"] == result["core_result_digest"]
    assert core.read_bytes() == core_bytes
    core.write_bytes(core_bytes + b" ")
    with pytest.raises(ValueError, match="integrity"):
        module.inspect_ops_case(admitted)


def test_fresh_with_agent_requires_approval_before_preparation(
    prepared_case, tmp_path, monkeypatch
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case, mode="with-agent")
    admitted = api().load_ops_plan(path, pin)
    monkeypatch.setattr(
        module, "load_normal_preparation", lambda **_: pytest.fail("unapproved preparation")
    )
    with pytest.raises(ValueError, match="approval"):
        module.run_ops_case(admitted, approved_plan_sha256=None)


def mock_ops_http(tmp_path, monkeypatch, *, fail_report=False):
    import httpx

    from tests.test_v2_live_ark import response
    from tests.test_v2_live_ops_provider import mocked

    wire = tmp_path / "wire"
    wire.mkdir(exist_ok=True)

    def http(request):
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        ctx = body["context"]
        is_report = "report" in body["protocol"]
        number = "report" if is_report else str(ctx["decision_index"])
        (wire / (number + ".json")).write_bytes(request.content)
        if is_report:
            if fail_report:
                return httpx.Response(400, json={"error": "synthetic"})
            fact = ctx["facts"][0]["fact"]
            field = fact["values"][0]
            output = dict(
                schema_version="nurse.v2.ops-layered-report-response/1",
                core_result_digest=ctx["core_result_digest"],
                claims=[
                    dict(
                        section="limitations",
                        claim=dict(
                            claim_id="c1",
                            statement_type="inference",
                            operation="explain_uncertainty",
                            fact_refs=[fact["fact_ref"]],
                            target_service_keys=fact["target_service_keys"],
                            interval=None,
                            requested_claim_strength="ranking_inference",
                            recommendation=None,
                            assertions=[
                                dict(
                                    fact_ref=fact["fact_ref"],
                                    field=field["name"],
                                    value=field["value"],
                                )
                            ],
                            surface_text="unknown",
                        ),
                    )
                ],
            )
        else:
            actions = (
                []
                if ctx["decision_index"] == 2
                else [
                    dict(
                        skill_id="ops-symptoms/1",
                        tool="measure_signal",
                        question="check observed symptom",
                        parameters={
                            "request_key": ctx["capabilities"]["requests"][0]["request_key"]
                        },
                    )
                ]
            )
            output = dict(
                actions=actions,
                stop_reason=None if actions else "done",
                public_summary="unknown",
                unresolved=[],
            )
        return httpx.Response(200, json=response(canonical_json_bytes(output).decode()))

    mocked(api("ops_provider"), tmp_path, monkeypatch, http)
    return wire


@pytest.mark.parametrize("report_failed_first", [False, True])
def test_saved_report_recovers_without_preparation_core_or_http_after_publication_crash(
    prepared_case,
    tmp_path,
    monkeypatch,
    report_failed_first,
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case, mode="with-agent")
    admitted = api().load_ops_plan(path, pin)
    wire = mock_ops_http(tmp_path, monkeypatch, fail_report=report_failed_first)
    if report_failed_first:
        before = module.run_ops_case(admitted, approved_plan_sha256=pin)
        assert before["channels"]["report"] == "unavailable"
        # A fatal request remains fatal in its lifetime slot. Simulate the
        # save-after-response interruption on a fresh report owner below instead.
        assert len(list(wire.glob("*.json"))) == 3
        core_bytes = (admitted.state_root / "core/result.json").read_bytes()
        monkeypatch.setattr(module, "load_normal_preparation", lambda **_: pytest.fail("reload"))
        after = module.run_ops_case(admitted, approved_plan_sha256=pin)
        assert after["channels"]["report"] == "unavailable"
        assert (admitted.state_root / "core/result.json").read_bytes() == core_bytes
        return
    original = module._finish
    monkeypatch.setattr(
        module,
        "_finish",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt("publication interrupted")),
    )
    with pytest.raises(KeyboardInterrupt):
        module.run_ops_case(admitted, approved_plan_sha256=pin)
    monkeypatch.setattr(module, "_finish", original)
    monkeypatch.setattr(module, "load_normal_preparation", lambda **_: pytest.fail("reload"))
    monkeypatch.setattr(module, "run_ops_core", lambda **_: pytest.fail("rerun core"))
    from nurse_rcl.agents import ark

    monkeypatch.setattr(ark, "load_ark_settings_from_key_file", lambda _: pytest.fail("reread key"))
    result = module.run_ops_case(admitted, approved_plan_sha256=None)
    assert result["channels"]["investigation"] == "completed"
    assert result["channels"]["report"] in {"completed", "partial"}
    assert len(list(wire.glob("*.json"))) == 3


@pytest.mark.parametrize("mode", ["core-only", "with-agent"])
def test_real_public_worker_phase_clock_progress_and_completed_resume(
    prepared_case,
    tmp_path,
    monkeypatch,
    mode,
):
    module = api("ops_execution")
    path, pin, plan = ops_plan(tmp_path, prepared_case, mode=mode)
    if mode == "with-agent":
        actual = module.supervise_phases
        script = (
            "import pytest; from pathlib import Path; "
            "from tests.test_v2_live_ops_execution import mock_ops_http; "
            f"mock_ops_http(Path({str(tmp_path)!r}),pytest.MonkeyPatch()); "
            "from nurse_rcl.v2.live.ops_execution import main; raise SystemExit(main())"
        )

        def supervised(command, **kwargs):
            return actual([command[0], "-c", script, *command[3:]], **kwargs)

        monkeypatch.setattr(module, "supervise_phases", supervised)
    events = []
    result = module.run_round(path, pin, approved_plan_sha256=pin, progress=events.append)
    assert result["channels"]["core"] == "completed", result
    assert result["channels"]["report"] in {"completed", "partial", "disabled"}, result
    assert result["timing"]["preparation_seconds"] > 0
    assert result["timing"]["method_seconds"] > 0
    assert result["timing"]["method_remaining"] is None
    assert result["timing"]["active"] is False
    root = Path(plan["output_root"]) / plan["case"]["opaque_case_id"]
    table = json.loads((root / "module-timing.json").read_bytes())
    assert table["aggregation"] == "per_process_not_wall_sum"
    assert table["complete"]
    assert len(table["processes"]) == (2 if mode == "core-only" else 4)
    assert any(e["event"] == "stage_process_started" for e in events)
    monkeypatch.setattr(module, "supervise_phases", lambda *a, **kw: pytest.fail("reran complete"))
    assert module.run_round(path, pin, progress=events.append) == result
    assert module.status(path, pin) == result


def test_report_recovery_interruption_does_not_make_old_partial_receipt_a_blocker(
    prepared_case,
    tmp_path,
    monkeypatch,
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case, mode="with-agent")
    admitted = api().load_ops_plan(path, pin)
    from nurse_rcl.agents import ark

    monkeypatch.setattr(
        ark,
        "load_ark_settings_from_key_file",
        lambda _: (_ for _ in ()).throw(ValueError("missing KEY2")),
    )
    first = module.run_ops_case(admitted, approved_plan_sha256=pin)
    assert first["channels"]["report"] == "unavailable"
    mock_ops_http(tmp_path, monkeypatch)
    original = module._finish
    monkeypatch.setattr(
        module,
        "_finish",
        lambda *args: (_ for _ in ()).throw(
            KeyboardInterrupt("report saved, case summary not saved")
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        module.run_ops_case(admitted, approved_plan_sha256=pin)
    monkeypatch.setattr(module, "_finish", original)
    monkeypatch.setattr(module, "load_normal_preparation", lambda **_: pytest.fail("reload"))
    monkeypatch.setattr(
        ark, "load_ark_settings_from_key_file", lambda _: pytest.fail("new request")
    )
    final = module.run_ops_case(admitted, approved_plan_sha256=None)
    assert final["core_result_digest"] == first["core_result_digest"]
    assert final["channels"]["report"] in {"completed", "partial"}


def test_core_only_publication_resume_does_not_reload_preparation(
    prepared_case,
    tmp_path,
    monkeypatch,
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case)
    admitted = api().load_ops_plan(path, pin)
    original = module._finish
    monkeypatch.setattr(module, "_finish", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        module.run_ops_case(admitted, approved_plan_sha256=None)
    monkeypatch.setattr(module, "_finish", original)
    monkeypatch.setattr(module, "load_normal_preparation", lambda **_: pytest.fail("reload"))
    assert (
        module.run_ops_case(admitted, approved_plan_sha256=None)["channels"]["core"] == "completed"
    )


def test_public_failure_after_core_can_be_inspected_and_resumed(
    prepared_case, tmp_path, monkeypatch
):
    module = api("ops_execution")
    path, pin, _ = ops_plan(tmp_path, prepared_case)
    actual = module.supervise_phases
    script = (
        "from nurse_rcl.v2.live import ops_case; "
        "ops_case._finish=lambda *a:(_ for _ in ()).throw("
        "RuntimeError('synthetic publish failure')); "
        "from nurse_rcl.v2.live.ops_execution import main; raise SystemExit(main())"
    )
    monkeypatch.setattr(
        module,
        "supervise_phases",
        lambda command, **kw: actual([command[0], "-c", script, *command[3:]], **kw),
    )
    failed = module.run_round(path, pin, progress=lambda _: None)
    assert failed["status"] == "failed" and failed["core_result_saved"]
    assert module.status(path, pin) == failed
    monkeypatch.setattr(module, "supervise_phases", actual)
    recovered = module.run_round(path, pin, progress=lambda _: None)
    assert recovered["status"] == "completed"
    assert recovered["timing"]["method_seconds"] >= failed["timing"]["method_seconds"]


def test_tmux_wrapper_keeps_log_and_exit_receipt(prepared_case, tmp_path):
    import os
    import subprocess
    import sys

    wrapper = Path(__file__).resolve().parents[1] / "scripts/dev/run_ops_isolated.sh"
    assert wrapper.is_file(), "missing tmux-friendly isolated wrapper"
    path, pin, _ = ops_plan(tmp_path, prepared_case)
    logs = tmp_path / "logs"
    logs.mkdir()
    result = subprocess.run(
        ["bash", str(wrapper), "single", str(path), pin, str(logs)],
        env={**os.environ, "NURSE_OPS_PYTHON": sys.executable},
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    attempts = tuple(logs.glob("attempt.*"))
    assert len(attempts) == 1
    assert (attempts[0] / "runner.exit").read_text().strip() == "EXIT_CODE=0"
    assert '"event": "ops_terminal"' in (attempts[0] / "run.log").read_text()


def test_preparation_timeout_still_publishes_terminal(prepared_case, tmp_path, monkeypatch):
    from time import monotonic

    from nurse_rcl.v2.live.repair_timing import CaseBudgetExpired

    module = api("ops_execution")
    path, pin, _ = ops_plan(tmp_path, prepared_case)
    now = monotonic()
    monkeypatch.setattr(module, "monotonic", lambda: now)

    def expired(command, *, timing, **kwargs):
        nonlocal now
        timing.switch("preparation", reason="raw_input_prepare", now=now)
        now += 3601.0
        with pytest.raises(CaseBudgetExpired):
            timing.checkpoint(now=now)
        return 124

    monkeypatch.setattr(module, "supervise_phases", expired)
    result = module.run_round(path, pin, progress=lambda _: None)
    assert result["status"] == "failed"
    assert result["failure_type"] == "preparation_timeout"
    assert result["timing"]["preparation_remaining"] == 0
    assert module.status(path, pin) == result
    monkeypatch.setattr(module, "supervise_phases", lambda *a, **kw: pytest.fail("expired restart"))
    assert module.run_round(path, pin, progress=lambda _: None) == result


def test_unclean_preparation_gap_cannot_reset_budget_or_lose_terminal(
    prepared_case,
    tmp_path,
    monkeypatch,
):
    from time import monotonic

    from nurse_rcl.v2.live.repair_timing import DEV_TIMING_POLICY, TimingJournal

    module = api("ops_execution")
    path, pin, plan = ops_plan(tmp_path, prepared_case)
    root = Path(plan["output_root"]) / plan["case"]["opaque_case_id"]
    root.mkdir()
    start = monotonic() - 3700.0
    timing = TimingJournal.open(
        root / "timing",
        identity=pin,
        now=start,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        expected_policy=DEV_TIMING_POLICY,
    )
    timing.switch("preparation", reason="raw_input_prepare", now=start + 0.1)
    monkeypatch.setattr(module, "supervise_phases", lambda *a, **kw: pytest.fail("budget reset"))
    result = module.run_round(path, pin, progress=lambda _: None)
    assert result["failure_type"] == "preparation_timeout"
    assert result["timing"]["preparation_remaining"] == 0
    assert module.status(path, pin) == result


def test_expired_cleanup_crash_before_terminal_is_recoverable(
    prepared_case,
    tmp_path,
    monkeypatch,
):
    from time import monotonic

    from nurse_rcl.v2.live.repair_timing import CaseBudgetExpired

    module = api("ops_execution")
    path, pin, _ = ops_plan(tmp_path, prepared_case)
    now = monotonic()
    monkeypatch.setattr(module, "monotonic", lambda: now)

    def expired(command, *, timing, **kwargs):
        nonlocal now
        timing.switch("preparation", reason="raw_input_prepare", now=now)
        now += 3601.0
        with pytest.raises(CaseBudgetExpired):
            timing.checkpoint(now=now)
        return 124

    original = module._publish_terminal
    monkeypatch.setattr(module, "supervise_phases", expired)
    monkeypatch.setattr(
        module,
        "_publish_terminal",
        lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic terminal publication crash")),
    )
    with pytest.raises(RuntimeError, match="publication crash"):
        module.run_round(path, pin, progress=lambda _: None)
    monkeypatch.setattr(module, "_publish_terminal", original)
    monkeypatch.setattr(module, "supervise_phases", lambda *a, **kw: pytest.fail("budget reset"))
    result = module.run_round(path, pin, progress=lambda _: None)
    assert result["failure_type"] == "preparation_timeout"
    assert result["timing"]["preparation_remaining"] == 0
    assert module.status(path, pin) == result


def test_unavailable_projection_stays_unavailable_on_offline_resume(
    prepared_case,
    tmp_path,
    monkeypatch,
):
    module = api("ops_case")
    path, pin, _ = ops_plan(tmp_path, prepared_case, mode="with-agent")
    admitted = api().load_ops_plan(path, pin)
    original = module.build_report_context
    monkeypatch.setattr(
        module,
        "build_report_context",
        lambda **kw: original(**kw).model_copy(update={"formal_projection_available": False}),
    )
    mock_ops_http(tmp_path, monkeypatch)
    first = module.run_ops_case(admitted, approved_plan_sha256=pin)
    assert first["channels"]["report"] == "unavailable"
    core = admitted.state_root / "core/result.json"
    raw = core.read_bytes()
    monkeypatch.setattr(module, "load_normal_preparation", lambda **_: pytest.fail("reload"))
    monkeypatch.setattr(
        api("ops_provider").ark,
        "load_ark_settings_from_key_file",
        lambda _: pytest.fail("offline credential access"),
    )
    final = module.run_ops_case(admitted, approved_plan_sha256=None)
    assert final["channels"]["report"] == "unavailable"
    assert final["channels"]["core"] == "completed"
    assert core.read_bytes() == raw
