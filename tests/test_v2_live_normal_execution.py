"""Normal runtime identities do not assert independent source/score replay."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def artifacts():
    name = "nurse_rcl.v2.live.normal_artifacts"
    assert importlib.util.find_spec(name), "missing normal execution identity"
    return importlib.import_module(name)


def test_normal_assurance_does_not_claim_independent_replay():
    module = artifacts()
    assert module.NORMAL_POLICY == "live-repair-normal-rcl/1"
    assert module.NormalAssurance().model_dump() == {
        "verification_level": "boundary_checked",
        "full_recompute": "not_performed",
        "interpretability_status": "source_bound_traceable",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"verification_level": "source_and_score_replayed"},
        {"verification_level": "preparation_bound_source_and_score_replayed"},
        {"full_recompute": "passed"},
        {"audit_passed": True},
    ],
)
def test_normal_assurance_rejects_unperformed_checks(changes):
    module = artifacts()
    with pytest.raises(ValidationError):
        module.NormalAssurance(**changes)


def execution():
    name = "nurse_rcl.v2.live.normal_execution"
    assert importlib.util.find_spec(name), "missing normal runner admission and timing"
    return importlib.import_module(name)


@pytest.fixture(scope="module", params=["rcabench", "aiops25"])
def prepared_case(tmp_path_factory, request):
    from tests.test_v2_live_repair_preparation_archive import build, load_arguments, setup

    root = tmp_path_factory.mktemp("normal-execution-" + request.param)
    archive, pair, args = setup(root, request.param)
    _, manifest, receipt = build(archive, pair, args)
    return pair, load_arguments(args, manifest, receipt)


def save(path, value):
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return sha256_hex(raw)


def freeze_plan(tmp_path, prepared):
    from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun

    module = execution()
    pair, args = prepared
    legacy = pair["legacy_stage_args"]
    source = tmp_path / "input.json"
    input_pin = save(source, legacy["specification"])
    case = dict(dataset=legacy["specification"]["dataset"],
                opaque_case_id=legacy["specification"]["opaque_case_id"],
                input_path=str(source), input_sha256=input_pin)
    code_root = Path(__file__).resolve().parents[1]
    code_files = {p.relative_to(code_root).as_posix(): sha256_hex(p.read_bytes())
                  for folder in ("src", "third_party/hermes_runtime")
                  for p in (code_root / folder).rglob("*.py")}
    configuration = tmp_path / "profile.json"
    # Use the actual pretty-printed profile bytes, not a canonicalized surrogate.
    profile_bytes = (code_root / "configs/v2/live_evidence_repair_normal.json").read_bytes()
    configuration.write_bytes(profile_bytes)
    configuration_pin = sha256_hex(profile_bytes)
    parent = tmp_path / "parent.json"
    parent_pin = save(parent, dict(cases=[case], output_root=str(tmp_path / "old")))
    lineage = tmp_path / "lineage.json"
    lineage_pin = save(lineage, dict(
        schema_version="nurse.v2.normal-rcl-lineage/1", opaque_case_id=case["opaque_case_id"],
        parent_plan_path=str(parent), parent_plan_sha256=parent_pin,
        budget_relationship="new_development_run_not_resume_or_refund",
        previous_runs=[], new_query_budget=8, new_report_requests=1, new_max_attempts=5,
    ))
    out = tmp_path / "normal"
    out.mkdir()
    run = RepairPreparationRun(
        key=args["expected_key"], archive_root=str(args["archive_root"]), mode="load",
        receipt_path=str(args["receipt_path"]),
        manifest_sha256=args["expected_manifest_sha256"],
        receipt_sha256=args["expected_receipt_sha256"],
    )
    plan = dict(
        schema_version=module.SCHEMA, code_root=str(code_root), code_files=code_files,
        configuration_path=str(configuration), configuration_sha256=configuration_pin,
        identity_manifest_path=str(legacy["identity_manifest_path"]),
        identity_artifact_sha256=legacy["expected_identity_artifact_sha256"],
        case=case, output_root=str(out), python_executable=sys.executable,
        method_identity=module.method_identity(), parent_plan_path=str(parent),
        parent_plan_sha256=parent_pin, lineage_path=str(lineage), lineage_sha256=lineage_pin,
        preparation=run.model_dump(mode="json"),
    )
    path = tmp_path / "plan.json"
    return path, save(path, plan), plan


def test_plan_admission_is_metadata_only(prepared_case, tmp_path, monkeypatch):
    module = execution()
    path, pin, plan = freeze_plan(tmp_path, prepared_case)
    from nurse_rcl.v2.live import repair_stage

    monkeypatch.setattr(repair_stage, "validate_input", lambda **_: pytest.fail("raw rescan"))
    admitted = module.load_execution(path, pin)
    assert admitted["preparation"].key.key_sha256 == plan["preparation"]["key"]["key_sha256"]
    assert admitted["raw_arguments"]["dataset"] == plan["case"]["dataset"]
    profile = json.loads(Path(plan["configuration_path"]).read_bytes())
    assert module.require_profile(profile) == dict(preparation=3600, method=None, case=None)
    for key, value in (("method_case_budget_seconds", 600), ("credential_variable", "ARK_API_KEY"),
                       ("max_request_attempts", 6), ("worker_count", 2)):
        with pytest.raises(ValueError):
            module.require_profile({**profile, key: value})
    with pytest.raises(ValueError):
        module.load_execution(path, "0" * 64)
    changed = {**plan, "skip_checks": True}
    with pytest.raises(ValueError):
        module.load_execution(path, save(path, changed))


@pytest.mark.parametrize("network_ok", [True, False])
def test_real_worker_completes_with_parent_timing_and_saved_results(
    prepared_case, tmp_path, monkeypatch, network_ok,
):
    module = execution()
    path, pin, plan = freeze_plan(tmp_path, prepared_case)
    actual = module.supervise_phases
    wire = tmp_path / "wire"
    wire.mkdir()
    # Only HTTP is mocked inside the real child. The public worker, phase socket,
    # preparation, query, scoring, publication and parent manifest check all run.
    http = ("lambda r:httpx.Response(200,json=json.loads(response_for(r.content)))"
            if network_ok else "lambda r:httpx.Response(400,json={'error':'synthetic'})")
    script = (
        "import json,httpx,pytest; from pathlib import Path; "
        "from tests.test_v2_live_repair_case_report import transport; "
        "from tests.test_v2_live_report import response_for; "
        f"mp=pytest.MonkeyPatch(); transport(Path({str(wire)!r}),mp,"
        f"{http}); "
        "from nurse_rcl.v2.live.normal_execution import main; raise SystemExit(main())"
    )

    def supervised(command, **kwargs):
        return actual([command[0], "-c", script, *command[3:]], **kwargs)

    monkeypatch.setattr(module, "supervise_phases", supervised)
    events = []
    result = module.run_round(path, pin, progress=events.append)
    assert result["status"] == ("completed" if network_ok else "failed"), result
    assert result["worker_exit_code"] == (0 if network_ok else 1)
    assert result["timing"]["preparation_seconds"] > 0
    assert result["timing"]["method_seconds"] > 0
    assert result["timing"]["method_remaining"] is None
    state = Path(plan["output_root"]) / plan["case"]["opaque_case_id"]
    table = json.loads((state / "module-timing.json").read_bytes())
    assert table["complete"]
    modules = {r["module"] for r in table["modules"]}
    assert {"fixed_facts", "error_facts", "local_scoring", "preparation_read"} <= modules
    if network_ok:
        assert "explanation_checks" in modules
    assert not {"independent_replay", "original_exit", "fresh_exit"} & modules
    assert any(e["event"] == "stage_process_started" for e in events)
    assert (state / "result.json").is_file() == network_ok
    monkeypatch.setattr(module, "supervise_phases", lambda *a, **k: pytest.fail("reran completed"))
    assert module.run_round(path, pin, progress=events.append) == result


def test_diagnostic_clock_keeps_cumulative_preparation_but_no_method_deadline(tmp_path):
    from nurse_rcl.v2.live.repair_timing import (
        DIAGNOSTIC_TIMING_POLICY,
        CaseBudgetExpired,
        TimingJournal,
    )

    execution()
    root = tmp_path / "clock"
    clock = TimingJournal.open(root, identity="b" * 64, now=0.0, boot_id="test",
                               expected_policy=DIAGNOSTIC_TIMING_POLICY)
    clock.switch("preparation", reason="raw_input_prepare", now=1.0)
    clock.switch("method", reason="orchestration", now=3501.0)
    resumed = TimingJournal.open(root, identity="b" * 64, now=20000.0, boot_id="test",
                                 expected_policy=DIAGNOSTIC_TIMING_POLICY)
    assert resumed.clock.used_seconds["preparation"] == 3500
    assert resumed.clock.remaining_seconds is None
    resumed.switch("preparation", reason="raw_input_prepare", now=20001.0)
    with pytest.raises(CaseBudgetExpired, match="preparation_timeout"):
        resumed.checkpoint(now=20101.0)


# Historical single-case administrative freezer coverage remains in the source
# worktree; its case-specific deployment script is not part of this release.
