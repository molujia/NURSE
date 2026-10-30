"""Whole synthetic raw cases reach persisted stages, never a real ARK endpoint."""

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.repair_raw_audit import _file_pins
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config
from tests.test_v2_live_report import response_for
from tests.test_v2_live_staging import provision, setup_case


def api():
    name = "nurse_rcl.v2.live.repair_stage"
    assert importlib.util.find_spec(name), "missing source-bound whole-case stage entry"
    return importlib.import_module(name)


def inputs(tmp_path):
    from nurse_rcl.v2.live.staged_case import load_staged_rcabench_case

    args, opaque = setup_case(tmp_path)
    stage = provision(args)
    root = args["stage_root"] / opaque
    source = load_staged_rcabench_case(
        root,
        expected_stage_sha256=stage.stage_sha256,
        inference_manifest_path=args["inference_manifest_path"],
        expected_inference_artifact_sha256=args["expected_inference_artifact_sha256"],
        configuration=build_rcabench_adapter_config(window_seconds=240),
    )
    spec = dict(
        schema_version="nurse.v2.live-repair-case-input/1",
        dataset="rcabench",
        opaque_case_id=opaque,
        source_root=str(root),
        stage_sha256=stage.stage_sha256,
        source_descriptor_sha256=source.descriptor()["descriptor_sha256"],
        source_files=[list(pair) for pair in _file_pins(root, "rcabench")],
        configuration_sha256=source.adapter.configuration.configuration_sha256,
        source_case_id=None,
        authorized_queue=None,
    )
    state = tmp_path / "state"
    state.mkdir()
    # The inference stage must continue after all private mapping/answer inputs
    # become unavailable. Only the immutable export and inference manifest remain.
    args["evaluator_map_path"].rename(args["evaluator_map_path"].with_suffix(".offline"))
    args["source_root"].rename(args["source_root"].with_name("offline"))
    return dict(
        specification=spec,
        expected_input_sha256=sha256_hex(canonical_json_bytes(spec)),
        identity_manifest_path=args["inference_manifest_path"],
        expected_identity_artifact_sha256=args["expected_inference_artifact_sha256"],
        state_root=state,
        execution_sha256="a" * 64,
    )


def test_raw_queries_audit_report_publish_once_and_resume_without_redraw(tmp_path):
    module = api()
    args = inputs(tmp_path)
    calls, progress = [], []

    def complete(payload):
        calls.append(payload)
        assert b"SECRET_LABEL" not in payload and b"answer-bearing-case" not in payload
        return response_for(payload)

    for stage in ("queries", "audit", "report"):
        module.run_stage(stage=stage, **args, complete=complete, progress=progress.append)
    assert len(calls) == 1
    summary = json.loads((args["state_root"] / "audit/summary.json").read_bytes())
    audit = json.loads((args["state_root"] / "audit/audit.json").read_bytes())
    assert summary["audit_sha256"] == audit["result_sha256"]
    report = json.loads((args["state_root"] / "report/report.json").read_bytes())
    assert report["exchange_attempted"] and report["status"] == "full"
    assert report["pre_ranking_semantics"] is False
    assert any(row["event"] == "queries_completed" for row in progress)
    assert any(row["event"] == "audit_completed" for row in progress)
    before = (args["state_root"] / "report/report.json").read_bytes()
    for stage in ("queries", "audit", "report"):
        module.run_stage(
            stage=stage, **args, complete=lambda _: pytest.fail("redraw"), progress=progress.append
        )
    assert (args["state_root"] / "report/report.json").read_bytes() == before


@pytest.mark.parametrize("change", ["input", "files", "identity", "phase"])
def test_stage_rejects_drift_or_out_of_order_before_query_or_network(tmp_path, change):
    module = api()
    args = inputs(tmp_path)
    stage = "queries"
    if change == "input":
        args["specification"]["opaque_case_id"] = "opq_" + "f" * 32
    elif change == "files":
        from pathlib import Path

        path = Path(args["specification"]["source_root"]) / "normal_metrics.parquet"
        path.write_bytes(path.read_bytes() + b"drift")
    elif change == "identity":
        args["expected_identity_artifact_sha256"] = "f" * 64
    else:
        stage = "report"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.run_stage(stage=stage, **args, complete=lambda _: pytest.fail("sent invalid case"))
    assert not (args["state_root"] / "queries").exists()
    assert not (args["state_root"] / "report").exists()


def test_stage_retains_query_failure_and_never_publishes_success(tmp_path, monkeypatch):
    module = api()
    args = inputs(tmp_path)
    from nurse_rcl.v2.live.repair_query_session import RepairQueryRuntime

    def failure(self, *values, **kwargs):
        raise ValueError("controlled query integrity failure")

    monkeypatch.setattr(RepairQueryRuntime, "_advance_values", failure)
    with pytest.raises(ValueError):
        module.run_stage(stage="queries", **args, complete=lambda _: pytest.fail("sent"))
    assert not (args["state_root"] / "queries").exists()
    assert (args["state_root"] / "failure-queries.json").is_file()


def test_frozen_success_still_checks_source_drift_on_resume(tmp_path):
    module = api()
    args = inputs(tmp_path)
    module.run_stage(stage="queries", **args, complete=lambda _: pytest.fail("sent"))
    from pathlib import Path

    path = Path(args["specification"]["source_root"]) / "normal_metrics.parquet"
    path.write_bytes(path.read_bytes() + b"late drift")
    with pytest.raises(ValueError):
        module.run_stage(stage="queries", **args, complete=lambda _: pytest.fail("sent"))


def aiops25_inputs(tmp_path):
    from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
    from tests.test_v2_live_repair_raw_audit import arguments

    raw_args = arguments(tmp_path, "aiops25")
    opaque = raw_args["opaque_case_id"]
    identity = dict(
        schema_version="nurse.v2.inference-identity-manifest/1",
        method="hmac_sha256_128",
        case_count=1,
        rcabench_case_count=0,
        aiops25_case_count=1,
        records=[dict(dataset="aiops25", opaque_case_id=opaque, method="hmac_sha256_128")],
    )
    identity["manifest_sha256"] = sha256_hex(canonical_json_bytes(identity))
    assert InferenceIdentityManifest.model_validate(identity).verify()
    path = tmp_path / "identity.json"
    path.write_bytes(canonical_json_bytes(identity))
    spec = dict(
        schema_version="nurse.v2.live-repair-case-input/1",
        dataset="aiops25",
        opaque_case_id=opaque,
        source_root=str(raw_args["source_root"]),
        stage_sha256=None,
        source_descriptor_sha256=raw_args["expected_source_descriptor_sha256"],
        source_files=[list(p) for p in _file_pins(raw_args["source_root"], "aiops25")],
        configuration_sha256=raw_args["configuration"].configuration_sha256,
        source_case_id=raw_args["source_case_id"],
        authorized_queue=raw_args["authorized_queue"].model_dump(mode="json"),
    )
    state = tmp_path / "state"
    state.mkdir()
    return dict(
        specification=spec,
        expected_input_sha256=sha256_hex(canonical_json_bytes(spec)),
        identity_manifest_path=path,
        expected_identity_artifact_sha256=sha256_hex(path.read_bytes()),
        state_root=state,
        execution_sha256="a" * 64,
    )


def test_aiops25_authorized_whole_prepared_case_reaches_the_same_report_stages(tmp_path):
    module = api()
    args = aiops25_inputs(tmp_path)
    for stage in ("queries", "audit", "report"):
        module.run_stage(
            stage=stage,
            **args,
            complete=response_for,
        )
    result = json.loads((args["state_root"] / "report/report.json").read_bytes())
    assert result["status"] == "full" and result["exchange_status"] == "response_ready"


def test_real_stage_worker_uses_durable_repair_bridge_after_source_replay(tmp_path, monkeypatch):
    import httpx

    from nurse_rcl.agents.ark import load_ark_settings
    from nurse_rcl.v2.live import ark_bridge, repair_execution

    assert hasattr(repair_execution, "run_worker"), (
        "missing repair stage worker with actual bridge wiring"
    )
    args = inputs(tmp_path)
    opaque = args["specification"]["opaque_case_id"]
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "third_party/hermes_runtime").mkdir(parents=True)
    (code / "src/main.py").write_bytes(b"pass\n")
    out = tmp_path / "round"
    out.mkdir()
    args["state_root"].rename(out / opaque)
    input_path = tmp_path / "input.json"
    input_path.write_bytes(canonical_json_bytes(args["specification"]))
    plan = dict(
        schema_version="nurse.v2.live-repair-execution-plan/2",
        code_root=str(code),
        code_files={"src/main.py": sha256_hex(b"pass\n")},
        output_root=str(out),
        cases=[
            dict(
                dataset="rcabench",
                opaque_case_id=opaque,
                input_path=str(input_path),
                input_sha256=args["expected_input_sha256"],
            )
        ],
        identity_manifest_path=str(args["identity_manifest_path"]),
        identity_artifact_sha256=args["expected_identity_artifact_sha256"],
    )
    plan["cases"].extend(
        {**plan["cases"][0], "opaque_case_id": "opq_" + str(i) * 32} for i in range(2, 5)
    )
    calls = []

    def http(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer synthetic-key2"
        return httpx.Response(200, json=json.loads(response_for(request.content)))

    original = ark_bridge.LiveArkBridge

    def bridge(**kwargs):
        kwargs.update(transport=httpx.MockTransport(http), lock_path=tmp_path / "test-ark.lock")
        return original(**kwargs)

    monkeypatch.setattr(ark_bridge, "LiveArkBridge", bridge)
    monkeypatch.setattr(
        "nurse_rcl.agents.ark.load_ark_settings_from_key_file",
        lambda _: load_ark_settings({"ARK_API_KEY2": "synthetic-key2"}),
    )
    original_read = repair_execution.read_json

    def bounded_control_read(path, *positional, **kwargs):
        assert path.name != "audit.json", (
            "the small control reader must not parse the full live audit"
        )
        return original_read(path, *positional, **kwargs)

    monkeypatch.setattr(repair_execution, "read_json", bounded_control_read)
    for stage in ("queries", "audit", "report"):
        repair_execution.run_worker(plan, "a" * 64, 0, stage, progress=lambda _: None)
    repair_execution.run_worker(plan, "a" * 64, 0, "report", progress=lambda _: None)
    assert len(calls) == 1
    assert list((out / opaque).glob("ark-receipt-*.json"))
    repair_execution._require_live_closure(out / opaque, plan["cases"][0])
    report_path = out / opaque / "report/report.json"
    report = json.loads(report_path.read_bytes())
    report["exchange_attempted"] = False
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError):
        repair_execution._require_live_closure(out / opaque, plan["cases"][0])
