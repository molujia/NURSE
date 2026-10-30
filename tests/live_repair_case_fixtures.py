"""Complete synthetic stage inputs for old/new case execution, never real data."""

import json
import sys
from pathlib import Path
from uuid import UUID

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.repair_stage import validate_input
from nurse_rcl.v2.live.repair_timing import TIMING_POLICY, TimingJournal
from tests.test_v2_live_repair_stage import aiops25_inputs, inputs


def make_case_pair(tmp_path, dataset, *, eight_queries=False, timing_policy=TIMING_POLICY):
    if eight_queries:
        from unittest.mock import patch

        import pyarrow.parquet as pq

        from tests import test_v2_live_repair_stage as stage_fixture
        from tests.test_v2_live_sources import T0, write

        assert dataset == "rcabench"
        setup = stage_fixture.setup_case

        def setup_eight(root):
            values, opaque = setup(root)
            raw = values["source_root"]
            metadata = pq.read_table(raw / "normal_metrics.parquet").schema.metadata
            for window, start in (("normal", T0 - 240), ("abnormal", T0)):
                write(
                    raw,
                    f"{window}_metrics.parquet",
                    [start + n * 30 for _ in range(9) for n in range(8)],
                    metric=["container.cpu.usage"] * 72,
                    value=(
                        [10.0] * 8
                        if window == "normal"
                        else [10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
                    )
                    * 9,
                    service_name=[f"service-{i}" for i in range(9) for _ in range(8)],
                    answer=["SECRET_LABEL"] * 72,
                )
            path = raw / "normal_metrics.parquet"
            pq.write_table(pq.read_table(path).replace_schema_metadata(metadata), path)
            return values, opaque

        # Fixture provisioning only: no runtime function or guard is mocked.
        with patch.object(stage_fixture, "setup_case", setup_eight):
            args = inputs(tmp_path)
    else:
        args = (inputs if dataset == "rcabench" else aiops25_inputs)(tmp_path)
    opaque = args["specification"]["opaque_case_id"]
    owner_state = tmp_path / "owner" / opaque
    owner_state.mkdir(parents=True)
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "third_party/hermes_runtime").mkdir(parents=True)
    (code / "src/main.py").write_bytes(b"pass\n")
    timing_identity = sha256_hex(
        canonical_json_bytes([args["execution_sha256"], args["expected_input_sha256"], opaque])
    )
    TimingJournal.open(
        owner_state / "timing",
        identity=timing_identity,
        now=0.0,
        boot_id="synthetic",
        expected_policy=timing_policy,
    )
    owned = {**args, "state_root": owner_state}
    _, _, raw = validate_input(**{k: v for k, v in owned.items() if k != "execution_sha256"})
    return dict(
        owner_stage_args={
            **owned,
            "code_root": code,
            "code_files": {"src/main.py": sha256_hex(b"pass\n")},
            "timing_identity_sha256": timing_identity,
        },
        owner_state=owner_state,
        legacy_stage_args=args,
        raw_arguments=raw,
    )


def make_first_case_plan(tmp_path):
    """Freeze four metadata entries before staging; only first telemetry is active.

    All metadata comes from actual complete synthetic source descriptors. Tests
    additionally refuse nonselected telemetry reads during the owning invocation.
    """
    from nurse_rcl.v2.live.repair_execution import FIRST_CASE_ID
    from nurse_rcl.v2.live.repair_raw_audit import _file_pins
    from nurse_rcl.v2.live.staged_case import load_staged_rcabench_case
    from nurse_rcl.v2.opaque_ids import EvaluatorIdentitySource, build_identity_artifacts
    from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config
    from tests.test_v2_live_staging import provision, setup_case

    def save(path, value):
        data = canonical_json_bytes(value)
        path.write_bytes(data)
        return sha256_hex(data)

    args, _ = setup_case(tmp_path)
    ids = (FIRST_CASE_ID, "opq_" + "1" * 32, "opq_" + "2" * 32, "opq_" + "3" * 32)
    # Generation order is dataset/original-id, independent from active queue order.
    uuids = iter(UUID(hex=ids[i][4:], version=4) for i in (2, 3, 0, 1))
    sources = tuple(
        EvaluatorIdentitySource(
            dataset="rcabench" if i < 2 else "aiops25",
            original_case_id=args["source_root"].name if i == 0 else f"sibling-{i}",
            incident_group_id="grp_" + str(i) * 24,
            answer_services=("SECRET_LABEL",),
        )
        for i in range(4)
    )
    mapping, manifest = build_identity_artifacts(
        sources, method="random_uuid_v4", uuid_factory=lambda: next(uuids)
    )
    by_original = {r.original_case_id: r.opaque_case_id for r in mapping.records}
    ordered_ids = [by_original[s.original_case_id] for s in sources]
    assert ordered_ids[0] == FIRST_CASE_ID
    args["expected_evaluator_artifact_sha256"] = save(
        args["evaluator_map_path"], mapping.model_dump(mode="json")
    )
    args["expected_inference_artifact_sha256"] = save(
        args["inference_manifest_path"], manifest.model_dump(mode="json")
    )
    stage = provision(args)
    root = args["stage_root"] / FIRST_CASE_ID
    config = build_rcabench_adapter_config(window_seconds=240)
    source = load_staged_rcabench_case(
        root,
        expected_stage_sha256=stage.stage_sha256,
        inference_manifest_path=args["inference_manifest_path"],
        expected_inference_artifact_sha256=args["expected_inference_artifact_sha256"],
        configuration=config,
    )
    spec = dict(
        schema_version="nurse.v2.live-repair-case-input/1",
        dataset="rcabench",
        opaque_case_id=FIRST_CASE_ID,
        source_root=str(root),
        stage_sha256=stage.stage_sha256,
        source_descriptor_sha256=source.descriptor()["descriptor_sha256"],
        source_files=[list(pair) for pair in _file_pins(root, "rcabench")],
        configuration_sha256=config.configuration_sha256,
        source_case_id=None,
        authorized_queue=None,
    )
    # Inference has no evaluator map or private original case available.
    args["evaluator_map_path"].rename(args["evaluator_map_path"].with_suffix(".offline"))
    args["source_root"].rename(args["source_root"].with_name("offline"))
    queue = []
    for i, opaque in enumerate(ordered_ids):
        value = spec
        if i == 1:
            sibling = tmp_path / "sibling-rca"
            sibling.mkdir()
            sibling_args, _ = setup_case(sibling)
            source_root = sibling_args["source_root"].with_name("sibling-1")
            sibling_args["source_root"].rename(source_root)
            sibling_args.update(
                source_root=source_root,
                evaluator_map_path=args["evaluator_map_path"].with_suffix(".offline"),
                expected_evaluator_artifact_sha256=args["expected_evaluator_artifact_sha256"],
                inference_manifest_path=args["inference_manifest_path"],
                expected_inference_artifact_sha256=args["expected_inference_artifact_sha256"],
            )
            sibling_stage = provision(sibling_args)
            sibling_root = sibling_args["stage_root"] / opaque
            sibling_source = load_staged_rcabench_case(
                sibling_root,
                expected_stage_sha256=sibling_stage.stage_sha256,
                inference_manifest_path=args["inference_manifest_path"],
                expected_inference_artifact_sha256=args["expected_inference_artifact_sha256"],
                configuration=config,
            )
            value = {
                **spec,
                "opaque_case_id": opaque,
                "source_root": str(sibling_root),
                "stage_sha256": sibling_stage.stage_sha256,
                "source_descriptor_sha256": sibling_source.descriptor()["descriptor_sha256"],
                "source_files": [list(p) for p in _file_pins(sibling_root, "rcabench")],
            }
            source_root.rename(source_root.with_name("offline"))
        elif i > 1:
            from nurse_rcl.v2.live.aiops25_sources import read_aiops25_live_sources
            from tests.test_v2_live_repair_raw_audit import arguments

            sibling = tmp_path / f"sibling-aiops-{i}"
            sibling.mkdir()
            raw = arguments(sibling, "aiops25")
            source25 = read_aiops25_live_sources(
                raw["source_root"],
                source_case_id=raw["source_case_id"],
                opaque_case_id=opaque,
                authorized_queue=raw["authorized_queue"],
                configuration=raw["configuration"],
            )
            value = dict(
                schema_version=spec["schema_version"],
                dataset="aiops25",
                opaque_case_id=opaque,
                source_root=str(raw["source_root"]),
                stage_sha256=None,
                source_descriptor_sha256=source25.descriptor()["descriptor_sha256"],
                source_files=[list(p) for p in _file_pins(raw["source_root"], "aiops25")],
                configuration_sha256=raw["configuration"].configuration_sha256,
                source_case_id=raw["source_case_id"],
                authorized_queue=raw["authorized_queue"].model_dump(mode="json"),
            )
        path = tmp_path / f"input-{i}.json"
        queue.append(
            dict(
                dataset=value["dataset"],
                opaque_case_id=opaque,
                input_path=str(path),
                input_sha256=save(path, value),
            )
        )
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "third_party/hermes_runtime").mkdir(parents=True)
    (code / "src/main.py").write_bytes(b"pass\n")
    profile = json.loads(Path("configs/v2/live_evidence_repair.json").read_bytes())
    profile.pop("stage_order")
    profile.pop("preparation_execution_version")
    profile.update(
        schema_version="nurse.v2.live-evidence-repair-profile/2",
        case_execution_version="live-repair-case-owner/1",
        worker_stage_order=["case"],
    )
    profile_path = code / "owner-profile.json"
    profile_pin = save(profile_path, profile)
    out = tmp_path / "out"
    out.mkdir()
    parent_path = tmp_path / "original-plan.json"
    common = dict(
        cases=queue,
        identity_manifest_path=str(args["inference_manifest_path"]),
        identity_artifact_sha256=args["expected_inference_artifact_sha256"],
    )
    parent_pin = save(
        parent_path,
        dict(
            **common,
            schema_version="nurse.v2.live-repair-execution-plan/1",
            output_root=str(tmp_path / "old-out"),
        ),
    )
    # Expected new identity is test-owned, not obtained from the new loader.
    from nurse_rcl.v2.live.repair_case_artifacts import REPAIR_CASE_AUDIT_SEMANTICS_SHA256
    from nurse_rcl.v2.live.repair_execution import method_identity

    method = method_identity()
    method.pop("raw_audit")
    method.update(
        case_execution="live-repair-case-owner/1",
        case_audit=REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
        report_schema="nurse.v2.live-repair-report/2",
        projection_schema="nurse.v2.live-repair-report-projection/2",
        closure_schema="nurse.v2.live-repair-case-closure/1",
    )
    plan = dict(
        **common,
        schema_version="nurse.v2.live-repair-case-owner-single-plan/1",
        code_root=str(code),
        code_files={"src/main.py": sha256_hex(b"pass\n")},
        configuration_path=str(profile_path),
        configuration_sha256=profile_pin,
        output_root=str(out),
        python_executable=sys.executable,
        method_identity=method,
        parent_plan_path=str(parent_path),
        parent_plan_sha256=parent_pin,
        active_case_indexes=[0],
    )
    # Loader tests use a frozen metadata binding. The freezer tests separately
    # construct and reconstruct actual failed /1 query and timing journals.
    lineage_path = tmp_path / "synthetic-lineage.json"
    lineage = dict(
        schema_version="nurse.v2.live-repair-case-owner-lineage/1",
        opaque_case_id=ids[0],
        parent_plan_path=str(parent_path),
        parent_plan_sha256=parent_pin,
        budget_relationship="new_development_run_not_resume_or_refund",
        initial_only=True,
        old_budget_exhausted=True,
        query_reservations=0,
        measured=0,
        completed=0,
        ark_attempts=0,
    )
    plan.update(lineage_path=str(lineage_path), lineage_sha256=save(lineage_path, lineage))
    path = tmp_path / "owner-plan.json"
    return dict(plan_path=path, plan_sha256=save(path, plan), plan=plan)
