"""Fixed Dev paired queue, not a general scheduler or permission to launch a batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.live.dev_batch import (
    ASSIGNED_FAILURE,
    load_dev_membership,
    read_pair,
    seal_dev_pairs,
    verify_dev_seal,
)
from nurse_rcl.v2.live.dev_budget import DevBudget
from nurse_rcl.v2.live.dev_execution import DevPreparationTimeout, DevQueuePaused, execute_branch
from nurse_rcl.v2.live.normal_b2_execution import SCHEMA as B2_SCHEMA
from nurse_rcl.v2.live.normal_b2_execution import configuration as b2_configuration
from nurse_rcl.v2.live.normal_execution import (
    B0_SCHEMA,
    _metadata,
    _pinned,
    b0_configuration,
    b0_method_identity,
)
from nurse_rcl.v2.live.normal_semantic_indexed_exchange import INDEXED_SEMANTIC_EXCHANGE_VERSION
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_execution import (
    install_termination_handlers,
    require_code,
    supervise_phases,
)
from nurse_rcl.v2.live.repair_module_timing import (
    _module_span,
    _module_timing_scope,
    summarize_module_timing,
)
from nurse_rcl.v2.live.repair_preparation_archive import preparation_key
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_stage import RepairCaseInput
from nurse_rcl.v2.live.repair_timing import (
    DEV_TIMING_POLICY,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

SCHEMA = "nurse.v2.dev-paired-batch/1"


def code_manifest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): file_sha256(require_plain_path(p))
        for folder in ("src", "third_party/hermes_runtime")
        for p in sorted((root / folder).rglob("*.py"))
    }


def _budgets(new: int) -> dict[str, Any]:
    return dict(
        max_logical_requests=new * 3,
        max_attempts=new * 15,
        preparation_per_case_seconds=3600,
        method_hard_deadline=None,
        method_warning_seconds=600,
        timing_policy=DEV_TIMING_POLICY,
        worker_count=1,
        llm_in_flight=1,
        memory_limit_bytes=8589934592,
    )


def freeze_batch(
    *,
    settings: dict[str, Any],
    code_root: Path,
    output: Path,
    imports: list[dict[str, Any]],
) -> tuple[Path, str]:
    require_plain_path(output)
    if any(output.iterdir()):
        raise ValueError("Dev freeze needs a new empty output directory")
    members = load_dev_membership(Path(settings["split_path"]), settings["split_sha256"])
    imported = {(p["dataset"], p["opaque_case_id"]): p for p in imports}
    if len(imported) != len(imports) or not imported.keys() <= set(members):
        raise ValueError("Dev imported pair membership differs")
    for pair in imports:
        read_pair(pair)
    files = code_manifest(code_root)
    require_code(code_root, files)
    rows = [
        dict(index=i, dataset=d, opaque_case_id=opaque, imported_pair=imported.get((d, opaque)))
        for i, (d, opaque) in enumerate(members, 1)
    ]
    batch = dict(
        schema_version=SCHEMA,
        code_root=str(code_root),
        code_files=files,
        output_root=str(output),
        python_executable=sys.executable,
        settings=settings,
        rows=rows,
        B0_configuration=b0_configuration(),
        B2_configuration=b2_configuration(semantic_protocol=INDEXED_SEMANTIC_EXCHANGE_VERSION),
        budgets=_budgets(len(rows) - len(imports)),
    )
    path = output / "batch.json"
    publish_json(path, batch)
    return path, file_sha256(path)


def read_batch(path: Path, pin: str) -> dict[str, Any]:
    batch = _metadata(_pinned(path, pin))
    if (
        set(batch)
        != {
            "schema_version",
            "code_root",
            "code_files",
            "output_root",
            "python_executable",
            "settings",
            "rows",
            "B0_configuration",
            "B2_configuration",
            "budgets",
        }
        or batch["schema_version"] != SCHEMA
        or (
            batch["python_executable"] != sys.executable
            or Path(batch["output_root"]) != path.parent
            or batch["B0_configuration"] != b0_configuration()
            or batch["B2_configuration"]
            != b2_configuration(
                semantic_protocol=INDEXED_SEMANTIC_EXCHANGE_VERSION,
            )
        )
    ):
        raise ValueError("Dev batch frozen fields differ")
    root = require_plain_path(Path(batch["code_root"]))
    if not Path(__file__).resolve().is_relative_to(root):
        raise ValueError("Dev controller is outside frozen code root")
    require_code(root, batch["code_files"])
    settings, rows = batch["settings"], batch["rows"]
    members = load_dev_membership(Path(settings["split_path"]), settings["split_sha256"])
    if (
        type(rows) is not list
        or len(rows) != len(members)
        or any(
            type(r) is not dict
            or set(r) != {"index", "dataset", "opaque_case_id", "imported_pair"}
            or type(r["index"]) is not int
            or r["index"] != i
            or (r["dataset"], r["opaque_case_id"]) != member
            for i, (r, member) in enumerate(zip(rows, members, strict=True), 1)
        )
    ):
        raise ValueError("Dev batch complete case membership/order differs")
    for row in rows:
        pair = row["imported_pair"]
        if pair is not None and any(pair.get(k) != row[k] for k in ("dataset", "opaque_case_id")):
            raise ValueError("Dev imported pair differs from assigned member")
    if batch["budgets"] != _budgets(sum(r["imported_pair"] is None for r in rows)):
        raise ValueError("Dev batch budget differs")
    return batch


def _case_root(batch: dict[str, Any], row: dict[str, Any]) -> Path:
    return Path(batch["output_root"]) / f"case-{row['index']:04d}"


def _case_identity(pin: str, row: dict[str, Any]) -> str:
    return sha256_hex(
        canonical_json_bytes([pin, row["index"], row["dataset"], row["opaque_case_id"]])
    )


def _supply_and_freeze(batch: dict[str, Any], row: dict[str, Any]) -> None:
    # The only import of private map access. This worker exits before B0/B2.
    from nurse_rcl.v2.live.dev_provision import provision_one

    root = _case_root(batch, row)
    supply = root / "supply"
    supply.mkdir(mode=0o700, exist_ok=True)
    receipt = provision_one(
        batch["settings"],
        dataset=row["dataset"],
        opaque_case_id=row["opaque_case_id"],
        output=supply,
    )
    case = receipt["case"]
    spec = RepairCaseInput.model_validate_json(
        _pinned(Path(case["input_path"]), case["input_sha256"])
    )
    configuration: Any = build_rcabench_adapter_config(window_seconds=240)
    extra: dict[str, Any] = {}
    if spec.dataset == "aiops25":
        configuration = build_aiops25_adapter_config(window_seconds=480)
        extra = dict(source_case_id=spec.source_case_id, authorized_queue=spec.authorized_queue)
    raw = dict(
        source_root=Path(spec.source_root),
        dataset=spec.dataset,
        opaque_case_id=spec.opaque_case_id,
        configuration=configuration,
        expected_source_descriptor_sha256=spec.source_descriptor_sha256,
        **extra,
    )
    key = preparation_key(
        raw_arguments=raw, input_sha256=case["input_sha256"], code_root=Path(batch["code_root"])
    )
    if key.source_files != spec.source_files:
        raise ValueError("Dev complete source inventory changed during supply")
    out = root / "b0"
    out.mkdir(mode=0o700, exist_ok=True)
    state = out / case["opaque_case_id"]
    state.mkdir(mode=0o700, exist_ok=True)
    (root / "preparations").mkdir(mode=0o700, exist_ok=True)
    run = RepairPreparationRun(
        key=key,
        archive_root=str(root / "preparations" / key.key_sha256),
        mode="build",
        receipt_path=str(state / "preparation-receipt.json"),
        manifest_sha256=None,
        receipt_sha256=None,
    )
    parent = supply / "parent.json"
    publish_json(parent, dict(cases=[case], output_root=str(supply)))
    lineage = dict(
        schema_version="nurse.v2.normal-b0-lineage/1",
        opaque_case_id=case["opaque_case_id"],
        parent_plan_path=str(parent),
        parent_plan_sha256=file_sha256(parent),
        budget_relationship="new_development_run_not_resume_or_refund",
        previous_runs=[],
        new_query_budget=8,
        new_report_requests=0,
        new_max_attempts=0,
    )
    publish_json(out / "lineage.json", lineage)
    publish_json(out / "configuration.json", batch["B0_configuration"])
    plan = dict(
        schema_version=B0_SCHEMA,
        code_root=batch["code_root"],
        code_files=batch["code_files"],
        configuration_path=str(out / "configuration.json"),
        configuration_sha256=file_sha256(out / "configuration.json"),
        identity_manifest_path=batch["settings"]["identity_manifest_path"],
        identity_artifact_sha256=batch["settings"]["identity_artifact_sha256"],
        case=case,
        output_root=str(out),
        python_executable=batch["python_executable"],
        method_identity=b0_method_identity(),
        parent_plan_path=str(parent),
        parent_plan_sha256=file_sha256(parent),
        lineage_path=str(out / "lineage.json"),
        lineage_sha256=file_sha256(out / "lineage.json"),
        preparation=run.model_dump(mode="json"),
    )
    publish_json(out / "plan.json", plan)
    publish_json(
        root / "supplied.json",
        dict(plan_path=str(out / "plan.json"), plan_sha256=file_sha256(out / "plan.json")),
    )


def _execute_case(
    batch: dict[str, Any],
    row: dict[str, Any],
    path: Path,
    pin: str,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    root = _case_root(batch, row)
    root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(root)
    identity = _case_identity(pin, row)
    if (root / "pair.json").exists():
        retained = read_json(root / "pair.json")
        if retained.get("identity") != identity:
            raise ValueError("Dev retained case identity differs")
        read_pair(retained["pair"])
        return dict(retained["pair"])
    timing = DevBudget.open(
        root / "timing",
        identity=identity,
        now=monotonic(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    b0: dict[str, Any] | None = None

    def finish_pair(pair: dict[str, Any], *, expired: bool = False) -> dict[str, Any]:
        read_pair(pair)
        if expired:
            # Keep incomplete spans rather than inventing their end times after termination.
            table = dict(
                status="preparation_timeout",
                complete=False,
                event_files={
                    p.name: file_sha256(p) for p in (root / "module-spans").glob("*.jsonl")
                },
                timing=timing.clock.summary(),
            )
        else:
            timing.pause(now=monotonic())
            table = summarize_module_timing(
                event_paths=tuple(sorted((root / "module-spans").glob("*.jsonl"))),
                timing_root=timing.root,
                expected_timing_identity=identity,
                expected_policy=DEV_TIMING_POLICY,
            )
        publish_json(root / "module-timing.json", table)
        record = dict(
            pair=pair,
            timing=timing.clock.summary(),
            module_timing=dict(
                path=str(root / "module-timing.json"),
                sha256=file_sha256(root / "module-timing.json"),
            ),
            branch_account_deltas={
                name: read_json(root / (name.lower() + "-execution.json"))["account_delta"]
                if (root / (name.lower() + "-execution.json")).exists()
                else None
                for name in ("B0", "B2")
            },
        )
        publish_json(root / "execution.json", record)
        pair = dict(
            pair,
            execution=dict(
                path=str(root / "execution.json"), sha256=file_sha256(root / "execution.json")
            ),
        )
        publish_json(
            root / "pair.json",
            dict(
                identity=identity,
                pair=pair,
                timing=timing.clock.summary(),
                timing_checkpoint_sha256=timing.previous,
            ),
        )
        return pair

    try:
        if not (root / "supplied.json").exists():
            with (
                _module_timing_scope(
                    root / "module-spans", identity=identity, boot_id=timing.boot_id
                ) as recorder,
                _module_span("worker_supervision", role="orchestration"),
            ):

                def event(value: dict[str, Any]) -> None:
                    recorder.observe()
                    progress(dict(value, stage="supply"))

                code = supervise_phases(
                    [
                        sys.executable,
                        "-m",
                        "nurse_rcl.v2.live.dev_control",
                        "--supply-worker",
                        "--batch",
                        str(path),
                        "--batch-sha256",
                        pin,
                        "--index",
                        str(row["index"]),
                        "--module-parent",
                        str(recorder.current_span_id),
                    ],
                    log_path=root / "supply.log",
                    timing=timing,
                    progress=event,
                )
            if code == 124 and timing.clock.used_seconds["preparation"] >= 3600:
                raise DevPreparationTimeout("preparation_timeout")
            if code != 0:
                raise DevQueuePaused("Dev supply did not complete; retained for diagnosis")
        supplied = read_json(root / "supplied.json")
        bp, bpin = Path(supplied["plan_path"]), supplied["plan_sha256"]
        b0 = execute_branch(bp, bpin, branch="B0", case_root=root, timing=timing, progress=progress)
        if b0["status"] == "failed":
            return finish_pair(
                dict(
                    dataset=row["dataset"],
                    opaque_case_id=row["opaque_case_id"],
                    B0=b0["reference"],
                    B2=None,
                )
            )
        base = _metadata(_pinned(bp, bpin))
        out = root / "b2"
        out.mkdir(mode=0o700, exist_ok=True)
        plan = dict(
            schema_version=B2_SCHEMA,
            code_root=batch["code_root"],
            code_files=batch["code_files"],
            python_executable=batch["python_executable"],
            configuration=batch["B2_configuration"],
            baseline_plan_path=str(bp),
            baseline_plan_sha256=bpin,
            baseline_result_path=b0["reference"]["result_path"],
            baseline_result_sha256=b0["reference"]["result_sha256"],
            output_root=str(out),
            case=base["case"],
        )
        publish_json(out / "plan.json", plan)
        b2 = execute_branch(
            out / "plan.json",
            file_sha256(out / "plan.json"),
            branch="B2",
            case_root=root,
            timing=timing,
            progress=progress,
        )
        pair = dict(
            dataset=row["dataset"],
            opaque_case_id=row["opaque_case_id"],
            B0=b0["reference"],
            B2=b2["reference"],
        )
        return finish_pair(pair)
    except DevPreparationTimeout:
        branch = "B0" if b0 is None else "B2"
        extra = (
            {}
            if b0 is None
            else dict(
                baseline_plan_sha256=b0["reference"]["plan_sha256"],
                baseline_result_sha256=b0["reference"]["result_sha256"],
            )
        )
        case = {k: row[k] for k in ("dataset", "opaque_case_id")} if b0 is None else base["case"]
        failure_plan = dict(
            schema_version=ASSIGNED_FAILURE,
            batch_path=str(path),
            batch_sha256=pin,
            index=row["index"],
            case=case,
            branch=branch,
            output_root=str(root),
            **extra,
        )
        fp = root / (branch.lower() + "-failure-plan.json")
        publish_json(fp, failure_plan)
        failed = root / (branch.lower() + "-failure.json")
        publish_json(
            failed,
            dict(
                schema_version="nurse.v2.dev-assigned-failure/1",
                status="failed",
                plan_sha256=file_sha256(fp),
                reason="preparation_timeout",
                timing=TimingJournal._inspect_expired(
                    root / "timing",
                    identity=identity,
                    boot_id=timing.boot_id,
                    expected_policy=DEV_TIMING_POLICY,
                ),
            ),
        )
        reference = dict(
            plan_path=str(fp),
            plan_sha256=file_sha256(fp),
            result_path=str(failed),
            result_sha256=file_sha256(failed),
        )
        return finish_pair(
            dict(
                dataset=row["dataset"],
                opaque_case_id=row["opaque_case_id"],
                B0=reference if b0 is None else b0["reference"],
                B2=None if b0 is None else reference,
            ),
            expired=True,
        )
    except Exception:
        # A live parent can close a known interval; an abruptly lost parent cannot.
        if timing.clock.active and timing.clock.account == "method":
            timing.pause(now=monotonic())
        raise


def run_batch(
    path: Path,
    pin: str,
    *,
    approved_batch_sha256: str | None,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if approved_batch_sha256 != pin:
        raise ValueError("Dev launch requires approval for this exact frozen batch")
    batch = read_batch(path, pin)
    out = path.parent
    with (out / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "approval.json", dict(batch_sha256=pin))
        if (out / "result.json").exists():
            result = read_json(out / "result.json")
            if result.get("batch_sha256") != pin:
                raise ValueError("Dev completed batch identity differs")
            verify_dev_seal(Path(result["seal_path"]), result["seal_sha256"])
            return result
        pairs = []
        for row in batch["rows"]:

            def event(value: dict[str, Any], row: dict[str, Any] = row) -> None:
                progress(
                    dict(
                        value,
                        case_index=row["index"],
                        total_cases=len(batch["rows"]),
                        dataset=row["dataset"],
                        opaque_case_id=row["opaque_case_id"],
                    )
                )

            event(dict(event="dev_case_started"))
            if row["imported_pair"] is not None:
                pair = row["imported_pair"]
                read_pair(pair)
            else:
                pair = _execute_case(batch, row, path, pin, event)
            pairs.append(pair)
            event(
                dict(
                    event="dev_case_completed",
                    completed=len(pairs),
                    imported=row["imported_pair"] is not None,
                )
            )
        seal_path = out / "predictions.json"
        settings = batch["settings"]
        seal_pin = seal_dev_pairs(
            split_path=Path(settings["split_path"]),
            split_sha256=settings["split_sha256"],
            pairs=pairs,
            output=seal_path,
        )
        result = dict(
            schema_version="nurse.v2.dev-paired-completion/1",
            status="completed",
            batch_sha256=pin,
            case_count=len(pairs),
            seal_path=str(seal_path),
            seal_sha256=seal_pin,
        )
        publish_json(out / "result.json", result)
        progress(dict(event="dev_batch_completed", **result))
        return result


def batch_status(path: Path, pin: str) -> dict[str, Any]:
    """Read small operator records only; never prepare, evaluate, or reserve work."""
    batch = read_batch(path, pin)
    completed = 0
    imported = 0
    current = None
    for row in batch["rows"]:
        root = _case_root(batch, row)
        if row["imported_pair"] is not None:
            imported += 1
            continue
        if (root / "pair.json").exists():
            retained = read_json(root / "pair.json")
            if retained.get("identity") != _case_identity(pin, row):
                raise ValueError("Dev status retained identity differs")
            completed += 1
        elif current is None:
            journals = sorted((root / "timing").glob("event-*.json"))
            current = dict(
                index=row["index"],
                dataset=row["dataset"],
                opaque_case_id=row["opaque_case_id"],
                last_timing_event=read_json(journals[-1]) if journals else None,
                method_warning=bool(list(root.glob("method-warning-*.json"))),
            )
    state = (
        "incomplete_requires_log_check"
        if (path.parent / "approval.json").exists()
        else "awaiting_approval"
    )
    if (path.parent / "result.json").exists():
        result = read_json(path.parent / "result.json")
        if result.get("batch_sha256") != pin:
            raise ValueError("Dev status completion identity differs")
        state = "completed"
    return dict(
        state=state,
        batch_sha256=pin,
        completed=completed + imported,
        newly_completed=completed,
        imported=imported,
        total=len(batch["rows"]),
        current=current,
        verification="small_operator_records_not_full_seal",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supply-worker", action="store_true", required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--batch-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--module-parent", required=True)
    parser.add_argument("--timing-fd", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))
    client = PhaseClient(args.timing_fd)
    try:
        batch = read_batch(args.batch, args.batch_sha256)
        if not 1 <= args.index <= len(batch["rows"]):
            raise ValueError("Dev supply index differs")
        row = batch["rows"][args.index - 1]
        if row["imported_pair"] is not None:
            raise ValueError("Dev imported case must not be provisioned again")
        root = _case_root(batch, row)
        with (
            _use_phase_client(client),
            _module_timing_scope(
                root / "module-spans",
                identity=_case_identity(args.batch_sha256, row),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                parent_id=args.module_parent,
            ),
            _module_span("case_worker"),
            _phase("preparation", "raw_input_prepare"),
            _module_span("preparation_sources"),
        ):
            _supply_and_freeze(batch, row)
        client.finish()
        return 0
    except Exception as error:
        print(
            json.dumps(dict(event="dev_supply_failed", error_type=type(error).__name__)), flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
