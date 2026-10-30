"""One-case, pinned B0-to-B2 execution; no raw rebuild and no full audit mode."""

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

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalEvidence
from nurse_rcl.v2.live.normal_b0_case import B0_RESULT, inspect_b0_result
from nurse_rcl.v2.live.normal_b2_report import NormalB2ReportRun, run_normal_b2_report
from nurse_rcl.v2.live.normal_b2_report_prompt import B2_REPORT_EXCHANGE_VERSION
from nurse_rcl.v2.live.normal_b2_result import NormalB2Evidence, score_normal_b2
from nurse_rcl.v2.live.normal_case import _identity as _base_identity
from nurse_rcl.v2.live.normal_case import _save_json
from nurse_rcl.v2.live.normal_execution import B0_SCHEMA as B0_PLAN_SCHEMA
from nurse_rcl.v2.live.normal_execution import _internal as _base_internal
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.normal_preparation import _require_compatible_key, load_normal_preparation
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.normal_semantic_disabled_exchange import DISABLED_SEMANTIC_EXCHANGE_VERSION
from nurse_rcl.v2.live.normal_semantic_indexed_exchange import INDEXED_SEMANTIC_EXCHANGE_VERSION
from nurse_rcl.v2.live.normal_semantic_owner import run_normal_hybrid_semantics
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
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_preparation_codec import member_digest
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_stage import RepairCaseInput
from nurse_rcl.v2.live.repair_timing import (
    DIAGNOSTIC_TIMING_POLICY,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

SCHEMA = "nurse.v2.normal-b2-single-plan/1"
_FIELDS = {
    "schema_version", "code_root", "code_files", "python_executable", "configuration",
    "baseline_plan_path", "baseline_plan_sha256", "baseline_result_path",
    "baseline_result_sha256", "output_root", "case",
}
_TERMINALS = {"completed", "completed_partial_report", "completed_fallback"}


def configuration(
    *, semantic_protocol: str = DISABLED_SEMANTIC_EXCHANGE_VERSION,
) -> dict[str, Any]:
    if semantic_protocol not in (
        DISABLED_SEMANTIC_EXCHANGE_VERSION, INDEXED_SEMANTIC_EXCHANGE_VERSION,
    ):
        raise ValueError("unregistered B2 semantic configuration")
    return dict(
        profile="v2-live-hybrid-normal", semantic_scope="query_priorities_only",
        semantic_protocol=semantic_protocol,
        report_protocol=B2_REPORT_EXCHANGE_VERSION,
        credential_variable="ARK_API_KEY2", endpoint="https://ark.cn-beijing.volces.com/api/v3",
        model=ARK_MODEL, llm_in_flight=1, worker_count=1, max_request_attempts=5,
        branch_tool_budget=8, paired_physical_union_limit=16,
        semantic_requests=2, report_requests=1, preparation_case_budget_seconds=3600,
        method_case_budget_seconds=None, timing_policy=DIAGNOSTIC_TIMING_POLICY,
        reuse="pinned_complete_preparation_and_B0_only", full_recompute="not_performed",
    )


def _read_plan(path: Path, pin: str) -> dict[str, Any]:
    plan = _metadata(_pinned(path, pin))
    if set(plan) != _FIELDS or plan["schema_version"] != SCHEMA or (
        plan["python_executable"] != sys.executable
        or type(plan["configuration"]) is not dict
        or canonical_json_bytes(plan["configuration"]) != canonical_json_bytes(configuration(
            semantic_protocol=plan["configuration"].get("semantic_protocol", ""),
        ))
    ):
        raise ValueError("B2 plan/configuration/interpreter differs")
    case = plan["case"]
    if type(case) is not dict or set(case) != {
        "dataset", "opaque_case_id", "input_path", "input_sha256",
    }:
        raise ValueError("B2 plan requires one complete case")
    if not Path(__file__).resolve().is_relative_to(require_plain_path(Path(plan["code_root"]))):
        raise ValueError("B2 implementation is outside frozen code root")
    require_plain_path(Path(plan["output_root"]))
    return plan


def _identity(plan: dict[str, Any], pin: str) -> dict[str, Any]:
    return dict(
        profile="v2-live-hybrid-normal", execution_sha256=pin,
        code_sha256=sha256_hex(canonical_json_bytes(plan["code_files"])),
        baseline_plan_sha256=plan["baseline_plan_sha256"],
        baseline_result_sha256=plan["baseline_result_sha256"],
        input_sha256=plan["case"]["input_sha256"], opaque_case_id=plan["case"]["opaque_case_id"],
    )


def _entry(files: list[dict[str, Any]], name: str) -> dict[str, Any]:
    rows = [r for r in files if r.get("name") == name]
    if len(rows) != 1 or set(rows[0]) != {"name", "size_bytes", "sha256"} or (
        type(rows[0]["size_bytes"]) is not int
        or not 0 <= rows[0]["size_bytes"] <= REPAIR_MAX_JSON_BYTES
    ):
        raise ValueError("B2 pinned artifact member is missing or invalid")
    return rows[0]


def _member(root: Path, entry: dict[str, Any]) -> bytes:
    name = entry["name"]
    if type(name) is not str or Path(name).is_absolute() or "\\" in name or (
        any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("B2 artifact member path is unsafe")
    raw = _pinned(root / name, entry["sha256"], entry["size_bytes"])
    if len(raw) != entry["size_bytes"]:
        raise ValueError("B2 artifact member size differs")
    return raw


def load_execution(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Read the old run as pinned input, never resume it under changed source code."""
    plan = _read_plan(path, expected_sha256)
    require_code(Path(plan["code_root"]), plan["code_files"])
    old = _metadata(_pinned(Path(plan["baseline_plan_path"]), plan["baseline_plan_sha256"]))
    pure = old.get("schema_version") == B0_PLAN_SCHEMA
    if old.get("schema_version") not in {"nurse.v2.normal-rcl-single-plan/1", B0_PLAN_SCHEMA} or (
        old.get("case") != plan["case"]
    ):
        raise ValueError("B2 baseline plan has another case or schema")
    base_root = require_plain_path(Path(old["output_root"]) / plan["case"]["opaque_case_id"])
    if Path(plan["baseline_result_path"]) != base_root / "result.json":
        raise ValueError("B2 baseline result is outside its original run")
    result = _metadata(_pinned(base_root / "result.json", plan["baseline_result_sha256"]))
    expected_base = _base_identity(_base_internal(old), plan["baseline_plan_sha256"])
    if result.get("schema_version") != (B0_RESULT if pure else "nurse.v2.normal-rcl-result/1") or (
        result.get("identity") != expected_base
        or result.get("status") not in (
            {"completed"} if pure else {"completed", "completed_partial_report"}
        )
    ):
        raise ValueError("B2 baseline is not a completed bound normal result")
    if pure:
        inspect_b0_result(base_root, expected_base)
    preparation = RepairPreparationRun.model_validate_json(canonical_json_bytes(old["preparation"]))
    if Path(plan["output_root"]).is_relative_to(Path(old["output_root"])) or (
        Path(plan["output_root"]).is_relative_to(Path(preparation.archive_root))
    ):
        raise ValueError("B2 cannot write into its saved baseline or preparation")
    if preparation.mode == "build":
        entry = _entry(result["files"], "preparation-receipt.json")
        receipt = _metadata(_member(base_root, entry))
        if receipt.get("execution_sha256") != plan["baseline_plan_sha256"] or (
            receipt.get("execution_code_sha256") != expected_base["code_sha256"]
        ):
            raise ValueError("B2 saved build receipt has another owner")
        preparation = preparation.model_copy(update=dict(
            mode="load", receipt_path=str(base_root / entry["name"]),
            receipt_sha256=entry["sha256"], manifest_sha256=receipt["manifest_sha256"],
        ))
    base = NormalEvidence.model_validate_json(
        _member(base_root, _entry(result["files"], "evidence.json")),
    )
    manifest = InferenceIdentityManifest.model_validate_json(_pinned(
        Path(old["identity_manifest_path"]), old["identity_artifact_sha256"],
    ))
    case = plan["case"]
    spec = RepairCaseInput.model_validate_json(
        _pinned(Path(case["input_path"]), case["input_sha256"]),
    )
    if not manifest.verify() or not any(
        (r.dataset, r.opaque_case_id) == (spec.dataset, spec.opaque_case_id)
        for r in manifest.records
    ) or (spec.dataset, spec.opaque_case_id) != (case["dataset"], case["opaque_case_id"]) or (
        (base.dataset, base.opaque_case_id) != (spec.dataset, spec.opaque_case_id)
        or base.result_sha256 != result["evidence_sha256"]
        or base.ranking.result_sha256 != result["ranking_sha256"]
        or preparation.key.input_sha256 != case["input_sha256"]
        or preparation.key.source_files != spec.source_files
        or base.source_descriptor_sha256 != spec.source_descriptor_sha256
    ):
        raise ValueError("B2 independent case/source/preparation identity differs")
    extra: dict[str, Any] = {}
    adapter: Any
    if spec.dataset == "rcabench":
        adapter = build_rcabench_adapter_config(window_seconds=240)
        if spec.stage_sha256 is None or spec.source_case_id is not None or (
            spec.authorized_queue is not None
        ):
            raise ValueError("B2 RCABench requires opaque staged input")
    else:
        adapter = build_aiops25_adapter_config(window_seconds=480)
        if spec.stage_sha256 is not None or spec.source_case_id is None or (
            spec.authorized_queue is None
        ):
            raise ValueError("B2 AIOps25 requires a complete authorized case")
        extra = dict(source_case_id=spec.source_case_id, authorized_queue=spec.authorized_queue)
    if adapter.configuration_sha256 != spec.configuration_sha256:
        raise ValueError("B2 adapter configuration differs")
    raw = dict(dataset=spec.dataset, opaque_case_id=spec.opaque_case_id, configuration=adapter,
               expected_source_descriptor_sha256=spec.source_descriptor_sha256, **extra)
    _require_compatible_key(preparation.key, raw)
    return dict(
        state_root=Path(plan["output_root"]) / spec.opaque_case_id,
        identity=_identity(plan, expected_sha256), preparation=preparation, raw_arguments=raw,
        identity_manifest=manifest, base_evidence=base,
        baseline_result_file_sha256=plan["baseline_result_sha256"],
        code_sha256=sha256_hex(canonical_json_bytes(plan["code_files"])),
        semantic_protocol=plan["configuration"]["semantic_protocol"],
    )


def _body_summary(name: str, value: dict[str, Any]) -> dict[str, Any]:
    if name == "evidence":
        return dict(
            evidence_sha256=value["result_sha256"],
            ranking_sha256=value["ranking"]["result_sha256"],
            candidate_count=len(value["ranking"]["scores"]),
            **{k: value[k] for k in (
                "semantic_result_sha256", "semantic_status", "baseline_fallback",
                "logical_tools_used", "card_count", "verification_count", "reused_count",
                "executed_count", "physical_union_count",
            )},
        )
    if name == "ranking":
        return dict(evidence_sha256=value["source_evidence_sha256"],
                    ranking_sha256=value["source_ranking_sha256"],
                    candidate_count=len(value["scores"]))
    return dict(evidence_sha256=value["source_evidence_sha256"],
                ranking_sha256=value["source_ranking_sha256"],
                report_sha256=value["result_sha256"], report_status=value["status"])


def _save_body(root: Path, name: str, value: dict[str, Any], identity: dict[str, Any]) -> None:
    member = _save_json(root / (name + ".json"), value)
    publish_json(root / (name + "-receipt.json"), dict(
        identity=identity, member=member, summary=_body_summary(name, value),
    ))


def _load_body(root: Path, name: str, identity: dict[str, Any]) -> bytes:
    receipt = _metadata(_read_bytes(root / (name + "-receipt.json"), 16384))
    if set(receipt) != {"identity", "member", "summary"} or receipt["identity"] != identity:
        raise ValueError("B2 saved body owner differs")
    return _member(root, _entry([receipt["member"]], name + ".json"))


def _files(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if not path.is_file() or name.split("/")[0] in {"timing", "module-spans"} or (
            path.name in {"result.json", "failure.json", "module-timing.json"}
            or (path.suffix != ".json" and path.name != "head")
        ):
            continue
        size, pin = member_digest(path)
        entries.append(dict(name=name, size_bytes=size, sha256=pin))
    # Path ordering compares components; the persisted contract compares POSIX names.
    return sorted(entries, key=lambda row: row["name"])


def inspect_b2_result(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    result = _metadata(_read_bytes(root / "result.json", 131072))
    if result.get("schema_version") != "nurse.v2.normal-b2-result/1" or (
        result.get("identity") != identity or result.get("status") not in _TERMINALS
        or any(result.get(k) != v for k, v in NormalAssurance().model_dump().items())
    ):
        raise ValueError("B2 completion identity/status differs")
    files = result.get("files")
    if type(files) is not list or not 1 <= len(files) <= 128:
        raise ValueError("B2 completion file domain differs")
    names: list[str] = [r["name"] for r in files
                       if type(r) is dict and type(r.get("name")) is str]
    if len(names) != len(files) or (
        names != sorted(set(names)) or not {
            "identity.json", "semantic-identity.json",
            "semantic-first.json", "semantic-result.json",
            "evidence.json", "evidence-receipt.json", "ranking.json", "ranking-receipt.json",
            "report.json", "report-receipt.json",
        } <= set(names)
    ):
        raise ValueError("B2 completion is missing required method artifacts")
    for row in files:
        _entry([row], row["name"])
        name = row["name"]
        if Path(name).is_absolute() or "\\" in name or any(
            p in ("", ".", "..") for p in name.split("/")
        ) or member_digest(root / name) != (row["size_bytes"], row["sha256"]):
            raise ValueError("B2 completion member integrity differs")
    for name in ("evidence", "ranking", "report"):
        receipt = _metadata(_read_bytes(root / (name + "-receipt.json"), 16384))
        if set(receipt) != {"identity", "member", "summary"} or (
            receipt["identity"] != identity
            or receipt["member"] != _entry(files, name + ".json")
        ):
            raise ValueError("B2 completion receipt binding differs")
        summary = receipt["summary"]
        if type(summary) is not dict or not summary or canonical_json_bytes(summary) != (
            canonical_json_bytes({k: result.get(k) for k in summary})
        ):
            raise ValueError("B2 completion summary differs from saved body receipts")
    return result


def run_b2_case(*, plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    root = require_plain_path(plan["state_root"])
    identity = plan["identity"]
    if identity["execution_sha256"] != plan_sha256 or root.name != identity["opaque_case_id"]:
        raise ValueError("B2 worker identity differs")
    lock = os.open(root / "case.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    started, stage = monotonic(), "admission"
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(root / "identity.json", identity)
        if (root / "failure.json").exists():
            raise ValueError("B2 case has a retained terminal failure")
        if (root / "result.json").exists():
            return inspect_b2_result(root, identity)
        try:
            preparation = plan["preparation"]
            base = plan["base_evidence"]
            stage, prep_started = "preparation", monotonic()
            with _phase("preparation", "raw_input_prepare"), _module_span("preparation_load"):
                data = load_normal_preparation(
                    archive_root=Path(preparation.archive_root), expected_key=preparation.key,
                    raw_arguments=plan["raw_arguments"],
                    receipt_path=Path(preparation.receipt_path),
                    expected_manifest_sha256=preparation.manifest_sha256,
                    expected_receipt_sha256=preparation.receipt_sha256,
                )
            prep_seconds = monotonic() - prep_started
            print(json.dumps(dict(event="b2_preparation_loaded", opaque_case_id=root.name,
                                  reused=True, seconds=prep_seconds)), flush=True)
            stage = "semantics"
            semantic = run_normal_hybrid_semantics(
                data=data, block_root=Path(preparation.archive_root) / "blocks",
                paired_base_session=base.session,
                expected_base_session_sha256=base.session_sha256,
                expected_source_descriptor_sha256=base.source_descriptor_sha256,
                expected_catalog_sha256=base.catalog_sha256,
                identity_manifest=plan["identity_manifest"],
                expected_identity_manifest_sha256=plan["identity_manifest"].manifest_sha256,
                state_root=root, execution_sha256=plan_sha256, code_sha256=plan["code_sha256"],
                protocol=plan["semantic_protocol"],
            )
            stage = "scoring"
            if (root / "evidence.json").exists():
                evidence = NormalB2Evidence.model_validate_json(
                    _load_body(root, "evidence", identity),
                )
                if evidence.semantic_result_sha256 != semantic.result_sha256 or (
                    evidence.paired_base_evidence_sha256 != base.result_sha256
                    or evidence.catalog_sha256 != data.catalog.catalog_sha256
                ):
                    raise ValueError("B2 saved evidence lineage differs")
            else:
                evidence = score_normal_b2(
                    data=data, semantic_run=semantic, paired_base_evidence=base,
                    expected_semantic_result_sha256=semantic.result_sha256,
                    expected_base_evidence_sha256=base.result_sha256,
                )
                _save_body(root, "evidence", evidence.model_dump(mode="json"), identity)
            ranking = evidence.ranking
            if (root / "ranking.json").exists():
                _load_body(root, "ranking", identity)
            else:
                _save_body(root, "ranking", dict(
                    schema_version="nurse.v2.normal-b2-ranking/1",
                    source_evidence_sha256=evidence.result_sha256,
                    source_ranking_sha256=ranking.result_sha256,
                    candidate_service_keys=ranking.candidate_service_keys,
                    ordered_service_keys=ranking.ordered_service_keys,
                    top_k_service_keys=ranking.top_k_service_keys,
                    scores=[s.model_dump(mode="json") for s in ranking.scores],
                    tie_groups=[t.model_dump(mode="json") for t in ranking.tie_groups],
                ), identity)
            stage = "report"
            if (root / "report.json").exists():
                report = NormalB2ReportRun.model_validate_json(_load_body(root, "report", identity))
                if report.source_evidence_sha256 != evidence.result_sha256:
                    raise ValueError("B2 saved report lineage differs")
            else:
                with _module_span("report"):
                    report = run_normal_b2_report(
                        evidence=evidence, sources=data.sources,
                        identity_manifest=plan["identity_manifest"],
                        expected_identity_manifest_sha256=plan["identity_manifest"].manifest_sha256,
                        state_root=root, execution_sha256=plan_sha256,
                        code_sha256=plan["code_sha256"],
                    )
                _save_body(root, "report", report.model_dump(mode="json"), identity)
            stage = "publication"
            delivered = (
                report.status in {"full", "partial"} and report.presentation is not None
                and bool(report.presentation.safe_display_blocks)
                and report.raw_response_hex is not None
            )
            status = ("completed_fallback" if not delivered or evidence.baseline_fallback else
                      "completed" if report.status == "full" else "completed_partial_report")
            elapsed = monotonic() - started
            result = dict(
                schema_version="nurse.v2.normal-b2-result/1", identity=identity,
                **NormalAssurance().model_dump(), status=status, files=_files(root),
                candidate_count=len(ranking.scores), evidence_sha256=evidence.result_sha256,
                ranking_sha256=ranking.result_sha256, semantic_status=semantic.status,
                semantic_result_sha256=semantic.result_sha256, report_status=report.status,
                report_sha256=report.result_sha256, baseline_fallback=evidence.baseline_fallback,
                logical_tools_used=evidence.logical_tools_used, card_count=evidence.card_count,
                verification_count=evidence.verification_count,
                reused_count=evidence.reused_count, executed_count=evidence.executed_count,
                physical_union_count=evidence.physical_union_count,
                timing=dict(preparation_seconds=prep_seconds, method_seconds=elapsed - prep_seconds,
                            elapsed_seconds=elapsed,
                            scope="this_B2_invocation_incremental_not_B0_cost"),
            )
            _save_json(root / "result.json", result)
            return result
        except Exception as exc:
            from nurse_rcl.v2.exchange_json import ExchangeBoundaryError

            failure = dict(schema_version="nurse.v2.normal-b2-failure/1", identity=identity,
                           status="failed", stage=stage, error_type=type(exc).__name__,
                           elapsed_seconds=monotonic() - started)
            if type(exc) is ExchangeBoundaryError and exc.code == "json_limit":
                failure["capacity_reason"] = "json_limit"
            publish_json(root / "failure.json", failure)
            return failure
    finally:
        os.close(lock)


def run_round(
    path: Path, pin: str, *, progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    plan = _read_plan(path, pin)
    out, identity = Path(plan["output_root"]), _identity(plan, pin)
    state = out / plan["case"]["opaque_case_id"]
    state.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(state)
    lock = os.open(out / "round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "identity.json", identity)
        if (out / "terminal.json").exists():
            retained = read_json(out / "terminal.json")
            if retained.get("identity") != identity:
                raise ValueError("B2 retained terminal identity differs")
            if retained["status"] in _TERMINALS:
                inspect_b2_result(state, identity)
            return dict(retained, timing=read_json(out / "timing-summary.json"))
        timing = TimingJournal.open(
            state / "timing", identity=pin, now=monotonic(),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            expected_policy=DIAGNOSTIC_TIMING_POLICY,
        )
        terminal: dict[str, Any] = dict(identity=identity, status="failed", worker_exit_code=None)
        with _module_timing_scope(state / "module-spans", identity=pin,
                                  boot_id=timing.boot_id) as recorder:
            def event(row: dict[str, Any]) -> None:
                recorder.observe()
                progress(dict(row, case_index=1, total_cases=1, opaque_case_id=state.name))

            try:
                with _module_span("worker_supervision", role="orchestration"):
                    code = supervise_phases(
                        [sys.executable, "-m", "nurse_rcl.v2.live.normal_b2_execution",
                         "--plan", str(path), "--plan-sha256", pin, "--worker",
                         "--module-parent", str(recorder.current_span_id)],
                        log_path=state / "case.log", timing=timing, progress=event,
                    )
                terminal["worker_exit_code"] = code
                with _module_span("parent_finalize", role="verification"):
                    if code != 0:
                        raise ValueError("B2 worker did not finish")
                    result = inspect_b2_result(state, identity)
                    terminal.update(status=result["status"],
                                    semantic_status=result["semantic_status"],
                                    report_status=result["report_status"],
                                    candidate_count=result["candidate_count"])
            except Exception as exc:
                terminal.update(status="failed", error_type=type(exc).__name__)
        timing.checkpoint(now=monotonic())
        publish_json(state / "module-timing.json", summarize_module_timing(
            event_paths=tuple(sorted((state / "module-spans").glob("*.jsonl"))),
            timing_root=timing.root, expected_timing_identity=pin,
            expected_policy=DIAGNOSTIC_TIMING_POLICY,
        ))
        publish_json(out / "terminal.json", terminal)
        timing.checkpoint(now=monotonic())
        summary = timing.clock.summary()
        publish_json(out / "timing-summary.json", summary)
        progress(dict(event="b2_terminal", **terminal, timing=summary))
        return dict(terminal, timing=summary)
    finally:
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timing-fd", type=int)
    parser.add_argument("--module-parent")
    args = parser.parse_args()
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))

    def emit(row: dict[str, Any]) -> None:
        print(json.dumps(row), flush=True)

    try:
        if args.worker != (args.timing_fd is not None and args.module_parent is not None):
            raise ValueError("B2 worker requires private parent channels")
        if not args.worker:
            result = run_round(args.plan, args.plan_sha256, progress=emit)
        else:
            envelope = _read_plan(args.plan, args.plan_sha256)
            state = Path(envelope["output_root"]) / envelope["case"]["opaque_case_id"]
            client = PhaseClient(args.timing_fd)
            with _use_phase_client(client), _module_timing_scope(
                state / "module-spans", identity=args.plan_sha256,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                parent_id=args.module_parent,
            ), _module_span("case_worker"):
                with _phase("preparation", "input_validation"), _module_span("input_guard"):
                    plan = load_execution(args.plan, args.plan_sha256)
                result = run_b2_case(plan=plan, plan_sha256=args.plan_sha256)
            client.finish()
        return int(result["status"] not in _TERMINALS)
    except KeyboardInterrupt:
        emit(dict(event="b2_interrupted"))
        return 130
    except Exception as exc:
        emit(dict(event="b2_failed", error_type=type(exc).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
