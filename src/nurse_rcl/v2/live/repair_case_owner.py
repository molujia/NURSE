"""One raw lifetime through ranking, independent replay, report and publication.

The private body is only called after the strict plan loader in the worker.
Its returned closure is data, not parent/timing success or reusable authority.
"""

from __future__ import annotations

import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live import repair_query_session as queries
from nurse_rcl.v2.live import repair_raw_audit as raw
from nurse_rcl.v2.live.repair_case_artifacts import (
    REPAIR_CASE_POLICY,
    RepairCaseAudit,
    RepairCaseClosure,
)
from nurse_rcl.v2.live.repair_case_guards import (
    _case_inputs_scope,
    _CaseInputs,
    _prepared_case_inputs_scope,
)
from nurse_rcl.v2.live.repair_case_replay import _build_owned_case_audit, _replay_case_evidence
from nurse_rcl.v2.live.repair_case_report import (
    RepairCaseReportRun,
    _run_owned_report,
)
from nurse_rcl.v2.live.repair_case_report import (
    _snapshot as report_snapshot,
)
from nurse_rcl.v2.live.repair_case_report import (
    _unchanged as report_unchanged,
)
from nurse_rcl.v2.live.repair_driver import (
    publish_bundle,
    publish_json,
    read_bundle,
    read_json,
    report_binding,
)
from nurse_rcl.v2.live.repair_execution import require_code
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_archive import _require_preparation_files
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationBinding,
    RepairPreparationRun,
)
from nurse_rcl.v2.live.repair_prepared_case_artifacts import (
    REPAIR_PREPARED_POLICY,
    RepairPreparedCaseAudit,
    RepairPreparedCaseClosure,
    RepairPreparedReportRun,
)
from nurse_rcl.v2.live.repair_queries import _seal
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_query_session import _run_owned_queries
from nurse_rcl.v2.live.repair_report_projection import _seal as _report_seal
from nurse_rcl.v2.live.repair_safety import canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_stage import RepairCaseInput, validate_input
from nurse_rcl.v2.live.repair_timing import (
    DIAGNOSTIC_TIMING_POLICY,
    TIMING_POLICY,
    TimingJournal,
    _preparation_call,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import read_inference_identity, require_plain_path


def run_repair_case_from_raw(
    *, plan_path: Path, expected_plan_sha256: str, case_index: int
) -> RepairCaseClosure:
    """Complete pinned case entry, only inside the worker's active timing owner."""
    from nurse_rcl.v2.live.repair_case_execution import (
        active_case_indexes,
        entry_file_pins,
        load_execution,
    )
    from nurse_rcl.v2.live.repair_timing import _CLIENT

    if _CLIENT.get() is None:
        raise ValueError("case entry requires the private worker timing owner")
    plan = load_execution(plan_path, expected_plan_sha256)
    if type(case_index) is not int or case_index not in active_case_indexes(plan):
        raise ValueError("case entry is outside the frozen active queue")
    case = plan["cases"][case_index]
    state = require_plain_path(Path(plan["output_root"]) / case["opaque_case_id"])
    if plan["schema_version"] != "nurse.v2.live-repair-case-owner-single-plan/1":
        raise ValueError("raw case entry requires the old exact plan policy")
    return cast(
        RepairCaseClosure,
        _run_owned_case_body(
            specification=read_json(Path(case["input_path"])),
            expected_input_sha256=case["input_sha256"],
            identity_manifest_path=Path(plan["identity_manifest_path"]),
            expected_identity_artifact_sha256=plan["identity_artifact_sha256"],
            state_root=state,
            execution_sha256=expected_plan_sha256,
            code_root=Path(plan["code_root"]),
            code_files=plan["code_files"],
            timing_identity_sha256=sha256_hex(canonical_json_bytes([expected_plan_sha256, case])),
            frozen_entry_files=entry_file_pins(plan, plan_path, expected_plan_sha256),
        ),
    )


def run_repair_case_from_preparation(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    case_index: int,
) -> RepairPreparedCaseClosure:
    from nurse_rcl.v2.live.repair_case_execution import (
        active_case_indexes,
        entry_file_pins,
        load_execution,
    )
    from nurse_rcl.v2.live.repair_timing import _CLIENT

    if _CLIENT.get() is None:
        raise ValueError("prepared case entry requires the private worker timing owner")
    plan = load_execution(plan_path, expected_plan_sha256)
    if plan["schema_version"] != "nurse.v2.live-repair-prepared-single-plan/1":
        raise ValueError("prepared case entry requires the exact preparation plan")
    if type(case_index) is not int or case_index not in active_case_indexes(plan):
        raise ValueError("prepared case entry is outside the frozen active queue")
    case = plan["cases"][case_index]
    state = require_plain_path(Path(plan["output_root"]) / case["opaque_case_id"])
    return _run_prepared_case_body(
        preparation=RepairPreparationRun.model_validate_json(
            canonical_json_bytes(plan["preparation"])
        ),
        specification=read_json(Path(case["input_path"])),
        expected_input_sha256=case["input_sha256"],
        identity_manifest_path=Path(plan["identity_manifest_path"]),
        expected_identity_artifact_sha256=plan["identity_artifact_sha256"],
        state_root=state,
        execution_sha256=expected_plan_sha256,
        code_root=Path(plan["code_root"]),
        code_files=plan["code_files"],
        timing_identity_sha256=sha256_hex(canonical_json_bytes([expected_plan_sha256, case])),
        frozen_entry_files=entry_file_pins(plan, plan_path, expected_plan_sha256),
    )


def _require_response_receipt(
    state: Path,
    report: RepairCaseReportRun | RepairPreparedReportRun,
    audit: RepairCaseAudit | RepairPreparedCaseAudit,
) -> None:
    from nurse_rcl.v2.live.ark_bridge import _Ledger

    report_snapshot(report)
    if (
        type(audit) is RepairPreparedCaseAudit
        and (type(report) is not RepairPreparedReportRun or report.preparation != audit.preparation)
    ) or (type(audit) is RepairCaseAudit and type(report) is not RepairCaseReportRun):
        raise ValueError("case report source policy/binding differs from audit")
    if (
        report != _report_seal(report, "result_sha256")
        or report.status not in ("full", "partial")
        or not report.exchange_attempted
        or report.exchange_status != "response_ready"
        or report.source_audit_sha256 != audit.result_sha256
        or report.source_ranking_sha256 != audit.ranking.result_sha256
        or report.presentation is None
        or not report.presentation.safe_display_blocks
        or report.request_hex is None
        or report.raw_response_hex is None
        or sha256_hex(bytes.fromhex(report.raw_response_hex)) != report.response_sha256
        or sha256_hex(bytes.fromhex(report.request_hex)) != report.request_sha256
    ):
        raise ValueError("case report closure incomplete")
    expected = report_binding(
        bytes.fromhex(report.request_hex),
        opaque_case_id=audit.opaque_case_id,
        source_sha256=audit.source_descriptor_sha256,
        audit_sha256=audit.result_sha256,
    ).model_dump(mode="json")
    if read_json(state / "request-binding.json") != expected:
        raise ValueError("case request binding differs")
    found = False
    for path in state.glob("ark-receipt-*.json"):
        receipt = read_json(path)
        if path.name != f"ark-receipt-{sha256_hex(canonical_json_bytes(receipt))[:16]}.json":
            raise ValueError("case ARK receipt pin differs")
        ledger = _Ledger.model_validate(receipt["ledger"])
        if ledger.binding.model_dump(mode="json") != expected or receipt["attempt_count"] != len(
            ledger.attempts
        ):
            raise ValueError("case ARK receipt binding differs")
        found |= any(
            attempt.response_sha256 == report.response_sha256
            and attempt.response_json is not None
            and attempt.response_json.encode() == bytes.fromhex(report.raw_response_hex)
            and attempt.record is not None
            and attempt.record.outcome == "success"
            for attempt in ledger.attempts
        )
    if not found:
        raise ValueError("case report has no actual response receipt")


def _timing_prefix(
    state: Path, identity: str, *, expected_policy: str = TIMING_POLICY
) -> TimingJournal:
    root = require_plain_path(state / "timing")
    first = read_json(root / "event-00000000.json", limit=4096)
    return TimingJournal._read_existing(
        root, identity=identity, boot_id=first["boot_id"], expected_policy=expected_policy
    )


def _retain_bundle(target: Path, files: dict[str, bytes], identity: str) -> None:
    expected = dict(
        identity=identity, files={name: sha256_hex(data) for name, data in sorted(files.items())}
    )
    if not target.exists() and not target.is_symlink():
        publish_bundle(target, files, identity=identity)
    if read_bundle(target, identity=identity) != expected:
        raise ValueError("case checkpoint differs from independently rebuilt values")


def _terminal_journal_guard(root: Path, session: queries.RepairQuerySession) -> Callable[[], None]:
    """Protect the terminal prefix through all remaining exits/publications.

    This rereads the bounded structural journal, not its semantic algorithms.
    The already held ledger lock remains owned by the surrounding case scope.
    """

    def identity() -> tuple[int, int, int, int]:
        directory, lock = root.lstat(), (root / "lock").lstat()
        if not stat.S_ISDIR(directory.st_mode) or not stat.S_ISREG(lock.st_mode):
            raise ValueError("case journal root or lock is not a plain owned binding")
        return directory.st_dev, directory.st_ino, lock.st_dev, lock.st_ino

    bound = identity()
    entries = queries._read_journal(root)
    if entries[-1].entry_sha256 != session.journal_sha256:
        raise ValueError("case journal tail differs from the terminal session")
    retained = tuple(canonical(entry, queries._ROOTS) for entry in entries)
    head = canonical_json_bytes([entries[-1].sequence, entries[-1].entry_sha256])
    names = {"lock", "head", *(f"{entry.sequence:03d}.json" for entry in entries)}

    def unchanged() -> None:
        if (
            identity() != bound
            or {path.name for path in root.iterdir()} != names
            or queries._read_bytes(root / "head", 256) != head
            or tuple(canonical(entry, queries._ROOTS) for entry in queries._read_journal(root))
            != retained
        ):
            raise ValueError("case terminal journal changed during owner publication")

    unchanged()
    return unchanged


@contextmanager
def _logged_case_inputs(
    *,
    preparation: RepairPreparationRun | None = None,
    execution_sha256: str = "",
    execution_code_sha256: str = "",
    **arguments: Any,
) -> Iterator[_CaseInputs]:
    with ExitStack() as exits:
        with _module_span("preparation_validate"):
            inputs = exits.enter_context(
                _case_inputs_scope(**arguments)
                if preparation is None
                else _prepared_case_inputs_scope(
                    preparation=preparation,
                    raw_arguments=arguments,
                    execution_sha256=execution_sha256,
                    execution_code_sha256=execution_code_sha256,
                )
            )
        try:
            yield inputs
        finally:
            error = sys.exc_info()
            # ExitStack remains a fallback if logging itself fails.
            with query_step("owner_exit"):
                exits.__exit__(*error)


def _run_owned_case_body(
    *,
    specification: dict[str, Any],
    expected_input_sha256: str,
    identity_manifest_path: Path,
    expected_identity_artifact_sha256: str,
    state_root: Path,
    execution_sha256: str,
    code_root: Path,
    code_files: dict[str, str],
    timing_identity_sha256: str,
    frozen_entry_files: tuple[tuple[str, str], ...] = (),
    _preparation: RepairPreparationRun | None = None,
) -> RepairCaseClosure | RepairPreparedCaseClosure:
    state = require_plain_path(state_root)
    if _preparation is not None and type(_preparation) is not RepairPreparationRun:
        raise ValueError("case owner preparation plan type differs")
    policy = REPAIR_CASE_POLICY if _preparation is None else REPAIR_PREPARED_POLICY
    timing_policy = TIMING_POLICY if _preparation is None else DIAGNOSTIC_TIMING_POLICY
    audit_type = RepairCaseAudit if _preparation is None else RepairPreparedCaseAudit
    namespace = state / ("case-owner-v1" if _preparation is None else "prepared-case-v1")
    namespace.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(namespace)
    failure = namespace / "failure.json"
    if failure.exists() or failure.is_symlink() or (state / "terminal.json").exists():
        raise ValueError("case has a retained terminal failure or terminal record")
    if (namespace / "closure.json").exists() or (namespace / "result").exists():
        raise ValueError("case publication is terminal or incomplete; parent verification required")
    case, manifest, arguments = validate_input(
        specification=specification,
        expected_input_sha256=expected_input_sha256,
        identity_manifest_path=identity_manifest_path,
        expected_identity_artifact_sha256=expected_identity_artifact_sha256,
        state_root=state,
    )
    if state.name != case.opaque_case_id:
        raise ValueError("case output identity differs")
    if _preparation is not None and (
        _preparation.key.input_sha256 != expected_input_sha256
        or _preparation.key.opaque_case_id != case.opaque_case_id
        or (
            _preparation.mode == "build"
            and Path(_preparation.receipt_path) != state / "preparation-receipt.json"
        )
    ):
        raise ValueError("case preparation differs from independently verified input/run")
    prepared_binding: RepairPreparationBinding | None = None
    code_pin = sha256_hex(canonical_json_bytes(code_files))
    # File pins originate at the verified entry, never from the report prompt.
    entry_files = (
        *frozen_entry_files,
        (str(identity_manifest_path), expected_identity_artifact_sha256),
    )
    if case.dataset == "rcabench":
        stage_path = require_plain_path(Path(case.source_root) / "stage.json")
        entry_files += ((str(stage_path), file_sha256(stage_path)),)
    input_before = snapshot((specification, case, code_files), (RepairCaseInput,))

    def unchanged() -> None:
        with ExitStack() as checks:
            checks.callback(require_code, code_root, code_files)
            checks.callback(
                require_unchanged,
                (specification, case, code_files),
                (RepairCaseInput,),
                input_before,
            )
            checks.callback(
                read_inference_identity, identity_manifest_path, expected_identity_artifact_sha256
            )

            def files() -> None:
                if raw._file_pins(Path(case.source_root), case.dataset) != case.source_files:
                    raise ValueError("case source changed during owner")

            checks.callback(files)

            def metadata() -> None:
                for name, pin in entry_files:
                    if file_sha256(require_plain_path(Path(name))) != pin:
                        raise ValueError("case entry identity file changed")

            checks.callback(metadata)
            if _preparation is not None and prepared_binding is not None:
                checks.callback(
                    _require_preparation_files, _preparation, prepared_binding, arguments
                )

    input_exit = _preparation_call("stage_input_exit")(unchanged)
    try:
        unchanged()
        _timing_prefix(state, timing_identity_sha256, expected_policy=timing_policy)
        publish_json(
            state / "identity.json",
            dict(
                execution_sha256=execution_sha256,
                input_sha256=expected_input_sha256,
                opaque_case_id=case.opaque_case_id,
            ),
        )
        with ExitStack() as locks:
            # Outputs remain guarded across original raw exits, while the lock
            # remains held through result and closure publication.
            with ExitStack() as output_guards:
                output_guards.callback(input_exit)
                with _logged_case_inputs(
                    **arguments,
                    preparation=_preparation,
                    execution_sha256=execution_sha256,
                    execution_code_sha256=code_pin,
                ) as inputs:
                    prepared_binding = inputs.preparation
                    if prepared_binding is not None:
                        output_guards.callback(
                            require_unchanged,
                            prepared_binding,
                            (RepairPreparationBinding,),
                            snapshot(prepared_binding, (RepairPreparationBinding,)),
                        )
                    created = locks.enter_context(inputs.prepared.runtime._locked_ledger())
                    session = _run_owned_queries(inputs, created=created)
                    journal_check = _terminal_journal_guard(inputs.prepared.runtime.root, session)
                    locks.callback(journal_check)
                    output_guards.callback(journal_check)
                    with query_step("production_scoring"):
                        audit = _build_owned_case_audit(inputs, session)
                    audit_before = snapshot(audit, (audit_type,))
                    output_guards.callback(require_unchanged, audit, (audit_type,), audit_before)
                    _retain_bundle(
                        namespace / "draft", {"audit.json": audit_before[0]}, execution_sha256
                    )
                    unchanged()
                    with ExitStack() as replay_scope:
                        with query_step("verification_replay"):
                            fresh = replay_scope.enter_context(
                                _replay_case_evidence(inputs, candidate_audit=audit)
                            )
                        unchanged()
                        with query_step("report"):
                            report = _run_owned_report(
                                inputs,
                                audit=fresh,
                                identity_manifest=manifest,
                                expected_identity_manifest_sha256=manifest.manifest_sha256,
                                state_root=state,
                                execution_sha256=execution_sha256,
                                code_root=code_root,
                                code_files=code_files,
                                entry_file_pins=entry_files,
                                entry_specification=specification,
                                expected_input_sha256=expected_input_sha256,
                            )
                        output_guards.callback(report_unchanged, report, report_snapshot(report))
                        _require_response_receipt(state, report, audit)
                    unchanged()
                # raw and fresh source exits are complete, outputs still guarded.
                files = {
                    "audit.json": canonical(audit, (audit_type,)),
                    "report.json": report_snapshot(report)[0],
                }
            unchanged()
            verification = dict(
                policy=policy,
                audit_sha256=audit.result_sha256,
                session_sha256=session.session_sha256,
                replay_scope=audit.replay_scope,
            )
            exits = dict(
                policy=policy,
                input_sha256=expected_input_sha256,
                code_sha256=code_pin,
                source_files_sha256=audit.source_files_sha256,
                audit_sha256=audit.result_sha256,
                report_sha256=report.result_sha256,
            )
            files.update(
                {
                    "verification.json": canonical_json_bytes(verification),
                    "exit-checks.json": canonical_json_bytes(exits),
                }
            )
            with ExitStack() as final_checks:
                final_checks.callback(unchanged)
                final_checks.callback(journal_check)
                final_checks.callback(_require_response_receipt, state, report, audit)
                receipt = dict(
                    identity=execution_sha256,
                    files={name: sha256_hex(data) for name, data in sorted(files.items())},
                )
                with _module_span("artifact_publish"):
                    publish_bundle(namespace / "result", files, identity=execution_sha256)
                if read_bundle(namespace / "result", identity=execution_sha256) != receipt:
                    raise ValueError("case result publication differs")
            timing = _timing_prefix(state, timing_identity_sha256, expected_policy=timing_policy)
            if report.request_sha256 is None or report.response_sha256 is None:
                raise ValueError("case report has no request or response identity")
            closure_type = RepairCaseClosure if _preparation is None else RepairPreparedCaseClosure
            binding: dict[str, Any] = (
                {} if prepared_binding is None else {"preparation": prepared_binding}
            )
            closure = _seal(
                closure_type(
                    **binding,
                    dataset=case.dataset,
                    opaque_case_id=case.opaque_case_id,
                    execution_sha256=execution_sha256,
                    code_sha256=code_pin,
                    input_sha256=expected_input_sha256,
                    source_descriptor_sha256=audit.source_descriptor_sha256,
                    source_files_sha256=audit.source_files_sha256,
                    candidate_freeze_sha256=audit.candidate_freeze_sha256,
                    catalog_sha256=audit.catalog_sha256,
                    journal_sha256=session.journal_sha256,
                    session_sha256=session.session_sha256,
                    audit_sha256=audit.result_sha256,
                    ranking_sha256=audit.ranking.result_sha256,
                    report_sha256=report.result_sha256,
                    request_sha256=report.request_sha256,
                    response_sha256=report.response_sha256,
                    verification_sha256=sha256_hex(files["verification.json"]),
                    exit_checks_sha256=sha256_hex(files["exit-checks.json"]),
                    publication_sha256=sha256_hex(canonical_json_bytes(receipt)),
                    timing_identity_sha256=timing_identity_sha256,
                    timing_checkpoint_sha256=timing.previous,
                    result_sha256="0" * 64,
                ),
                "result_sha256",
            )
            # Parent still requires finish, exit0, final clock and receipt checks.
            with _module_span("artifact_publish"):
                publish_json(namespace / "closure.json", closure.model_dump(mode="json"))
            if read_json(namespace / "closure.json") != closure.model_dump(mode="json"):
                raise ValueError("case closure publication differs")
            return closure
    except BaseException as error:
        publish_json(
            failure,
            dict(
                execution_sha256=execution_sha256, status="failed", error_type=type(error).__name__
            ),
        )
        raise


def _run_prepared_case_body(
    *,
    preparation: RepairPreparationRun,
    **arguments: Any,
) -> RepairPreparedCaseClosure:
    if type(preparation) is not RepairPreparationRun:
        raise ValueError("prepared owner requires exact pinned preparation data")
    return cast(
        RepairPreparedCaseClosure, _run_owned_case_body(_preparation=preparation, **arguments)
    )
