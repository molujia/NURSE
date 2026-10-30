"""Supervised decision-only prefixes; stage grants never reset lifetime requests."""

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
from uuid import uuid4

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_agent_state import OpsWorkingAgentResult, load_conversation
from nurse_rcl.v2.live.ops_case import _admit_display, _preparation, case_identity, plan_identity
from nurse_rcl.v2.live.ops_contracts import (
    OpsCoreIdentity,
    OpsSreDirectedProfile,
    OpsSreEntryProfile,
    OpsSreFeedbackFactProfile,
    OpsSreFocusedProfile,
    OpsSreObservableProfile,
    OpsSreSingleOwnerProfile,
    OpsSreTypedPurposeProfile,
)
from nurse_rcl.v2.live.ops_execution import _clock, _invocation_usage, _modules
from nurse_rcl.v2.live.ops_failure import (
    FailureStage,
    OpsFailureDetail,
    failure_detail,
    secondary_failures,
)
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_plan import (
    OpsAdmittedPlan,
    load_ops_plan,
    read_ops_plan,
    require_ops_code,
)
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_atomic_selector_provider import OpsSreAtomicSelectorDecisionProvider
from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import (
    OpsSreAtomicSelectorLedgerState,
    OpsSreAtomicSelectorPublicationReader,
    OpsSreAtomicSelectorToolbox,
)
from nurse_rcl.v2.live.ops_sre_core_import import CORE_MEMBERS, import_source_core
from nurse_rcl.v2.live.ops_sre_directed_provider import (
    OpsSreDirectedDecisionProvider,
    stage_grant,
)
from nurse_rcl.v2.live.ops_sre_directed_tools import (
    OpsSreDirectedLedgerState,
    OpsSreDirectedPublicationReader,
    OpsSreDirectedToolbox,
)
from nurse_rcl.v2.live.ops_sre_entry_provider import OpsSreEntryDecisionProvider
from nurse_rcl.v2.live.ops_sre_entry_tools import (
    OpsSreEntryLedgerState,
    OpsSreEntryPublicationReader,
    OpsSreEntryToolbox,
)
from nurse_rcl.v2.live.ops_sre_feedback_fact_provider import OpsSreFeedbackFactDecisionProvider
from nurse_rcl.v2.live.ops_sre_observable_provider import OpsSreObservableDecisionProvider
from nurse_rcl.v2.live.ops_sre_observable_tools import (
    OpsSreObservableLedgerState,
    OpsSreObservablePublicationReader,
    OpsSreObservableToolbox,
)
from nurse_rcl.v2.live.ops_sre_single_owner_provider import OpsSreSingleOwnerDecisionProvider
from nurse_rcl.v2.live.ops_sre_single_owner_tools import (
    OpsSreSingleOwnerLedgerState,
    OpsSreSingleOwnerPublicationReader,
    OpsSreSingleOwnerToolbox,
)
from nurse_rcl.v2.live.ops_sre_typed_purpose_provider import OpsSreTypedPurposeDecisionProvider
from nurse_rcl.v2.live.ops_sre_typed_purpose_tools import (
    OpsSreTypedPurposeLedgerState,
    OpsSreTypedPurposePublicationReader,
    OpsSreTypedPurposeToolbox,
)
from nurse_rcl.v2.live.ops_worker_outcome import read_outcome, write_outcome
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_execution import install_termination_handlers, supervise_phases
from nurse_rcl.v2.live.repair_module_timing import _module_span, _module_timing_scope
from nurse_rcl.v2.live.repair_preparation_codec import member_digest
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_timing import PhaseClient, _phase, _use_phase_client
from nurse_rcl.v2.live.staged_case import require_plain_path

STAGE_MEMBERS = (
    *CORE_MEMBERS,
    "source-core.json",
    "sre-score-lineage.json",
    "investigation/state.json",
    "investigation/conversation.json",
)


def stage_provider_type(profile: object) -> type[OpsSreDirectedDecisionProvider]:
    from nurse_rcl.v2.live.ops_contracts import OpsSreAtomicSelectorProfile

    if isinstance(profile, OpsSreFocusedProfile):
        from nurse_rcl.v2.live.ops_sre_focused_provider import OpsSreFocusedDecisionProvider

        return OpsSreFocusedDecisionProvider
    if isinstance(profile, OpsSreFeedbackFactProfile):
        return OpsSreFeedbackFactDecisionProvider
    if isinstance(profile, OpsSreAtomicSelectorProfile):
        return OpsSreAtomicSelectorDecisionProvider
    if isinstance(profile, OpsSreTypedPurposeProfile):
        return OpsSreTypedPurposeDecisionProvider
    if isinstance(profile, OpsSreSingleOwnerProfile):
        return OpsSreSingleOwnerDecisionProvider
    if isinstance(profile, OpsSreObservableProfile):
        return OpsSreObservableDecisionProvider
    if isinstance(profile, OpsSreEntryProfile):
        return OpsSreEntryDecisionProvider
    if isinstance(profile, OpsSreDirectedProfile):
        return OpsSreDirectedDecisionProvider
    raise ValueError("unregistered bounded stage profile")


def _admit_stage(
    admitted: OpsAdmittedPlan,
    *,
    stage_decision_limit: int,
    approved_plan_sha256: str | None,
    approved_stage_sha256: str | None,
) -> dict[str, Any]:
    plan = admitted.plan
    if (
        not isinstance(plan.profile, OpsSreDirectedProfile | OpsSreEntryProfile)
        or plan.mode != "with-agent"
        or plan.preparation.mode != "load"
        or plan.source_core is None
    ):
        raise ValueError("bounded stage requires directed profile and compatible prepared core")
    if type(stage_decision_limit) is not int or not 1 <= stage_decision_limit <= 8:
        raise ValueError("stage ceiling differs")
    grant = stage_grant(admitted.plan_sha256, stage_decision_limit)
    if approved_plan_sha256 != admitted.plan_sha256 or approved_stage_sha256 != sha256_hex(
        canonical_json_bytes(grant),
    ):
        raise ValueError("bounded stage requires explicit execution and stage approval")
    require_ops_code(plan)
    return grant


def _prefix_pins(root: Path, identity: dict[str, Any]) -> dict[str, list[str]]:
    focused = identity.get("profile") == "v2-ops-sre/15"
    atomic_selector = identity.get("profile") in {"v2-ops-sre/13", "v2-ops-sre/14"} or focused
    typed_purpose = identity.get("profile") == "v2-ops-sre/12"
    single_owner = identity.get("profile") == "v2-ops-sre/11"
    observable = (
        identity.get("profile") == "v2-ops-sre/10"
        or single_owner
        or typed_purpose
        or atomic_selector
    )
    entry = identity.get("profile") == "v2-ops-sre/9" or observable
    if identity.get("profile") not in {
        "v2-ops-sre/8",
        "v2-ops-sre/9",
        "v2-ops-sre/10",
        "v2-ops-sre/11",
        "v2-ops-sre/12",
        "v2-ops-sre/13",
        "v2-ops-sre/14",
        "v2-ops-sre/15",
    }:
        raise ValueError("bounded stage prefix profile differs")
    core = OpsCoreResult.model_validate_json(
        _read_bytes(root / "core/result.json", REPAIR_MAX_JSON_BYTES),
    )
    conversation = load_conversation(
        root / "investigation/conversation.json",
        snapshot=core.identity.case_snapshot_id,
        code=identity["code_sha256"],
        opaque_case_id=identity["opaque_case_id"],
        empirical=True,
        sre=True,
        sre_profile="hermes-sre-tools/25"
        if focused
        else "hermes-sre-tools/23"
        if atomic_selector
        else "hermes-sre-tools/22"
        if typed_purpose
        else "hermes-sre-tools/21"
        if single_owner
        else "hermes-sre-tools/20"
        if observable
        else "hermes-sre-tools/19"
        if entry
        else "hermes-sre-tools/18",
    )
    raw = _read_bytes(root / "investigation/state.json", 64 * 1024**2)
    if focused:
        from nurse_rcl.v2.live.ops_sre_focused_tools import OpsSreFocusedLedgerState

    state_type = (
        OpsSreFocusedLedgerState
        if focused
        else OpsSreAtomicSelectorLedgerState
        if atomic_selector
        else OpsSreTypedPurposeLedgerState
        if typed_purpose
        else OpsSreSingleOwnerLedgerState
        if single_owner
        else OpsSreObservableLedgerState
        if observable
        else OpsSreEntryLedgerState
        if entry
        else OpsSreDirectedLedgerState
    )
    ledger = state_type.model_validate_json(raw)
    if (
        ledger.code_sha256 != identity["code_sha256"]
        or ledger.case_snapshot_id != core.identity.case_snapshot_id
        or raw != canonical_json_bytes(ledger.model_dump(mode="json"))
        or ledger.state_sha256
        != sha256_hex(
            canonical_json_bytes(
                ledger.model_dump(mode="json", exclude={"state_sha256"}),
            )
        )
    ):
        raise ValueError("stage prefix ledger integrity differs")
    return dict(
        decisions=[
            sha256_hex(canonical_json_bytes(row.model_dump(mode="json")))
            for row in conversation.decisions
        ],
        actions=[
            sha256_hex(canonical_json_bytes(row.model_dump(mode="json"))) for row in ledger.records
        ],
    )


def _inspect_stage(
    root: Path,
    identity: dict[str, Any],
    *,
    resume_ceiling: int | None = None,
) -> dict[str, Any]:
    result = read_json(root / "stage-outcome.json")
    unsigned = {k: v for k, v in result.items() if k != "stage_result_sha256"}
    files = result.get("files")
    if (
        result.get("schema_version") != "nurse.v2.sre-stage-outcome/1"
        or result.get("identity") != identity
        or result.get("stage_result_sha256") != sha256_hex(canonical_json_bytes(unsigned))
        or type(files) is not dict
        or set(files) != set(STAGE_MEMBERS)
    ):
        raise ValueError("stage outcome integrity differs")
    advanced = False
    for name, value in files.items():
        if list(member_digest(root / name)) != value and (
            resume_ceiling is not None
            and resume_ceiling > result["decision_count"]
            and result["termination"] == "stage_pause"
            and name in {"investigation/state.json", "investigation/conversation.json"}
        ):
            advanced = True
        elif list(member_digest(root / name)) != value:
            raise ValueError("stage saved member integrity differs")
    if advanced:
        assert resume_ceiling is not None
        current = _prefix_pins(root, identity)
        old = result["prefix_pins"]
        if len(current["decisions"]) > resume_ceiling:
            raise ValueError("stage ceiling is below saved prefix")
        if any(current[k][: len(old[k])] != old[k] for k in ("decisions", "actions")):
            raise ValueError("stage completed prefix integrity differs")
    return result


def stage_status(path: Path, pin: str) -> dict[str, Any]:
    """Read exact saved work only, without telemetry, credentials or provider resume."""
    plan = read_ops_plan(path, pin)
    return _inspect_stage(
        Path(plan.output_root) / plan.case.opaque_case_id, plan_identity(plan, pin)
    )


def _reuse(result: dict[str, Any], ceiling: int) -> bool:
    if result["decision_count"] > ceiling:
        raise ValueError("stage ceiling is below saved prefix")
    return bool(result["termination"] != "stage_pause" or result["decision_count"] == ceiling)


def _failed_round(
    root: Path,
    identity: dict[str, Any],
    grant: dict[str, Any],
) -> dict[str, Any]:
    """A joined early failure is terminal, not permission to start another worker."""
    row = read_json(root / "stage-round.json")
    unsigned = {k: v for k, v in row.items() if k != "round_result_sha256"}
    if (
        row.get("identity") != identity
        or row.get("requested_grant") != grant
        or row.get("status") not in {"failed", "interrupted"}
        or row.get("round_result_sha256") != sha256_hex(canonical_json_bytes(unsigned))
    ):
        raise ValueError("saved failed round identity, grant or integrity differs")
    receipt = read_outcome(
        root,
        dict(
            identity,
            stage_grant_sha256=sha256_hex(canonical_json_bytes(grant)),
            worker_invocation_id=row["worker_invocation_id"],
        ),
    )
    if (
        receipt.primary_failure is None
        or row.get("worker_receipt_sha256") != receipt.state_sha256
        or row.get("primary_failure") != receipt.primary_failure.model_dump(mode="json")
    ):
        raise ValueError("saved failed round worker receipt differs")
    return row


def run_directed_stage(
    admitted: OpsAdmittedPlan,
    *,
    stage_decision_limit: int,
    approved_plan_sha256: str | None,
    approved_stage_sha256: str | None,
) -> dict[str, Any]:
    """Worker path: source-admitted fixed core, original HERMES owner, no report."""
    grant = _admit_stage(
        admitted,
        stage_decision_limit=stage_decision_limit,
        approved_plan_sha256=approved_plan_sha256,
        approved_stage_sha256=approved_stage_sha256,
    )
    root, plan = admitted.state_root, admitted.plan
    root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(root)
    lock = os.open(root / "case.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "stage-outcome.json").exists():
            retained = _inspect_stage(
                root, case_identity(admitted), resume_ceiling=stage_decision_limit
            )
            if _reuse(retained, stage_decision_limit):
                return retained
        publish_json(root / "identity.json", case_identity(admitted))
        grants = root / "stage-grants"
        grants.mkdir(mode=0o700, exist_ok=True)
        require_plain_path(grants)
        grant_path = grants / f"grant-{approved_stage_sha256}.json"
        if grant_path.exists() or grant_path.is_symlink():
            if read_json(grant_path) != grant:
                raise ValueError("saved stage grant differs")
        else:
            publish_json(grant_path, grant)
        data = _preparation(admitted)
        with _module_span("input_guard"):
            _admit_display(data, admitted)
            identity = OpsCoreIdentity(
                dataset=plan.case.dataset,
                opaque_case_id=plan.case.opaque_case_id,
                case_snapshot_id=prepared_snapshot_id(data),
                configuration_sha256=plan.core_configuration_sha256,
                environment_sha256=plan.environment_sha256,
            )
        with _module_span("fixed_core_import"):
            core = import_source_core(admitted, identity)
        entry = isinstance(plan.profile, OpsSreEntryProfile)
        observable = isinstance(plan.profile, OpsSreObservableProfile)
        single_owner = isinstance(plan.profile, OpsSreSingleOwnerProfile)
        typed_purpose = isinstance(plan.profile, OpsSreTypedPurposeProfile)
        from nurse_rcl.v2.live.ops_contracts import OpsSreAtomicSelectorProfile

        focused = isinstance(plan.profile, OpsSreFocusedProfile)
        atomic_selector = isinstance(plan.profile, OpsSreAtomicSelectorProfile)
        if focused:
            from nurse_rcl.v2.live.ops_sre_focused_tools import (
                OpsSreFocusedPublicationReader, OpsSreFocusedToolbox,
            )
        toolbox_type = (
            OpsSreFocusedToolbox
            if focused
            else OpsSreAtomicSelectorToolbox
            if atomic_selector
            else OpsSreTypedPurposeToolbox
            if typed_purpose
            else OpsSreSingleOwnerToolbox
            if single_owner
            else OpsSreObservableToolbox
            if observable
            else OpsSreEntryToolbox
            if entry
            else OpsSreDirectedToolbox
        )
        reader_type = (
            OpsSreFocusedPublicationReader
            if focused
            else OpsSreAtomicSelectorPublicationReader
            if atomic_selector
            else OpsSreTypedPurposePublicationReader
            if typed_purpose
            else OpsSreSingleOwnerPublicationReader
            if single_owner
            else OpsSreObservablePublicationReader
            if observable
            else OpsSreEntryPublicationReader
            if entry
            else OpsSreDirectedPublicationReader
        )
        provider_type = stage_provider_type(plan.profile)
        toolbox = toolbox_type(
            data=data,
            block_root=Path(plan.preparation.archive_root) / "blocks",
        )
        reader = reader_type(root=root, data=data, identity=identity)
        with _module_span("frozen_source_publication"):
            reader.bind_checkpoint(toolbox, core=core)
            reader.publish_frozen(toolbox, core)
        provider = provider_type(
            root=root / "provider",
            code_sha256=plan.code_sha256,
            execution_sha256=admitted.plan_sha256,
            approved_execution_sha256=approved_plan_sha256,
            opaque_case_id=identity.opaque_case_id,
            case_snapshot_id=identity.case_snapshot_id,
            stage_decision_limit=stage_decision_limit,
            approved_stage_sha256=approved_stage_sha256,
        )
        with _phase("method", "orchestration"), _module_span("ops_investigation"):
            result = run_ops_agent(
                root=root,
                toolbox=toolbox,
                code_sha256=plan.code_sha256,
                complete=provider.decision,
                recover_complete=provider.saved_decision,
                empirical=True,
                sre=True,
                sre_profile="hermes-sre-tools/25"
                if focused
                else "hermes-sre-tools/23"
                if atomic_selector
                else "hermes-sre-tools/22"
                if typed_purpose
                else "hermes-sre-tools/21"
                if single_owner
                else "hermes-sre-tools/20"
                if observable
                else "hermes-sre-tools/19"
                if entry
                else "hermes-sre-tools/18",
                opaque_case_id=identity.opaque_case_id,
                stage_decision_limit=stage_decision_limit,
                feedback_fact=isinstance(plan.profile, OpsSreFeedbackFactProfile),
            )
        assert isinstance(result, OpsWorkingAgentResult)
        require_ops_code(plan)
        body = dict(
            schema_version="nurse.v2.sre-stage-outcome/1",
            identity=case_identity(admitted),
            approved_grant=grant,
            core_result_digest=core.core_result_digest,
            channels=dict(
                core="completed", investigation=result.investigation_status, report="disabled"
            ),
            termination=result.termination,
            decision_count=result.decision_count,
            action_count=result.action_count,
            primary_failure=(
                result.primary_failure.model_dump(mode="json")
                if result.primary_failure is not None
                else None
            ),
            secondary_failures=[r.model_dump(mode="json") for r in result.secondary_failures],
            prefix_pins=_prefix_pins(root, case_identity(admitted)),
            files={name: list(member_digest(root / name)) for name in STAGE_MEMBERS},
        )
        body["stage_result_sha256"] = sha256_hex(canonical_json_bytes(body))
        publish_json(root / f"stage-outcome-{body['stage_result_sha256']}.json", body)
        atomic_write_bytes(root / "stage-outcome.json", canonical_json_bytes(body))
        return body
    finally:
        os.close(lock)


def run_stage_round(
    path: Path,
    pin: str,
    *,
    stage_decision_limit: int,
    approved_plan_sha256: str | None,
    approved_stage_sha256: str | None,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Reuse existing phase supervision, one-hour prepare and method warn-only policy."""
    admitted = load_ops_plan(path, pin)
    grant = _admit_stage(
        admitted,
        stage_decision_limit=stage_decision_limit,
        approved_plan_sha256=approved_plan_sha256,
        approved_stage_sha256=approved_stage_sha256,
    )
    root, plan = admitted.state_root, admitted.plan
    root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(root)
    lock = os.open(root / "stage-round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = _invocation_usage(plan, pin, root, None)
        if (root / "stage-outcome.json").exists():
            retained = _inspect_stage(
                root, case_identity(admitted), resume_ceiling=stage_decision_limit
            )
            if _reuse(retained, stage_decision_limit):
                return dict(
                    retained,
                    status="stage_paused"
                    if retained["termination"] == "stage_pause"
                    else (
                        "failed" if retained["termination"] == "unavailable" else "stage_terminal"
                    ),
                    invocation_kind="saved_stage_reuse",
                    usage=_invocation_usage(plan, pin, root, baseline),
                )
        elif (root / "worker-outcome.json").exists() or (root / "stage-round.json").exists():
            retained = _failed_round(root, case_identity(admitted), grant)
            return dict(
                retained,
                status="failed",
                invocation_kind="saved_stage_failure_reuse",
                usage=_invocation_usage(plan, pin, root, baseline, worker_terminated=True),
            )
        timing = _clock(root, pin)
        invocation_id = uuid4().hex
        body: dict[str, Any] = dict(
            identity=case_identity(admitted),
            status="failed",
            worker_invocation_id=invocation_id,
            requested_grant=grant,
            channels=dict(core="pending", investigation="unavailable", report="disabled"),
        )
        with _module_timing_scope(root / "module-spans", identity=pin, boot_id=timing.boot_id):
            try:
                command = [
                    sys.executable,
                    "-m",
                    "nurse_rcl.v2.live.ops_sre_directed_stage",
                    "--plan",
                    str(path),
                    "--plan-sha256",
                    pin,
                    "--worker",
                    "--through-decision-index",
                    str(stage_decision_limit),
                    "--approved-plan-sha256",
                    str(approved_plan_sha256),
                    "--approved-stage-sha256",
                    str(approved_stage_sha256),
                    "--worker-invocation-id",
                    invocation_id,
                ]
                with _module_span("worker_supervision", role="orchestration"):
                    code = supervise_phases(
                        command, log_path=root / "stage.log", timing=timing, progress=progress
                    )
                body["worker_exit_code"] = code
                receipt_path = root / "worker-outcome.json"
                if code != 0 and (receipt_path.exists() or receipt_path.is_symlink()):
                    receipt = read_outcome(
                        root,
                        dict(
                            case_identity(admitted),
                            stage_grant_sha256=sha256_hex(canonical_json_bytes(grant)),
                            worker_invocation_id=invocation_id,
                        ),
                    )
                    if receipt.primary_failure is None:
                        raise ValueError("failed stage worker receipt has no primary failure")
                    body.update(
                        primary_failure=receipt.primary_failure.model_dump(mode="json"),
                        secondary_failures=[
                            r.model_dump(mode="json") for r in receipt.secondary_failures
                        ],
                        worker_receipt_sha256=receipt.state_sha256,
                    )
                # A failed worker may already have saved the actual provider/tool cause.
                # Never promote a previous, smaller grant's outcome to this invocation.
                retained = _inspect_stage(root, case_identity(admitted))
                if retained["approved_grant"] != grant:
                    raise ValueError("worker outcome belongs to a different stage grant")
                body.update(retained)
                if code != 0:
                    raise ValueError("bounded stage worker incomplete")
                body["status"] = (
                    "stage_paused"
                    if body["termination"] == "stage_pause"
                    else ("stage_terminal" if body["termination"] != "unavailable" else "failed")
                )
            except BaseException as exc:
                primary = body.get("primary_failure")
                if primary is not None:
                    rows = body.setdefault("secondary_failures", [])
                    detail = failure_detail("case_worker", exc).model_dump(mode="json")
                    if detail != primary and detail not in rows and len(rows) < 16:
                        rows.append(detail)
                body.update(
                    failure_type=primary["failure_type"]
                    if primary is not None
                    else type(exc).__name__,
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                )
                if (root / "core/result.json").exists():
                    body["channels"]["core"] = "completed"
                else:
                    body["channels"]["core"] = "failed"
            finally:
                timing._record("pause", "method", "orchestration", monotonic())
                body["timing"] = timing.clock.summary()
                body["usage"] = _invocation_usage(plan, pin, root, baseline, worker_terminated=True)
                body["module_timing"] = _modules(root, pin)
                body["round_result_sha256"] = sha256_hex(canonical_json_bytes(body))
                publish_json(root / f"stage-round-{body['round_result_sha256']}.json", body)
                atomic_write_bytes(root / "stage-round.json", canonical_json_bytes(body))
                progress(dict(event="sre_stage_round_saved", **body))
        return body
    finally:
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--through-decision-index", type=int)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--approved-stage-sha256")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-invocation-id")
    parser.add_argument("--timing-fd", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))

    def emit(value: dict[str, Any]) -> None:
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        if args.worker != (args.timing_fd is not None) or (args.status and args.worker):
            raise ValueError("stage worker requires private timing channel")
        if args.worker:
            if (
                type(args.worker_invocation_id) is not str
                or len(args.worker_invocation_id) != 32
                or any(c not in "0123456789abcdef" for c in args.worker_invocation_id)
            ):
                raise ValueError("stage worker requires supervisor invocation identity")
        elif args.worker_invocation_id is not None:
            raise ValueError("worker invocation identity is private to the supervisor")
        if args.status:
            emit(stage_status(args.plan, args.plan_sha256))
            return 0
        options = dict(
            stage_decision_limit=args.through_decision_index,
            approved_plan_sha256=args.approved_plan_sha256,
            approved_stage_sha256=args.approved_stage_sha256,
        )
        if args.worker:
            envelope = read_ops_plan(args.plan, args.plan_sha256)
            root = Path(envelope.output_root) / envelope.case.opaque_case_id
            binding = dict(
                plan_identity(envelope, args.plan_sha256),
                stage_grant_sha256=args.approved_stage_sha256,
                worker_invocation_id=args.worker_invocation_id,
            )
            client = PhaseClient(args.timing_fd)
            primary: BaseException | None = None
            secondary: list[OpsFailureDetail] = []

            def retain_failure() -> None:
                assert primary is not None
                try:
                    write_outcome(
                        root, binding, failure_detail("case_worker", primary), tuple(secondary[:16])
                    )
                except Exception as exc:
                    emit(
                        dict(
                            event="sre_stage_failure_receipt_unavailable",
                            failure_type=type(exc).__name__,
                        )
                    )

            try:
                with _use_phase_client(client):
                    admitted = load_ops_plan(args.plan, args.plan_sha256)
                    with _module_timing_scope(
                        admitted.state_root / "module-spans",
                        identity=args.plan_sha256,
                        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    ):
                        result = run_directed_stage(admitted, **options)
            except BaseException as exc:
                primary = exc
                secondary.extend(secondary_failures(exc))
                retain_failure()
            finally:
                cleanups: tuple[tuple[FailureStage, Callable[[], None]], ...] = (
                    ("timing_finish", client.finish),
                    ("timing_close", client.channel.close),
                )
                for stage, cleanup in cleanups:
                    try:
                        cleanup()
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                        else:
                            secondary.append(failure_detail(stage, exc))
                        retain_failure()
            if primary is not None:
                raise primary
            emit(result)
            return int(result["termination"] == "unavailable")
        result = run_stage_round(args.plan, args.plan_sha256, progress=emit, **options)
        return int(result["status"] not in {"stage_paused", "stage_terminal"})
    except KeyboardInterrupt:
        emit(dict(event="sre_stage_interrupted"))
        return 130
    except Exception as exc:
        emit(dict(event="sre_stage_failed", failure_type=type(exc).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
