"""Private Linux process owner for deterministic core and isolated investigation.

This is not a public live CLI or an OS sandbox for arbitrary Python. Only the
project-owned completion adapter may be supplied in production; the model gets
JSON and the registered tools, never Python, filesystem or process capabilities.
Admission, authorization, run freezing and full progress belong to the live CLI.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import resource
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent import OpsCompletion, run_ops_agent
from nurse_rcl.v2.live.ops_agent_state import (
    OpsAgentResult,
    OpsNavigationAgentResult,
    OpsReplanningAgentResult,
    OpsWorkingAgentResult,
    load_conversation,
    save_conversation,
)
from nurse_rcl.v2.live.ops_contracts import (
    OpsCoreIdentity,
    OpsNodeEvidence,
    SreProfileId,
    resolve_sre_profile,
)
from nurse_rcl.v2.live.ops_core import run_ops_core
from nurse_rcl.v2.live.ops_failure import failure_detail, retain_secondary, secondary_failures
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolbox
from nurse_rcl.v2.live.ops_publication import OpsPublicationReader
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_tools import OpsToolbox
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path


def _save(path: Path, body: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(body))


@contextmanager
def _worker_timing(module: str) -> Iterator[None]:
    """Detach fork-inherited diagnostic FDs; only the case owner can switch phases."""
    from nurse_rcl.v2.live import repair_module_timing as spans
    from nurse_rcl.v2.live import repair_timing as phases

    recorder, client = spans._SESSION.get(), phases._CLIENT.get()
    if isinstance(client, phases.PhaseClient):
        client.channel.close()
    phases._CLIENT.set(None)
    spans._SESSION.set(None)
    spans._STACK.set(())
    if recorder is None:
        yield
    else:
        os.close(recorder.fd)  # Child FD only; parent's recorder remains valid.
        with (
            spans._module_timing_scope(
                recorder.path.parent, identity=recorder.identity, boot_id=recorder.boot_id
            ),
            spans._module_span(module),
        ):
            yield


def _core_worker(
    data: RepairPreparationData,
    identity: OpsCoreIdentity,
    root: Path,
    blocks: Path,
) -> None:
    started = monotonic()
    _save(root / "core-process.json", dict(status="running", pid=os.getpid()))
    try:
        with _worker_timing("ops_core"):
            run_ops_core(data=data, identity=identity, root=root, block_root=blocks)
    except BaseException as exc:
        _save(
            root / "core-process.json",
            dict(
                status="failed",
                failure_type=type(exc).__name__,
                elapsed_seconds=monotonic() - started,
            ),
        )
        raise
    _save(
        root / "core-process.json", dict(status="completed", elapsed_seconds=monotonic() - started)
    )


def _investigation_worker(
    data: RepairPreparationData,
    identity: OpsCoreIdentity,
    root: Path,
    blocks: Path,
    code: str,
    complete: OpsCompletion,
    empirical: bool = False,
    sre: bool = False,
    frozen_core: OpsCoreResult | None = None,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
    execution_sha256: str | None = None,
    invocation_id: str | None = None,
    prior_state_present: bool = False,
    recover_complete: Callable[[int, bytes], bytes | None] | None = None,
) -> None:
    selected = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    started = monotonic()
    directory = root / "investigation"
    _save(directory / "process.json", dict(status="running", pid=os.getpid()))
    provider_entered = False

    def bounded_complete(index: int, context: dict[str, Any] | bytes) -> bytes:
        nonlocal provider_entered
        # This marker is set before the sole trusted completion entry, including
        # local provider failure or reuse. It is deliberately conservative.
        provider_entered = True
        return complete(index, context)

    try:
        os.nice(10)  # Lower-priority supplementary work, never throttle the core.
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = min(8 * 1024**3, soft if soft != resource.RLIM_INFINITY else 8 * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        with _worker_timing("ops_investigation"):
            toolbox = (
                OpsIndexedToolbox(data=data, block_root=blocks, soft_targets=True)
                if empirical
                else OpsToolbox(data=data, block_root=blocks)
            )
            if sre:
                assert selected is not None
                from nurse_rcl.v2.live.ops_sre_action_tools import (
                    OpsSreActionPublicationReader,
                    OpsSreActionToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_brief_tools import OpsSreBriefToolbox
                from nurse_rcl.v2.live.ops_sre_comparable_tools import OpsSreComparableToolbox
                from nurse_rcl.v2.live.ops_sre_condition_tools import OpsSreConditionToolbox
                from nurse_rcl.v2.live.ops_sre_d21_tools import (
                    OpsSreUsedSourcePublicationReader,
                    OpsSreUsedSourceToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_d22_tools import (
                    OpsSreReceiptReusePublicationReader,
                    OpsSreReceiptReuseToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_d23_tools import (
                    OpsSreBalancedPublicationReader,
                    OpsSreBalancedToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_d25_tools import (
                    OpsSreReplanningPublicationReader,
                    OpsSreReplanningToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_d26_tools import (
                    OpsSreQuestionPublicationReader,
                    OpsSreQuestionToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_navigation_tools import (
                    OpsSreNavigationPublicationReader,
                    OpsSreNavigationToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsSreRecoveryToolbox
                from nurse_rcl.v2.live.ops_sre_tools import OpsSreToolbox
                from nurse_rcl.v2.live.ops_sre_workbench_tools import (
                    OpsSreWorkbenchPublicationReader,
                    OpsSreWorkbenchToolbox,
                )
                from nurse_rcl.v2.live.ops_sre_working_tools import OpsSreWorkingToolbox

                toolbox_type = {
                    "hermes-sre-tools/4": OpsSreToolbox,
                    "hermes-sre-tools/5": OpsSreWorkingToolbox,
                    "hermes-sre-tools/6": OpsSreBriefToolbox,
                    "hermes-sre-tools/7": OpsSreConditionToolbox,
                    "hermes-sre-tools/8": OpsSreRecoveryToolbox,
                    "hermes-sre-tools/9": OpsSreComparableToolbox,
                    "hermes-sre-tools/10": OpsSreNavigationToolbox,
                    "hermes-sre-tools/11": OpsSreWorkbenchToolbox,
                    "hermes-sre-tools/12": OpsSreUsedSourceToolbox,
                    "hermes-sre-tools/13": OpsSreReceiptReuseToolbox,
                    "hermes-sre-tools/14": OpsSreBalancedToolbox,
                    "hermes-sre-tools/15": OpsSreReplanningToolbox,
                    "hermes-sre-tools/16": OpsSreQuestionToolbox,
                    "hermes-sre-tools/17": OpsSreActionToolbox,
                }[selected]
                toolbox = toolbox_type(data=data, block_root=blocks)
            reader: OpsPublicationReader
            if selected in {
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            }:
                from nurse_rcl.v2.live.ops_sre_comparable_publication import (
                    OpsSreComparablePublicationReader,
                )

                reader = (
                    OpsSreActionPublicationReader
                    if selected == "hermes-sre-tools/17"
                    else OpsSreQuestionPublicationReader
                    if selected == "hermes-sre-tools/16"
                    else OpsSreReplanningPublicationReader
                    if selected == "hermes-sre-tools/15"
                    else OpsSreBalancedPublicationReader
                    if selected == "hermes-sre-tools/14"
                    else OpsSreUsedSourcePublicationReader
                    if selected == "hermes-sre-tools/12"
                    else OpsSreReceiptReusePublicationReader
                    if selected == "hermes-sre-tools/13"
                    else OpsSreWorkbenchPublicationReader
                    if selected == "hermes-sre-tools/11"
                    else OpsSreNavigationPublicationReader
                    if selected == "hermes-sre-tools/10"
                    else OpsSreComparablePublicationReader
                )(root=root, data=data, identity=identity)
                assert isinstance(toolbox, OpsSreComparableToolbox)
                reader.bind_checkpoint(toolbox, core=frozen_core)
            else:
                reader = OpsPublicationReader(root=root, data=data, identity=identity)
            if frozen_core is not None and selected not in {
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            }:
                if not sre or frozen_core.identity != identity:
                    raise ValueError("frozen SRE core identity differs")
                diagnostics = json.loads(
                    _read_bytes(root / "core/diagnostics.json", REPAIR_MAX_JSON_BYTES)
                )
                if diagnostics.get("core_result_digest") != frozen_core.core_result_digest:
                    raise ValueError("frozen SRE diagnostics identity differs")
                toolbox.admit_core(
                    nodes=tuple(
                        OpsNodeEvidence.model_validate_json(canonical_json_bytes(n))
                        for n in diagnostics["nodes"]
                    ),
                    steps=frozen_core.evidence.session.steps,
                )
            refresh = (lambda: reader.poll(toolbox)) if frozen_core is None else None
            if (
                selected
                in {
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                    "hermes-sre-tools/14",
                    "hermes-sre-tools/15",
                    "hermes-sre-tools/16",
                    "hermes-sre-tools/17",
                }
                and frozen_core is not None
            ):
                assert isinstance(reader, OpsSreComparablePublicationReader)
                assert isinstance(toolbox, OpsSreComparableToolbox)

                def refresh() -> None:
                    reader.publish_frozen(toolbox, frozen_core)

                checkpoint = directory / "state.json"
                if not checkpoint.exists() and not checkpoint.is_symlink():
                    refresh()
            result = run_ops_agent(
                root=root,
                toolbox=toolbox,
                code_sha256=code,
                complete=bounded_complete
                if selected
                in {
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                    "hermes-sre-tools/14",
                    "hermes-sre-tools/15",
                    "hermes-sre-tools/16",
                    "hermes-sre-tools/17",
                }
                else complete,
                recover_complete=recover_complete,
                refresh=refresh,
                empirical=empirical,
                sre=sre,
                working=working,
                sre_profile=selected,
                opaque_case_id=identity.opaque_case_id if empirical else None,
            )
            _save(directory / "result.json", result.model_dump(mode="json"))
    except BaseException as exc:
        details: dict[str, Any] = {}
        startup = (
            selected
            in {
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            }
            and not (directory / "conversation.json").exists()
        )
        if startup:
            assert (
                selected == "hermes-sre-tools/9"
                or selected == "hermes-sre-tools/10"
                or selected == "hermes-sre-tools/11"
                or selected == "hermes-sre-tools/12"
                or selected == "hermes-sre-tools/13"
                or selected == "hermes-sre-tools/14"
                or selected == "hermes-sre-tools/15"
                or selected == "hermes-sre-tools/16"
                or selected == "hermes-sre-tools/17"
            )
            from nurse_rcl.v2.live.ops_startup_failure import save_startup_failure

            try:
                save_startup_failure(
                    root,
                    identity=identity,
                    code=code,
                    execution=execution_sha256,
                    primary=failure_detail("investigation_worker", exc),
                    secondary=secondary_failures(exc),
                    provider_boundary_entered=provider_entered,
                    invocation_id=invocation_id,
                    prior_state_present=prior_state_present,
                    profile=selected,
                )
            except BaseException as receipt_error:
                retain_secondary(exc, "investigation_worker", receipt_error)
            details["primary_failure"] = failure_detail("investigation_worker", exc).model_dump(
                mode="json"
            )
            details["secondary_failures"] = [
                r.model_dump(mode="json") for r in secondary_failures(exc)
            ]
            details["startup_binding"] = dict(
                identity=identity.model_dump(mode="json"),
                code=code,
                execution=execution_sha256,
                invocation_id=invocation_id,
            )
        if (
            selected
            in {
                "hermes-sre-tools/5",
                "hermes-sre-tools/6",
                "hermes-sre-tools/7",
                "hermes-sre-tools/8",
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            }
            and not startup
        ):
            # The loop saved its terminal receipt before propagating an interrupt.
            # Do not replace consumed actions with an outer zero-count fallback.
            try:
                state = load_conversation(
                    directory / "conversation.json",
                    snapshot=identity.case_snapshot_id,
                    code=code,
                    empirical=True,
                    sre=True,
                    sre_profile=selected,
                    opaque_case_id=identity.opaque_case_id,
                )
                if isinstance(state.result, OpsWorkingAgentResult):
                    result = state.result
                    outer = failure_detail("investigation_worker", exc)
                    extras = secondary_failures(exc)
                    if result.primary_failure is None or (
                        result.primary_failure.failure_type != outer.failure_type
                    ):
                        extras = (outer, *extras)
                    result = result.model_copy(
                        update={
                            "secondary_failures": tuple(
                                dict.fromkeys((*result.secondary_failures, *extras))
                            )[:16]
                        }
                    )
                    state = state.model_copy(update={"result": result})
                    save_conversation(directory / "conversation.json", state)
                    _save(directory / "result.json", result.model_dump(mode="json"))
                    details["investigation_status"] = result.investigation_status
                    details["primary_failure"] = (
                        result.primary_failure.model_dump(mode="json")
                        if result.primary_failure
                        else None
                    )
            except Exception as recovery:
                details["recovery_failure_type"] = type(recovery).__name__
        _save(
            directory / "process.json",
            dict(
                status="failed",
                failure_type=type(exc).__name__,
                elapsed_seconds=monotonic() - started,
                **details,
            ),
        )
        raise
    _save(
        directory / "process.json",
        dict(
            status=result.termination,
            elapsed_seconds=monotonic() - started,
            **(
                {"investigation_status": result.investigation_status}
                if isinstance(result, OpsWorkingAgentResult)
                else {}
            ),
        ),
    )


@dataclass
class OpsPair:
    """Supervisor-owned handles, never passed to either model or core worker."""

    root: Path
    core_process: BaseProcess
    investigation_process: BaseProcess
    empirical: bool = False
    working: bool = False
    identity: OpsCoreIdentity | None = None
    code_sha256: str | None = None
    sre_profile: SreProfileId | None = None
    execution_sha256: str | None = None
    invocation_id: str | None = None

    def join_core(self, *, timeout: float | None = None) -> OpsCoreResult:
        self.core_process.join(timeout)
        if self.core_process.is_alive():
            raise TimeoutError("core still running; no method cutoff applied")
        if self.core_process.exitcode != 0:
            raise RuntimeError("core failed; no replacement ranking")
        return OpsCoreResult.model_validate_json(
            _read_bytes(
                self.root / "core" / "result.json",
                REPAIR_MAX_JSON_BYTES,
            )
        )

    def join_investigation(self, *, timeout: float | None = None) -> OpsAgentResult:
        self.investigation_process.join(timeout)
        if self.investigation_process.is_alive():
            raise TimeoutError("investigation still running; core remains independent")
        if self.sre_profile in {
            "hermes-sre-tools/9",
            "hermes-sre-tools/10",
            "hermes-sre-tools/11",
            "hermes-sre-tools/12",
            "hermes-sre-tools/13",
            "hermes-sre-tools/14",
            "hermes-sre-tools/15",
            "hermes-sre-tools/16",
            "hermes-sre-tools/17",
        } and (
            not (self.root / "investigation/conversation.json").exists()
            and not (self.root / "investigation/result.json").exists()
        ):
            from nurse_rcl.v2.live.ops_startup_failure import (
                InvestigationStartupFailure,
                read_startup_failure,
            )

            assert self.identity is not None and self.code_sha256 is not None
            path = self.root / "investigation/startup-failure.json"
            if not path.exists() and not path.is_symlink():
                # A failed receipt write cannot prove non-send. The supervisor
                # may still retain the current worker's typed first failure.
                from nurse_rcl.v2.live.ops_failure import OpsFailureDetail

                body = json.loads(_read_bytes(self.root / "investigation/process.json", 65536))
                expected = dict(
                    identity=self.identity.model_dump(mode="json"),
                    code=self.code_sha256,
                    execution=self.execution_sha256,
                    invocation_id=self.invocation_id,
                )
                if body.get("status") != "failed" or body.get("startup_binding") != expected:
                    raise ValueError("startup failure unavailable for terminated worker")
                raise InvestigationStartupFailure(
                    OpsFailureDetail.model_validate(body["primary_failure"]),
                    receipt_sha256=None,
                    secondary=tuple(
                        OpsFailureDetail.model_validate(r)
                        for r in body.get("secondary_failures", [])
                    ),
                )
            assert (
                self.sre_profile == "hermes-sre-tools/9"
                or self.sre_profile == "hermes-sre-tools/10"
                or self.sre_profile == "hermes-sre-tools/11"
                or self.sre_profile == "hermes-sre-tools/12"
                or self.sre_profile == "hermes-sre-tools/13"
                or self.sre_profile == "hermes-sre-tools/14"
                or self.sre_profile == "hermes-sre-tools/15"
                or self.sre_profile == "hermes-sre-tools/16"
                or self.sre_profile == "hermes-sre-tools/17"
            )
            receipt = read_startup_failure(
                self.root,
                identity=self.identity,
                code=self.code_sha256,
                execution=self.execution_sha256,
                invocation_id=self.invocation_id,
                profile=self.sre_profile,
            )
            raise InvestigationStartupFailure(
                receipt.primary_failure,
                receipt_sha256=receipt.receipt_sha256,
                secondary=receipt.secondary_failures,
            )
        if self.working or self.sre_profile in {
            "hermes-sre-tools/5",
            "hermes-sre-tools/6",
            "hermes-sre-tools/7",
            "hermes-sre-tools/8",
            "hermes-sre-tools/9",
            "hermes-sre-tools/10",
            "hermes-sre-tools/11",
            "hermes-sre-tools/12",
            "hermes-sre-tools/13",
            "hermes-sre-tools/14",
            "hermes-sre-tools/15",
            "hermes-sre-tools/16",
            "hermes-sre-tools/17",
        }:
            assert self.identity is not None and self.code_sha256 is not None
            state = load_conversation(
                self.root / "investigation/conversation.json",
                snapshot=self.identity.case_snapshot_id,
                code=self.code_sha256,
                empirical=True,
                sre=True,
                working=self.working,
                sre_profile=self.sre_profile,
                opaque_case_id=self.identity.opaque_case_id,
            )
            retained = (
                OpsReplanningAgentResult
                if self.sre_profile
                in {"hermes-sre-tools/15", "hermes-sre-tools/16", "hermes-sre-tools/17"}
                else OpsNavigationAgentResult
                if self.sre_profile
                in {
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                    "hermes-sre-tools/14",
                    "hermes-sre-tools/15",
                    "hermes-sre-tools/16",
                    "hermes-sre-tools/17",
                }
                else OpsWorkingAgentResult
            ).model_validate_json(
                _read_bytes(self.root / "investigation/result.json", 128 * 1024 * 1024)
            )
            if state.result != retained:
                raise ValueError("working process/conversation outcome differs")
            return retained
        if self.investigation_process.exitcode != 0:
            raise RuntimeError("investigation failed; keep completed core")
        return OpsAgentResult.model_validate_json(
            _read_bytes(
                self.root / "investigation" / "result.json",
                (128 if self.empirical else 48) * 1024 * 1024,
            )
        )

    def close(self) -> None:
        """Explicit supervisor cancellation/cleanup, not model-controlled stop."""
        for child in (self.investigation_process, self.core_process):
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)


def start_ops_pair(
    *,
    data: RepairPreparationData,
    identity: OpsCoreIdentity,
    root: Path,
    block_root: Path,
    code_sha256: str,
    complete: OpsCompletion,
    empirical: bool = False,
    sre: bool = False,
    frozen_core: OpsCoreResult | None = None,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
    execution_sha256: str | None = None,
    recover_complete: Callable[[int, bytes], bytes | None] | None = None,
) -> OpsPair:
    """Start trusted workers without queues/acks or serializing whole telemetry.

    Linux fork supplies private copy-on-write preparation, not shared mutable
    Python state. Start from a single-threaded admitted owner before constructing
    any live HTTP client in the investigation child. No core wall-time deadline.
    """
    if threading.active_count() != 1:
        raise ValueError("isolated fork requires a single-threaded owner")
    if sre and not empirical:
        raise ValueError("SRE requires empirical wire policy")
    selected = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    require_plain_path(root)
    directory = root / "investigation"
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    prior_state_present = any(directory.iterdir())
    ctx = multiprocessing.get_context("fork")
    invocation_id = uuid4().hex
    core = ctx.Process(target=_core_worker, args=(data, identity, root, block_root))
    investigation = ctx.Process(
        target=_investigation_worker,
        args=(
            data,
            identity,
            root,
            block_root,
            code_sha256,
            complete,
            empirical,
            sre,
            frozen_core,
            working,
            selected,
            execution_sha256,
            invocation_id,
            prior_state_present,
            recover_complete,
        ),
    )
    core.start()
    try:
        investigation.start()
    except BaseException:
        # Failed supplementary process creation must not terminate core work.
        _save(directory / "process.json", dict(status="unavailable", reason="start_failed"))
        core.join()
        raise
    return OpsPair(
        root,
        core,
        investigation,
        empirical,
        working,
        identity,
        code_sha256,
        selected,
        execution_sha256,
        invocation_id,
    )
