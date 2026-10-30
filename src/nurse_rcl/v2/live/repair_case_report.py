"""Owner-bound report data and code-owned ARK transport, not a public trust API."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.live import ark_bridge, repair_safety
from nurse_rcl.v2.live.repair_case_artifacts import RepairCaseAudit
from nurse_rcl.v2.live.repair_case_guards import _CaseInputs
from nurse_rcl.v2.live.repair_driver import (
    publish_bundle,
    publish_json,
    read_bundle,
    report_binding,
)
from nurse_rcl.v2.live.repair_execution import require_code
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_prepared_case_artifacts import (
    RepairPreparedCaseAudit,
    RepairPreparedReportProjection,
    RepairPreparedReportRun,
)
from nurse_rcl.v2.live.repair_report import (
    _REPORT_ROOTS,
    RepairReportRun,
    _compose_values,
    _RepairReportFields,
    _snapshot_surface,
)
from nurse_rcl.v2.live.repair_report_projection import (
    POLICY,
    RepairReportProjection,
    _projection_values,
    _RepairProjectionFields,
    _seal,
)
from nurse_rcl.v2.live.repair_report_prompt import REPAIR_EXCHANGE_VERSION
from nurse_rcl.v2.live.repair_report_request import RUN_POLICY
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest


class RepairCaseReportProjection(_RepairProjectionFields):
    schema_version: Literal["nurse.v2.live-repair-report-projection/2"] = (
        "nurse.v2.live-repair-report-projection/2"
    )


class RepairCaseReportRun(_RepairReportFields):
    schema_version: Literal["nurse.v2.live-repair-report/2"] = "nurse.v2.live-repair-report/2"
    projection: RepairCaseReportProjection
    interpretability_status: Literal["owner_bound_source_and_score_replayed"] = (
        "owner_bound_source_and_score_replayed"
    )


_ROOTS: tuple[type[BaseModel], ...] = (
    RepairCaseReportRun,
    RepairCaseReportProjection,
    RepairPreparedReportProjection,
    RepairPreparedReportRun,
    *(r for r in _REPORT_ROOTS if r not in (RepairReportRun, RepairReportProjection)),
)


def _snapshot(value: BaseModel) -> tuple[bytes, str]:
    return _snapshot_surface(
        value,
        _ROOTS,
        RepairPreparedReportRun if type(value) is RepairPreparedReportRun else RepairCaseReportRun,
    )


def _unchanged(value: BaseModel, before: tuple[bytes, str]) -> None:
    try:
        same = _snapshot(value) == before
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        same = False
    if not same:
        raise ValueError("case report content or representation changed")


@_module_call("report_projection")
def _project_case_audit(
    audit: RepairCaseAudit | RepairPreparedCaseAudit,
) -> RepairCaseReportProjection | RepairPreparedReportProjection:
    if type(audit) not in (RepairCaseAudit, RepairPreparedCaseAudit):
        raise ValueError("case report requires the exact new audit identity")
    cls = (
        RepairPreparedReportProjection
        if type(audit) is RepairPreparedCaseAudit
        else RepairCaseReportProjection
    )
    binding: dict[str, Any] = (
        {"preparation": audit.preparation} if type(audit) is RepairPreparedCaseAudit else {}
    )
    return _seal(
        cls(
            **binding,
            **_projection_values(
                ranking=audit.ranking,
                graph=audit.graph,
                opaque_case_id=audit.opaque_case_id,
                audit_sha256=audit.result_sha256,
                audit_fields=tuple(sorted(type(audit).model_fields)),
            ),
        ),
        "projection_sha256",
    )


def _run_owned_report(
    inputs: _CaseInputs,
    *,
    audit: RepairCaseAudit | RepairPreparedCaseAudit,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    state_root: Path,
    execution_sha256: str,
    code_root: Path,
    code_files: dict[str, str],
    entry_file_pins: tuple[tuple[str, str], ...],
    entry_specification: dict[str, Any],
    expected_input_sha256: str,
) -> RepairCaseReportRun | RepairPreparedReportRun:
    """Called inside the owner's independently rebuilt evidence scope only."""
    from nurse_rcl.v2.live.repair_case_owner import _terminal_journal_guard

    audit_type = RepairCaseAudit if inputs.preparation is None else RepairPreparedCaseAudit
    if type(audit) is not audit_type or type(identity_manifest) is not InferenceIdentityManifest:
        raise ValueError("case report requires exact audit and manifest")
    require_pin(execution_sha256)
    require_pin(expected_identity_manifest_sha256)
    state_root = require_plain_path(state_root)
    namespace = state_root / ("case-owner-v1" if inputs.preparation is None else "prepared-case-v1")
    namespace.mkdir(exist_ok=True)
    require_plain_path(namespace)
    checkpoint = namespace / "report-input"
    started = monotonic()
    with ExitStack() as guards:
        checks: list[Any] = []

        def register(check: Any) -> None:
            checks.append(check)
            guards.callback(check)

        def keep(value: Any) -> Any:
            before = _snapshot(value)
            register(lambda: _unchanged(value, before))
            return value

        def barrier() -> None:
            with ExitStack() as all_checks:
                for check in checks:
                    all_checks.callback(check)

        audit_before = repair_safety.snapshot(audit, (audit_type,))
        code_before = repair_safety.snapshot(code_files, (RepairCaseAudit,))
        metadata_before = repair_safety.snapshot(
            (entry_file_pins, entry_specification), (RepairCaseAudit,)
        )
        require_pin(expected_input_sha256)

        def check_entry() -> None:
            repair_safety.require_unchanged(
                (entry_file_pins, entry_specification), (RepairCaseAudit,), metadata_before
            )
            if sha256_hex(canonical_json_bytes(entry_specification)) != expected_input_sha256:
                raise ValueError("case input changed before report disclosure")
            if type(entry_file_pins) is not tuple or not entry_file_pins:
                raise ValueError("case report requires independently pinned entry files")
            for item in entry_file_pins:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or any(type(s) is not str for s in item)
                ):
                    raise ValueError("invalid case entry file binding")
                path, pin = item
                require_pin(pin)
                if file_sha256(require_plain_path(Path(path))) != pin:
                    raise ValueError("case entry identity file changed")

        register(check_entry)
        register(_terminal_journal_guard(inputs.prepared.runtime.root, audit.session))
        register(lambda: require_code(code_root, code_files))
        register(
            lambda: repair_safety.require_unchanged(code_files, (RepairCaseAudit,), code_before)
        )
        register(inputs.check_full_method)
        register(lambda: repair_safety.require_unchanged(audit, (audit_type,), audit_before))
        keep(identity_manifest)
        keep(POLICY)
        keep(RUN_POLICY)
        manifest = keep(
            InferenceIdentityManifest.model_validate_json(
                canonical_json_bytes(plain(identity_manifest))
            )
        )
        if (
            not manifest.verify()
            or manifest.manifest_sha256 != expected_identity_manifest_sha256
            or not any(
                r.opaque_case_id == audit.opaque_case_id and r.dataset == audit.dataset
                for r in manifest.records
            )
            or audit.opaque_case_id != state_root.name
        ):
            raise ValueError("case report independent identity or dataset differs")
        barrier()
        projection = keep(_project_case_audit(audit))
        timings = {"report_projection": monotonic() - started}
        files = {
            "audit.json": audit_before[0],
            "projection.json": _snapshot(projection)[0],
        }
        expected_receipt = dict(
            identity=execution_sha256,
            files={name: sha256_hex(data) for name, data in sorted(files.items())},
        )

        def check_checkpoint() -> None:
            if read_bundle(checkpoint, identity=execution_sha256) != expected_receipt:
                raise ValueError("case report-input checkpoint differs from current verification")

        barrier()
        if not checkpoint.exists() and not checkpoint.is_symlink():
            publish_bundle(checkpoint, files, identity=execution_sha256)
        check_checkpoint()
        register(check_checkpoint)

        def complete(payload: bytes) -> bytes:
            # The pin is from this protected, freshly verified audit, never a
            # prompt field or a residual successful legacy stage marker.
            barrier()
            binding = report_binding(
                payload,
                opaque_case_id=audit.opaque_case_id,
                source_sha256=audit.source_descriptor_sha256,
                audit_sha256=audit.result_sha256,
            )
            publish_json(state_root / "request-binding.json", binding.model_dump(mode="json"))
            with ark_bridge.LiveArkBridge(
                root=state_root.parent / "ark",
                lock_path=Path(
                    "/home/wangrunzhou/NURSE/development/live-rcl-20260918/locks/ark-in-flight.lock"
                ),
                settings=ark.load_ark_settings_from_key_file(
                    Path("/home/wangrunzhou/.config/nurse/ark.env")
                ),
                code_sha256=sha256_hex(canonical_json_bytes(code_files)),
                config_sha256=execution_sha256,
                authorized_case_ids=(audit.opaque_case_id,),
                request_protocol=REPAIR_EXCHANGE_VERSION,
            ) as bridge:
                try:
                    barrier()
                    return bridge.complete(binding, payload)
                finally:
                    # Attempt/response consumption survives every late failure.
                    with ExitStack() as exits:
                        exits.callback(barrier)
                        receipt = bridge.receipt(binding)
                        pin = sha256_hex(canonical_json_bytes(receipt))
                        publish_json(state_root / f"ark-receipt-{pin[:16]}.json", receipt)

        values = _compose_values(
            audit_sha256=audit.result_sha256,
            ranking_sha256=audit.ranking.result_sha256,
            projection=projection,
            identity_manifest=manifest,
            identity_pin=expected_identity_manifest_sha256,
            complete=complete,
            keep=keep,
            timings=timings,
        )
        report_type = RepairCaseReportRun if inputs.preparation is None else RepairPreparedReportRun
        binding: dict[str, Any] = (
            {} if inputs.preparation is None else {"preparation": inputs.preparation}
        )
        result = _seal(report_type(**values, **binding), "result_sha256")
        keep(result)
        barrier()
    return result
