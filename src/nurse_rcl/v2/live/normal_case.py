"""Private normal worker composition with a single evidence body and finite receipts.

The strict normal plan loader supplies the typed, admitted plan. This module
does not accept a user-supplied runtime callback or authenticate itself as an audit.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from time import monotonic
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding
from nurse_rcl.v2.live.normal_artifacts import (
    NORMAL_POLICY,
    NormalAssurance,
    NormalEvidence,
    NormalPreparationReceipt,
)
from nurse_rcl.v2.live.normal_preparation import build_normal_preparation, load_normal_preparation
from nurse_rcl.v2.live.normal_queries import _read_bytes, _seal, run_normal_queries
from nurse_rcl.v2.live.normal_report import NormalReportRun, run_normal_report
from nurse_rcl.v2.live.normal_scoring import score_normal_case
from nurse_rcl.v2.live.repair_driver import _fsync, publish_json
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_preparation_codec import _decode_json_member, member_digest
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_timing import _phase
from nurse_rcl.v2.live.staged_case import require_plain_path

_SCHEMAS = {
    "evidence": "nurse.v2.normal-rcl-evidence/1",
    "ranking": "nurse.v2.normal-rcl-ranking-result/1",
    "report": "nurse.v2.normal-rcl-report/1",
}


def _save_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Encode once, exclusive atomic publish, return the bytes' manifest entry."""
    raw = canonical_json_bytes(value)
    if len(raw) > REPAIR_MAX_JSON_BYTES:
        raise ValueError("normal result member exceeds capacity")
    descriptor, name = tempfile.mkstemp(prefix=".normal-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(name=path.name, size_bytes=len(raw), sha256=sha256_hex(raw))


def _save_member(root: Path, name: str, value: dict[str, Any], identity: dict[str, Any]) -> None:
    member = _save_json(root / (name + ".json"), value)
    if name == "evidence":
        summary = dict(evidence_sha256=value["result_sha256"],
                       ranking_sha256=value["ranking"]["result_sha256"],
                       paid_count=value["session"]["paid_count"],
                       candidate_count=len(value["ranking"]["scores"]))
    elif name == "ranking":
        summary = dict(evidence_sha256=value["source_evidence_sha256"],
                       ranking_sha256=value["source_ranking_sha256"],
                       candidate_count=len(value["scores"]))
    else:
        summary = dict(evidence_sha256=value["source_evidence_sha256"],
                       ranking_sha256=value["source_ranking_sha256"],
                       report_sha256=value["result_sha256"], report_status=value["status"])
    publish_json(root / (name + "-receipt.json"), dict(
        identity=identity, member=member, body_schema=_SCHEMAS[name], summary=summary,
    ))


def _load_member(root: Path, name: str, identity: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    receipt = _decode_json_member(_read_bytes(root / (name + "-receipt.json"), 16384))
    if set(receipt) != {"identity", "member", "body_schema", "summary"} or (
        receipt["identity"] != identity or receipt["body_schema"] != _SCHEMAS[name]
    ):
        raise ValueError("normal saved member execution identity differs")
    member = receipt["member"]
    if type(member) is not dict or set(member) != {"name", "size_bytes", "sha256"} or (
        member["name"] != name + ".json" or type(member["size_bytes"]) is not int
        or not 0 <= member["size_bytes"] <= REPAIR_MAX_JSON_BYTES
    ):
        raise ValueError("normal saved member descriptor differs")
    raw = _read_bytes(root / member["name"], member["size_bytes"])
    if len(raw) != member["size_bytes"] or sha256_hex(raw) != member["sha256"]:
        raise ValueError("normal saved member integrity differs")
    return raw, member


def _identity(plan: dict[str, Any], pin: str) -> dict[str, Any]:
    preparation: RepairPreparationRun = plan["preparation"]
    policy = NORMAL_POLICY
    if plan.get("b0_only") is True:
        from nurse_rcl.v2.live.normal_b0_case import B0_POLICY

        policy = B0_POLICY
    return dict(
        policy=policy, execution_sha256=pin, code_sha256=plan["code_sha256"],
        input_sha256=preparation.key.input_sha256,
        preparation_key_sha256=preparation.key.key_sha256,
        opaque_case_id=preparation.key.opaque_case_id,
    )


def inspect_normal_result(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Final manifest/bytes only: no scoring, source rebuild or semantic audit."""
    if (root / "failure.json").exists():
        raise ValueError("normal case has a retained terminal failure")
    result = _decode_json_member(_read_bytes(root / "result.json", 65536))
    expected = {
        "schema_version", "identity", "status", "files", "paid_count", "candidate_count",
        "evidence_sha256", "ranking_sha256", "report_sha256", "report_status", "timing",
        *NormalAssurance.model_fields,
    }
    if set(result) != expected or result["schema_version"] != "nurse.v2.normal-rcl-result/1" or (
        result["identity"] != identity
        or result["status"] not in {"completed", "completed_partial_report"}
        or any(result[k] != v for k, v in NormalAssurance().model_dump().items())
    ):
        raise ValueError("normal completion manifest identity/status differs")
    if type(result["paid_count"]) is not int or not 0 <= result["paid_count"] <= 8 or (
        type(result["candidate_count"]) is not int or result["candidate_count"] <= 0
        or result["report_status"] != (
            "full" if result["status"] == "completed" else "partial"
        )
    ):
        raise ValueError("normal completion counts/report status differ")
    files = result["files"]
    binding = LiveRequestBinding.model_validate_json(
        _read_bytes(root / "request-binding.json", 16384)
    )
    required = {"evidence.json", "evidence-receipt.json", "ranking.json", "ranking-receipt.json",
                "report.json", "report-receipt.json", "request-binding.json", "report-input.json",
                "query-ledger/head", "ark/run-identity.json", f"ark/request-{binding.slot}.json"}
    if binding.opaque_case_id != identity["opaque_case_id"] or (
        binding.context_sha256 != result["evidence_sha256"]
    ):
        raise ValueError("normal completion request binding differs")
    if type(files) is not list or not 1 <= len(files) <= 96:
        raise ValueError("normal completion manifest file capacity differs")
    names = tuple(m.get("name") for m in files if type(m) is dict)
    if len(names) != len(files) or any(type(n) is not str for n in names) or (
        names != tuple(sorted(set(cast(tuple[str, ...], names)))) or not required <= set(names)
    ):
        raise ValueError("normal completion manifest file domain differs")
    for member in files:
        name = member["name"]
        if set(member) != {"name", "size_bytes", "sha256"} or (
            Path(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))
            or "\\" in name or type(member["size_bytes"]) is not int
            or not 0 <= member["size_bytes"] <= REPAIR_MAX_JSON_BYTES
        ):
            raise ValueError("normal completion manifest unsafe member")
        if member_digest(root / name) != (member["size_bytes"], member["sha256"]):
            raise ValueError("normal completion member integrity differs")
    entries = {m["name"]: m for m in files}
    for name in _SCHEMAS:
        receipt = _decode_json_member(_read_bytes(root / (name + "-receipt.json"), 16384))
        fields = {"evidence_sha256", "ranking_sha256"} | (
            {"paid_count", "candidate_count"} if name == "evidence"
            else {"candidate_count"} if name == "ranking" else {"report_sha256", "report_status"}
        )
        if set(receipt) != {"identity", "member", "body_schema", "summary"} or (
            receipt["identity"] != identity or receipt["body_schema"] != _SCHEMAS[name]
            or receipt["member"] != entries[name + ".json"]
            or receipt["summary"] != {k: result[k] for k in fields}
        ):
            raise ValueError("normal completion receipt binding differs")
    return result


def _complete_files(root: Path, *, b0_only: bool = False) -> list[dict[str, Any]]:
    files = []
    # Complete method trace is retained; exclude locks, unfinished temporary files,
    # parent timing logs and the final manifest itself.
    for name in (("evidence", "ranking") if b0_only else ("evidence", "ranking", "report")):
        receipt_name = name + "-receipt.json"
        receipt = _decode_json_member(_read_bytes(root / receipt_name, 16384))
        files.append(receipt["member"])
    paths = [*(root.glob("*-receipt*.json")),
             *sorted((root / "query-ledger").glob("*.json")), root / "query-ledger/head"]
    if not b0_only:
        paths.extend([root / "request-binding.json", root / "report-input.json",
                      *sorted((root / "ark").glob("*.json"))])
    for path in paths:
        size, digest = member_digest(path)
        files.append(dict(name=path.relative_to(root).as_posix(), size_bytes=size, sha256=digest))
    return sorted(files, key=lambda row: row["name"])


def run_normal_case(*, plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    """Use only the strict loader's typed internal plan; no public CLI in this module."""
    if "b0_only" in plan:
        raise ValueError("legacy normal worker does not accept a pure B0 plan")
    return _run_case(plan=plan, plan_sha256=plan_sha256)


def _run_case(*, plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    """Shared numerical path; only the registered owner chooses the terminal schema."""
    b0_only = plan.get("b0_only") is True
    inspect_result = inspect_normal_result
    if b0_only:
        from nurse_rcl.v2.live.normal_b0_case import inspect_b0_result

        inspect_result = inspect_b0_result
    require_pin(plan_sha256)
    root = require_plain_path(Path(plan["state_root"]))
    identity = _identity(plan, plan_sha256)
    if root.name != identity["opaque_case_id"]:
        raise ValueError("normal case output identity differs")
    lock = os.open(root / "case.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    started = monotonic()
    stage = "admission"
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(root / "identity.json", identity)
        if (root / "failure.json").exists():
            raise ValueError("normal case has a retained terminal failure")
        if (root / "result.json").exists():
            return inspect_result(root, identity)
        try:
            stage = "preparation"
            preparation: RepairPreparationRun = plan["preparation"]
            preparation_started = monotonic()
            module = "preparation_build" if preparation.mode == "build" and not (
                Path(preparation.receipt_path).exists()
            ) else "preparation_load"
            with _phase("preparation", "raw_input_prepare"), _module_span(module):
                arguments = dict(
                    archive_root=Path(preparation.archive_root), expected_key=preparation.key,
                    raw_arguments=plan["raw_arguments"],
                    receipt_path=Path(preparation.receipt_path),
                )
                if preparation.mode == "load":
                    assert preparation.manifest_sha256 is not None
                    assert preparation.receipt_sha256 is not None
                    data = load_normal_preparation(
                        **arguments, expected_manifest_sha256=preparation.manifest_sha256,
                        expected_receipt_sha256=preparation.receipt_sha256,
                    )
                elif Path(preparation.receipt_path).exists():
                    raw_receipt = _read_bytes(Path(preparation.receipt_path), 16384)
                    saved = NormalPreparationReceipt.model_validate_json(raw_receipt)
                    if saved.execution_sha256 != plan_sha256 or (
                        saved.execution_code_sha256 != plan["code_sha256"]
                    ):
                        raise ValueError("normal cold recovery has a different build owner")
                    data = load_normal_preparation(
                        **arguments, expected_manifest_sha256=saved.manifest_sha256,
                        expected_receipt_sha256=sha256_hex(raw_receipt),
                    )
                else:
                    data, _, _ = build_normal_preparation(
                        **arguments, execution_sha256=plan_sha256,
                        execution_code_sha256=plan["code_sha256"],
                    )
            preparation_seconds = monotonic() - preparation_started
            stage = "queries"
            session = run_normal_queries(data=data, ledger_root=root / "query-ledger",
                                         block_root=Path(preparation.archive_root) / "blocks")
            stage = "scoring"
            if (root / "evidence.json").exists():
                raw, _ = _load_member(root, "evidence", identity)
                _decode_json_member(raw)
                evidence = NormalEvidence.model_validate_json(raw)
                if _seal(evidence, "result_sha256") != evidence or evidence.session != session or (
                    evidence.catalog_sha256 != data.catalog.catalog_sha256
                    or evidence.source_descriptor_sha256 != data.source_descriptor_sha256
                ):
                    raise ValueError("normal saved evidence identity differs")
            else:
                evidence = score_normal_case(data=data, session=session)
                with _module_span("artifact_publish"):
                    _save_member(root, "evidence", evidence.model_dump(mode="json"), identity)
            ranking = evidence.ranking
            if (root / "ranking.json").exists():
                _load_member(root, "ranking", identity)
            else:
                _save_member(root, "ranking", dict(
                    schema_version="nurse.v2.normal-rcl-ranking-result/1",
                    source_evidence_sha256=evidence.result_sha256,
                    source_ranking_sha256=ranking.result_sha256,
                    ordered_service_keys=ranking.ordered_service_keys,
                    candidate_service_keys=ranking.candidate_service_keys,
                    top_k_service_keys=ranking.top_k_service_keys,
                    scores=[s.model_dump(mode="json") for s in ranking.scores],
                    tie_groups=[t.model_dump(mode="json") for t in ranking.tie_groups],
                ), identity)
            if b0_only:
                from nurse_rcl.v2.live.normal_b0_case import complete_b0

                stage = "publication"
                return complete_b0(
                    root=root, identity=identity, evidence=evidence,
                    preparation_seconds=preparation_seconds, started=started,
                )
            stage = "report"
            if (root / "report.json").exists():
                raw, _ = _load_member(root, "report", identity)
                report = NormalReportRun.model_validate_json(raw)
                if report.source_evidence_sha256 != evidence.result_sha256 or (
                    _seal(report, "result_sha256") != report
                ):
                    raise ValueError("normal saved report binding differs")
            else:
                with _module_span("report"):
                    report = run_normal_report(
                        sources=data.sources,
                        evidence=evidence, identity_manifest=plan["identity_manifest"],
                        expected_identity_manifest_sha256=plan["identity_manifest"].manifest_sha256,
                        state_root=root, execution_sha256=plan_sha256,
                        code_sha256=plan["code_sha256"],
                    )
                _save_member(root, "report", report.model_dump(mode="json"), identity)
            if report.status not in ("full", "partial") or not report.exchange_attempted or (
                report.raw_response_hex is None or report.presentation is None
                or not report.presentation.safe_display_blocks
            ):
                raise ValueError("normal case has no displayable real Report Agent response")
            stage = "publication"
            files = _complete_files(root)
            elapsed = monotonic() - started
            result = dict(
                schema_version="nurse.v2.normal-rcl-result/1", identity=identity,
                **NormalAssurance().model_dump(),
                status="completed" if report.status == "full" else "completed_partial_report",
                files=files, paid_count=session.paid_count, candidate_count=len(ranking.scores),
                evidence_sha256=evidence.result_sha256, ranking_sha256=ranking.result_sha256,
                report_sha256=report.result_sha256, report_status=report.status,
                timing=dict(preparation_seconds=preparation_seconds,
                            method_seconds=elapsed - preparation_seconds, elapsed_seconds=elapsed),
            )
            _save_json(root / "result.json", result)
            return result
        except Exception as error:
            from nurse_rcl.v2.exchange_json import ExchangeBoundaryError

            failure = dict(
                schema_version=("nurse.v2.normal-b0-failure/1" if b0_only else
                                "nurse.v2.normal-rcl-failure/1"), identity=identity,
                status="failed", stage=stage, error_type=type(error).__name__,
                elapsed_seconds=monotonic() - started,
            )
            if b0_only and type(error) is ExchangeBoundaryError and error.code == "json_limit":
                failure["capacity_reason"] = "json_limit"
            publish_json(root / "failure.json", failure)
            return failure
    finally:
        os.close(lock)
