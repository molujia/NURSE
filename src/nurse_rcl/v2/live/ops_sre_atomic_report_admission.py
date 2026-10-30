"""Read-only bytes from an independently pinned source inventory, not source discovery.

Manifest hashes bind an already admitted source; they are not signatures or consent.
No owner, ledger, toolbox, preparation, core import or provider is instantiated here.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ops_agent_state import OpsAtomicSelectorConversation, OpsDecisionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import OpsAtomicSelectorActionRecord

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_brief_report import BriefSreReportContext


def verified_member_bytes(
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
    members: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Check a caller-selected subset against an external, unchanged full inventory.

    This does not assert that the subset is a complete investigation or authorize
    any member as a fact. The enclosing admission checks the required member set.
    """
    index = _manifest_index(manifest_raw, expected_manifest_sha256)
    if not members or not members.keys() <= index.keys():
        raise ValueError("source member set is not in the manifest")
    checked: dict[str, bytes] = {}
    for path, raw in members.items():
        if type(raw) is not bytes or (
            len(raw) != index[path]["bytes"] or sha256_hex(raw) != index[path]["sha256"]
        ):
            raise ValueError("source member bytes differ from the manifest")
        checked[path] = raw
    return checked


def _manifest_index(
    manifest_raw: bytes, expected_manifest_sha256: str
) -> dict[str, dict[str, Any]]:
    require_pin(expected_manifest_sha256)
    if len(manifest_raw) > 4 * 1024**2 or sha256_hex(manifest_raw) != expected_manifest_sha256:
        raise ValueError("source manifest external pin differs")
    manifest = json.loads(manifest_raw)
    if type(manifest) is not dict or canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("source manifest is not canonical")
    rows = manifest.get("members")
    if type(rows) is not list or not rows or len(rows) > 4096:
        raise ValueError("source manifest member inventory differs")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("source manifest member shape differs")
        path = row["path"]
        if (
            type(path) is not str
            or not path
            or (
                PurePosixPath(path).is_absolute()
                or str(PurePosixPath(path)) != path
                or ".." in PurePosixPath(path).parts
                or "\\" in path
                or ":" in path
                or path in index
            )
        ):
            raise ValueError("source manifest member path differs")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("source manifest member size differs")
        require_pin(row["sha256"])
        index[path] = row
    return index


@dataclass(frozen=True)
class AdmittedAtomicMetadata:
    """Original pinned bytes, not an executable investigation owner or score source."""

    manifest_raw: bytes
    manifest_pin: str
    members: Mapping[str, bytes]
    expected: Mapping[str, str]
    decisions: tuple[OpsDecisionRecord, ...]
    records: tuple[OpsActionRecord, ...]
    references: dict[str, Any]
    round_result: dict[str, Any]
    core_raw: bytes


def _signed_json(raw: bytes, field: str) -> dict[str, Any]:
    value = json.loads(raw)
    if (
        type(value) is not dict
        or canonical_json_bytes(value) != raw
        or (
            value.get(field)
            != sha256_hex(
                canonical_json_bytes({key: child for key, child in value.items() if key != field})
            )
        )
    ):
        raise ValueError("source envelope content signature differs")
    return value


def admit_terminal_metadata(
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
    members: Mapping[str, bytes],
    expected: Mapping[str, str],
) -> AdmittedAtomicMetadata:
    """Admit fixed signed terminal/prefix metadata; complete core admission is separate.

    Only caller-selected original members are decoded. Navigation is not restored,
    and a valid metadata result alone does not grant core or factual authority.
    """
    checked = verified_member_bytes(
        manifest_raw, expected_manifest_sha256=expected_manifest_sha256, members=members
    )
    if set(expected) != {
        "case_id",
        "plan_sha256",
        "code_sha256",
        "grant_sha256",
        "worker_invocation_id",
    }:
        raise ValueError("source expected identity fields differ")
    if not re.fullmatch(r"opq_[a-f0-9]{32}", expected["case_id"]) or not re.fullmatch(
        r"[a-f0-9]{32}", expected["worker_invocation_id"]
    ):
        raise ValueError("source expected case/invocation differs")
    for key in ("plan_sha256", "code_sha256", "grant_sha256"):
        require_pin(expected[key])
    manifest = json.loads(manifest_raw)
    if manifest.get("schema_version") != "nurse.m3-feedback-terminal-evidence/1" or any(
        manifest.get(key) != expected[key]
        for key in ("plan_sha256", "grant_sha256", "worker_invocation_id")
    ):
        raise ValueError("source manifest identity differs")
    base = f"run/results/{expected['case_id']}/"
    names = {
        "conversation": base + "investigation/conversation.json",
        "ledger": base + "investigation/state.json",
        "round": base + "stage-round.json",
        "outcome": base + "stage-outcome.json",
        "core": base + "core/result.json",
        "plan": "run/plan.json",
        "grant": "run/stage-grant-through-3.json",
    }
    exits = [
        name for name in checked if name.startswith("run/logs/") and name.endswith("/runner.exit")
    ]
    if not set(names.values()) <= checked.keys() or len(exits) != 1:
        raise ValueError("source required terminal members are missing")
    if checked[exits[0]].strip() not in {b"0", b"EXIT_CODE=0"} or manifest.get("runner_exit") != 0:
        raise ValueError("source runner terminal exit differs")
    if len(checked[names["ledger"]]) > 64 * 1024**2 or (
        len(checked[names["conversation"]]) > 128 * 1024**2
        or len(checked[names["core"]]) > 256 * 1024**2
    ):
        raise ValueError("source member exceeds its original resource boundary")
    plan = json.loads(checked[names["plan"]])
    grant = json.loads(checked[names["grant"]])
    if (
        sha256_hex(checked[names["plan"]]) != expected["plan_sha256"]
        or sha256_hex(canonical_json_bytes(plan["code_files"])) != expected["code_sha256"]
        or plan["case"]["opaque_case_id"] != expected["case_id"]
        or sha256_hex(checked[names["grant"]]) != expected["grant_sha256"]
        or grant.get("execution_sha256") != expected["plan_sha256"]
        or grant.get("report_limit") != 0
    ):
        raise ValueError("source plan/code/grant binding differs")
    conv = _signed_json(checked[names["conversation"]], "state_sha256")
    ledger = _signed_json(checked[names["ledger"]], "state_sha256")
    round_result = _signed_json(checked[names["round"]], "round_result_sha256")
    outcome = _signed_json(checked[names["outcome"]], "stage_result_sha256")
    identity = dict(
        opaque_case_id=expected["case_id"],
        plan_sha256=expected["plan_sha256"],
        code_sha256=expected["code_sha256"],
        mode="with-agent",
        profile="v2-ops-sre/14",
    )
    prefix = dict(
        decisions=[sha256_hex(canonical_json_bytes(d)) for d in conv["decisions"]],
        actions=[sha256_hex(canonical_json_bytes(r)) for r in ledger["records"]],
    )
    channels = dict(core="completed", investigation="partial", report="disabled")
    if (
        round_result.get("identity") != identity
        or outcome.get("identity") != identity
        or round_result.get("requested_grant") != grant
        or round_result.get("approved_grant") != grant
        or outcome.get("approved_grant") != grant
        or round_result.get("worker_invocation_id") != expected["worker_invocation_id"]
        or round_result.get("worker_exit_code") != 0
        or round_result.get("status") != "stage_paused"
        or round_result.get("stage_result_sha256") != outcome.get("stage_result_sha256")
        or any(
            row.get("prefix_pins") != prefix
            or row.get("channels") != channels
            or row.get("termination") != "stage_pause"
            for row in (round_result, outcome)
        )
        or round_result.get("decision_count") != len(conv["decisions"])
        or round_result.get("action_count") != len(ledger["records"])
        or not 1 <= len(conv["decisions"]) <= 8
        or len(ledger["records"]) > len(conv["decisions"])
    ):
        raise ValueError("source signed terminal identity/prefix differs")
    if conv.get("schema_version") != "nurse.v2.sre-conversation/21" or (
        ledger.get("schema_version") != "nurse.v2.sre-investigation-ledger/24"
        or conv.get("case_snapshot_id") != ledger.get("case_snapshot_id")
        or conv.get("opaque_case_id") != expected["case_id"]
        or any(
            row.get("code_sha256") != expected["code_sha256"]
            or row.get("investigation_profile_id") != "hermes-sre-tools/23"
            for row in (conv, ledger)
        )
    ):
        raise ValueError("source conversation/ledger identity differs")
    conversation = OpsAtomicSelectorConversation.model_validate_json(checked[names["conversation"]])
    records = tuple(
        OpsAtomicSelectorActionRecord.model_validate_json(canonical_json_bytes(r))
        for r in ledger["records"]
    )
    return AdmittedAtomicMetadata(
        manifest_raw,
        expected_manifest_sha256,
        MappingProxyType(checked),
        MappingProxyType(dict(expected)),
        conversation.decisions,
        records,
        conversation.reference_state,
        round_result,
        checked[names["core"]],
    )


def read_source_archive(
    *,
    archive: Path,
    manifest_raw: bytes,
    expected_manifest_sha256: str,
    expected_archive_sha256: str,
    required: set[str],
) -> dict[str, bytes]:
    """Stream/hash the entire old archive; decode only selected bytes, never extract."""
    from nurse_rcl.v2.live.staged_case import require_plain_path

    index = _manifest_index(manifest_raw, expected_manifest_sha256)
    require_pin(expected_archive_sha256)
    if not required or not required <= index.keys():
        raise ValueError("archive required member set differs")
    label = json.loads(manifest_raw).get("label")
    if type(label) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,80}", label):
        raise ValueError("archive label differs")
    metadata = {f"{label}-manifest.json", f"{label}-reservation.json"}
    before = require_plain_path(archive).stat()
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while block := handle.read(1024**2):
            digest.update(block)
    if digest.hexdigest() != expected_archive_sha256:
        raise ValueError("archive external bytes pin differs")
    found: set[str] = set()
    selected: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            if (
                not member.isfile()
                or member.name in found
                or (member.name not in index and member.name not in metadata)
            ):
                raise ValueError("archive has duplicate, nonregular or unlisted members")
            found.add(member.name)
            expected_size = index[member.name]["bytes"] if member.name in index else 4 * 1024**2
            if member.size > expected_size or (
                member.name in index and member.size != expected_size
            ):
                raise ValueError("archive member size differs")
            member_handle = bundle.extractfile(member)
            if member_handle is None:
                raise ValueError("archive regular member is unreadable")
            hashed, count, chunks = hashlib.sha256(), 0, []
            retain = member.name in required or member.name == f"{label}-manifest.json"
            while block := member_handle.read(1024**2):
                count += len(block)
                hashed.update(block)
                if retain:
                    chunks.append(block)
            if count != member.size or (
                member.name in index and hashed.hexdigest() != index[member.name]["sha256"]
            ):
                raise ValueError("archive original member bytes differ")
            if retain:
                raw = b"".join(chunks)
                if member.name == f"{label}-manifest.json":
                    if raw != manifest_raw:
                        raise ValueError("archive embedded manifest differs")
                else:
                    selected[member.name] = raw
    after = archive.stat()
    if found != index.keys() | metadata or (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("archive set or original file changed")
    return verified_member_bytes(
        manifest_raw, expected_manifest_sha256=expected_manifest_sha256, members=selected
    )


def assemble_admitted_atomic_source(admitted: AdmittedAtomicMetadata) -> BriefSreReportContext:
    """Pure projection of a complete pinned core and actual saved receipts, no replay."""
    from nurse_rcl.v2.live.ops_report_context import build_report_context
    from nurse_rcl.v2.live.ops_result import OpsCoreResult
    from nurse_rcl.v2.live.ops_sre_atomic_report import (
        AtomicReportSourceIdentity,
        assemble_atomic_report_context,
    )
    from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

    if type(admitted) is not AdmittedAtomicMetadata:
        raise ValueError("source admission type differs")
    source = admit_terminal_metadata(
        admitted.manifest_raw,
        expected_manifest_sha256=admitted.manifest_pin,
        members=admitted.members,
        expected=admitted.expected,
    )
    core = OpsCoreResult.model_validate_json(source.core_raw)
    if (
        core.identity.opaque_case_id != source.expected["case_id"]
        or core.identity.case_snapshot_id != source.references["snapshot"]
        or core.core_result_digest != source.round_result["core_result_digest"]
    ):
        raise ValueError("complete source core/snapshot differs")
    # No original measurement blocks are loaded. This existing pure projection
    # retains core-known facts and its original omissions, not invented context.
    formal = build_report_context(core=core, sources=(), investigation=(), soft_targets=True)
    if not formal.formal_projection_available or not formal.facts:
        raise ValueError(
            "source formal projection is unavailable; empty formal is not a substitute"
        )
    for record in source.records:
        if record.model_view_json is None:  # type: ignore[attr-defined]
            continue
        rows = _delivered_records(working_evidence_view(json.loads(record.model_view_json)))  # type: ignore[attr-defined]
        if any(row.get("representation") == BRIEF_REPRESENTATION for row in rows):
            raise ValueError(
                "source needs complete retained brief material; minimal decoder will not invent it"
            )

    def digest(value: Any) -> str:
        return sha256_hex(canonical_json_bytes(value))

    identity = AtomicReportSourceIdentity(
        source_plan_sha256=source.expected["plan_sha256"],
        source_code_sha256=source.expected["code_sha256"],
        stage_grant_sha256=source.expected["grant_sha256"],
        stage_round_sha256=source.round_result["round_result_sha256"],
        worker_invocation_id=source.expected["worker_invocation_id"],
        opaque_case_id=source.expected["case_id"],
        case_snapshot_id=core.identity.case_snapshot_id,
        source_run_id=source.references["run_id"],
        core_result_digest=core.core_result_digest,
        formal_context_sha256=formal.context_sha256,
        reference_state_sha256=digest(source.references),
        decision_pins=tuple(digest(row.model_dump(mode="json")) for row in source.decisions),
        action_pins=tuple(digest(row.model_dump(mode="json")) for row in source.records),
        retained_sources_sha256=digest([]),
        termination="stage_pause",
        investigation_status="partial",
    )
    return assemble_atomic_report_context(
        formal=formal,
        decisions=source.decisions,
        records=source.records,
        references=source.references,
        retained_sources=(),
        source_identity=identity,
        expected_source_sha256=digest(identity.model_dump(mode="json")),
    )
