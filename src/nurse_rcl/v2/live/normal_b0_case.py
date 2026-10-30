"""Explicit zero-model B0 terminals sharing the unchanged normal numerical path."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalEvidence
from nurse_rcl.v2.live.normal_case import _SCHEMAS, _complete_files, _run_case, _save_json
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.repair_preparation_codec import _decode_json_member, member_digest
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES

B0_POLICY = "live-normal-b0/1"
B0_RESULT = "nurse.v2.normal-b0-result/1"


def inspect_b0_result(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Check saved bytes/receipts, never replay preparation, queries or scoring."""
    if (root / "failure.json").exists():
        raise ValueError("B0 has a retained terminal failure")
    result = _decode_json_member(_read_bytes(root / "result.json", 65536))
    expected = {
        "schema_version", "identity", "status", "files", "paid_count", "candidate_count",
        "evidence_sha256", "ranking_sha256", "llm_requests", "explanation_status", "timing",
        *NormalAssurance.model_fields,
    }
    if set(result) != expected or result["schema_version"] != B0_RESULT or (
        result["identity"] != identity or identity.get("policy") != B0_POLICY
        or result["status"] != "completed" or type(result["llm_requests"]) is not int
        or result["llm_requests"] != 0 or result["explanation_status"] != "not_applicable"
        or any(result[k] != v for k, v in NormalAssurance().model_dump().items())
        or type(result["paid_count"]) is not int or not 0 <= result["paid_count"] <= 8
        or type(result["candidate_count"]) is not int or result["candidate_count"] <= 0
    ):
        raise ValueError("B0 completion identity/status/count differs")
    files = result["files"]
    if type(files) is not list or not 1 <= len(files) <= 96:
        raise ValueError("B0 completion file domain differs")
    entries: dict[str, dict[str, Any]] = {}
    for row in files:
        if type(row) is not dict or set(row) != {"name", "size_bytes", "sha256"}:
            raise ValueError("B0 member descriptor differs")
        name = row["name"]
        if type(name) is not str or Path(name).is_absolute() or "\\" in name or (
            any(p in ("", ".", "..") for p in name.split("/"))
            or type(row["size_bytes"]) is not int
            or not 0 <= row["size_bytes"] <= REPAIR_MAX_JSON_BYTES
            or name in entries or "ark" in name or "report" in name
            or name == "request-binding.json"
        ):
            raise ValueError("B0 unsafe or non-B0 member")
        if member_digest(root / name) != (row["size_bytes"], row["sha256"]):
            raise ValueError("B0 saved member integrity differs")
        entries[name] = row
    required = {"evidence.json", "evidence-receipt.json", "ranking.json",
                "ranking-receipt.json", "query-ledger/head"}
    if list(entries) != sorted(entries) or not required <= entries.keys():
        raise ValueError("B0 completion is missing ordered required artifacts")
    for name in ("evidence", "ranking"):
        receipt = _decode_json_member(_read_bytes(root / (name + "-receipt.json"), 16384))
        fields = {"evidence_sha256", "ranking_sha256", "candidate_count"}
        if name == "evidence":
            fields.add("paid_count")
        if set(receipt) != {"identity", "member", "body_schema", "summary"} or (
            receipt["identity"] != identity or receipt["body_schema"] != _SCHEMAS[name]
            or receipt["member"] != entries[name + ".json"]
            or receipt["summary"] != {k: result[k] for k in fields}
        ):
            raise ValueError("B0 completion summary/receipt differs")
    return result


def complete_b0(
    *, root: Path, identity: dict[str, Any], evidence: NormalEvidence,
    preparation_seconds: float, started: float,
) -> dict[str, Any]:
    files = _complete_files(root, b0_only=True)
    elapsed = monotonic() - started
    result = dict(
        schema_version=B0_RESULT, identity=identity, status="completed",
        **NormalAssurance().model_dump(), files=files,
        paid_count=evidence.session.paid_count, candidate_count=len(evidence.ranking.scores),
        evidence_sha256=evidence.result_sha256, ranking_sha256=evidence.ranking.result_sha256,
        llm_requests=0, explanation_status="not_applicable",
        timing=dict(preparation_seconds=preparation_seconds,
                    method_seconds=elapsed - preparation_seconds, elapsed_seconds=elapsed),
    )
    _save_json(root / "result.json", result)
    return result


def run_b0_case(*, plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    if plan.get("b0_only") is not True:
        raise ValueError("B0 requires its explicit admitted plan")
    return _run_case(plan=plan, plan_sha256=plan_sha256)
