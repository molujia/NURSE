"""Guardrailed evidence-tool authoring support for the NURSE agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from nurse.schemas import FORBIDDEN_INFERENCE_FIELDS, _assert_no_forbidden


def _reject_forbidden_source(source: str) -> None:
    lowered = source.lower()
    for token in FORBIDDEN_INFERENCE_FIELDS:
        if token.lower() in lowered:
            raise ValueError(f"forbidden inference field `{token}` in generated tool source")


def _state_path(draft_dir: Path, tool_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tool_id)
    return draft_dir / f"{safe_id}.json"


def _load_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    _assert_no_forbidden({k: v for k, v in payload.items() if k != "source"})
    action = str(payload.get("action") or "draft")
    tool_id = str(payload["tool_id"])
    draft_dir = Path(str(payload.get("draft_dir") or "outputs/agent_tool_drafts"))
    draft_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(draft_dir, tool_id)

    if action == "draft":
        source = str(payload.get("source") or "")
        _reject_forbidden_source(source)
        contract_ok = bool(payload.get("candidate_schema_valid", False))
        smoke_ok = bool(payload.get("smoke_test_passed", False))
        record = {
            "tool_id": tool_id,
            "approved": False,
            "contract_check_passed": contract_ok,
            "smoke_test_passed": smoke_ok,
            "ranking_status": "advisory_not_ranked",
            "state": "draft",
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    if action == "approve":
        previous = _load_record(path)
        contract_ok = bool(payload.get("contract_check_passed", previous.get("contract_check_passed", False)))
        smoke_ok = bool(payload.get("smoke_test_passed", previous.get("smoke_test_passed", False)))
        approved = contract_ok and smoke_ok
        record = {
            "tool_id": tool_id,
            "approved": approved,
            "contract_check_passed": contract_ok,
            "smoke_test_passed": smoke_ok,
            "ranking_status": "approved_for_ranking" if approved else "advisory_not_ranked",
            "state": "approved" if approved else "blocked",
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    if action == "reject":
        record = {
            "tool_id": tool_id,
            "approved": False,
            "ranking_status": "advisory_not_ranked",
            "state": "rejected",
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    raise ValueError(f"unsupported tool-authoring action: {action}")
