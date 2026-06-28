"""State and trace contracts for the NURSE agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nurse.schemas import _assert_no_forbidden


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_dict(value: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(value or {})
    _assert_no_forbidden(payload)
    return payload


@dataclass
class AgentRequest:
    """A user request accepted by the NURSE agent console."""

    text: str
    run_id: str
    output_root: str
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        _assert_no_forbidden(payload)
        return payload


@dataclass
class AgentPlan:
    """The agent's interpreted intent and ordered tool plan."""

    intent: str
    tool_names: List[str]
    dataset: str = ""
    case_id: str = ""
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        _assert_no_forbidden(payload)
        return payload


@dataclass
class AgentToolCall:
    """A compact trace entry for one tool invocation."""

    tool_name: str
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    status: str = "ok"
    started_at: str = field(default_factory=_now)
    finished_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        _assert_no_forbidden(payload)
        return payload


@dataclass
class AgentTrace:
    """Append-only execution trace for one agent run."""

    run_id: str
    user_request: str
    tool_calls: List[AgentToolCall] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def record_tool_call(
        self,
        tool_name: str,
        input_summary: Dict[str, Any] | None,
        output_summary: Dict[str, Any] | None,
        *,
        status: str = "ok",
    ) -> AgentToolCall:
        call = AgentToolCall(
            tool_name=tool_name,
            input_summary=_safe_dict(input_summary),
            output_summary=_safe_dict(output_summary),
            status=status,
        )
        self.tool_calls.append(call)
        return call

    def record_decision(self, decision: Dict[str, Any]) -> None:
        self.decisions.append(_safe_dict(decision))

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "user_request": self.user_request,
            "created_at": self.created_at,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "decisions": list(self.decisions),
        }
        _assert_no_forbidden(payload)
        return payload


@dataclass
class AgentRunResult:
    """Return value for console/orchestrator runs."""

    run_id: str
    intent: str
    dataset: str
    trace: AgentTrace
    output_root: Path
    report_path: Optional[Path] = None
    report_json_path: Optional[Path] = None
    experiment_config: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "intent": self.intent,
            "dataset": self.dataset,
            "output_root": str(self.output_root),
            "report_path": str(self.report_path) if self.report_path else "",
            "report_json_path": str(self.report_json_path) if self.report_json_path else "",
            "experiment_config": dict(self.experiment_config),
            "artifact_paths": dict(self.artifact_paths),
            "trace": self.trace.to_dict(),
        }
        _assert_no_forbidden(payload)
        return payload
