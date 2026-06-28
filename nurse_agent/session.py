"""Persistent conversational session state for the NURSE agent console."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from nurse.schemas import _assert_no_forbidden

from .orchestrator import (
    NurseAgentOrchestrator,
    infer_dataset,
    infer_metric_score_backend,
    is_full_benchmark_request,
)
from .state import AgentPlan, AgentRunResult


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_json_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    _assert_no_forbidden(payload)
    return payload


@dataclass
class SessionTurn:
    """One user-visible command handled by a NURSE agent session."""

    text: str
    intent: str
    dataset: str
    run_id: str
    metric_score_backend: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_json_dict(asdict(self))


@dataclass
class NurseSessionState:
    """Serializable memory shared by one interactive NURSE session."""

    session_id: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    command_history: List[Dict[str, Any]] = field(default_factory=list)
    last_intent: str = ""
    last_dataset: str = ""
    last_metric_score_backend: str = ""
    last_experiment_config: Dict[str, Any] = field(default_factory=dict)
    last_artifact_paths: Dict[str, str] = field(default_factory=dict)
    last_result: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, session_id: str) -> "NurseSessionState":
        data = dict(payload or {})
        data.setdefault("session_id", session_id)
        data.setdefault("created_at", _now())
        data.setdefault("updated_at", _now())
        data.setdefault("command_history", [])
        data.setdefault("last_intent", "")
        data.setdefault("last_dataset", "")
        data.setdefault("last_metric_score_backend", "")
        data.setdefault("last_experiment_config", {})
        data.setdefault("last_artifact_paths", {})
        data.setdefault("last_result", {})
        return cls(**data)

    def record_result(self, text: str, result: AgentRunResult) -> None:
        metric_backend = str(result.experiment_config.get("metric_score_backend") or self.last_metric_score_backend or "")
        turn = SessionTurn(
            text=text,
            intent=result.intent,
            dataset=result.dataset,
            run_id=result.run_id,
            metric_score_backend=metric_backend,
        )
        self.command_history.append(turn.to_dict())
        self.command_history = self.command_history[-100:]
        self.last_intent = result.intent
        self.last_dataset = result.dataset
        if metric_backend:
            self.last_metric_score_backend = metric_backend
        self.last_experiment_config = dict(result.experiment_config)
        self.last_artifact_paths = dict(result.artifact_paths)
        self.last_result = {
            "run_id": result.run_id,
            "intent": result.intent,
            "dataset": result.dataset,
            "output_root": str(result.output_root),
            "report_path": str(result.report_path) if result.report_path else "",
            "report_json_path": str(result.report_json_path) if result.report_json_path else "",
            "artifact_paths": dict(result.artifact_paths),
        }
        self.updated_at = _now()
        _assert_no_forbidden(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return _safe_json_dict(asdict(self))


class NurseAgentSession:
    """Interactive NURSE session with persistent command memory."""

    def __init__(
        self,
        *,
        output_root: str | Path = "outputs/agent_runs",
        session_id: str = "default",
        llm_required: bool = True,
        model: str | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.session_id = session_id
        self.session_dir = self.output_root / "agent_sessions" / session_id
        self.state_path = self.session_dir / "session_state.json"
        self.orchestrator = NurseAgentOrchestrator(
            output_root=self.output_root,
            llm_required=llm_required,
            model=model,
        )
        self.state = self._load_state()

    def run_command(self, text: str, *, dry_run: bool = False) -> AgentRunResult:
        plan = self.plan_with_memory(text, dry_run=dry_run)
        result = self.orchestrator.run_plan(text, plan)
        self.state.record_result(text, result)
        self.save()
        return result

    def plan_with_memory(self, text: str, *, dry_run: bool = False) -> AgentPlan:
        if self._is_follow_up(text) and self.state.last_intent == "full_benchmark":
            return self._follow_up_experiment_plan(text, dry_run=dry_run)
        plan = self.orchestrator.plan(text, dry_run=dry_run)
        if plan.intent == "full_benchmark" and self._should_inherit_backend(text):
            backend = self.state.last_metric_score_backend or plan.experiment_config.get("metric_score_backend") or "evidence_robust"
            plan.experiment_config["metric_score_backend"] = backend
        return plan

    def status_lines(self) -> List[str]:
        if not self.state.last_result:
            return [f"session_id={self.session_id}", "last_result=none"]
        lines = [
            f"session_id={self.session_id}",
            f"last_intent={self.state.last_intent}",
            f"last_dataset={self.state.last_dataset}",
            f"last_metric_score_backend={self.state.last_metric_score_backend or 'n/a'}",
            f"last_run_id={self.state.last_result.get('run_id', '')}",
        ]
        return lines

    def artifact_lines(self) -> List[str]:
        if not self.state.last_artifact_paths:
            return ["no artifacts yet"]
        return [f"{name}: {path}" for name, path in sorted(self.state.last_artifact_paths.items())]

    def memory_lines(self) -> List[str]:
        history = self.state.command_history[-10:]
        if not history:
            return ["no command history yet"]
        return [
            f"{turn['created_at']} | {turn['intent']} | {turn['dataset']} | {turn.get('metric_score_backend') or 'n/a'} | {turn['run_id']}"
            for turn in history
        ]

    def reset(self) -> None:
        self.state = NurseSessionState(session_id=self.session_id)
        self.save()

    def save(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_state(self) -> NurseSessionState:
        if not self.state_path.exists():
            return NurseSessionState(session_id=self.session_id)
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        _assert_no_forbidden(payload)
        return NurseSessionState.from_dict(payload, session_id=self.session_id)

    def _follow_up_experiment_plan(self, text: str, *, dry_run: bool) -> AgentPlan:
        previous = dict(self.state.last_experiment_config)
        dataset = infer_dataset(text, default=self.state.last_dataset or previous.get("dataset") or "rcaeval-tt")
        backend = infer_metric_score_backend(
            text,
            default=self.state.last_metric_score_backend or previous.get("metric_score_backend") or "evidence_robust",
        )
        config = {
            "dataset": dataset,
            "metric_score_backend": backend,
            "dry_run": dry_run,
            "shuffle": bool(previous.get("shuffle", True)),
            "seed": int(previous.get("seed", 20260621)),
            "workers": int(previous.get("workers", 6)),
            "methods": list(previous.get("methods") or ["nurse"]),
        }
        return AgentPlan(
            intent="full_benchmark",
            tool_names=["experiment_runner"],
            dataset=dataset,
            case_id="",
            experiment_config=config,
        )

    def _is_follow_up(self, text: str) -> bool:
        normalized = text.lower()
        return any(
            phrase in normalized
            for phrase in (
                "same configuration",
                "same config",
                "same settings",
                "same mode",
                "same backend",
                "that configuration",
                "that mode",
            )
        )

    def _should_inherit_backend(self, text: str) -> bool:
        normalized = text.lower()
        if not self.state.last_metric_score_backend:
            return False
        if infer_metric_score_backend(text, default=""):
            return False
        return is_full_benchmark_request(text) and any(
            phrase in normalized
            for phrase in ("same", "reuse", "again", "another dataset")
        )