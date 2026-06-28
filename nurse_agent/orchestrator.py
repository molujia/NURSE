"""State-machine orchestrator for the agentic NURSE console."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

from nurse.schemas import _assert_no_forbidden

from .state import AgentPlan, AgentRunResult, AgentTrace
from .tools import (
    alarm_context_adapter_tool,
    evidence_admission_tool,
    evidence_candidates_builder_tool,
    evidence_graph_builder_tool,
    evidence_ranker_tool,
    experiment_runner_tool,
    incident_prior_tool,
    rcl_context_packager_tool,
    recovery_advice_tool,
    report_writer_tool,
)
from .tools.common import load_sample_case


ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


DATASET_ALIASES = {
    "aiops": "aiops25-retained-nonnode",
    "aiops25": "aiops25-retained-nonnode",
    "aiops25-retained-nonnode": "aiops25-retained-nonnode",
    "rcaeval-tt": "rcaeval-tt",
    "rcaeval tt": "rcaeval-tt",
    "tt": "rcaeval-tt",
    "trainticket": "rcaeval-tt",
    "train ticket": "rcaeval-tt",
}


SINGLE_CASE_TOOLS = [
    "alarm_context_adapter",
    "evidence_candidates_builder",
    "evidence_admission",
    "evidence_graph_builder",
    "evidence_ranker",
    "incident_prior",
    "rcl_context_packager",
    "recovery_advice",
    "report_writer",
]


def infer_dataset(text: str, *, default: str = "rcaeval-tt") -> str:
    normalized = text.lower().replace("_", "-")
    for alias, dataset in DATASET_ALIASES.items():
        if alias in normalized:
            return dataset
    return default


def infer_metric_score_backend(text: str, *, default: str = "evidence_robust") -> str:
    normalized = text.lower()
    if "alarm_fixed_window" in normalized or "ad reuse" in normalized or "ad-reuse" in normalized:
        return "alarm_fixed_window"
    if (
        "sliding-window" in normalized
        or "sliding window" in normalized
        or "3-sigma" in normalized
        or "3sigma" in normalized
        or "three-sigma" in normalized
        or "three sigma" in normalized
    ):
        return "sliding_3sigma_window"
    if "default evidence" in normalized or "default backend" in normalized:
        return "evidence_robust"
    return default


def is_full_benchmark_request(text: str) -> bool:
    normalized = text.lower()
    return any(
        phrase in normalized
        for phrase in (
            "full benchmark",
            "full experiment",
            "full aiops",
            "full rcaeval",
            "all cases",
            "benchmark",
            "alarm_fixed_window",
            "ad reuse",
            "sliding-window",
            "sliding window",
            "3-sigma",
            "3sigma",
        )
    )


class NurseAgentOrchestrator:
    """Small, explicit agent loop that plans and calls NURSE tools."""

    def __init__(
        self,
        *,
        output_root: str | Path = "outputs/agent_runs",
        llm_required: bool = True,
        model: str | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.llm_required = bool(llm_required)
        self.model = model or os.environ.get("NURSE_OPENAI_MODEL") or "gpt-5-nano"
        self._tools: Dict[str, ToolHandler] = {
            "alarm_context_adapter": alarm_context_adapter_tool.run,
            "evidence_candidates_builder": evidence_candidates_builder_tool.run,
            "evidence_admission": evidence_admission_tool.run,
            "evidence_graph_builder": evidence_graph_builder_tool.run,
            "evidence_ranker": evidence_ranker_tool.run,
            "incident_prior": incident_prior_tool.run,
            "rcl_context_packager": rcl_context_packager_tool.run,
            "recovery_advice": recovery_advice_tool.run,
            "report_writer": report_writer_tool.run,
            "experiment_runner": experiment_runner_tool.run,
        }

    def run_script(self, text: str, *, dry_run: bool = False) -> AgentRunResult:
        if self.llm_required and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "NURSE agent public mode requires OPENAI_API_KEY. "
                "Use llm_required=False only for internal tests."
            )
        plan = self.plan(text, dry_run=dry_run)
        return self.run_plan(text, plan)

    def run_plan(self, text: str, plan: AgentPlan) -> AgentRunResult:
        if plan.intent == "single_case":
            return self._run_single_case(text, plan)
        if plan.intent == "full_benchmark":
            return self._run_experiment(text, plan)
        raise ValueError(f"unsupported NURSE agent intent: {plan.intent}")

    def plan(self, text: str, *, dry_run: bool = False) -> AgentPlan:
        if is_full_benchmark_request(text):
            dataset = infer_dataset(text, default="rcaeval-tt")
            metric_score_backend = infer_metric_score_backend(text)
            return AgentPlan(
                intent="full_benchmark",
                tool_names=["experiment_runner"],
                dataset=dataset,
                case_id="",
                experiment_config={
                    "dataset": dataset,
                    "metric_score_backend": metric_score_backend,
                    "dry_run": dry_run,
                    "shuffle": True,
                    "seed": 20260621,
                    "workers": 6,
                    "methods": ["nurse"],
                },
            )
        normalized = text.lower()
        dataset = infer_dataset(text, default="aiops25-retained-nonnode")
        if dataset == "aiops25-retained-nonnode" or "single_case_1" in normalized or "case 1" in normalized:
            manifest = load_sample_case("single_case_1_aiops25.json")
        elif dataset == "rcaeval-tt" or "single_case_2" in normalized or "case 2" in normalized:
            manifest = load_sample_case("single_case_2_rcaeval_tt.json")
        else:
            manifest = load_sample_case("single_case_1_aiops25.json")
        return AgentPlan(
            intent="single_case",
            tool_names=list(SINGLE_CASE_TOOLS),
            dataset=str(manifest["dataset"]),
            case_id=str(manifest["case_id"]),
            experiment_config={"dry_run": dry_run},
        )

    def _run_single_case(self, text: str, plan: AgentPlan) -> AgentRunResult:
        run_id = self._new_run_id(plan)
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "agent_trace.json"
        trace = AgentTrace(run_id=run_id, user_request=text)
        trace.record_decision(
            {"intent": plan.intent, "dataset": plan.dataset, "case_id": plan.case_id, "tool_plan": list(plan.tool_names)}
        )

        manifest_name = (
            "single_case_2_rcaeval_tt.json"
            if plan.dataset == "rcaeval-tt"
            else "single_case_1_aiops25.json"
        )
        state = load_sample_case(manifest_name)
        state.update(
            {
                "output_dir": str(run_dir),
                "trace_path": str(trace_path),
                "model": self.model,
                "artifact_paths": {},
                "use_llm": self.llm_required,
            }
        )

        for tool_name in plan.tool_names:
            payload = self._payload_for(tool_name, state)
            output = self._tools[tool_name](payload)
            _assert_no_forbidden(output)
            trace.record_tool_call(
                tool_name,
                self._summarize(payload),
                self._summarize(output),
            )
            state.update(output)

        trace_path.write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifact_paths = dict(state.get("artifact_paths") or {})
        return AgentRunResult(
            run_id=run_id,
            intent=plan.intent,
            dataset=plan.dataset,
            trace=trace,
            output_root=run_dir,
            report_path=Path(artifact_paths["agent_report_markdown"]),
            report_json_path=Path(artifact_paths["agent_report_json"]),
            experiment_config=plan.experiment_config,
            artifact_paths=artifact_paths,
        )

    def _run_experiment(self, text: str, plan: AgentPlan) -> AgentRunResult:
        run_id = self._new_run_id(plan)
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "agent_trace.json"
        trace = AgentTrace(run_id=run_id, user_request=text)
        config = dict(plan.experiment_config)
        config.update({"output_dir": str(run_dir), "model": self.model})
        output = self._tools["experiment_runner"](config)
        trace.record_decision({"intent": plan.intent, "dataset": plan.dataset, "tool_plan": list(plan.tool_names)})
        trace.record_tool_call("experiment_runner", self._summarize(config), self._summarize(output))
        trace_path.write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifact_paths = dict(output.get("artifact_paths") or {})
        artifact_paths["agent_trace"] = trace_path.name
        return AgentRunResult(
            run_id=run_id,
            intent=plan.intent,
            dataset=plan.dataset,
            trace=trace,
            output_root=run_dir,
            experiment_config=dict(output.get("experiment_config") or plan.experiment_config),
            artifact_paths=artifact_paths,
        )

    def _payload_for(self, tool_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "alarm_context_adapter":
            keys = ["dataset", "case_id"]
        elif tool_name == "evidence_candidates_builder":
            keys = ["dataset", "case_id", "case_dir", "alarm", "metric_score_backend"]
        elif tool_name == "evidence_admission":
            keys = ["dataset", "case_id", "case_dir", "alarm", "service_candidates", "candidates"]
        elif tool_name == "evidence_graph_builder":
            keys = ["dataset", "case_id", "alarm", "service_candidates", "verified_evidence", "topology_edges"]
        elif tool_name == "evidence_ranker":
            keys = ["dataset", "case_id", "graph"]
        elif tool_name == "incident_prior":
            keys = ["dataset", "case_id", "service_candidates", "ranked", "history_index"]
        elif tool_name == "rcl_context_packager":
            keys = ["dataset", "case_id", "graph", "prediction", "rejected_items", "llm_advisory"]
        elif tool_name == "recovery_advice":
            keys = ["dataset", "case_id", "context", "model", "use_llm"]
        elif tool_name == "report_writer":
            keys = [
                "dataset",
                "case_id",
                "prediction",
                "context",
                "incident_prior",
                "recovery_advice",
                "trace_path",
                "artifact_paths",
                "output_dir",
                "model",
                "llm_explanation",
            ]
        else:
            raise KeyError(f"unknown tool: {tool_name}")
        return {key: state[key] for key in keys if key in state}

    def _new_run_id(self, plan: AgentPlan) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{stamp}_{plan.dataset}_{plan.case_id}".replace(":", "_").replace("/", "_")

    def _summarize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        passthrough = {
            "dataset",
            "case_id",
            "case_dir",
            "metric_score_backend",
            "top1",
            "ranked_count",
            "verified_count",
            "rejected_count",
            "candidate_count",
            "graph_nodes",
            "graph_relations",
            "evidence_summary_count",
        }
        for key in passthrough:
            if key in payload:
                summary[key] = payload[key]
        if "artifact_paths" in payload:
            summary["artifact_paths"] = dict(payload["artifact_paths"])
        if "experiment_config" in payload:
            config = dict(payload["experiment_config"])
            summary["experiment_config"] = {
                key: config[key]
                for key in ("dataset", "metric_score_backend", "dry_run", "workers")
                if key in config
            }
        if "incident_prior" in payload:
            prior = dict(payload["incident_prior"])
            summary["incident_prior"] = {
                "active": bool(prior.get("active")),
                "method": str(prior.get("method") or ""),
            }
        if "recovery_advice" in payload:
            summary["recovery_advice_count"] = len(payload["recovery_advice"])
        if "llm_explanation" in payload:
            summary["llm_explanation_present"] = bool(payload.get("llm_explanation"))
        if "prediction" in payload:
            ranked = list(payload["prediction"].get("ranked", []))
            summary["ranked_count"] = len(ranked)
            summary["top1"] = ranked[0] if ranked else ""
        _assert_no_forbidden(summary)
        return summary