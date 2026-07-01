"""Tool registry and contract checks for the NURSE agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol

from nurse.schemas import _assert_no_forbidden


ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


class NurseAgentTool(Protocol):
    name: str
    description: str

    def validate_input(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        ...

    def validate_output(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        ...

    def run(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        ...


def _validated_dict(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    data = dict(payload or {})
    _assert_no_forbidden(data)
    return data


@dataclass
class RegisteredTool:
    """A small typed tool wrapper used by the agent orchestrator."""

    name: str
    description: str
    handler: ToolHandler | None = None

    def validate_input(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return _validated_dict(payload)

    def validate_output(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return _validated_dict(payload)

    def run(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        validated = self.validate_input(payload)
        if self.handler is None:
            output = {"status": "not_implemented", "tool": self.name}
        else:
            output = self.handler(validated)
        return self.validate_output(output)


class ToolRegistry:
    """Ordered registry of NURSE agent tools."""

    def __init__(self, tools: Iterable[NurseAgentTool] = ()) -> None:
        self._tools: Dict[str, NurseAgentTool] = {}
        self._order: List[str] = []
        for tool in tools:
            self.register(tool)

    def register(self, tool: NurseAgentTool) -> None:
        if tool.name not in self._tools:
            self._order.append(tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> NurseAgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown NURSE agent tool: {name}") from exc

    def names(self) -> List[str]:
        return list(self._order)


def build_default_registry() -> ToolRegistry:
    """Build the paper-facing NURSE tool registry.

    Handlers are attached by the orchestrator layer. The registry itself is
    intentionally lightweight so contract tests can run without datasets.
    """

    tool_specs = [
        ("alarm_context_adapter", "Load or create the shared alarm artifact and benchmark window."),
        ("evidence_candidates_builder", "Build raw candidate evidence from configured telemetry sources."),
        ("evidence_admission", "Verify candidate legality, provenance, and raw-data grounding."),
        ("evidence_graph_builder", "Build the typed verified Evidence Graph and propagation context."),
        ("evidence_ranker", "Compute Top-K ranking and E/P/H/R score breakdown."),
        ("incident_prior", "Compute H: historical-case prior from approved historical incident material."),
        ("rcl_context_packager", "Package Verified RCL Context for the User Manager."),
        ("recovery_advice", "Generate grounded recovery advice for SRE action."),
        ("experiment_runner", "Launch or dry-run benchmark experiments with explicit configs."),
        ("report_writer", "Write JSON, Markdown, and trace artifacts."),
        ("tool_authoring", "Draft, test, and approval-gate evidence-generation tools."),
    ]
    return ToolRegistry(
        RegisteredTool(name=name, description=description) for name, description in tool_specs
    )
