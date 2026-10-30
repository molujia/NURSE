"""Application adapters that run both NURSE agents through the embedded runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from hermes_runtime.runtime import AgentRunResult, EmbeddedAgentRuntime, RegisteredTool
from pydantic import Field

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.agents.report import (
    AgentReportSubmission,
    VerifiedAgentReportArtifact,
    finalize_user_manager_report,
)
from nurse_rcl.agents.semantic_fact import (
    SemanticFactSubmission,
    build_semantic_fact_agent_request,
)
from nurse_rcl.agents.user_manager import (
    UserManagerToolDefinition,
    build_user_manager_orchestration_request,
    build_user_manager_report_request,
)
from nurse_rcl.agents.user_manager_state import (
    UserManagerCheckpoint,
    UserManagerTool,
    apply_user_manager_tool,
)
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact
from nurse_rcl.methods.paper_v1.verified_context import VerifiedRCLContextArtifact
from nurse_rcl.schemas import StrictFrozenModel

CompletionFunction = Callable[[dict[str, Any]], Any]


class StageToolOutput(StrictFrozenModel):
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: dict[str, Any]


StageHandler = Callable[
    [UserManagerTool, dict[str, Any]],
    StageToolOutput | Mapping[str, object],
]


@dataclass(frozen=True, slots=True)
class UserManagerToolStepResult:
    tool: UserManagerTool
    output: StageToolOutput
    checkpoint: UserManagerCheckpoint
    agent_run: AgentRunResult


@dataclass(frozen=True, slots=True)
class SemanticFactAgentLoopResult:
    submission: SemanticFactSubmission
    submission_hash: str
    agent_run: AgentRunResult


@dataclass(frozen=True, slots=True)
class UserManagerReportAgentResult:
    report: VerifiedAgentReportArtifact
    checkpoint: UserManagerCheckpoint
    agent_run: AgentRunResult


def run_user_manager_tool_step(
    *,
    checkpoint: UserManagerCheckpoint,
    input_artifact_hash: str,
    complete: CompletionFunction,
    handler: StageHandler,
) -> UserManagerToolStepResult:
    """Run exactly one state-authorized User Manager tool through HERMES."""

    request = build_user_manager_orchestration_request(
        checkpoint=checkpoint,
        input_artifact_hash=input_artifact_hash,
    )
    definition = request.tools[0]
    tool = UserManagerTool(_function_name(definition))
    expected_arguments = {
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "input_artifact_hash": input_artifact_hash,
    }
    captured: list[StageToolOutput] = []

    def dispatch(arguments: dict[str, Any]) -> dict[str, Any]:
        if captured:
            raise ValueError("User Manager tool may execute only once per request")
        if arguments != expected_arguments:
            raise ValueError("User Manager tool arguments do not match frozen request hashes")
        output = StageToolOutput.model_validate(handler(tool, arguments))
        captured.append(output)
        return output.model_dump(mode="json")

    agent_run = _run_agent(
        messages=tuple(message.model_dump(mode="json") for message in request.messages),
        definition=definition,
        complete=complete,
        handler=dispatch,
    )
    output = _require_single_execution(captured, agent_run=agent_run, tool=tool.value)
    updated_checkpoint = apply_user_manager_tool(
        checkpoint=checkpoint,
        tool=tool,
        output_artifact_hash=output.artifact_hash,
    )
    return UserManagerToolStepResult(
        tool=tool,
        output=output,
        checkpoint=updated_checkpoint,
        agent_run=agent_run,
    )


def run_semantic_fact_agent(
    *,
    artifact: SemanticPayloadArtifact,
    complete: CompletionFunction,
) -> SemanticFactAgentLoopResult:
    """Run the closed Semantic Fact Agent tool loop and hash its submission."""

    request = build_semantic_fact_agent_request(artifact)
    definition = UserManagerToolDefinition.model_validate(
        request.tools[0].model_dump(mode="json")
    )
    captured: list[SemanticFactSubmission] = []

    def dispatch(arguments: dict[str, Any]) -> dict[str, Any]:
        if captured:
            raise ValueError("Semantic Fact Agent may submit only once per request")
        submission = SemanticFactSubmission.model_validate(arguments)
        if submission.payload_hash != artifact.payload_hash:
            raise ValueError("semantic submission payload hash does not match request")
        captured.append(submission)
        submission_hash = _model_hash(submission)
        return {
            "fact_count": len(submission.facts),
            "payload_hash": submission.payload_hash,
            "submission_hash": submission_hash,
        }

    agent_run = _run_agent(
        messages=tuple(message.model_dump(mode="json") for message in request.messages),
        definition=definition,
        complete=complete,
        handler=dispatch,
    )
    submission = _require_single_execution(
        captured,
        agent_run=agent_run,
        tool=_function_name(definition),
    )
    return SemanticFactAgentLoopResult(
        submission=submission,
        submission_hash=_model_hash(submission),
        agent_run=agent_run,
    )


def run_user_manager_report_agent(
    *,
    checkpoint: UserManagerCheckpoint,
    context: VerifiedRCLContextArtifact,
    complete: CompletionFunction,
) -> UserManagerReportAgentResult:
    """Run the report tool loop, verify its submission, and close the state chain."""

    request = build_user_manager_report_request(
        checkpoint=checkpoint,
        verified_context=context,
    )
    definition = request.tools[0]
    captured: list[VerifiedAgentReportArtifact] = []

    def dispatch(arguments: dict[str, Any]) -> dict[str, Any]:
        if captured:
            raise ValueError("User Manager report tool may execute only once per request")
        submission = AgentReportSubmission.model_validate(arguments)
        report = finalize_user_manager_report(
            context=context,
            submission=submission,
        )
        captured.append(report)
        return {
            "artifact_hash": report.artifact_hash,
            "source": report.source,
            "verification_code": report.verification_code,
        }

    agent_run = _run_agent(
        messages=tuple(message.model_dump(mode="json") for message in request.messages),
        definition=definition,
        complete=complete,
        handler=dispatch,
    )
    report = _require_single_execution(
        captured,
        agent_run=agent_run,
        tool=UserManagerTool.WRITE_REPORT.value,
    )
    updated_checkpoint = apply_user_manager_tool(
        checkpoint=checkpoint,
        tool=UserManagerTool.WRITE_REPORT,
        output_artifact_hash=report.artifact_hash,
    )
    return UserManagerReportAgentResult(
        report=report,
        checkpoint=updated_checkpoint,
        agent_run=agent_run,
    )


def _run_agent(
    *,
    messages: tuple[dict[str, Any], ...],
    definition: UserManagerToolDefinition,
    complete: CompletionFunction,
    handler: Callable[[dict[str, Any]], Any],
) -> AgentRunResult:
    function = definition.function
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("agent tool parameters must be a JSON object")
    registered = RegisteredTool(
        name=_function_name(definition),
        description=str(function.get("description", "")),
        parameters=parameters,
        handler=handler,
    )
    runtime = EmbeddedAgentRuntime(
        model=ARK_MODEL,
        complete=complete,
        tools=(registered,),
    )
    return runtime.run(messages=messages)


def _function_name(definition: UserManagerToolDefinition) -> str:
    name = definition.function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("agent tool definition requires a function name")
    return name


def _require_single_execution(
    captured: list[Any],
    *,
    agent_run: AgentRunResult,
    tool: str,
) -> Any:
    executed = tuple(
        event
        for event in agent_run.audit_events
        if event.code == "tool_executed" and event.tool_name == tool
    )
    if len(captured) != 1 or len(executed) != 1 or len(agent_run.audit_events) != 1:
        raise ValueError(f"agent must execute exactly one authorized {tool} call")
    return captured[0]


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
