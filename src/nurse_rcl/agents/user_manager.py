"""Closed orchestration and report prompts for the NURSE User Manager."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nurse_rcl.agents.report import agent_report_submission_schema
from nurse_rcl.agents.user_manager_state import (
    UserManagerCheckpoint,
    UserManagerState,
    UserManagerTool,
    authorize_user_manager_tool,
)
from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.methods.paper_v1.verified_context import VerifiedRCLContextArtifact
from nurse_rcl.schemas import StrictFrozenModel

USER_MANAGER_ORCHESTRATION_PROMPT_VERSION: Literal["user_manager_orchestration_v1"] = (
    "user_manager_orchestration_v1"
)
USER_MANAGER_REPORT_PROMPT_VERSION: Literal["user_manager_report_v1"] = (
    "user_manager_report_v1"
)

USER_MANAGER_ORCHESTRATION_SYSTEM_PROMPT = " ".join(
    (
        "You are the NURSE User Manager in pre-ranking orchestration mode.",
        "Use the hash-bound case state in the user message.",
        "Make exactly one authorized tool call from the single supplied definition.",
        "Do not create evidence. Do not assign ranks or scores.",
        "Do not bypass, repeat, or reorder deterministic stages.",
        "Do not read files, execute commands, use a browser, search, or access memory.",
        "Do not delegate work or use labels. Do not call external services.",
        "Do not emit a report in orchestration mode.",
    )
)

USER_MANAGER_REPORT_SYSTEM_PROMPT = " ".join(
    (
        "You are the NURSE User Manager in report mode.",
        "Use only the Verified RCL Context in the user message.",
        "Do not reorder Top-K. Do not modify scores or score components.",
        "Do not add candidates. Do not invent evidence or evidence paths.",
        "Label recommendations unsupported unless directly supported by context evidence.",
        "Make exactly one write_report call.",
        "Do not read files, use labels, or call external services.",
    )
)


class UserManagerMessage(StrictFrozenModel):
    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class UserManagerToolDefinition(StrictFrozenModel):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class UserManagerOrchestrationRequest(StrictFrozenModel):
    prompt_version: Literal["user_manager_orchestration_v1"] = (
        USER_MANAGER_ORCHESTRATION_PROMPT_VERSION
    )
    mode: Literal["orchestration"] = "orchestration"
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    messages: tuple[UserManagerMessage, UserManagerMessage]
    tools: tuple[UserManagerToolDefinition]


class UserManagerReportRequest(StrictFrozenModel):
    prompt_version: Literal["user_manager_report_v1"] = USER_MANAGER_REPORT_PROMPT_VERSION
    mode: Literal["report"] = "report"
    verified_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    messages: tuple[UserManagerMessage, UserManagerMessage]
    tools: tuple[UserManagerToolDefinition]


def build_user_manager_orchestration_request(
    *,
    checkpoint: UserManagerCheckpoint,
    input_artifact_hash: str,
) -> UserManagerOrchestrationRequest:
    """Expose exactly the one tool authorized by a verified pre-ranking checkpoint."""

    if not checkpoint.verify():
        raise ValueError("User Manager checkpoint verification failed")
    if checkpoint.state in {
        UserManagerState.RANKING_FROZEN,
        UserManagerState.REPORT_WRITTEN,
    }:
        raise ValueError("orchestration requests require a pre-ranking state")
    allowed_tools = tuple(
        tool
        for tool in UserManagerTool
        if authorize_user_manager_tool(state=checkpoint.state, tool=tool).accepted
    )
    if len(allowed_tools) != 1:  # pragma: no cover - fixed transition-table invariant
        raise AssertionError("pre-ranking state must authorize exactly one tool")
    user_document = {
        "case_id": checkpoint.case_id,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "input_artifact_hash": input_artifact_hash,
        "state": checkpoint.state.value,
    }
    return UserManagerOrchestrationRequest(
        checkpoint_hash=checkpoint.checkpoint_hash,
        input_artifact_hash=input_artifact_hash,
        messages=(
            UserManagerMessage(
                role="system",
                content=USER_MANAGER_ORCHESTRATION_SYSTEM_PROMPT,
            ),
            UserManagerMessage(
                role="user",
                content=canonical_json_bytes(user_document).decode("utf-8"),
            ),
        ),
        tools=(
            UserManagerToolDefinition.model_validate(
                _orchestration_tool_definition(allowed_tools[0])
            ),
        ),
    )


def build_user_manager_report_request(
    *,
    checkpoint: UserManagerCheckpoint,
    verified_context: VerifiedRCLContextArtifact,
) -> UserManagerReportRequest:
    """Bind the report agent to only one verified frozen context and one report tool."""

    if not checkpoint.verify():
        raise ValueError("User Manager checkpoint verification failed")
    if checkpoint.state != UserManagerState.RANKING_FROZEN:
        raise ValueError("report requests require the RANKING_FROZEN state")
    if not verified_context.verify():
        raise ValueError("Verified RCL context hash verification failed")
    if verified_context.context.case_id != checkpoint.case_id:
        raise ValueError("Verified RCL context case identity does not match checkpoint")
    user_document = {
        "verified_context_hash": verified_context.context_hash,
        "verified_rcl_context": verified_context.context.model_dump(mode="json"),
    }
    return UserManagerReportRequest(
        verified_context_hash=verified_context.context_hash,
        messages=(
            UserManagerMessage(
                role="system",
                content=USER_MANAGER_REPORT_SYSTEM_PROMPT,
            ),
            UserManagerMessage(
                role="user",
                content=canonical_json_bytes(user_document).decode("utf-8"),
            ),
        ),
        tools=(UserManagerToolDefinition.model_validate(_report_tool_definition()),),
    )


def _orchestration_tool_definition(tool: UserManagerTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.value,
            "description": "Run the single deterministic stage authorized by the case state.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "checkpoint_hash": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "input_artifact_hash": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                },
                "required": ["checkpoint_hash", "input_artifact_hash"],
            },
        },
    }


def _report_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": UserManagerTool.WRITE_REPORT.value,
            "description": "Submit a bounded report for deterministic verification.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **agent_report_submission_schema().get("properties", {}),
                },
                "required": agent_report_submission_schema().get("required", []),
                "$defs": agent_report_submission_schema().get("$defs", {}),
            },
        },
    }
