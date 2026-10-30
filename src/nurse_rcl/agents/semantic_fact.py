"""Closed prompt and tool contract for the Semantic Fact Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact
from nurse_rcl.models import SemanticFact
from nurse_rcl.schemas import StrictFrozenModel

SEMANTIC_FACT_PROMPT_VERSION: Literal["semantic_fact_v1_1"] = "semantic_fact_v1_1"
SUBMIT_SEMANTIC_FACTS_TOOL = "submit_semantic_facts"

SEMANTIC_FACT_SYSTEM_PROMPT = """You are the NURSE Semantic Fact Agent.
Use only the hash-bound semantic payload in the user message. Every service must be an
existing candidate and every cited evidence ID must be existing evidence owned by that
candidate. Submit only root_support, source_support, or symptom_support facts by calling
submit_semantic_facts. Do not assign ranks or scores. Do not add candidates. Do not invent
or quote free-text evidence. Do not use labels, ground truth, injection records, or answer
keys. The relation shape is strict:
root_support must set related_service and related_evidence_id to null.
source_support requires both related fields; its primary
service/evidence is the earlier source and its related service/evidence is the later
symptom. symptom_support requires both related fields; its primary service/evidence is
the later symptom and its related service/evidence is the earlier source.
Do not submit a source_support or symptom_support fact without both related fields set to non-empty
payload identifiers. A single-candidate payload cannot support a source/symptom relation;
use root_support only when defensible, otherwise submit an empty facts array.
Do not read files, execute commands, use a browser, access memory, or delegate work.
Do not call external services. Make exactly one submit_semantic_facts call;
an empty facts array is permitted when no defensible fact exists.
"""


class SemanticFactSubmission(StrictFrozenModel):
    """The only accepted tool-argument envelope from the semantic agent."""

    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    facts: tuple[SemanticFact, ...]


class SemanticAgentMessage(StrictFrozenModel):
    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class SemanticAgentTool(StrictFrozenModel):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class SemanticFactAgentRequest(StrictFrozenModel):
    """Deterministic request surface passed to the embedded runtime."""

    prompt_version: Literal["semantic_fact_v1_1"] = SEMANTIC_FACT_PROMPT_VERSION
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    messages: tuple[SemanticAgentMessage, SemanticAgentMessage]
    tools: tuple[SemanticAgentTool]


def _fact_schema() -> dict[str, Any]:
    nullable_string = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "null"},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "fact_type": {
                "type": "string",
                "enum": [
                    "root_support",
                    "source_support",
                    "symptom_support",
                ],
            },
            "service": {"type": "string", "minLength": 1},
            "evidence_id": {"type": "string", "minLength": 1},
            "related_service": nullable_string,
            "related_evidence_id": nullable_string,
        },
        "required": [
            "fact_type",
            "service",
            "evidence_id",
            "related_service",
            "related_evidence_id",
        ],
        "oneOf": [
            {
                "properties": {
                    "fact_type": {"const": "root_support"},
                    "related_service": {"type": "null"},
                    "related_evidence_id": {"type": "null"},
                }
            },
            {
                "properties": {
                    "fact_type": {
                        "enum": ["source_support", "symptom_support"],
                    },
                    "related_service": {"type": "string", "minLength": 1},
                    "related_evidence_id": {"type": "string", "minLength": 1},
                }
            },
        ],
    }


def semantic_fact_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return the single immutable-by-convention function declaration."""

    return (
        {
            "type": "function",
            "function": {
                "name": SUBMIT_SEMANTIC_FACTS_TOOL,
                "description": (
                    "Submit candidate-scoped semantic facts for deterministic guard validation."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "payload_hash": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "facts": {
                            "type": "array",
                            "items": _fact_schema(),
                        },
                    },
                    "required": ["payload_hash", "facts"],
                },
            },
        },
    )


def build_semantic_fact_agent_request(
    artifact: SemanticPayloadArtifact,
) -> SemanticFactAgentRequest:
    """Bind the deterministic prompt and sole tool to one verified payload hash."""

    if not artifact.verify():
        raise ValueError("semantic payload hash verification failed")
    user_document = {
        "payload_hash": artifact.payload_hash,
        "semantic_payload": artifact.payload.model_dump(mode="json"),
    }
    user_content = canonical_json_bytes(user_document).decode("utf-8")
    tools = (
        SemanticAgentTool.model_validate(semantic_fact_tool_definitions()[0]),
    )
    return SemanticFactAgentRequest(
        payload_hash=artifact.payload_hash,
        messages=(
            SemanticAgentMessage(
                role="system",
                content=SEMANTIC_FACT_SYSTEM_PROMPT,
            ),
            SemanticAgentMessage(role="user", content=user_content),
        ),
        tools=tools,
    )
