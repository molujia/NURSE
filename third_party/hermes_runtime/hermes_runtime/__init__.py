"""Minimal HERMES-derived runtime embedded for NURSE.

Only the explicitly attributed modules in ``SOURCE_MANIFEST.json`` belong to
this package.  It is not a general HERMES compatibility layer.
"""

from .contracts import NormalizedResponse, ToolCall, Usage
from .errors import EmptyStreamError
from .loop import ConversationResult, run_conversation
from .runtime import AgentRunResult, EmbeddedAgentRuntime, RegisteredTool, RuntimeAuditEvent
from .transport import ChatCompletionsTransport, ProviderTransport
from .usage import CanonicalUsage, normalize_usage

__all__ = [
    "AgentRunResult",
    "CanonicalUsage",
    "ChatCompletionsTransport",
    "ConversationResult",
    "EmbeddedAgentRuntime",
    "EmptyStreamError",
    "NormalizedResponse",
    "ProviderTransport",
    "RegisteredTool",
    "RuntimeAuditEvent",
    "ToolCall",
    "Usage",
    "normalize_usage",
    "run_conversation",
]
