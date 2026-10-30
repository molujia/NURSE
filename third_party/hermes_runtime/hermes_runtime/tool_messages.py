# Derived from HERMES Agent 0.20.5 (agent/tool_dispatch_helpers.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: removed generic dispatch, timestamps, multimodal and web/MCP wrapping.
"""Canonical OpenAI-compatible tool-result message construction."""

from __future__ import annotations

import json
from typing import Any


def make_tool_result_message(
    name: str,
    content: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Build the only tool-result shape accepted by the embedded loop."""

    del name  # Tool identity is already bound by tool_call_id on the wire.
    canonical_id = tool_call_id.split("|", 1)[0].strip()
    serialized_content = (
        content
        if isinstance(content, str)
        else json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return {
        "role": "tool",
        "content": serialized_content,
        "tool_call_id": canonical_id,
    }
