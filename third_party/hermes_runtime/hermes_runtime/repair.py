# Derived from HERMES Agent 0.20.5 (agent/message_sanitization.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: deterministic JSON-object repair only; removed provider policies.
"""Conservative repair for malformed function-tool argument JSON."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from .contracts import ToolCall

_JSON_STRING_RE = re.compile(r'("(?:\\.|[^"\\])*")')


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(raw):
        character = raw[index]
        if in_string and character == "\\" and index + 1 < len(raw):
            output.extend((character, raw[index + 1]))
            index += 2
            continue
        if character == '"':
            in_string = not in_string
            output.append(character)
        elif in_string and ord(character) < 0x20:
            output.append(f"\\u{ord(character):04x}")
        else:
            output.append(character)
        index += 1
    return "".join(output)


def _canonical_object(raw: str) -> str | None:
    try:
        value = json.loads(raw, strict=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _replace_python_literals_outside_strings(raw: str) -> str:
    parts = _JSON_STRING_RE.split(raw)
    for index in range(0, len(parts), 2):
        segment = parts[index]
        segment = re.sub(r"\bNone\b", "null", segment)
        segment = re.sub(r"\bTrue\b", "true", segment)
        parts[index] = re.sub(r"\bFalse\b", "false", segment)
    return "".join(parts)


def repair_tool_call_arguments(raw_arguments: str, tool_name: str = "?") -> str:
    """Return canonical JSON-object arguments, using ``{}`` after bounded repairs fail."""

    del tool_name  # Names are intentionally not logged on this secret-sensitive path.
    stripped = raw_arguments.strip() if isinstance(raw_arguments, str) else ""
    if not stripped or stripped == "None":
        return "{}"

    repaired = _canonical_object(_escape_invalid_chars_in_json_strings(stripped))
    if repaired is not None:
        return repaired

    fixed = _replace_python_literals_outside_strings(stripped)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    open_braces = max(0, fixed.count("{") - fixed.count("}"))
    open_brackets = max(0, fixed.count("[") - fixed.count("]"))
    fixed = f"{fixed}{'}' * open_braces}{']' * open_brackets}"
    repaired = _canonical_object(_escape_invalid_chars_in_json_strings(fixed))
    return repaired if repaired is not None else "{}"


def deterministic_call_id(name: str, arguments: str, index: int = 0) -> str:
    """Construct a stable call ID without provider or process state."""

    digest = hashlib.sha256(f"{name}\0{arguments}\0{index}".encode()).hexdigest()[:24]
    return f"call_{digest}"


def uniquify_tool_call_ids(tool_calls: Sequence[ToolCall]) -> tuple[ToolCall, ...]:
    """Fill absent IDs and deterministically replace duplicates."""

    result: list[ToolCall] = []
    seen: set[str] = set()
    for index, tool_call in enumerate(tool_calls):
        call_id = tool_call.id or deterministic_call_id(
            tool_call.name, tool_call.arguments, index
        )
        collision = 0
        while call_id in seen:
            collision += 1
            call_id = deterministic_call_id(
                tool_call.name,
                tool_call.arguments,
                index + collision,
            )
        seen.add(call_id)
        result.append(
            ToolCall(
                id=call_id,
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
        )
    return tuple(result)
