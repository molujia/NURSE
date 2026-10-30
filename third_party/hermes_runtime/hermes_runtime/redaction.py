# Derived from HERMES Agent 0.20.5 (agent/redact.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: fixed runtime-only masking; removed plugins, terminal and browser paths.
"""Small secret-redaction boundary for runtime diagnostics and artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_DISPLAY_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f\x80-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064]"
)
_AUTHORIZATION_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s,;]+)")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)\b\s*=\s*)([^\s]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-|ark-|ghp_|github_pat_)[A-Za-z0-9._-]{8,}"
)


def mask_secret(
    value: str,
    *,
    head: int = 4,
    tail: int = 4,
    floor: int = 12,
    placeholder: str = "***",
    empty: str = "",
) -> str:
    """Mask a secret for display while removing control characters."""

    if not value:
        return empty
    cleaned = _DISPLAY_CONTROL_RE.sub("", value)
    if not cleaned:
        return empty
    if len(cleaned) < floor:
        return placeholder
    return f"{cleaned[:head]}...{cleaned[-tail:]}"


def redact_sensitive_text(text: str, *, secrets: Iterable[str] = ()) -> str:
    """Redact explicit in-memory secrets and common credential-shaped text."""

    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)
    redacted = text
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        redacted = redacted.replace(secret, "***")
    redacted = _AUTHORIZATION_RE.sub(r"\1***", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1***", redacted)
    return _KNOWN_TOKEN_RE.sub("***", redacted)


def redact_value(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    """Recursively redact strings while preserving JSON-compatible structure."""

    frozen_secrets = tuple(secrets)
    if isinstance(value, str):
        return redact_sensitive_text(value, secrets=frozen_secrets)
    if isinstance(value, Mapping):
        return {
            key: redact_value(item, secrets=frozen_secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, secrets=frozen_secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets=frozen_secrets) for item in value)
    return value
