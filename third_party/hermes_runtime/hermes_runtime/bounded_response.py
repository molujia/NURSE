# Derived from HERMES Agent 0.20.5 (agent/bounded_response.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: modern typing and no module logger; behavior remains bounded and best-effort.
"""Bounded reads of HTTP streaming error bodies."""

from __future__ import annotations

import threading
from contextlib import suppress

import httpx

DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
DEFAULT_ERROR_BODY_TIMEOUT_S = 10.0


def _safe_close(response: httpx.Response) -> None:
    with suppress(Exception):
        response.close()


def read_streaming_error_body(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> str:
    """Return a best-effort UTF-8 error snippet under byte and wall-clock bounds."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")

    chunks: list[bytes] = []
    done = threading.Event()

    def drain() -> None:
        total = 0
        try:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                accepted = chunk[:remaining]
                chunks.append(accepted)
                total += len(accepted)
                if len(accepted) < len(chunk):
                    break
        except Exception:
            pass
        finally:
            done.set()

    worker = threading.Thread(target=drain, name="bounded-error-body-read", daemon=True)
    worker.start()
    done.wait(timeout=timeout_s)
    _safe_close(response)
    return b"".join(chunks).decode("utf-8", errors="replace")


def read_error_body_or_default(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> str | None:
    """Return ``None`` when the bounded error body is empty."""

    text = read_streaming_error_body(
        response,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
    )
    return text or None
