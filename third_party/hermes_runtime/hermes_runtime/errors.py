# Derived from HERMES Agent 0.20.5 (agent/errors.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: retained only the empty-response error used by NURSE.
"""Narrow errors retained from the HERMES runtime."""


class EmptyStreamError(RuntimeError):
    """Raised when a provider returns no usable response."""
