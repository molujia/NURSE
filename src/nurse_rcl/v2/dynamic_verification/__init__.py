"""Private, deterministic dynamic verification; no LLM or registry side effects."""

from .contracts import VerificationSourceCatalog
from .source_catalog import build_verification_source_catalog, source_records_sha256

__all__ = [
    "VerificationSourceCatalog",
    "build_verification_source_catalog",
    "source_records_sha256",
]
