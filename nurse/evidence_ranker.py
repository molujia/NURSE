"""Paper-facing Evidence Ranker wrapper for NURSE."""

from __future__ import annotations

from .ranker import NurseRanker


class EvidenceRanker(NurseRanker):
    """Alias wrapper that exposes the paper module name in code."""

