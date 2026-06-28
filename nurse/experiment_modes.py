"""Release experiment-mode switches for NURSE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class NurseExperimentMode:
    name: str
    description: str
    call_llm: bool = True
    use_llm_candidates: bool = True
    verify_llm_candidates: bool = True
    use_llm_hypotheses: bool = True
    use_llm_semantic_prior: bool = True
    use_llm_rank_repair: bool = True
    include_llm_proposal_in_graph: bool = True
    symbolic_score_enabled: bool = True
    unsafe: bool = False
    ranker_overrides: Dict[str, Any] = field(default_factory=dict)
    prompt_profile: str = "balanced"

    def ranker_kwargs(self, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        kwargs = dict(self.ranker_overrides)
        kwargs["allow_llm_scoring"] = bool(self.use_llm_hypotheses or self.use_llm_semantic_prior)
        kwargs["symbolic_score_enabled"] = bool(self.symbolic_score_enabled)
        if not self.use_llm_semantic_prior:
            kwargs["llm_semantic_prior_weight"] = 0.0
        kwargs.update(overrides or {})
        return kwargs

    def sanitize_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(proposal or {})
        if not self.use_llm_candidates:
            sanitized["candidates"] = []
        if not self.use_llm_hypotheses:
            sanitized["hypotheses"] = []
            sanitized["candidate_rules"] = []
        if not self.use_llm_semantic_prior:
            sanitized["semantic_facts"] = []
        if not self.use_llm_rank_repair:
            sanitized["rank_repair"] = []
        if not self.include_llm_proposal_in_graph:
            return {}
        return sanitized


_MODES: Dict[str, NurseExperimentMode] = {
    "full": NurseExperimentMode(
        name="full",
        description="Default NURSE mode: call the LLM before final ranking so guarded semantic facts can participate in S.",
        use_llm_candidates=True,
        use_llm_hypotheses=True,
        use_llm_semantic_prior=True,
        use_llm_rank_repair=False,
        include_llm_proposal_in_graph=True,
        symbolic_score_enabled=True,
    ),
    "wo_S": NurseExperimentMode(
        name="wo_S",
        description="Disable symbolic rules and guarded semantic rank facts. The LLM may still be used later for explanation only.",
        use_llm_candidates=False,
        use_llm_hypotheses=False,
        use_llm_semantic_prior=False,
        use_llm_rank_repair=False,
        include_llm_proposal_in_graph=False,
        symbolic_score_enabled=False,
    ),
    "explanation_only": NurseExperimentMode(
        name="explanation_only",
        description="Call the LLM only for grounded explanation and recovery advice; no LLM output may affect ranking.",
        use_llm_candidates=False,
        use_llm_hypotheses=False,
        use_llm_semantic_prior=False,
        use_llm_rank_repair=False,
        include_llm_proposal_in_graph=False,
        symbolic_score_enabled=False,
    ),
}


def experiment_mode_choices() -> List[str]:
    return sorted(_MODES)


def load_experiment_mode(name: str | None) -> NurseExperimentMode:
    mode_name = str(name or "full").strip()
    if mode_name in {"default", "release"}:
        mode_name = "full"
    try:
        return _MODES[mode_name]
    except KeyError as exc:
        raise ValueError(f"unsupported NURSE experiment mode: {mode_name}") from exc


def parse_ranker_overrides(value: str | None) -> Dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    maybe_path = Path(text)
    if maybe_path.exists():
        return json.loads(maybe_path.read_text(encoding="utf-8-sig"))
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("ranker overrides must be a JSON object")
    return parsed
