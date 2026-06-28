"""Executable SOP-style rules for Nurse evidence facts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .evidence_program import EvidenceFact
from .matching import canonical_entity


RULE_DIR = Path(__file__).resolve().parent / "rules"
DEFAULT_RULE_PATH = RULE_DIR / "default_sop_rules.json"


def _load_rules(rule_path: str | Path = DEFAULT_RULE_PATH) -> list[dict]:
    path = Path(rule_path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_contains(metric: str, tokens: Iterable[str]) -> bool:
    text = str(metric or "").lower()
    return any(str(token).lower() in text for token in tokens)


def _allowed_services(service_candidates: Iterable[str] | None) -> set[str]:
    return {
        canonical_entity(service)
        for service in (service_candidates or [])
        if canonical_entity(service)
    }


def apply_default_rules(
    facts: Iterable[EvidenceFact],
    *,
    service_candidates: Iterable[str] | None = None,
    rule_path: str | Path = DEFAULT_RULE_PATH,
) -> List[EvidenceFact]:
    fact_list = list(facts)
    rules = _load_rules(rule_path)
    allowed = _allowed_services(service_candidates)
    propagation = {
        (canonical_entity(fact.subject), canonical_entity(fact.object))
        for fact in fact_list
        if fact.predicate == "propagates_to"
    }
    emitted: List[EvidenceFact] = []

    for fact in fact_list:
        service = canonical_entity(fact.subject)
        role = str(fact.attrs.get("role") or "")
        if fact.predicate == "llm_proposal" and allowed and service not in allowed:
            emitted.append(
                EvidenceFact(
                    "contradiction",
                    fact.subject,
                    "unsupported_root",
                    1.0,
                    {"role": "unsupported_proposal", "source": "rule_bank"},
                )
            )
            continue

        for rule in rules:
            when = dict(rule.get("when") or {})
            emit = dict(rule.get("emit") or {})
            if when.get("role") and role != when["role"]:
                continue
            if when.get("metric_contains") and not _metric_contains(fact.object, when["metric_contains"]):
                continue
            if when.get("has_incoming_propagation"):
                if not any(dst == service for _, dst in propagation):
                    continue
            weight = max(float(fact.weight or 0.0), 0.0) * float(emit.get("weight_multiplier", 0.5))
            emitted.append(
                EvidenceFact(
                    str(emit.get("predicate") or "rule_support"),
                    fact.subject,
                    str(rule.get("rule_id") or fact.object),
                    weight,
                    {
                        "role": str(emit.get("role") or "rule_supports_root"),
                        "rule_id": str(rule.get("rule_id") or ""),
                        "source": "rule_bank",
                    },
                )
            )
    return emitted
