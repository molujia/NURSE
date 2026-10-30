"""Finite D10 ranking-mutation diagnostics; not a natural-language truth verifier."""

# ruff: noqa: RUF001 -- intentional Chinese clause punctuation

from __future__ import annotations

import re

from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_presentation_schemas import claim_free_text_fields

FROZEN_RANKING_POLICY = "sre-frozen-ranking/1"
_CLAUSE = re.compile(r"[。！？!?；;，,\n]|(?<!\d)\.(?!\d)|但是|然而|但|而|\bbut\b", re.I)
_SUBJECT = r"(?:排名(?:区分度)?|排序|正式分数|\brank(?:ing)?\b|\bformal\s+score\b)"
_CHANGE = r"(?:上升|下降|提高|降低|增强|减弱|改变|变化|调整|更新|重排|提升|增加|减少|" \
          r"\b(?:change|increase|decrease|rise|fall|improve|update|reorder)(?:s|d)?\b)"
_MUTATION = re.compile(
    rf"{_SUBJECT}.{{0,24}}?(?P<after>{_CHANGE})|(?P<before>{_CHANGE}).{{0,24}}?{_SUBJECT}", re.I)
_NEGATED = re.compile(
    r"(?:不会|不能|不应|不再|不|未|无需|没有|不可|并不|"
    r"\bnot\s+|\bnever\s+|\bcannot\s+|\bno\s+longer\s+)"
    r"(?:会|可能|能够|需要|打算|实际|直接|间接|\s|directly\s+|indirectly\s+)*$", re.I)
_DOUBLE_NEGATION = re.compile(r"并非|不是|不保证|不能保证|不意味着|not\s+(?:true|guaranteed)", re.I)


def frozen_ranking_fields(claim: ReportCompositionClaim) -> tuple[str, ...]:
    """Return original field paths; no negation in another clause cancels a hit."""
    hits = []
    for path, text in claim_free_text_fields(claim):
        violation = False
        for clause in _CLAUSE.split(text):
            for match in _MUTATION.finditer(clause):
                verb = match.start("after") if match.group("after") else match.start("before")
                prefix = clause[:verb]
                if not _NEGATED.search(prefix) or _DOUBLE_NEGATION.search(prefix):
                    violation = True
                    break
            if violation:
                break
        if violation:
            hits.append(path)
    return tuple(hits)
