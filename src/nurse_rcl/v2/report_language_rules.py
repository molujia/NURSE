"""Finite, source-bound language diagnostics; passing never proves semantic correctness."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_evidence_schemas import (
    EvidenceScalar,
    ReportEvidenceFact,
    ReportEvidenceView,
)
from nurse_rcl.v2.report_presentation_schemas import (
    LanguageReason,
    ReportLanguageFieldCheck,
    claim_free_text_fields,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

_CLAUSE = re.compile(
    r"[;!?。\n,]|\.(?=\s|$)|\b(?:but|however|yet|and)\b|但是|然而|不过|但|并且|而且"
)
_ALIAS = re.compile(r"(?<![a-z0-9_])f[0-9]{1,6}(?![a-z0-9_])")
_TOKEN = (
    r"""(?:"[^"\n]{1,128}"|'[^'\n]{1,128}'|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?|"""
    r"[a-z_][a-z0-9_/-]{0,127})"
)
_ASSIGNMENT = re.compile(
    rf"(?<![a-z0-9_])(f[0-9]{{1,6}})(?:\.|\s*(?:的\s*)?)([a-z][a-z0-9_]{{0,63}})"
    rf"\s*(?:=|\bis\b|\bare\b|为|是)\s*({_TOKEN})(?:\s*([a-z%][a-z0-9%/_-]{{0,31}}))?"
)
_KNOWN_UNITS = frozenset(
    {
        "ns",
        "us",
        "ms",
        "s",
        "sec",
        "seconds",
        "milliseconds",
        "minutes",
        "hours",
        "bytes",
        "kb",
        "mb",
        "gb",
        "kib",
        "mib",
        "gib",
        "%",
        "percent",
        "cores",
        "requests/s",
        "count",
        "ratio",
    }
)
_BELONGS = re.compile(r"\b(f[0-9]{1,6})\s*(?:belongs\s+to|属于)\s*([a-z0-9_./:-]{1,256})")
_DIRECTION = re.compile(r"([a-z0-9_./:-]{1,256})\s*(?:->|→)\s*([a-z0-9_./:-]{1,256})")
_CONTRACTED_NEGATION = (
    r"\b(?:do|does|did|is|are|was|were|has|have|had|ca|could|wo|would|should|must)"
    r"n['\u2019]t\b(?!\s+only\b)"
)
_NEGATION = re.compile(
    rf"(?:(?:\b(?:not|never|neither|nor|no)\b|{_CONTRACTED_NEGATION})"
    r"(?!\s+only\b)(?:\s+[a-z]+){0,3}\s*|"
    r"(?:尚未|未曾|未经|没有|并非|而非|无(?!疑|可)|不是|不能|不代表|不(?!仅|但)|未)[^,;。:]{0,8})$"
)
_TENTATIVE = re.compile(
    r"\b(?:may|might|could|proposed|unverified|hypothetical|if|whether)\b|拟验证|待验证|尚待验证|可能|假设|是否"
)
_RULE_PATTERNS: tuple[tuple[LanguageReason, str], ...] = (
    (
        "causal_overclaim",
        r"\b(?:confirmed|proven|verified|established)\s+(?:root\s+cause|causality|causal\s+(?:link|relationship))\b|\b(?:causes|caused|directly\s+causing)\b|\b(?:is|was)\s+(?:the|a)\s+root\s+cause\b|根因(?:是|为)|是(?:唯一)?根因|(?:确认|证实|证明|确定)(?:为|了|的|是)?(?:根因|因果|导致)|(?:已确认|已证实)(?:的)?根因|(?:必然|直接)导致",
    ),
    # Keep subject and predicate inside one colon-delimited assertion, stopping at the
    # first predicate. Global clause splitting must retain colons in source identifiers.
    ("score_as_probability", r"(?:\bscore\b|分数|得分)[^;。:]{0,64}?(?:\bprobability\b|概率)"),
    (
        "tie_break_as_evidence",
        r"(?:tie[-_ ]?break|平局破除|破平局)[^;。:]{0,64}?"
        r"(?:stronger\s+evidence|evidence\s+advantage|证据更强|证据优势)",
    ),
    (
        "recommendation_executed",
        r"\b(?:already\s+(?:been\s+)?executed|has\s+been\s+executed|was\s+executed)\b|(?:已|已经)执行",
    ),
    (
        "recommendation_validated",
        r"\b(?:already\s+(?:been\s+)?validated|has\s+been\s+validated|verified\s+(?:successful|effective))\b|(?:已|已经)验证(?:成功|有效)?",
    ),
    (
        "recommendation_guarantee",
        r"\bguarantees?\s+(?:recovery|restoration|success)\b|\bwill\s+(?:fix|resolve|restore)\b|保证(?:恢复|修复|成功)|(?:必定|一定会)恢复",
    ),
)
_RULES = tuple((reason, re.compile(pattern)) for reason, pattern in _RULE_PATTERNS)

SRE_LANGUAGE_POLICY = "sre-future-conditional/1"
# Normalized punctuation; only an explicit future check followed by a consequent.
# Stop at an adversative, sentence or second conditional rather than exempting a paragraph.
_FUTURE_CHECK = re.compile(
    r"(?:若|如果)(?:后续|进一步)?(?:检查|核验|验证)"
    r"(?:(?!但是|然而|不过|但|并且|而且|若|如果|[;!?。\n]).){1,512}?"
    r"[, ]*则(?:(?!但是|然而|不过|但|并且|而且|若|如果|[;!?。\n]).){1,512}"
)
_PAST_ASSERTION = re.compile(r"已经|现已|已确认|已证实|已完成|已修复|已恢复|已执行|当前|目前")


def _future_check_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FUTURE_CHECK.finditer(text)
            if not _PAST_ASSERTION.search(m.group())]


@dataclass(frozen=True)
class _LanguageScope:
    facts: tuple[ReportEvidenceFact, ...]
    candidate_service_keys: tuple[str, ...]


@dataclass(frozen=True)
class LanguageFactIndex:
    """Complete alias/unit vocabulary; no retained expanded fact objects."""

    aliases: frozenset[str]
    units: frozenset[str]

    @classmethod
    def build(cls, facts: Iterable[ReportEvidenceFact]) -> LanguageFactIndex:
        aliases: set[str] = set()
        units = set(_KNOWN_UNITS)
        for fact in facts:
            if type(fact) is not ReportEvidenceFact or fact.fact_ref in aliases:
                raise ValueError("language index fact identity differs")
            aliases.add(fact.fact_ref)
            units.update(_normalized(str(value.value)) for value in fact.values
                if (value.name == "unit" or value.name.endswith("_unit"))
                and type(value.value) is str)
        return cls(frozenset(aliases), frozenset(units))


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _negated(clause: str, start: int, end: int) -> bool:
    # Only this adjacent embedded candidate question is denied. Use its full
    # clause prefix so whitespace cannot fall outside the generic 64-char window.
    if re.fullmatch(r"is\s+the\s+root\s+cause", clause[start:end]) and re.search(
        r"\bdoes\s+not\s+(?:by\s+itself\s+)?establish\s+which\s+candidate\s*$",
        clause[:start],
    ):
        return True
    prefix = clause[max(0, start - 64) : start]
    if _NEGATION.search(prefix):
        return True
    # A narrow interpretation denial. Do not exempt another assertion merely
    # because "cannot" appeared earlier in the same sentence or paragraph.
    if re.search(r"\bcannot\s+be\s+interpreted\s+as\s*$", prefix):
        return True
    # Patterns with a subject before the assertion need local negation inside the match.
    fragment = clause[start:end]
    return bool(
        re.search(
            rf"\b(?:is\s+not|does\s+not|not\s+an?)\b(?!\s+only\b)|"
            rf"{_CONTRACTED_NEGATION}|不代表|不构成|不是|并非|而非",
            fragment,
        )
    )


def _equal_text(value: EvidenceScalar, token: str, *, numeric_field: bool = False) -> bool:
    token = token.strip("\"'")
    if type(value) is bool:
        return token == ("true" if value else "false")
    if value is None:
        return token == "null"
    if type(value) in (int, float) or numeric_field:
        try:
            return Decimal(str(value)) == Decimal(token)
        except InvalidOperation:
            return False
    return _normalized(str(value)) == token


def _source_reasons(
    clause: str,
    claim: ReportCompositionClaim,
    facts: Mapping[str, ReportEvidenceFact],
    unit_vocabulary: frozenset[str],
    *,
    conditional_outcome: bool = False,
    future_confirmation: bool = False,
) -> set[LanguageReason]:
    reasons: set[LanguageReason] = set()
    for ref in _ALIAS.findall(clause):
        if ref not in facts:
            reasons.add("unknown_fact_alias")
        elif ref not in claim.fact_refs:
            reasons.add("uncited_fact_alias")
    for match in _ASSIGNMENT.finditer(clause):
        ref, field, token, unit = match.groups()
        if (
            ref not in facts
            or ref not in claim.fact_refs
            or _negated(clause, match.start(), match.end())
        ):
            continue
        values = {v.name: v.value for v in facts[ref].values}
        if field not in values:
            reasons.add("source_field_missing")
        elif not _equal_text(
            values[field],
            token,
            numeric_field=field
            in {
                "quantized_total_score",
                "quantized_event_evidence_score",
                "quantized_propagation_score",
                "total_score_gap",
                "event_evidence_gap",
                "propagation_gap",
                "rule_gap",
                "history_gap",
            },
        ):
            reasons.add("source_value_mismatch")
        expected_unit = values.get(field + "_unit", values.get("unit"))
        if field.startswith("measurement_") and field.endswith("_value"):
            expected_unit = values.get(field.removesuffix("_value") + "_unit")
        if (
            unit in unit_vocabulary
            and re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", token)
            and (expected_unit is not None or field + "_unit" in values)
            and (expected_unit is None or not _equal_text(expected_unit, unit))
        ):
            reasons.add("source_unit_mismatch")
    for match in _BELONGS.finditer(clause):
        ref, target = match.groups()
        if (
            ref in facts
            and ref in claim.fact_refs
            and target not in tuple(_normalized(t) for t in facts[ref].target_service_keys)
            and not _negated(clause, match.start(), match.end())
        ):
            reasons.add("fact_target_mismatch")
    cited = tuple(facts[ref] for ref in claim.fact_refs if ref in facts)
    if not _TENTATIVE.search(clause):
        for match in _DIRECTION.finditer(clause):
            if _negated(clause, match.start(), match.end()):
                continue
            directions = set()
            for fact in cited:
                values = {v.name: v.value for v in fact.values}
                for left, right in (
                    ("source_service_key", "target_service_key"),
                    ("caller_service_key", "callee_service_key"),
                ):
                    if type(values.get(left)) is str and type(values.get(right)) is str:
                        directions.add(
                            (_normalized(str(values[left])), _normalized(str(values[right])))
                        )
            if not directions:
                reasons.add("unsupported_relation")
            elif match.groups() not in directions:
                reasons.add("relation_direction_mismatch")
        mentioned = set(_ALIAS.findall(clause))
        described = (
            tuple(fact for fact in cited if fact.fact_ref in mentioned) if mentioned else cited
        )
        incomplete = any(
            _normalized(str(v.value)) in {"unknown", "no_record", "degraded"}
            for fact in described
            for v in fact.values
        )
        if incomplete:
            for match in re.finditer(r"\b(?:confirmed|verified|normal)\b|确认|证实|正常", clause):
                if future_confirmation and not _PAST_ASSERTION.search(clause):
                    continue  # Only this parsed future fragment, never other diagnostics.
                # This field describes a possible future result, not a measurement
                # already made. Explicit actual/confirmed claims remain checked.
                if (
                    conditional_outcome
                    and match.group() in {"normal", "正常"}
                    and not re.search(
                        r"\b(?:already|confirmed|verified|now|was|has\s+been)\b|已经|已确认|已证实|当前",
                        clause,
                    )
                ):
                    continue
                if not _negated(clause, match.start(), match.end()):
                    reasons.add("missing_evidence_upgraded")
    return reasons


def _prose_unit_reasons(
    text: str,
    claim: ReportCompositionClaim,
    facts: Mapping[str, ReportEvidenceFact],
    unit_vocabulary: frozenset[str],
) -> set[LanguageReason]:
    """Finite numeric-unit forms, bound to cited targets and matching source values.

    Ambiguous matching facts do not lend each other units. Explicit assignments
    are handled by their stricter field-specific rule. No conversion is inferred.
    """
    reasons: set[LanguageReason] = set()
    assignments = tuple(_ASSIGNMENT.finditer(text))
    for match in re.finditer(
        r"(?<![a-z0-9_])[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*([a-z%][a-z0-9%/_-]{0,31})", text
    ):
        unit = match[1]
        if unit not in unit_vocabulary or any(
            a.start() <= match.start() < a.end() for a in assignments
        ):
            continue
        token = text[match.start() : match.start(1)].strip()
        # Sentence scope retains a target preceding a comma ("For svc_a, ...").
        start = max(text.rfind(c, 0, match.start()) for c in (";", "。", "\n")) + 1
        prefix = text[start : match.start()]
        mentioned = set(_ALIAS.findall(prefix))
        targets = {t for t in claim.target_service_keys if _normalized(t) in prefix}
        scope = targets or set(claim.target_service_keys)
        candidates = [
            f
            for ref in facts
            if ref in claim.fact_refs
            for f in (facts[ref],)
            if (not mentioned or ref in mentioned)
            and (not scope or bool(scope.intersection(f.target_service_keys)))
        ]
        expected = []
        for fact in candidates:
            values = {v.name: v.value for v in fact.values}
            for field, value in values.items():
                if type(value) not in (int, float) or not _equal_text(value, token):
                    continue
                expected.append(
                    values.get(
                        field.removesuffix("_value") + "_unit"
                        if field.startswith("measurement_") and field.endswith("_value")
                        else field + "_unit",
                        values.get("unit"),
                    )
                )
        if not expected or any(value is None or not _equal_text(value, unit) for value in expected):
            reasons.add("source_unit_mismatch")
    return reasons


def _health_reasons(
    clause: str,
    claim: ReportCompositionClaim,
    facts: Mapping[str, ReportEvidenceFact],
    *,
    conditional_outcome: bool,
) -> set[LanguageReason]:
    # A median describes a location statistic, never absence of degradation in
    # the complete window. This finite pattern is not a semantic truth checker.
    summaries = [
        dict((v.name, v.value) for v in f.values)
        for ref in facts
        if ref in claim.fact_refs
        for f in (facts[ref],)
    ]
    if (
        not any(
            v.get("kind") == "window_summary" and v.get("primary_family") == "latency"
            for v in summaries
        )
        or conditional_outcome
    ):
        return set()
    pattern = (
        r"\b(?:without|no)\s+latency\s+(?:degradation|worsening)\b|"
        r"\blatency\s+(?:is|was|remains)\s+(?:normal|healthy)\b|"
        r"(?:不存在|没有|无)延迟(?:恶化|退化)|延迟(?:正常|健康)"
    )
    return (
        {"unsupported_health_inference"}
        if any(
            not _negated(clause, m.start(), m.end()) and not _TENTATIVE.search(clause[: m.start()])
            for m in re.finditer(pattern, clause)
        )
        else set()
    )


def _candidate_reasons(
    clause: str, claim: ReportCompositionClaim, view: _LanguageScope
) -> set[LanguageReason]:
    reasons: set[LanguageReason] = set()
    for target in view.candidate_service_keys:
        pattern = (
            rf"(?<![a-z0-9_]){re.escape(_normalized(target))}\s*(?:的\s*)?"
            rf"(rank|score|排名|分数)\s*(?:is|=|为|是)\s*({_TOKEN})"
        )
        for match in re.finditer(pattern, clause):
            if _negated(clause, match.start(), match.end()):
                continue
            field = "rank" if match[1] in {"rank", "排名"} else "quantized_total_score"
            values = [
                v.value
                for f in view.facts
                if f.fact_ref in claim.fact_refs
                and f.kind == "ranking"
                and target in f.target_service_keys
                for v in f.values
                if v.name == field
            ]
            if not values or not all(
                _equal_text(value, match[2], numeric_field=True) for value in values
            ):
                reasons.add("candidate_value_mismatch")
    return reasons


def _check_text(
    text: str,
    claim: ReportCompositionClaim,
    view: _LanguageScope,
    *,
    conditional_outcome: bool = False,
    language_policy: str | None = None,
    indexed_facts: Mapping[str, ReportEvidenceFact] | None = None,
    index: LanguageFactIndex | None = None,
) -> tuple[LanguageReason, ...]:
    normalized = _normalized(text)
    reasons: set[LanguageReason] = set()
    # These are active instructions/content irrespective of surrounding disclaimer prose.
    if re.search(
        r"(?:https?://|javascript:|data:text/html|www\.)|<\s*/?\s*[a-z!][^>]{0,512}>", normalized
    ):
        reasons.add("active_content")
    if re.search(
        r"\b(?:ignore|override|disregard)\b[^;。]{0,64}\b(?:instructions?|rules?|system|previous)\b|(?:忽略|覆盖|无视)[^;。]{0,32}(?:指令|规则|提示)|(?:hide|suppress)\s+(?:the\s+)?(?:uncertainty|counterevidence)",
        normalized,
    ):
        reasons.add("prompt_injection")
    if re.search(
        r"```|\b(?:rm\s+-[a-z]*r|curl\s+|wget\s+|bash\s+-c|powershell\b|invoke-expression\b|os\.system\s*\(|subprocess\.|eval\s*\(|exec\s*\()",
        normalized,
    ):
        reasons.add("executable_content")
    facts = ({f.fact_ref: f for f in view.facts} if indexed_facts is None else indexed_facts)
    # Source units extend recognition only; equality still uses the cited alias and field.
    # Unknown prose suffixes such as "because" are not inferred to be physical units.
    unit_vocabulary = index.units if index is not None else _KNOWN_UNITS | frozenset(
        _normalized(str(value.value))
        for fact in view.facts
        for value in fact.values
        if (value.name == "unit" or value.name.endswith("_unit")) and type(value.value) is str
    )
    reasons.update(_prose_unit_reasons(normalized, claim, facts, unit_vocabulary))
    spans = _future_check_spans(normalized) if (
        language_policy == SRE_LANGUAGE_POLICY and conditional_outcome) else []
    ends = [*[(m.start(), m.end()) for m in _CLAUSE.finditer(normalized)],
            (len(normalized), len(normalized))]
    start = 0
    for end, following in ends:
        clause = normalized[start:end]
        future = any(a <= start and end <= b for a, b in spans)
        start = following
        scoped_conditional = conditional_outcome and (language_policy is None or future)
        if language_policy == SRE_LANGUAGE_POLICY:
            for match in re.finditer(r"(?:已|已经)(?:修复|恢复|完成)", clause):
                if not _negated(clause, match.start(), match.end()):
                    reasons.add("recommendation_executed")
        for reason, pattern in _RULES:
            for match in pattern.finditer(clause):
                weak_conditional_cause = (
                    reason == "causal_overclaim"
                    and match.group() in {"causes", "caused", "directly causing"}
                    and _TENTATIVE.search(clause[: match.start()])
                )
                if not _negated(clause, match.start(), match.end()) and not weak_conditional_cause:
                    reasons.add(reason)
        reasons.update(
            _source_reasons(
                clause, claim, facts, unit_vocabulary, conditional_outcome=scoped_conditional,
                future_confirmation=future,
            )
        )
        reasons.update(
            _health_reasons(clause, claim, facts, conditional_outcome=scoped_conditional)
        )
        reasons.update(_candidate_reasons(clause, claim, view))
    return tuple(sorted(reasons))


def check_report_claim_language(
    claim: ReportCompositionClaim, *, view: ReportEvidenceView
) -> tuple[ReportLanguageFieldCheck, ...]:
    """Scan all free fields, including mechanically rejected drafts; grant no source authority."""
    if type(claim) is not ReportCompositionClaim or type(view) is not ReportEvidenceView:
        raise ValueError("language diagnostics require exact claim and view types")
    for value, expected in ((claim, ReportCompositionClaim), (view, ReportEvidenceView)):
        preflight_exact_pydantic_tree(
            value, root_model_types=(expected,), label="language diagnostic", max_string_length=None
        )
    return tuple(
        ReportLanguageFieldCheck(
            field_path=path,
            raw_text=text,
            reason_codes=_check_text(
                text,
                claim,
                _LanguageScope(view.facts, view.top_k),
                conditional_outcome=claim.recommendation is not None
                and claim.recommendation.status == "proposed"
                and path.startswith("recommendation.possible_outcomes["),
            ),
        )
        for path, text in claim_free_text_fields(claim)
    )


def check_fact_claim_language(
    claim: ReportCompositionClaim, *, facts: tuple[ReportEvidenceFact, ...],
    candidate_service_keys: tuple[str, ...],
    ops_empirical: bool = False,
    language_policy: str | None = None,
) -> tuple[ReportLanguageFieldCheck, ...]:
    """Finite diagnostics for an admitted fact set, not a redefined Top-K view.

    The caller separately owns fact provenance, section roles and exact scope.
    Passing does not establish natural-language truth or causal correctness.
    """
    if language_policy not in (None, SRE_LANGUAGE_POLICY):
        raise ValueError("unknown report language policy")
    claim_type: type[ReportCompositionClaim] = ReportCompositionClaim
    check_type: type[ReportLanguageFieldCheck] = ReportLanguageFieldCheck
    if ops_empirical:
        from nurse_rcl.v2.live.ops_report_proposals import (
            OpsCompositionClaim,
            OpsReportLanguageFieldCheck,
        )

        claim_type, check_type = OpsCompositionClaim, OpsReportLanguageFieldCheck
    if type(claim) is not claim_type or type(facts) is not tuple or (
        any(type(f) is not ReportEvidenceFact for f in facts)
        or type(candidate_service_keys) is not tuple
        or any(type(key) is not str for key in candidate_service_keys)
    ):
        raise ValueError("language diagnostics require exact claim and fact types")
    scope = _LanguageScope(facts, candidate_service_keys)
    return tuple(check_type(
        field_path=path, raw_text=text,
        reason_codes=_check_text(
            text, claim, scope, conditional_outcome=claim.recommendation is not None
            and claim.recommendation.status == "proposed"
            and path.startswith("recommendation.possible_outcomes[")
            and path.endswith(".interpretation"), language_policy=language_policy,
        ),
    ) for path, text in claim_free_text_fields(claim))


def check_indexed_fact_claim_language(
    claim: ReportCompositionClaim, *, facts: Mapping[str, ReportEvidenceFact],
    index: LanguageFactIndex, candidate_service_keys: tuple[str, ...],
    language_policy: str | None = None,
) -> tuple[ReportLanguageFieldCheck, ...]:
    """Empirical rules with a full immutable index and only locally resolved values."""
    from nurse_rcl.v2.live.ops_report_proposals import (
        OpsCompositionClaim,
        OpsReportLanguageFieldCheck,
    )

    if type(claim) is not OpsCompositionClaim or language_policy not in (None, SRE_LANGUAGE_POLICY):
        raise ValueError("language diagnostic claim or policy differs")
    if frozenset(facts) != index.aliases or type(candidate_service_keys) is not tuple or (
        any(type(key) is not str for key in candidate_service_keys)
    ):
        raise ValueError("language index scope differs")
    cited = tuple(facts[ref] for ref in claim.fact_refs if ref in facts)
    if any(type(f) is not ReportEvidenceFact for f in cited):
        raise ValueError("language diagnostic fact type differs")
    scope = _LanguageScope(cited, candidate_service_keys)
    return tuple(OpsReportLanguageFieldCheck(field_path=path, raw_text=text,
        reason_codes=_check_text(text, claim, scope,
            conditional_outcome=claim.recommendation is not None
            and claim.recommendation.status == "proposed"
            and path.startswith("recommendation.possible_outcomes[")
            and path.endswith(".interpretation"), language_policy=language_policy,
            indexed_facts=facts, index=index),
    ) for path, text in claim_free_text_fields(claim))
