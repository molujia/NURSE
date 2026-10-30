"""Narrow D22 check for explicit decimal text against cited, asserted supply."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal

from nurse_rcl.v2.report_presentation_schemas import claim_free_text_fields

_DECIMAL = re.compile(r"(?<![\w.-])[+-]?(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?(?![0-9.])")
_EXACT_DECIMAL_SCAN = re.compile(
    r"(?<![\w.-])[+-]?(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?(?![0-9]|\.[0-9])"
)
_CLOCK_FRACTION = re.compile(
    r"(?:(?<!\w)|(?<=T))\d{1,2}:\d{2}(?::\d{2})?\.\d+(?![0-9.])"
)
DecimalValuePolicy = Literal["float-only/1", "exact-decimal-string/2"]
FLOAT_ONLY: DecimalValuePolicy = "float-only/1"
EXACT_DECIMAL_STRING: DecimalValuePolicy = "exact-decimal-string/2"


def validate_decimal_value_policy(
    policy: DecimalValuePolicy, *, require_supplied_decimals: bool = True,
) -> None:
    if type(policy) is not str or policy not in {FLOAT_ONLY, EXACT_DECIMAL_STRING}:
        raise ValueError("decimal value policy differs")
    if policy == EXACT_DECIMAL_STRING and not require_supplied_decimals:
        raise ValueError("exact decimal policy requires supplied decimal checks")


def unsupplied_decimal_in_claim(
    claim: Any, *, briefs: Mapping[str, Any], facts: Mapping[str, Any],
    provided_brief_refs: set[str], provided_fact_refs: set[str],
    decimal_value_policy: DecimalValuePolicy = FLOAT_ONLY,
) -> bool:
    """Do not let a hidden or uncited numeric value license report prose."""
    validate_decimal_value_policy(decimal_value_policy)
    scanner = _EXACT_DECIMAL_SCAN if decimal_value_policy == EXACT_DECIMAL_STRING else _DECIMAL
    cited = set(claim.evidence_refs)
    supplied: set[str] = set()
    exact_strings: set[str] = set()
    for assertion in claim.assertions:
        ref = assertion.evidence_ref
        if ref not in cited:
            continue
        fields: dict[str, Any] | None = None
        if ref in provided_brief_refs and ref in briefs:
            fields = briefs[ref]["fields"]
        elif ref in provided_fact_refs and ref in facts:
            fields = {value.name: value.value for value in facts[ref].values}
        if fields is not None:
            actual = fields.get(assertion.field)
            if (
                type(actual) is float
                and type(assertion.value) is float
                and actual == assertion.value
            ):
                supplied.add(str(actual))
            elif (
                decimal_value_policy == EXACT_DECIMAL_STRING
                and type(actual) is str
                and type(assertion.value) is str
                and actual == assertion.value
                and _DECIMAL.fullmatch(actual) is not None
            ):
                exact_strings.add(actual)
    for _, text in claim_free_text_fields(claim):
        normalized = unicodedata.normalize("NFKC", text)
        # String permissions are literal, even though legacy float scanning uses NFKC.
        # Preserve positions so a raw spelling elsewhere cannot license a normalized one.
        raw_spans = set()
        if exact_strings:
            for raw in scanner.finditer(text):
                start = len(unicodedata.normalize("NFKC", text[:raw.start()]))
                raw_spans.add((start, start + len(raw.group()), raw.group()))
        clocks = tuple(
            (match.start(), match.end()) for match in _CLOCK_FRACTION.finditer(normalized)
        )
        for match in scanner.finditer(normalized):
            if any(start <= match.start() and match.end() <= end for start, end in clocks):
                continue
            if match.group() in supplied:
                continue
            if match.group() in exact_strings and (
                match.start(), match.end(), match.group()
            ) in raw_spans:
                continue
            return True
    return False
