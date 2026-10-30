"""Finite supplied-field and registered-condition boundaries; never a diagnosis."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.mechanism_library import build_mechanism_library

_IDENTIFIER = re.compile(r"(?<![\w])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w])")
_COUNT_PHRASES = tuple(
    (field, re.compile(r"(?<![\w.+-])[0-9]+\s+" + phrase, re.IGNORECASE))
    for field, phrase in (
        ("accepted_mechanism_count", r"accepted\s+mechanisms?\b"),
        ("refuting_predicate_count", r"refuting\s+predicates?\b"),
        ("unknown_predicate_count", r"unknown\s+predicates?\b"),
        ("unknown_mechanism_count", r"unknown\s+mechanisms?\b"),
    )
)
_EFFECTS = {
    "necessary": dict(
        true="necessary_condition_holds",
        false="necessary_condition_fails",
        unknown="condition_unknown",
    ),
    "optional": dict(
        true="optional_support_present",
        false="optional_support_absent",
        unknown="condition_unknown",
    ),
    "refuting": dict(true="refuter_present", false="refuter_absent", unknown="condition_unknown"),
}


def _rows(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    refs = supply["provided_fact_refs"]
    if len(refs) != len(set(refs)) or not set(refs) <= set(store.refs()):
        raise ValueError("grounding supplied refs differ")
    return {ref: store.resolve(ref) for ref in refs}


def _values(row: Any) -> dict[str, Any]:
    return {value.name: value.value for value in row.fact.values}


def scalar_field_scope(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """One shared name/type index, not another copy of the supplied values."""
    kinds = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    return {
        ref: {name: kinds[type(value)] for name, value in _values(row).items()}
        for ref, row in _rows(supply, store).items()
    }


def named_field_reasons(
    claim: Any,
    supply: dict[str, Any],
    store: NativeFactStore,
    *,
    field_scope: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Exact supplied identifiers plus four finite integer phrases, not semantic proof."""
    index = scalar_field_scope(supply, store) if field_scope is None else field_scope
    cited = set(claim.evidence_refs)
    if not cited <= index.keys():
        raise ValueError("grounding claim exceeds supplied fact refs")
    all_fields = {name for fields in index.values() for name in fields}
    selected = {name for ref in cited for name in index[ref]}
    texts = [claim.surface_text]
    advice = claim.recommendation
    if advice is not None:
        texts += [row.interpretation for row in advice.possible_outcomes]
    mentioned = {match.group() for text in texts for match in _IDENTIFIER.finditer(text)}
    required = {field for field, pattern in _COUNT_PHRASES if any(pattern.search(t) for t in texts)}
    outside = ((mentioned & all_fields) | required) - selected
    return ("named_field_outside_selected_sources",) if outside else ()


def condition_boundary(
    supply: dict[str, Any],
    store: NativeFactStore,
    subject_ref: str,
    interval: dict[str, int] | None,
) -> dict[str, Any]:
    """Public template roles plus exact same-target/window source locations only."""
    rows = _rows(supply, store)
    if subject_ref not in rows:
        raise ValueError("grounding condition subject not supplied")
    subject = rows[subject_ref]
    fields = _values(subject)
    windows = [window.model_dump(mode="json") for window in subject.fact.intervals]
    if (
        subject.fact.kind not in {"observation", "predicate"}
        or fields.get("truth_value") not in {"true", "false", "unknown"}
        or type(fields.get("predicate_kind")) is not str
        or not subject.fact.target_service_keys
        or interval not in windows
    ):
        raise ValueError("grounding requires original single predicate/window")
    mechanisms, roles = [], set()
    for definition in build_mechanism_library().definitions:
        matches = [
            role
            for role in _EFFECTS
            if fields["predicate_kind"] in getattr(definition, role + "_predicate_kinds")
        ]
        if not matches:
            continue
        roles.update(matches)
        necessary = []
        for kind in definition.necessary_predicate_kinds:
            sources = []
            for ref, row in rows.items():
                values = _values(row)
                if (
                    row.fact.target_service_keys == subject.fact.target_service_keys
                    and interval in [w.model_dump(mode="json") for w in row.fact.intervals]
                    and row.fact.kind in {"observation", "predicate"}
                    and values.get("predicate_kind") == kind
                    and values.get("truth_value") in {"true", "false", "unknown"}
                ):
                    sources.append(dict(evidence_ref=ref, truth_value=values["truth_value"]))
            necessary.append(
                dict(
                    predicate_kind=kind,
                    supply_state="supplied" if sources else "not_supplied",
                    sources=sources,
                )
            )
        mechanisms.append(
            dict(
                mechanism_key=definition.mechanism_key,
                mechanism_definition_sha256=definition.definition_sha256,
                mechanism_version=definition.mechanism_version,
                roles=matches,
                necessary_conditions=necessary,
            )
        )
    role = next(iter(roles)) if len(roles) == 1 else None
    return dict(
        subject_ref=subject_ref,
        predicate_kind=fields["predicate_kind"],
        role=role,
        effects={} if role is None else dict(_EFFECTS[role]),
        mechanisms=mechanisms,
        authority="registered_condition_boundary_not_case_verdict",
    )


def _condition_group(row: Any) -> tuple[str, str, bytes] | None:
    fields = _values(row)
    key = fields.get("mechanism_key")
    if (
        row.fact.kind != "hypothesis"
        or type(key) is not str
        or not {"necessary_conditions", "optional_conditions", "refuting_conditions"}
        <= fields.keys()
    ):
        return None
    try:
        definition = build_mechanism_library().lookup(key)
    except KeyError:
        return None
    if (
        fields.get("mechanism_definition_sha256", definition.definition_sha256)
        != definition.definition_sha256
    ):
        return None
    return (
        key,
        definition.definition_sha256,
        canonical_json_bytes([w.model_dump(mode="json") for w in row.fact.intervals]),
    )


def _is_condition_comparison(refs: Any, rows: dict[str, Any]) -> bool:
    if not set(refs) <= rows.keys():
        return False
    cited = [rows[ref] for ref in refs]
    keys = {_condition_group(row) for row in cited}
    return (
        len(keys) == 1
        and None not in keys
        and len({row.fact.target_service_keys for row in cited}) > 1
    )


def condition_expression_refs(
    expressions: list[dict[str, Any]],
    supply: dict[str, Any],
    store: NativeFactStore,
) -> list[str]:
    rows = _rows(supply, store)
    return [
        row["expression_ref"]
        for row in expressions
        if row["claim"]["operation"] == "synthesize_evidence"
        and _is_condition_comparison(row["claim"]["evidence_refs"], rows)
    ]


def competition_coverage(
    response: Any, supply: dict[str, Any], store: NativeFactStore
) -> dict[str, Any]:
    """Structural availability/coverage only. Never establishes distinguishability."""
    rows = _rows(supply, store)
    groups: dict[Any, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows.values():
        key = _condition_group(row)
        if key is not None and row.fact.target_service_keys:
            groups[key].add(row.fact.target_service_keys)
    available = sum(len(targets) > 1 for targets in groups.values())
    selected = []
    for claim in () if response is None else response.final_synthesis:
        if claim.operation != "synthesize_evidence" or not set(claim.evidence_refs) <= rows.keys():
            continue
        if _is_condition_comparison(claim.evidence_refs, rows):
            selected.append(claim.claim_id)
    status = "not_available" if not available else "structurally_covered" if selected else "missing"
    return dict(
        status=status,
        available_groups=available,
        selected_claim_ids=selected,
        reason_codes=["condition_competition_handoff_missing"] if status == "missing" else [],
        semantic_status="not_established",
    )


def grounding_diagnostics(
    response: Any,
    raw: str,
    supply: dict[str, Any],
    store: NativeFactStore,
    checks: Any,
) -> dict[str, Any]:
    """Bind source-derived boundaries and native effect codes, not rewritten text."""
    valid = {c.proposal.claim_id for c in checks if c.surface_status != "blocked"}
    covered = SimpleNamespace(
        final_synthesis=tuple(c for c in response.final_synthesis if c.claim_id in valid)
    )
    selections = json.loads(raw)
    native = {
        c["claim_id"]: c
        for section, items in selections.items()
        if section
        in {
            "executive_summary",
            "formal_interpretations",
            "final_synthesis",
            "limitations",
            "next_checks",
        }
        for c in items
    }
    branches = []
    for claim in response.next_checks:
        advice = claim.recommendation
        if advice is None or not advice.precondition_refs:
            continue
        subject = advice.outcome_subject_ref or advice.precondition_refs[0]
        boundary = condition_boundary(
            supply,
            store,
            subject,
            None if claim.interval is None else claim.interval.model_dump(mode="json"),
        )
        branches.append(
            dict(
                claim_id=claim.claim_id,
                subject_ref=subject,
                boundary=boundary,
                effects=[
                    dict(outcome=o["outcome"], condition_effect=o["condition_effect"])
                    for o in native[claim.claim_id]["outcomes"] or ()
                ],
            )
        )
    return dict(
        schema_version="nurse.v2.atomic-report-grounding/1",
        competition=competition_coverage(covered, supply, store),
        branches=branches,
        authority="program_boundaries_not_model_diagnosis",
    )
