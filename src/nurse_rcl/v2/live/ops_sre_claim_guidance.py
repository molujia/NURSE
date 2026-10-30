"""Advisory scopes over supplied facts, never permission or claim rewriting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_report_context import OpsReportFact

CLAIM_SCOPE_SCHEMA = "sre-claim-scope/2"


def claim_scope_guide(facts: Iterable[OpsReportFact]) -> dict[str, Any]:
    """Keep input order and exact provenance; omitted facts cannot enter this guide."""
    grouped: dict[bytes, dict[str, Any]] = {}
    layouts: dict[bytes, dict[str, Any]] = {}
    fact_layouts: dict[bytes, dict[str, Any]] = {}
    types = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    for item in facts:
        fact = item.fact
        # Names are literal schema data, not identity-bearing dictionary keys.
        # In particular source_refs/candidate_service_key must not be translated
        # as if their type label were a source/service identifier.
        fields = [dict(name=v.name, json_type=types[type(v.value)])
                  for v in sorted(fact.values, key=lambda v: v.name)]
        layout_key = canonical_json_bytes(fields)
        layout = layouts.setdefault(layout_key,
                                    dict(layout_id=f"l{len(layouts) + 1}", fields=fields))
        metadata = dict(layout_id=layout["layout_id"], kind=fact.kind, scope_basis=fact.scope_basis,
                        target_service_keys=list(fact.target_service_keys),
                        verification_precondition_kind=fact.kind in {
                            "observation", "predicate", "hypothesis"})
        fact_layouts.setdefault(canonical_json_bytes(metadata), dict(metadata, fact_refs=[]))[
            "fact_refs"].append(fact.fact_ref)
        required = {"ranking": ("explain_priority", 1),
                    "competition": ("compare_candidates", 2)}.get(fact.kind)
        if item.role != "formal_record" or required is None:
            continue
        operation, count = required
        if len(fact.target_service_keys) != count or len(set(fact.target_service_keys)) != count:
            continue
        source = item.model_dump(mode="json", exclude={"fact"})
        identity = canonical_json_bytes(dict(operation=operation, source=source,
            targets=fact.target_service_keys, intervals=[i.model_dump(mode="json")
                                                       for i in fact.intervals]))
        entry = grouped.setdefault(identity, dict(operation=operation,
            target_service_keys=list(fact.target_service_keys),
            source_fact_ref=fact.fact_ref, fact_refs=[]))
        entry["fact_refs"].append(fact.fact_ref)
    return dict(schema_version=CLAIM_SCOPE_SCHEMA, entries=list(grouped.values()),
                field_layouts=list(layouts.values()), fact_layouts=list(fact_layouts.values()),
                meaning="actual_values_fields_only; status_codes_not_boolean_fields; "
                        "kind_hint_not_permission; original value/target/time/claim checks apply")
