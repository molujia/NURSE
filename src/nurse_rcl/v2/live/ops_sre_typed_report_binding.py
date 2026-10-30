"""Complete normal legal typed closure, separate from model semantic validity.

No hardcoded case, step, service, target count, research sender or saved answer.
Raw native remains its own version; checked output is explicitly program-expanded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
    expression_catalog,
    resolve_expression_response,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_typed_report_contract import (
    SECTIONS,
    TYPED_REPORT_RESPONSE,
    parse_typed_response,
)

TYPED_CATALOG = "nurse.v2.sre-typed-report-catalog/1"
TYPED_RECEIPT = "nurse.v2.sre-typed-report-binding/1"


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _json_type(value: Any) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}[
        type(value)
    ]


def _unit(values: dict[str, Any], field: str) -> dict[str, Any]:
    if field not in values:
        return dict(state="not_supplied")
    value = values[field]
    return dict(
        state="supplied_null" if value is None else "supplied",
        field=field,
        value=deepcopy(value),
        json_type=_json_type(value),
    )


def typed_expression_catalog(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Retain the entire original permission set; annotate, never grant ancestors."""
    require_pin(supply["core_result_digest"])
    original = expression_catalog(supply, store)
    sources = {}
    for ref in supply["provided_fact_refs"]:
        item = store.resolve(ref)
        source = item.model_dump(mode="json")
        values = {v.name: v.value for v in item.fact.values}
        sources[ref] = dict(
            source=source,
            source_sha256=_digest(source),
            fields=[
                dict(
                    field=name,
                    value=deepcopy(value),
                    json_type=_json_type(value),
                    unit=_unit(values, name + "_unit"),
                )
                for name, value in values.items()
            ],
            record_unit=_unit(values, "unit"),
            authority="program_exact_source_row_not_model_diagnosis",
        )
    scope = _digest(dict(supply=supply, sources=sources))
    rows = []
    for choice in original["expressions"]:
        row = dict(
            expression=deepcopy(choice),
            source_scope_sha256=scope,
            source_versions={
                ref: sources[ref]["source_sha256"] for ref in choice["claim"]["evidence_refs"]
            },
        )
        rows.append(dict(row, row_version=_digest(row)))
    result = dict(
        schema_version=TYPED_CATALOG,
        core_result_digest=supply["core_result_digest"],
        source_scope_sha256=scope,
        rows=rows,
        sources=sources,
        field_scope=original["field_scope"],
        condition_boundaries=original["condition_boundaries"],
        authority="program_permissions_and_typed_companions_not_semantic_certification",
    )
    encode_json(result, REPORT_LIMITS)
    return result


def bind_typed_response(
    raw: bytes,
    supply: dict[str, Any],
    store: NativeFactStore,
    *,
    request_sha256: str,
    catalog_sha256: str,
    response_origin: dict[str, Any] | None,
) -> tuple[dict[str, Any], bytes]:
    """Admit the exact received selection; preserve all native explanations unchanged.

    The caller must obtain origin from its durable receive boundary. This function
    checks its binding, not the authenticity of an arbitrary caller-created dict.
    No transport/schema retry, semantic repair or source restoration occurs here.
    """
    require_pin(request_sha256)
    require_pin(catalog_sha256)
    if type(raw) is not bytes:
        raise TypeError("typed native requires original bytes")
    catalog = typed_expression_catalog(supply, store)
    if _digest(catalog) != catalog_sha256:
        raise ValueError("typed catalog version differs from current source")
    expected_origin = dict(
        request_sha256=request_sha256, catalog_sha256=catalog_sha256, native_sha256=sha256_hex(raw)
    )
    if type(response_origin) is not dict or response_origin != expected_origin:
        raise ValueError("typed response origin differs")
    response = parse_typed_response(raw)
    native = response.model_dump(mode="json")
    by_ref = {row["expression"]["expression_ref"]: row for row in catalog["rows"]}
    steps = {row["step_ref"] for row in supply["steps"]}
    if not response.step_findings.keys() <= steps:
        raise ValueError("typed expression names an absent step")
    bindings = []

    def admit(selection: dict[str, Any], section: str, step: str | None = None) -> dict[str, Any]:
        row = by_ref.get(selection["expression_ref"])
        if (
            row is None
            or section not in row["expression"]["sections"]
            or (section == "step_findings" and step not in row["expression"]["step_refs"])
        ):
            raise ValueError("typed selection exceeds expression/section/step permission")
        choice = row["expression"]
        refs = choice["claim"]["evidence_refs"]
        for primary in selection["primary_fields"]:
            ref = primary["evidence_ref"]
            if ref not in refs or primary["field"] not in catalog["field_scope"][ref]:
                raise ValueError("typed primary field exceeds selected exact source")
        bindings.append(
            dict(
                claim_id=selection["claim_id"],
                section=section,
                step_ref=step,
                expression_ref=selection["expression_ref"],
                row_version=row["row_version"],
                model_primary_fields=deepcopy(selection["primary_fields"]),
                primary_fields_covered=bool(selection["primary_fields"]),
                legacy_program_assertions=deepcopy(choice["claim"]["assertions"]),
                companion_sources={ref: deepcopy(catalog["sources"][ref]) for ref in refs},
                semantic_certified=False,
            )
        )
        # This is an explicit internal adapter to the unchanged original expansion
        # and guards, not rewriting/renaming the persisted typed native response.
        return {k: deepcopy(v) for k, v in selection.items() if k != "primary_fields"}

    internal = dict(
        schema_version="nurse.v2.atomic-report-expression-response/3",
        core_result_digest=supply["core_result_digest"],
    )
    for section in SECTIONS:
        internal[section] = (
            {
                step: [admit(c, section, step) for c in claims]
                for step, claims in native[section].items()
            }
            if section == "step_findings"
            else [admit(c, section) for c in native[section]]
        )
    checked = resolve_expression_response(encode_json(internal, REPORT_LIMITS), supply, store)
    receipt = dict(
        schema_version=TYPED_RECEIPT,
        native_schema=TYPED_REPORT_RESPONSE,
        origin=deepcopy(expected_origin),
        core_result_digest=supply["core_result_digest"],
        source_scope_sha256=catalog["source_scope_sha256"],
        checked_sha256=sha256_hex(checked),
        bindings=bindings,
        semantic_certified=False,
        raw_prose_policy="verbatim_no_repair",
        legacy_assertions_authority="program_expansion",
        companion_authority="constraints_not_independent_evidence_votes",
    )
    encode_json(receipt, REPORT_LIMITS)
    return receipt, checked
