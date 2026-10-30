"""Deterministic input-only projections; complete evidence stays with the owner.

This component is not a live protocol. Its trusted ``make_wire`` callback must
build the FINAL request, including system/schema/skills/native settings. The
returned bytes are the exact bytes checked, not an inner-JSON size estimate.
No source replay, measurement, model compression, output quota or core writes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, encode_json
from nurse_rcl.v2.live.ops_card_view import card_header
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedActionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_report_context import OpsReportContext, OpsReportFact
from nurse_rcl.v2.live.ops_skills import indexed_skill_index
from nurse_rcl.v2.live.ops_tools import card_id

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_failure import OpsFailureDetail

WireBuilder = Callable[[dict[str, Any]], dict[str, Any]]
INVESTIGATION_BYTES = 16384
REPORT_BYTES = 32768
# Existing JSON hard ceiling; depth/node limits remain the original view limits.
# Independent process/serialization protection, never an input-token allowance.
INPUT_RESOURCE_LIMITS = ExchangeLimits(max_bytes=4_194_304)


@dataclass(frozen=True)
class OpsInputProjection:
    status: Literal["ready", "context_budget", "resource_limit", "unavailable"]
    scope: str = "input_representation"
    view_json: str | None = None
    wire_bytes: bytes | None = None
    provided_fact_refs: tuple[str, ...] = ()
    omitted_fact_refs: tuple[str, ...] = ()
    target_bytes: int = 0
    target_exceeded: bool = False
    expansion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosedInputProjection(OpsInputProjection):
    """New-profile local diagnostics; the legacy projection shape is unchanged."""

    failure_detail: OpsFailureDetail | None = None


def _pack(
    view: dict[str, Any], make_wire: WireBuilder, limit: int, *, soft_targets: bool = False,
    diagnose: bool = False,
) -> OpsInputProjection:
    limits = INPUT_RESOURCE_LIMITS if soft_targets else ExchangeLimits(max_bytes=limit)
    try:
        # Check the object itself as well as the JSON-in-string envelope; string
        # escaping must not bypass the original depth/node limits.
        raw = encode_json(view, limits)
        payload = make_wire(json.loads(raw))
        wire = encode_json(payload, limits)
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        if diagnose:
            from nurse_rcl.v2.live.ops_failure import failure_detail

            return DiagnosedInputProjection(status="resource_limit", target_bytes=limit,
                failure_detail=failure_detail("investigation_input", exc))
        return OpsInputProjection(status="resource_limit" if soft_targets else "context_budget",
                                  target_bytes=limit)
    exceeded = len(wire) > limit
    return OpsInputProjection(status="ready", view_json=raw.decode(), wire_bytes=wire,
        target_bytes=limit, target_exceeded=exceeded,
        expansion_reasons=("required_semantics_or_complete_envelope",) if exceeded else ())


def _conditions(value: Any, path: tuple[str | int, ...] = ()) -> Iterator[dict[str, Any]]:
    """Keep the enclosing condition, not a truth token stripped of its scope."""
    if type(value) is dict:
        # A leaf does not carry its parent's necessary/refuting role or AND/OR
        # operator. Keep the full registered condition if ANY child is critical,
        # including a true OR whose other branch remains unknown.
        if value.get("truth_value") in ("false", "unknown") or (
            value.get("support_value") in ("false", "unknown")
        ) or (value.get("requirement") == "refuting" and value.get("truth_value") == "true"):
            yield dict(path=path, value=value)
            return
        if ("condition_id" in value or "formal_condition_id" in value) and (
            "requirement" in value
        ) and any(
            next(_conditions(child), None) is not None for child in value.values()
        ):
            yield dict(path=path, value=value)
            return
        for key in sorted(value):
            yield from _conditions(value[key], (*path, key))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _conditions(child, (*path, index))


def investigation_input(
    *, capabilities: dict[str, Any], records: tuple[OpsActionRecord, ...],
    decision_index: int, make_wire: WireBuilder, soft_targets: bool = False,
) -> OpsInputProjection:
    """Use only admitted ledger receipts; caller retains the complete HERMES trace.

The provider sees the previous actual view unchanged and cumulative critical
cards. Ordinary old content is referenced, not rebuilt from model summaries.
This does not refresh publications or issue cursors: the ledger owns that work.
"""
    if capabilities.get("investigation_profile_id") != "hermes-readonly-ops/2" or (
        type(decision_index) is not int or not 1 <= decision_index <= 8
        or len(records) >= decision_index
    ):
        raise ValueError("indexed investigation context identity differs")
    snapshot = capabilities["case_snapshot_id"]
    actions = []
    critical: dict[str, dict[str, Any]] = {}
    previous = None
    for index, record in enumerate(records, 1):
        if not isinstance(record, OpsIndexedActionRecord) or record.index != index:
            raise ValueError("indexed action sequence differs")
        action = record.action
        row: dict[str, Any] = dict(action_index=index, status=record.status,
                                   input_sha256=record.input_sha256,
                                   failure_type=record.failure_type)
        if action is not None:
            row.update(tool=action.tool, skill_id=action.skill_id,
                       parameters=json.loads(action.parameters_json))
            # A long legal model question remains on disk. This is an INPUT
            # projection choice, not a reason to reject or retry that output.
            question = action.question
            row["question_unverified"] = (
                question if len(question.encode("utf-8")) <= 512 else
                dict(sha256=sha256_hex(question.encode("utf-8")),
                     omitted_detail="model question retained in complete action record")
            )
        if record.result is None:
            if record.status != "interrupted":
                raise ValueError("unfinished action cannot become model feedback")
            previous = dict(status="interrupted", action_index=index,
                            limitations=["completion_unknown_no_refund", "no_factual_conclusion"])
            actions.append(row)
            continue
        receipt = record.result
        card = receipt.evidence
        if card.case_snapshot_id != snapshot or receipt.action_index != index:
            raise ValueError("investigation receipt snapshot or index differs")
        if record.model_view_json is None:
            raise ValueError("indexed action view missing")
        previous = json.loads(record.model_view_json)
        if previous.get("action_index") != index or previous.get("origin") != receipt.origin:
            raise ValueError("indexed action view binding differs")
        if previous.get("status") in {"context_budget", "resource_limit"}:
            return OpsInputProjection(
                status="resource_limit" if soft_targets else "context_budget", scope="tool_view",
                target_bytes=INVESTIGATION_BYTES,
            )
        key = card_id(card)
        row.update(card_id=key, origin=receipt.origin,
                   reused_core_evidence_ids=receipt.reused_core_evidence_ids)
        actions.append(row)
        if card.tool_version == "ops-index-navigation/2":
            row["meaning"] = "catalog_not_health_evidence"
            continue
        body = json.loads(card.payload_json)
        conditions = list(_conditions(body))
        if conditions or card.truth_value != "true" or card.availability != "available" or (
            card.limitations
        ):
            if key not in critical:
                critical[key] = dict(
                    **card_header(card, published=previous.get("published_by_core", False)),
                    parameters=json.loads(card.parameters_json), conditions=conditions,
                    # Timing/dependency truth lives in the card header; its
                    # payload is the directional/window/clock basis of that
                    # judgment, not dispensable ordinary history.
                    judgment_context=body if card.node in {"timing", "dependency"} else {
                        k: v for k, v in body.items() if type(v) not in (dict, list)
                        or k in {"baseline_window", "fault_window"}
                    },
                    origin=receipt.origin, action_indices=[],
                    omitted_detail="ordinary payload detail retained in complete registered card",
                )
            critical[key]["action_indices"].append(index)
    view = dict(
        schema_version="nurse.v2.ops-investigation-input/2", case_snapshot_id=snapshot,
        decision_index=decision_index, remaining_actions=8 - len(records),
        remaining_decisions=9 - decision_index, capabilities=capabilities,
        skills=indexed_skill_index(), actions=actions, previous_feedback=previous,
        critical_evidence=list(critical.values()),
        omitted_detail=["ordinary past tool details remain in full receipts and registered cards",
                        "model prose is not evidence; full conversation retained separately"],
    )
    return _pack(view, make_wire, INVESTIGATION_BYTES, soft_targets=soft_targets)


def _fact_tables(
    facts: tuple[OpsReportFact, ...], *, group_by_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Share field names and exact common cells, never combine truth judgments.

Tables group by kind and field layout, not rank or outcome. SRE may additionally
group by exact provenance so each record's shared metadata is not copied to
every scalar fact. This does not merge aliases or confer evidence authority.
A row expands to
one original fact: merge common with zip(columns, row), then nest fact./values.
Canonical JSON equality avoids equating false, 0 and 0.0 when sharing cells.
"""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for item in facts:
        row = item.model_dump(mode="json")
        fact = row.pop("fact")
        values = fact.pop("values")
        row.update({"fact." + key: value for key, value in fact.items()})
        row.update({"values." + v["name"]: v["value"] for v in values})
        source = canonical_json_bytes((item.role, item.evidence_ids, item.source_refs,
            item.action_index, item.origin, item.reused_core_evidence_ids)).decode() if (
                group_by_provenance) else ""
        key = (item.fact.kind, source, *sorted(row))
        groups.setdefault(key, []).append(row)
    tables = []
    for rows in groups.values():
        first = rows[0]
        common = {k: first[k] for k in sorted(first) if k != "fact.fact_ref" and all(
            canonical_json_bytes(row[k]) == canonical_json_bytes(first[k]) for row in rows[1:]
        )}
        columns = [k for k in sorted(first) if k not in common]
        tables.append(dict(common=common, columns=columns,
                           rows=[[row[k] for k in columns] for row in rows]))
    return tables


def expand_report_facts(view: dict[str, Any]) -> tuple[OpsReportFact, ...]:
    """Inverse of the code-owned table codec before egress and alias validation.

    Field names in columns are data until expanded; scanning just dictionary
    keys would both reject legitimate dotted names and miss malicious columns.
    """
    facts = []
    for table in view["fact_tables"]:
        if type(table) is not dict or set(table) != {"common", "columns", "rows"}:
            raise ValueError("report fact table shape differs")
        common, columns, rows = table["common"], table["columns"], table["rows"]
        if type(common) is not dict or type(columns) is not list or type(rows) is not list or (
            any(type(k) is not str for k in columns)
            or len(set(columns)) != len(columns) or set(common) & set(columns)
        ):
            raise ValueError("report fact table columns differ")
        if not rows:
            raise ValueError("report fact table must contain rows")
        if {"fact", "fact.values"} & (common.keys() | set(columns)):
            raise ValueError("report fact table reserved columns")
        for cells in rows:
            if type(cells) is not list or len(cells) != len(columns):
                raise ValueError("report fact table row differs")
            fields = {**common, **dict(zip(columns, cells, strict=True))}
            outer, fact, values = {}, {}, []
            for key, value in fields.items():
                if key.startswith("values."):
                    values.append(dict(name=key.removeprefix("values."), value=value))
                elif key.startswith("fact."):
                    fact[key.removeprefix("fact.")] = value
                else:
                    outer[key] = value
            fact["values"] = values
            outer["fact"] = fact
            facts.append(OpsReportFact.model_validate_json(
                encode_json(outer, INPUT_RESOURCE_LIMITS)))
    refs = [item.fact.fact_ref for item in facts]
    if len(set(refs)) != len(refs) or any(
        not set(item.fact.related_fact_refs) <= set(refs) for item in facts
    ) or not set(view["mandatory_fact_refs"]) <= set(refs):
        raise ValueError("report provided alias closure differs")
    return tuple(facts)


def required_report_refs(context: OpsReportContext) -> set[str]:
    """Required facts and reference closure from admitted memory, not from the wire."""
    facts = {f.fact.fact_ref: f for f in context.facts}
    required = set(context.mandatory_fact_refs) | {
        f.fact.fact_ref for f in context.facts
        if f.role == "supplementary" or f.fact.kind in {
            "ranking", "candidate_evidence", "competition", "uncertainty", "limit",
        }
    }
    pending = list(required)
    while pending:
        for ref in facts[pending.pop()].fact.related_fact_refs:
            if ref not in required:
                required.add(ref)
                pending.append(ref)
    return required


def report_input(
    *, context: OpsReportContext, make_wire: WireBuilder, soft_targets: bool = False,
    group_by_provenance: bool = False,
) -> OpsInputProjection:
    """Project saved facts with deterministic selection and reference closure.

All owner-designated mandatory facts are retained. Ranking, competition, limits,
unknowns and evaluated candidate contributions are always required. Optional
details are added in source order only when the complete wire fits. The returned
provided_fact_refs is the allowlist for the new protocol's report checker; the
complete context remains its authority and the renderer's input.
"""
    if not context.formal_projection_available:
        # The existing admitted builder obtains this flag from view=None:
        # the formal projector's mandatory_budget_exhausted result.
        return OpsInputProjection(status="resource_limit" if soft_targets else "context_budget",
                                  scope="formal_projection", target_bytes=REPORT_BYTES)
    facts = {f.fact.fact_ref: f for f in context.facts}
    if len(facts) != len(context.facts):
        raise ValueError("duplicate report alias")
    if not set(context.mandatory_fact_refs) <= facts.keys() or any(
        not set(f.fact.related_fact_refs) <= facts.keys() for f in context.facts
    ):
        raise ValueError("report reference closure differs")

    def closure(refs: set[str]) -> set[str]:
        pending = list(refs)
        result = set(refs)
        while pending:
            for ref in facts[pending.pop()].fact.related_fact_refs:
                if ref not in result:
                    result.add(ref)
                    pending.append(ref)
        return result

    required = required_report_refs(context)
    actions = [dict(
        action_index=r.action_index, card_id=card_id(r.evidence), node=r.evidence.node,
        origin=r.origin, reused_core_evidence_ids=r.reused_core_evidence_ids,
        meaning="catalog_not_health_evidence" if r.evidence.tool_version ==
        "ops-index-navigation/2" else "supplementary_not_formal_scoring_input",
    ) for r in context.investigation]

    def assemble(selected: set[str]) -> dict[str, Any]:
        omitted = [ref for ref in facts if ref not in selected]
        return dict(
            schema_version="nurse.v2.ops-report-input/2", opaque_case_id=context.opaque_case_id,
            case_snapshot_id=context.case_snapshot_id,
            core_result_digest=context.core_result_digest,
            ranking_count=len(context.ranking), formal_top_k=context.formal_top_k,
            ranking_note="full frozen ranking rendered by code; no alternative ranking",
            fact_encoding="each row = common + zip(columns,row); fact.* is fact metadata; "
                          "values.* are exact named scalars; other fields are provenance/role; "
                          "one row per original alias, never infer omitted facts",
            fact_tables=_fact_tables(tuple(facts[ref] for ref in facts if ref in selected),
                                     group_by_provenance=group_by_provenance),
            mandatory_fact_refs=[ref for ref in facts if ref in required],
            actions=actions, investigation_status=context.investigation_status,
            investigation_failure_type=context.investigation_failure_type,
            formal_detail_omissions=[v.model_dump(mode="json")
                                     for v in context.formal_detail_omissions],
            omitted_detail=dict(
                fact_refs=omitted, count=len(omitted),
                reason="ordinary details retained in full context; not normal or excluded",
                raw_receipts="retained on server; facts above are the only claim aliases",
                model_prose="unverified; retained on server, not factual input",
            ),
        )

    selected = required
    result = _pack(assemble(selected), make_wire, REPORT_BYTES, soft_targets=soft_targets)
    if result.status != "ready":
        return result
    for ref in (() if result.target_exceeded else facts):
        if ref not in selected:
            proposed = closure(selected | {ref})
            packed = _pack(assemble(proposed), make_wire, REPORT_BYTES, soft_targets=soft_targets)
            if packed.status == "ready" and not packed.target_exceeded:
                result, selected = packed, proposed
    return replace(
        result,
        provided_fact_refs=tuple(ref for ref in facts if ref in selected),
        omitted_fact_refs=tuple(ref for ref in facts if ref not in selected),
    )
