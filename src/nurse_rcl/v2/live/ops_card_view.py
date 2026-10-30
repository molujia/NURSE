"""Deterministic model-input views; complete immutable cards remain with the owner.

Sections use record paths, never caller-provided file paths or expressions.
No measurement, score, or model-authored summary is evaluated here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, encode_json
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_tools import card_id

VIEW_BYTES = 4032  # Final ledger envelope (including action_index) is checked at 4096.
VIEW_RESOURCE_BYTES = 4_194_304
SECTIONS = ("summary", "parameters", "payload", "evidence_ids", "source_refs", "limitations")
_ENTITY_FIELDS = {
    "candidate_service_key", "source_candidate", "target_candidate",
    "caller_service_key", "callee_service_key", "source_service_key", "target_service_key",
}


def entities(value: Any) -> tuple[str, ...]:
    """Retain both endpoints; do not assign a shared card to an arbitrary service."""
    result: set[str] = set()
    if type(value) is dict:
        result.update(v for k, v in value.items() if k in _ENTITY_FIELDS and type(v) is str)
        for child in value.values():
            result.update(entities(child))
    elif type(value) is list:
        for child in value:
            result.update(entities(child))
    return tuple(sorted(result))


def section_leaves(
    value: Any, path: tuple[str | int, ...] = (),
) -> Iterator[tuple[tuple[str | int, ...], Any]]:
    """Lossless scalar path/value stream, retaining empty containers explicitly."""
    if type(value) is dict and value:
        for key in sorted(value):
            yield from section_leaves(value[key], (*path, key))
    elif type(value) is list and value:
        for i, child in enumerate(value):
            yield from section_leaves(child, (*path, i))
    else:
        yield path, value


def section_value(card: OpsNodeEvidence, section: str) -> Any:
    if section == "parameters":
        return json.loads(card.parameters_json)
    if section == "payload":
        return json.loads(card.payload_json)
    if section in {"evidence_ids", "source_refs", "limitations"}:
        return list(getattr(card, section))
    raise ValueError("node section is not registered")


def reference_view(card: OpsNodeEvidence, section: str) -> dict[str, Any]:
    values = list(getattr(card, section))
    result: dict[str, Any] = dict(section=section, count=len(values),
                                  sha256=sha256_hex(canonical_json_bytes(values)))
    if len(canonical_json_bytes(values)) <= 512:
        result["values"] = values
    else:
        result["omitted_detail"] = "exact references available via read_node section"
    return result


def card_header(card: OpsNodeEvidence, *, published: bool) -> dict[str, Any]:
    parameters, body = json.loads(card.parameters_json), json.loads(card.payload_json)
    return dict(
        schema_version="nurse.v2.ops-tool-view/2", card_id=card_id(card),
        case_snapshot_id=card.case_snapshot_id, node=card.node, tool_version=card.tool_version,
        role="supplementary", score_authority=False, published_by_core=published,
        publication_note="publication does not establish formal scoring use",
        entities=sorted(set(entities(parameters)) | set(entities(body))),
        unit=card.unit, window_ns=card.window_ns, coverage=card.coverage,
        availability=card.availability, truth_value=card.truth_value,
        limitations=card.limitations,
        references={s: reference_view(card, s) for s in ("evidence_ids", "source_refs")},
        sections=SECTIONS,
    )


def critical_judgments(
    value: Any, path: tuple[str | int, ...] = (), *, preserve_scope: bool = False,
) -> Iterator[dict[str, Any]]:
    if type(value) is dict:
        if preserve_scope and (
            value.get("truth_value") in ("false", "unknown")
            or value.get("support_value") in ("false", "unknown")
            or (("condition_id" in value or "formal_condition_id" in value)
                and "requirement" in value and any(
                    next(critical_judgments(child, preserve_scope=True), None) is not None
                    for child in value.values()))
        ):
            yield dict(path=path, value=value)
            return
        # In the mechanism contract a TRUE refuting condition weakens the
        # explanation even when all necessary conditions (and the card) are
        # true. Retain its complete role, identity, reasons and source leaves.
        if value.get("requirement") == "refuting" and value.get("truth_value") == "true":
            yield dict(path=path, value=value)
            return
        for key in sorted(value):
            child = value[key]
            if key in {"truth_value", "support_value"} and child in ("false", "unknown"):
                yield dict(path=(*path, key), value=child)
            else:
                yield from critical_judgments(child, (*path, key), preserve_scope=preserve_scope)
    elif type(value) is list:
        for i, child in enumerate(value):
            yield from critical_judgments(child, (*path, i), preserve_scope=preserve_scope)


def summary_view(
    card: OpsNodeEvidence, *, published: bool, soft_targets: bool = False,
) -> dict[str, Any]:
    body = json.loads(card.payload_json)
    result = card_header(card, published=published)
    # Include all top-level scalars, named windows and actual false/unknown leaf
    # judgments. Optional detail is never substituted for a contrary judgment.
    selected = {k: v for k, v in body.items() if type(v) not in (dict, list)}
    selected.update({k: body[k] for k in ("baseline_window", "fault_window") if k in body})
    result.update(
        section="summary", parameters=json.loads(card.parameters_json), facts=selected,
        critical_judgments=list(critical_judgments(body, preserve_scope=soft_targets)),
        omitted_detail=["nested payload details except listed critical judgments; read sections"],
    )
    if soft_targets and card.node in {"timing", "dependency"}:
        result["judgment_context"] = body
    return result


def bounded_view(
    value: dict[str, Any], *, card: OpsNodeEvidence, soft_targets: bool = False,
) -> str:
    """Capacity failure is a representation state, not a failed/healthy measurement."""
    if soft_targets:
        return serialize_tool_view(value, soft_targets=True)
    try:
        return encode_json(value, ExchangeLimits(max_bytes=VIEW_BYTES)).decode("utf-8")
    except ExchangeBoundaryError:
        return encode_json(dict(
            schema_version="nurse.v2.ops-tool-view/2", status="context_budget",
            scope="input_representation", card_id=card_id(card),
            case_snapshot_id=card.case_snapshot_id, role="supplementary", score_authority=False,
            limitations=["required facts cannot fit; full evidence retained",
                         "not evidence absence or normality; stop model channel"],
        ), ExchangeLimits(max_bytes=VIEW_BYTES)).decode("utf-8")


def serialize_tool_view(
    value: dict[str, Any], *, soft_targets: bool, diagnose: bool = False,
) -> str:
    """Check the complete envelope, also on restore; navigation is never expanded."""
    navigation = "page" in value and "card_id" not in value
    if not soft_targets or navigation:
        return encode_json(value, ExchangeLimits(max_bytes=4096)).decode()
    if value.get("status") == "resource_limit":
        return encode_json(value, ExchangeLimits(max_bytes=4096)).decode()
    # Replace projection metadata before counting so repeated serialization is stable.
    body = {k: v for k, v in value.items()
            if k not in {"target_bytes", "target_exceeded", "expansion_reasons"}}
    try:
        limits = ExchangeLimits(max_bytes=VIEW_RESOURCE_BYTES)
        baseline = encode_json(body, limits)
        expanded = len(baseline) > VIEW_BYTES
        body.update(target_bytes=4096, target_exceeded=expanded,
                    expansion_reasons=["required_evidence_view"] if expanded else [])
        # The metadata itself is part of the actual complete model-side envelope.
        encoded = encode_json(body, limits)
        if not expanded and len(encoded) > 4096:
            body.update(target_exceeded=True, expansion_reasons=["required_evidence_view"])
            encoded = encode_json(body, limits)
        return encoded.decode()
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        details: dict[str, Any] = {}
        if diagnose:
            from nurse_rcl.v2.live.ops_failure import failure_detail

            details["failure_detail"] = failure_detail(
                "investigation_tool", exc).model_dump(mode="json")
        return encode_json(dict(
            schema_version="nurse.v2.ops-tool-view/2", status="resource_limit",
            scope="evidence_view_resource", card_id=value.get("card_id"),
            case_snapshot_id=value.get("case_snapshot_id"),
            action_index=value.get("action_index"), origin=value.get("origin"),
            role="supplementary", score_authority=False,
            limitations=["full evidence retained; JSON resource boundary exceeded",
                         "not observation absence or normality"],
            **details,
        ), ExchangeLimits(max_bytes=4096)).decode()
