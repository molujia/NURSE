"""Model path hypotheses, independent of factual prose admission and gold labels.

Relation evidence is not confirmed propagation. This module never removes a
recognizable prediction merely because its evidence is absent or contradicted.
"""

from __future__ import annotations

import html
import re
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

PREDICTION_SCHEMA = "nurse.v2.report-path-predictions/1"
Identifier = Annotated[str, Field(min_length=1, max_length=512)]
Explanation = Annotated[str, Field(min_length=1, max_length=8192)]


class PredictedPathEdge(StrictFrozenModel):
    source_service_key: Identifier
    target_service_key: Identifier
    basis: Literal["observed_relation", "inferred"]
    evidence_refs: tuple[Identifier, ...]
    rationale: Explanation


class PredictedPath(StrictFrozenModel):
    path_id: Identifier
    service_sequence: tuple[Identifier, ...] = Field(min_length=2)
    edges: tuple[PredictedPathEdge, ...]
    rationale: Explanation

    @model_validator(mode="after")
    def full_simple_path(self) -> Self:
        if len(set(self.service_sequence)) != len(self.service_sequence):
            raise ValueError("prediction must be a simple service path")
        expected = list(pairwise(self.service_sequence))
        if [(e.source_service_key, e.target_service_key) for e in self.edges] != expected:
            raise ValueError("prediction edges must cover adjacent services in order")
        if any(e.basis == "observed_relation" and not e.evidence_refs for e in self.edges):
            raise ValueError("observed relation declaration requires evidence references")
        return self


def prediction_prompt() -> str:
    return (
        " In the separate predicted_paths channel, propose your final root-to-symptom "
        "propagation hypotheses. This channel permits model inference, not confirmed facts. "
        "Always return predicted_paths and path_abstention_reason. Each path has path_id, "
        "service_sequence, edges and rationale; each adjacent edge has source_service_key, "
        "target_service_key, basis (observed_relation or inferred), evidence_refs and rationale. "
        "Use the supplied service identifiers, a complete directed sequence with at least two "
        "distinct services and no repeated service. Multiple candidate roots and paths longer "
        "than two hops are allowed, independently of core selected_path_ids. You may infer a "
        "missing relation or an order within a strongly connected component; mark such edges "
        "inferred. observed_relation requires exact delivered relation references, correctly "
        "oriented for propagation; an observed call/dependency still does not prove fault "
        "propagation. Never invent an observation or a source reference. Keep all explicitly "
        "proposed final paths in this collection, including uncertain inferred ones; their "
        "correctness is externally evaluated and unsupported paths are not silently removed. "
        "Do not place partial propagation fragments here or alter formal ranking/scores. "
        "Use a nonempty path_abstention_reason only when predicted_paths is empty; otherwise "
        "set it to null. Keep ordinary factual prose under its existing evidence permissions."
    )


def _pin(value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("path prediction provenance requires a SHA256 pin")


def _sequence(value: Any) -> bool:
    return type(value) is list and bool(value) and all(
        type(k) is str and bool(k) and k == k.strip() and len(k) <= 512 for k in value
    )


def _edge_audit(
    edges: Any, sequence: list[str], aliases: dict[str, str],
    visible: set[str], relations: dict[str, list[tuple[str, str]]],
) -> list[str]:
    issues = []
    if type(edges) is not list:
        return ["edge_declarations_missing_or_invalid"]
    expected = list(pairwise(sequence))
    if len(edges) != len(expected):
        issues.append("edge_declarations_do_not_cover_sequence")
    for index, edge in enumerate(edges):
        if type(edge) is not dict:
            issues.append("edge_declaration_invalid")
            continue
        source, target = edge.get("source_service_key"), edge.get("target_service_key")
        if type(source) is not str or type(target) is not str:
            issues.append("edge_endpoints_invalid")
            continue
        pair = (aliases.get(source, source), aliases.get(target, target))
        if index >= len(expected) or pair != expected[index]:
            issues.append("edge_endpoints_differ_from_sequence")
        refs = edge.get("evidence_refs")
        if type(refs) is not list or any(type(ref) is not str for ref in refs):
            issues.append("edge_evidence_refs_invalid")
            refs = []
        if any(ref not in visible for ref in refs):
            issues.append("evidence_ref_not_delivered")
        basis = edge.get("basis")
        if basis not in ("observed_relation", "inferred"):
            issues.append("edge_basis_invalid")
        if basis == "observed_relation":
            found = any(
                ref in visible and pair in [
                    (aliases.get(a, a), aliases.get(b, b)) for a, b in relations.get(ref, [])
                ] for ref in refs
            )
            if not found:
                issues.append("observed_relation_not_verified")
        if type(edge.get("rationale")) is not str or not edge["rationale"].strip():
            issues.append("edge_rationale_missing")
    return sorted(set(issues))


def build_prediction_export(
    native: bytes | None,
    *,
    opaque_case_id: str,
    core_result_digest: str,
    context_sha256: str,
    request_sha256: str | None,
    service_aliases: dict[str, str] | None = None,
    known_service_keys: tuple[str, ...] = (),
    visible_evidence_refs: tuple[str, ...] = (),
    observed_relations: dict[str, list[tuple[str, str]]] | None = None,
    expected_schema: str | None = None,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Export before prose guards, without accessing core files or references.

    Caller supplies only the current request's trusted alias book and delivered
    relations. Bad evidence annotations remain auditable, scoreable predictions.
    Unrecoverable sequence rows make the complete set invalid instead of allowing
    a cherry-picked subset score. Recognizable unknown/cyclic rows remain errors.
    """
    for pin in (core_result_digest, context_sha256):
        _pin(pin)
    _pin(request_sha256, optional=native is None)
    if (type(opaque_case_id) is not str or not opaque_case_id
            or opaque_case_id != opaque_case_id.strip()):
        raise ValueError("path prediction requires a case identity")
    if type(is_mock) is not bool or (native is not None and type(native) is not bytes):
        raise TypeError("original native bytes and explicit mock boolean required")
    aliases = {} if service_aliases is None else dict(service_aliases)
    if any(type(k) is not str or type(v) is not str for k, v in aliases.items()):
        raise ValueError("service aliases must be exact string identities")
    known = set(known_service_keys) | set(aliases.values())
    value: dict[str, Any] = dict(
        schema_version=PREDICTION_SCHEMA, opaque_case_id=opaque_case_id,
        core_result_digest=core_result_digest, context_sha256=context_sha256,
        request_sha256=request_sha256, native_sha256=None if native is None else sha256_hex(native),
        direction="root_to_symptom", granularity="service", status="unavailable",
        paths=[], raw_predicted_paths=None, abstention_reason=None, raw_abstention_reason=None,
        audit_issues=[], is_mock=is_mock,
    )
    if native is not None:
        try:
            body = decode_json_object(native, REPORT_LIMITS)
        except (ExchangeBoundaryError, UnicodeError, ValueError):
            value.update(status="invalid", audit_issues=["native_not_decodable"])
        else:
            _collect_predictions(
                value, body, aliases, known, set(visible_evidence_refs),
                {} if observed_relations is None else observed_relations, expected_schema,
            )
    value["artifact_sha256"] = sha256_hex(canonical_json_bytes(value))
    return value


def _collect_predictions(
    value: dict[str, Any], body: dict[str, Any], aliases: dict[str, str],
    known: set[str], visible: set[str], relations: dict[str, list[tuple[str, str]]],
    expected_schema: str | None,
) -> None:
    value["raw_predicted_paths"] = body.get("predicted_paths")
    reason = body.get("path_abstention_reason")
    value["raw_abstention_reason"] = reason
    value["abstention_reason"] = reason.strip() if type(reason) is str else None
    fatal = []
    if expected_schema is not None and body.get("schema_version") != expected_schema:
        fatal.append("native_response_schema_mismatch")
    if "predicted_paths" not in body:
        value.update(status="invalid" if fatal else "missing",
                     audit_issues=[*fatal, "paths_missing"])
        return
    rows = body["predicted_paths"]
    if type(rows) is not list:
        value.update(status="invalid", audit_issues=[*fatal, "paths_not_array"])
        return
    if "path_abstention_reason" not in body:
        fatal.append("abstention_field_missing")
    elif rows and reason is not None:
        value["audit_issues"].append("nonempty_prediction_has_abstention_reason")
    elif not rows and (type(reason) is not str or not reason.strip()):
        fatal.append("empty_prediction_requires_abstention_reason")
    identities = []
    for row in rows:
        if type(row) is not dict or not _sequence(row.get("service_sequence")):
            fatal.append("unrecoverable_path_sequence")
            continue
        original = row["service_sequence"]
        sequence = [aliases.get(key, key) for key in original]
        invalid = []
        if len(sequence) < 2:
            invalid.append("not_a_root_to_symptom_path")
        if len(sequence) != len(set(sequence)):
            invalid.append("repeated_service")
        if any(key not in known for key in sequence):
            invalid.append("unknown_service")
        issues = _edge_audit(row.get("edges"), sequence, aliases, visible, relations)
        path_id = row.get("path_id")
        if type(path_id) is not str or not path_id.strip():
            issues.append("path_identity_missing")
        elif path_id in identities:
            issues.append("duplicate_path_identity")
        identities.append(path_id)
        value["paths"].append(dict(
            path_id=path_id, service_sequence=sequence, model_service_sequence=original,
            edges=row.get("edges"), rationale=row.get("rationale"),
            audit_issues=sorted(set(issues + invalid)), invalid_for_scoring=bool(invalid),
        ))
    value["audit_issues"] = sorted(set(value["audit_issues"] + fatal))
    value["status"] = "invalid" if fatal else "available" if rows else "abstained"


def render_prediction_export(value: dict[str, Any]) -> str:
    """Render the same complete collection, escaping model-controlled text."""
    def safe(item: Any) -> str:
        text = html.escape(" ".join(str(item).splitlines()), quote=True)
        for char in "|`[]":
            text = text.replace(char, f"&#{ord(char)};")
        return text

    lines = ["## Model-predicted root-to-symptom paths", "",
             "MOCK / SYNTHETIC / NOT EMPIRICAL" if value["is_mock"] else "Model predictions",
             "", "These are hypotheses, not confirmed causal chains. Core ranking is unchanged.",
             "", f"Prediction status: {safe(value['status'])}.", ""]
    if value.get("abstention_reason") is not None:
        lines.extend([f"Abstention: {safe(value['abstention_reason'])}", ""])
    for index, row in enumerate(value["paths"], 1):
        seq = " → ".join(safe(k) for k in row["service_sequence"])
        lines.extend([f"{index}. {seq}", f"   - Hypothesis: {safe(row['rationale'])}"])
        edges = row.get("edges")
        if type(edges) is list:
            for edge in edges:
                if type(edge) is dict:
                    lines.append(
                        "   - Edge: " + safe(edge.get("source_service_key")) + " → "
                        + safe(edge.get("target_service_key")) + "; " + safe(edge.get("basis"))
                        + "; evidence: " + safe(edge.get("evidence_refs"))
                    )
        if row["audit_issues"]:
            lines.append("   - Audit: " + "; ".join(safe(k) for k in row["audit_issues"]))
        if row["invalid_for_scoring"]:
            lines.append("   - Retained as an incorrect prediction; not silently removed.")
    if value["audit_issues"]:
        lines.extend(["", "Collection audit: " + "; ".join(safe(k) for k in value["audit_issues"])])
    return "\n".join(lines) + "\n"
