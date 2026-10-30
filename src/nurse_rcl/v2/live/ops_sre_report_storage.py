"""One bounded, canonical shared file; no migration or live execution authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_fact_view import pack_fact_records, unpack_fact_records
from nurse_rcl.v2.live.ops_sre_history_view import pack_history_view, unpack_history_view
from nurse_rcl.v2.live.ops_sre_report_bounds import METADATA_LIMITS, validate_report_parts
from nurse_rcl.v2.live.ops_sre_report_context import (
    LegacyOpsSreReportContext,
    OpsSreReportContext,
    critical_fact,
)
from nurse_rcl.v2.live.ops_sre_report_navigation import validate_navigation_group

STORAGE_SCHEMA = "nurse.v2.sre-report-storage/1"


def _refs(values: Any, available: set[str]) -> set[str]:
    if type(values) is not list or any(type(v) is not str for v in values) or (
        len(values) != len(set(values)) or not set(values) <= available
    ):
        raise ValueError("SRE stored reference closure differs")
    return set(values)


def validate_sre_report_context(context: OpsSreReportContext) -> dict[str, Any]:
    """Validate model_copy values and full logical content at storage boundaries."""
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    if type(context) is not OpsSreReportContext:
        raise ValueError("SRE new report context required")
    value = context.model_dump(mode="json")
    facts = value["evidence"].pop("facts")
    validate_report_parts(value, facts, layout="context")
    value["evidence"]["facts"] = facts
    # Safe to serialize after complete partition preflight, without the old
    # aggregate quota. Revalidation includes strict scalar and enum contracts.
    admitted = OpsSreReportContext.model_validate_json(canonical_json_bytes(value))
    _require_outbound(value, case=admitted.evidence.opaque_case_id,
                      snapshot=admitted.evidence.case_snapshot_id)
    by_ref = {f.fact.fact_ref: f for f in admitted.evidence.facts}
    if len(by_ref) != len(facts):
        raise ValueError("SRE duplicate fact alias")
    validate_report_bindings(value, by_ref)
    return value


def validate_report_bindings(
    value: dict[str, Any], by_ref: Mapping[str, OpsReportFact], *, working: bool = False,
) -> None:
    """Shared permission rules; native callers provide an uncached local lookup."""
    aliases = set(by_ref)
    candidates = {r["candidate_service_key"] for r in value["evidence"]["ranking"]}
    critical_expected = set()
    for f in by_ref.values():
        if critical_fact(f):
            critical_expected.add(f.fact.fact_ref)
        if not set(f.fact.related_fact_refs) <= aliases or (
            not set(f.fact.target_service_keys) <= candidates
        ):
            raise ValueError("SRE fact scope or closure differs")
    critical = _refs(value["critical_fact_refs"], aliases)
    mandatory = _refs(value["evidence"]["mandatory_fact_refs"], aliases)
    if critical != critical_expected or (
        not critical <= mandatory
    ):
        raise ValueError("SRE critical facts differ")
    steps = value["steps"]
    if [s.get("step_ref") for s in steps] != [f"s{i}" for i in range(1, len(steps) + 1)]:
        raise ValueError("SRE stored step order differs")
    observed: dict[int, set[str]] = {}
    for index, step in enumerate(steps, 1):
        if step.get("decision_index") != index:
            raise ValueError("SRE stored decision order differs")
        refs = _refs(step.get("observed_fact_refs"), aliases)
        action = step.get("action_index")
        if action is None:
            if refs:
                raise ValueError("SRE undispatched step has observations")
        elif type(action) is not int or action in observed or not 1 <= action <= 8:
            raise ValueError("SRE stored action identity differs")
        else:
            if refs and step.get("executed", {}).get("status") != "completed":
                raise ValueError("SRE incomplete step has observations")
            observed[action] = refs
    grouped: dict[int, set[str]] = {a: set() for a in observed}
    executions = {s["action_index"]: s["executed"] for s in steps if s["action_index"] is not None}
    identities: set[str] = set()
    for group in value["record_groups"]:
        refs = _refs(group.get("fact_refs"), aliases)
        identity = group.get("record_sha256")
        actions = group.get("action_indices")
        if type(identity) is not str or identity in identities or type(actions) is not list or (
            not actions or any(type(a) is not int or a not in observed for a in actions)
        ):
            raise ValueError("SRE stored record binding differs")
        identities.add(identity)
        validate_navigation_group(group, tuple(by_ref[r] for r in refs), working=working)
        if group.get("source_role") == "query_scope_metadata":
            from nurse_rcl.v2.dynamic_verification.safety import require_pin

            require_pin(group.get("scope_content_sha256"))
            binding, deliveries = group.get("scope_binding"), group.get("deliveries")
            if group.get("card_id") is not None or type(binding) is not dict or set(binding) != {
                "case_snapshot_id", "tool", "parameters", "publication_prefix",
            } or binding["case_snapshot_id"] != value["evidence"]["case_snapshot_id"] or (
                type(deliveries) is not list or len(deliveries) != len(actions)
                or len(set(actions)) != len(actions)
            ):
                raise ValueError("SRE scope content binding differs")
            for action, delivery in zip(actions, deliveries, strict=True):
                execution = executions[action]
                params = execution.get("parameters", {})
                if type(delivery) is not dict or set(delivery) != {
                    "action_index", "receipt_card_id", "cursor",
                } or delivery != dict(action_index=action,
                        receipt_card_id=execution.get("receipt_card_id"),
                        cursor=params.get("cursor")) or execution.get("status") != "completed" or (
                    execution.get("tool") != binding["tool"]
                    or execution.get("publication_prefix") != binding["publication_prefix"]
                    or canonical_json_bytes({k: v for k, v in params.items() if k != "cursor"})
                    != canonical_json_bytes(binding["parameters"])
                ):
                    raise ValueError("SRE scope delivery differs from actual step")
        for ref in refs:
            fact = by_ref[ref]
            if fact.role != "supplementary" or fact.origin != group.get("source_role") or (
                fact.action_index != min(actions)
            ):
                raise ValueError("SRE stored record provenance differs")
        for action in actions:
            grouped[action].update(refs)
    if grouped != observed:
        raise ValueError("SRE stored step and record facts differ")


def encode_sre_report_storage(context: OpsSreReportContext) -> bytes:
    value = validate_sre_report_context(context)
    header = value.pop("evidence")
    header.pop("facts")
    packet = dict(schema_version=STORAGE_SCHEMA, evidence_header=header,
        fact_records=pack_fact_records(context.evidence.facts),
        history=pack_history_view(value, limits=METADATA_LIMITS))
    return encode_json(packet, REPORT_LIMITS)


def decode_sre_report_storage(raw: bytes) -> OpsSreReportContext:
    packet = decode_json_object(raw, REPORT_LIMITS)
    if set(packet) != {"schema_version", "evidence_header", "fact_records", "history"} or (
        packet["schema_version"] != STORAGE_SCHEMA or type(packet["evidence_header"]) is not dict
        or "facts" in packet["evidence_header"]
    ):
        raise ValueError("SRE report storage identity differs")
    value = unpack_history_view(packet["history"], limits=METADATA_LIMITS)
    if "evidence" in value:
        raise ValueError("SRE report storage header collision")
    value["evidence"] = packet["evidence_header"]
    facts = unpack_fact_records(packet["fact_records"])
    rows = [f.model_dump(mode="json") for f in facts]
    validate_report_parts(value, rows, layout="context")
    value["evidence"]["facts"] = rows
    context = OpsSreReportContext.model_validate_json(canonical_json_bytes(value))
    if encode_sre_report_storage(context) != raw:
        raise ValueError("SRE report storage is not canonical")
    return context


def read_legacy_sre_report_context(raw: bytes) -> LegacyOpsSreReportContext:
    value = decode_json_object(raw, REPORT_LIMITS)
    context = LegacyOpsSreReportContext.model_validate_json(encode_json(value, REPORT_LIMITS))
    if encode_json(context.model_dump(mode="json"), REPORT_LIMITS) != raw:
        raise ValueError("SRE legacy report context is not canonical")
    return context
