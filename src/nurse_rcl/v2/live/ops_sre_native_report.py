"""D12 native report context and canonical storage, without flat fact adapters.

The original input schema remains explicit for saved-material diagnostics. Such
diagnostics grant no new investigation execution or paid-owner authority.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_report_context import OpsReportContext, OpsReportFact, OpsReportHeader
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_context import (
    SRE_INPUT_SCHEMA,
    WORKING_INPUT_SCHEMA,
    _delivered_records,
    _receipt,
    investigation_steps,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore, build_fact_store
from nurse_rcl.v2.live.ops_sre_report_bounds import _measure
from nurse_rcl.v2.live.ops_sre_report_context import (
    _candidate_scopes,
    _inspection_scope,
    _iter_record_facts,
    critical_fact,
)
from nurse_rcl.v2.live.ops_sre_report_navigation import OVERVIEW, split_candidate_navigation
from nurse_rcl.v2.live.ops_sre_report_storage import validate_report_bindings
from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

NATIVE_STORAGE_SCHEMA = "nurse.v2.sre-report-storage/2"
# Approved D12 whole-report budget; never alter old metadata/investigation/tools.
NATIVE_REPORT_INPUT_LIMITS = ExchangeLimits(max_bytes=4_194_304, max_depth=32, max_nodes=100_000)


class NativeReportMetadata(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-report-context/5"] = "nurse.v2.sre-report-context/5"
    profile: Literal["hermes-sre-tools/5"] = "hermes-sre-tools/5"
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/5", "nurse.v2.sre-investigation-input/6"]
    evidence: OpsReportHeader
    steps: tuple[dict[str, Any], ...] = Field(max_length=8)
    record_groups: tuple[dict[str, Any], ...]
    critical_fact_refs: tuple[str, ...]
    unexpanded_critical_overviews: tuple[dict[str, Any], ...]
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"]
    action_seconds: tuple[float, ...]
    decision_seconds: tuple[float, ...]


class NativeFactLookup(Mapping[str, OpsReportFact]):
    """Mapping API with alias-only membership; no expanded-fact cache."""

    def __init__(self, store: NativeFactStore) -> None:
        self.store = store
        self._refs = store.refs()
        self._aliases = frozenset(self._refs)

    def __getitem__(self, key: str) -> OpsReportFact:
        return self.store.resolve(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __contains__(self, key: object) -> bool:
        return key in self._aliases


@dataclass(frozen=True)
class NativeSreReportContext:
    metadata: NativeReportMetadata
    fact_store: NativeFactStore

    @property
    def core_result_digest(self) -> str:
        return self.metadata.evidence.core_result_digest

    @property
    def context_sha256(self) -> str:
        return sha256_hex(encode_native_report_storage(self))


def validate_native_report(context: NativeSreReportContext) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    if type(context) is not NativeSreReportContext or type(context.metadata) is not (
        NativeReportMetadata
    ) or type(context.fact_store) is not NativeFactStore:
        raise ValueError("SRE native context type differs")
    metadata = context.metadata.model_dump(mode="json")
    packet = dict(schema_version=NATIVE_STORAGE_SCHEMA, metadata=metadata,
                  fact_records=context.fact_store.to_packet())
    # Count actual combined representation, including all envelope fields.
    _measure(packet, node_limit=100_000, boundary="storage_nodes")
    raw = encode_json(metadata, REPORT_LIMITS)
    admitted = NativeReportMetadata.model_validate_json(raw, strict=True)
    _require_outbound(packet, case=admitted.evidence.opaque_case_id,
                      snapshot=admitted.evidence.case_snapshot_id)
    validate_report_bindings(metadata, NativeFactLookup(context.fact_store),
                             working=admitted.investigation_input_schema == WORKING_INPUT_SCHEMA)
    return packet


def encode_native_report_storage(context: NativeSreReportContext) -> bytes:
    return encode_json(validate_native_report(context), REPORT_LIMITS)


def decode_native_report_storage(raw: bytes) -> NativeSreReportContext:
    packet = decode_json_object(raw, REPORT_LIMITS)
    if set(packet) != {"schema_version", "metadata", "fact_records"} or (
        packet["schema_version"] != NATIVE_STORAGE_SCHEMA
    ):
        raise ValueError("SRE native storage identity differs")
    metadata = NativeReportMetadata.model_validate_json(
        encode_json(packet["metadata"], REPORT_LIMITS), strict=True)
    store = NativeFactStore.from_packet(packet["fact_records"])
    context = NativeSreReportContext(metadata=metadata, fact_store=store)
    if encode_native_report_storage(context) != raw:
        raise ValueError("SRE native storage is not canonical")
    return context


def assemble_native_report_context(
    *, formal: OpsReportContext, decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"],
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"],
    failure_type: str | None = None, input_schema: str = WORKING_INPUT_SCHEMA,
) -> NativeSreReportContext:
    """Stream admitted facts; source input identity is not a migration switch."""
    if input_schema not in {SRE_INPUT_SCHEMA, WORKING_INPUT_SCHEMA}:
        raise ValueError("SRE source input identity differs")
    working = input_schema == WORKING_INPUT_SCHEMA
    steps = investigation_steps(decisions=decisions, records=records, input_schema=input_schema)
    candidates = {r.candidate_service_key for r in formal.ranking}
    groups: dict[str, dict[str, Any]] = {}
    observed: dict[int, list[str]] = {}
    overviews: list[dict[str, Any]] = []
    critical: list[str] = []

    def facts() -> Iterator[OpsReportFact]:
        count = 0
        for item in formal.facts:
            count += 1
            if critical_fact(item):
                critical.append(item.fact.fact_ref)
            yield item
        for record in records:
            _, view = _receipt(record)
            observed[record.index] = []
            if record.result is None:
                continue
            if record.result.evidence.case_snapshot_id != formal.case_snapshot_id:
                raise ValueError("SRE report receipt snapshot differs")
            if record.status != "completed":
                continue
            evidence_view = working_evidence_view(view) if working else view
            rows = _delivered_records(evidence_view)
            if not rows and all(k in evidence_view for k in ("node", "card_id", "truth_value")):
                rows = [dict(card_id=view["card_id"], node=view["node"],
                    source_role="delivered_partial_card", truth_value=view["truth_value"],
                    representation="delivered_section_not_full_judgment",
                    payload=dict(delivered_view=evidence_view),
                    limitations=view.get("limitations", []))]
            scope = _inspection_scope(view, record, working=working)
            if scope is not None:
                rows.append(scope)
                rows.extend(_candidate_scopes(view, scope))
            for row in rows:
                identity = sha256_hex(canonical_json_bytes(row))
                if identity not in groups:
                    fact_row, navigation = row, None
                    if row.get("representation") == OVERVIEW:
                        body, navigation = split_candidate_navigation(
                            row["payload"], working=working)
                        fact_row = dict(row, payload=body)
                    refs = []
                    for item in _iter_record_facts(fact_row, action_index=record.index,
                                                   candidates=candidates):
                        count += 1
                        ref = f"f{count}"
                        fields = dict(fact_ref=ref)
                        if row["source_role"] == "query_scope_metadata":
                            fields.update(kind="uncertainty", scope_basis="frozen_assessment",
                                          allowed_claim_strength="ranking_inference")
                        item = item.model_copy(update={"fact": item.fact.model_copy(update=fields)})
                        refs.append(ref)
                        if critical_fact(item):
                            critical.append(ref)
                        yield item
                    groups[identity] = dict(card_id=row["card_id"], record_sha256=identity,
                        representation=row.get("representation", "full_semantic_record"),
                        source_role=row["source_role"], action_indices=[], fact_refs=refs,
                        meaning="one_record_not_independent_votes; record_path_preserves_tree")
                    if navigation is not None:
                        groups[identity]["navigation"] = navigation
                    if row["source_role"] == "query_scope_metadata":
                        groups[identity].update(scope_content_sha256=row["scope_content_sha256"],
                            scope_binding={k: row[k] for k in (
                                "case_snapshot_id", "tool", "parameters", "publication_prefix")},
                            deliveries=[])
                group = groups[identity]
                group["action_indices"].append(record.index)
                if row["source_role"] == "query_scope_metadata":
                    group["deliveries"].append(dict(action_index=record.index,
                        receipt_card_id=view["card_id"], cursor=view["parameters"].get("cursor")))
                observed[record.index].extend(group["fact_refs"])
                # Early lower bound, without duplicating facts or resetting final quota.
                _measure(list(groups.values()), node_limit=100_000, boundary="storage_nodes")
            if "critical_overview" in view:
                overviews.append(dict(action_index=record.index,
                                      overview=view["critical_overview"]))

    store = build_fact_store(facts())
    for step in steps:
        step["observed_fact_refs"] = observed.get(step["action_index"], [])
    header = formal.model_dump(mode="json", exclude={"facts"})
    header.update(mandatory_fact_refs=list(dict.fromkeys((*formal.mandatory_fact_refs, *critical))),
                  investigation_status=investigation_status,
                  investigation_failure_type=failure_type)
    metadata = NativeReportMetadata.model_validate_json(canonical_json_bytes(dict(
        investigation_input_schema=input_schema, evidence=header, steps=steps,
        record_groups=list(groups.values()), critical_fact_refs=critical,
        unexpanded_critical_overviews=overviews, termination=termination,
        action_seconds=[r.elapsed_seconds for r in records],
        decision_seconds=[r.provider_seconds for r in decisions])))
    result = NativeSreReportContext(metadata=metadata, fact_store=store)
    validate_native_report(result)
    return result
