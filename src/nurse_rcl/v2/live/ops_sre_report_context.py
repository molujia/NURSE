"""New report envelope over terminal core and actually delivered SRE records.

The inner /2 evidence value is a private compatibility adapter for the existing
pure checker/table codec, not permission to resume an old report or conversation.
No full-card rereads, telemetry queries or model statements enter the fact table.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_report_context import (
    OpsReportContext,
    OpsReportFact,
    _field_unit,
    _scalars,
    _targets,
    build_report_context,
)
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_context import _delivered_records, _receipt, investigation_steps
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_report_navigation import OVERVIEW, split_candidate_navigation
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact, ReportEvidenceValue


class _SreReportFields(StrictFrozenModel):
    profile: Literal["hermes-sre-tools/1", "hermes-sre-tools/2", "hermes-sre-tools/3",
                     "hermes-sre-tools/4"] = "hermes-sre-tools/4"
    evidence: OpsReportContext
    steps: tuple[dict[str, Any], ...] = Field(max_length=8)
    record_groups: tuple[dict[str, Any], ...]
    critical_fact_refs: tuple[str, ...]
    unexpanded_critical_overviews: tuple[dict[str, Any], ...]
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"]
    action_seconds: tuple[float, ...]
    decision_seconds: tuple[float, ...]

    @property
    def core_result_digest(self) -> str:
        return self.evidence.core_result_digest


class LegacyOpsSreReportContext(_SreReportFields):
    """Explicit read-only historical value; never accepted by the new owner."""

    schema_version: Literal["nurse.v2.sre-report-context/1"] = "nurse.v2.sre-report-context/1"

    @property
    def context_sha256(self) -> str:
        return sha256_hex(encode_json(self.model_dump(mode="json"), REPORT_LIMITS))


class OpsSreReportContext(_SreReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/4"] = "nurse.v2.sre-report-context/4"

    @property
    def context_sha256(self) -> str:
        from nurse_rcl.v2.live.ops_sre_report_storage import encode_sre_report_storage

        return sha256_hex(encode_sre_report_storage(self))


def critical_fact(item: OpsReportFact) -> bool:
    fact = item.fact
    values = {v.name: v.value for v in fact.values}
    return bool(fact.kind in {"uncertainty", "limit"} or fact.status_codes or any(
        values.get(k) in {"false", "unknown"} for k in ("truth_value", "support_value")
    ) or (values.get("requirement") == "refuting" and values.get("truth_value") == "true"))


def _record_facts(
    row: dict[str, Any], *, action_index: int, candidates: set[str],
) -> list[OpsReportFact]:
    return list(_iter_record_facts(row, action_index=action_index, candidates=candidates))


def _iter_record_facts(
    row: dict[str, Any], *, action_index: int, candidates: set[str],
) -> Iterator[OpsReportFact]:
    body = row["payload"]
    targets = (_targets(body) or _targets(row.get("parameters", {}))
               or tuple(row.get("targets", ())))
    # A partial read keeps only its delivered header/section. No hidden expansion.
    partial = body.get("delivered_view", {})
    if not targets and partial:
        targets = tuple(partial.get("entities", ()))
    bounds = row.get("window_ns", partial.get("window_ns"))
    window = None if bounds is None else TimeInterval(start_unix_ns=bounds[0],
                                                     end_unix_ns=bounds[1])
    representation = row.get("representation", "full_semantic_record")
    limitations = tuple(row.get("limitations", ()))
    for path, values, scoped, interval, unit in _scalars(body, targets=targets,
            interval=window, unit=row.get("unit", partial.get("unit"))):
        if not set(scoped) <= candidates:
            raise ValueError("SRE report record candidate differs")
        metadata = dict(values)
        if path == "payload":
            # The observed header remains authoritative for the record judgment.
            metadata.update(truth_value=row["truth_value"],
                            availability=row.get("availability", partial.get("availability")))
        metadata.update(record_path=path, representation=representation,
                        source_role=row["source_role"])
        metadata.setdefault("unit", unit)
        for name, value in values.items():
            if type(value) in (int, float) and len(name) <= 58:
                metadata.setdefault(name + "_unit", _field_unit(name, unit))
        codes = set(limitations)
        if representation != "full_semantic_record":
            codes.add(representation)
        yield OpsReportFact(fact=ReportEvidenceFact(fact_ref="f1", kind="observation",
            target_service_keys=scoped, intervals=() if interval is None else (interval,),
            values=tuple(ReportEvidenceValue(name=k, value=v) for k, v in metadata.items()),
            status_codes=tuple(sorted(codes)), allowed_claim_strength="frozen_observation",
            scope_basis="bound_observation"), role="supplementary",
            evidence_ids=tuple(row.get("evidence_ids", ())),
            source_refs=tuple(row.get("source_refs", ())),
            source_path=path, action_index=action_index, origin=row["source_role"],
            reused_core_evidence_ids=tuple(row.get("evidence_ids", ()))
                if row["source_role"] == "investigation_core_reuse" else ())
    # Overview counts are metadata, never independent health observations.
    counts = row.get("judgment_status_counts")
    if counts:
        yield OpsReportFact(fact=ReportEvidenceFact(fact_ref="f1", kind="uncertainty",
            target_service_keys=targets, values=tuple(ReportEvidenceValue(
                name=k + "_count", value=v) for k, v in counts.items()),
            status_codes=("overview_not_full_judgment",),
            allowed_claim_strength="ranking_inference", scope_basis="frozen_assessment"),
            role="supplementary", evidence_ids=tuple(row.get("evidence_ids", ())),
            source_refs=tuple(row.get("source_refs", ())), source_path="judgment_status_counts",
            action_index=action_index, origin=row["source_role"])


def _inspection_scope(
    view: dict[str, Any], record: OpsActionRecord, *, working: bool = False, brief: bool = False,
) -> dict[str, Any] | None:
    """The actual inspection's coverage, never a new health measurement."""
    schemas = {"nurse.v2.sre-tool-view/2", "nurse.v2.sre-tool-view/3"}
    if working:
        schemas.add("nurse.v2.sre-tool-view/4")
    if brief:
        schemas.add("nurse.v2.sre-tool-view/5")
    if view.get("schema_version") not in schemas:
        return None
    view = expand(view)
    assert record.result is not None
    payload = dict(boundary=view["boundary"])
    # D11 case sections are page-local. Their candidate content has a separate
    # metadata record; a shared boundary must not depend on which page was read.
    if view["schema_version"] == "nurse.v2.sre-tool-view/2":
        payload["sections"] = [dict(candidate_service_key=k, **v)
                               for k, v in view["sections_state"].items()]
    if "dependency" in view:
        payload["dependency"] = view["dependency"]
    row = dict(card_id=None, node="boundary",
        parameters={k: v for k, v in view["parameters"].items() if k != "cursor"},
        case_snapshot_id=view["case_snapshot_id"], tool=view["tool"],
        publication_prefix=view["publication_prefix"],
        targets=view["scope"], payload=payload, source_role="query_scope_metadata",
        representation="query_scope_not_health_observation", truth_value=view["truth_value"],
        status=view["status"], availability=view["availability"],
        limitations=[*view["limitations"], "query_scope_not_health_observation"],
        source_refs=record.result.evidence.source_refs, evidence_ids=())
    row["scope_content_sha256"] = sha256_hex(canonical_json_bytes(row))
    return row


def _candidate_scopes(view: dict[str, Any], scope: dict[str, Any]) -> list[dict[str, Any]]:
    view = expand(view)
    summaries = (view["page"]["records"] if view.get("page", {}).get("unit") == "candidate_overview"
                 else view.get("candidate_overviews", []))
    rows = []
    for summary in summaries:
        # Structural paths are data, not new scalar field names. Preserve every
        # level in a list; do not overwrite the fact's own record_path.
        states = [dict(state, judgment_status_by_group=[dict(condition_path=path, **counts)
            for path, counts in state["judgment_status_by_group"].items()])
            for state in summary["mechanisms"]["states"]]
        summary = dict(summary, mechanisms=dict(summary["mechanisms"], states=states))
        row = {k: v for k, v in scope.items() if k != "scope_content_sha256"}
        row.update(payload=summary, targets=[summary["candidate_service_key"]],
                   representation="candidate_overview_not_full_judgment")
        row["scope_content_sha256"] = sha256_hex(canonical_json_bytes(row))
        rows.append(row)
    return rows


def build_sre_report_context(
    *, core: OpsCoreResult, sources: tuple[RepairMeasurementSource, ...],
    decisions: tuple[OpsDecisionRecord, ...], records: tuple[OpsActionRecord, ...],
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"],
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"],
    failure_type: str | None = None,
) -> OpsSreReportContext:
    formal = build_report_context(core=core, sources=sources, investigation=(), soft_targets=True)
    return assemble_sre_report_context(formal=formal, decisions=decisions, records=records,
        investigation_status=investigation_status, termination=termination,
        failure_type=failure_type)


def assemble_sre_report_context(
    *, formal: OpsReportContext, decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"],
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"],
    failure_type: str | None = None,
) -> OpsSreReportContext:
    """Pure assembly of admitted projections; does not grant source authority."""
    from nurse_rcl.v2.live.ops_sre_report_storage import validate_sre_report_context

    steps = investigation_steps(decisions=decisions, records=records)
    facts = list(formal.facts)
    candidates = {r.candidate_service_key for r in formal.ranking}
    # Identity is card + representation + exact content, not text authored by the LLM.
    dedup: dict[str, dict[str, Any]] = {}
    observed: dict[int, list[str]] = {}
    overviews = []
    for record in records:
        _, view = _receipt(record)
        observed[record.index] = []
        if record.result is None:
            continue
        if record.result.evidence.case_snapshot_id != formal.case_snapshot_id:
            raise ValueError("SRE report receipt snapshot differs")
        if record.status != "completed":
            continue
        rows = _delivered_records(view)
        if not rows and "node" in view and "card_id" in view and "truth_value" in view:
            rows = [dict(card_id=view["card_id"], node=view["node"],
                source_role="delivered_partial_card", truth_value=view["truth_value"],
                representation="delivered_section_not_full_judgment",
                payload=dict(delivered_view=view), limitations=view.get("limitations", []))]
        scope_record = _inspection_scope(view, record)
        if scope_record is not None:
            rows.append(scope_record)
            rows.extend(_candidate_scopes(view, scope_record))
        for row in rows:
            identity = sha256_hex(canonical_json_bytes(row))
            if identity not in dedup:
                fact_row, navigation = row, None
                if row.get("representation") == OVERVIEW:
                    body, navigation = split_candidate_navigation(row["payload"])
                    fact_row = dict(row, payload=body)
                new = _record_facts(fact_row, action_index=record.index, candidates=candidates)
                if row["source_role"] == "query_scope_metadata":
                    new = [f.model_copy(update={"fact": f.fact.model_copy(update=dict(
                        kind="uncertainty", scope_basis="frozen_assessment",
                        allowed_claim_strength="ranking_inference"))}) for f in new]
                refs = []
                for item in new:
                    ref = f"f{len(facts) + 1}"
                    facts.append(item.model_copy(update={"fact": item.fact.model_copy(
                        update={"fact_ref": ref})}))
                    refs.append(ref)
                dedup[identity] = dict(card_id=row["card_id"], record_sha256=identity,
                    representation=row.get("representation", "full_semantic_record"),
                    source_role=row["source_role"], action_indices=[], fact_refs=refs,
                    meaning="one_record_not_independent_votes; record_path_preserves_tree")
                if navigation is not None:
                    dedup[identity]["navigation"] = navigation
                if row["source_role"] == "query_scope_metadata":
                    dedup[identity].update(scope_content_sha256=row["scope_content_sha256"],
                        scope_binding={k: row[k] for k in (
                            "case_snapshot_id", "tool", "parameters", "publication_prefix")},
                        deliveries=[])
            group = dedup[identity]
            group["action_indices"].append(record.index)
            if row["source_role"] == "query_scope_metadata":
                group["deliveries"].append(dict(action_index=record.index,
                    receipt_card_id=view["card_id"], cursor=view["parameters"].get("cursor")))
            observed[record.index].extend(group["fact_refs"])
        if "critical_overview" in view:
            overviews.append(dict(action_index=record.index, overview=view["critical_overview"]))
    for step in steps:
        step["observed_fact_refs"] = observed.get(step["action_index"], [])
    critical = tuple(f.fact.fact_ref for f in facts if critical_fact(f))
    evidence = formal.model_copy(update=dict(facts=tuple(facts),
        mandatory_fact_refs=tuple(dict.fromkeys((*formal.mandatory_fact_refs, *critical))),
        investigation_status=investigation_status, investigation_failure_type=failure_type))
    result = OpsSreReportContext(evidence=evidence, steps=tuple(steps),
        record_groups=tuple(dedup.values()), critical_fact_refs=critical,
        unexpanded_critical_overviews=tuple(overviews), termination=termination,
        action_seconds=tuple(r.elapsed_seconds for r in records),
        decision_seconds=tuple(r.provider_seconds for r in decisions))
    validate_sre_report_context(result)
    return result
