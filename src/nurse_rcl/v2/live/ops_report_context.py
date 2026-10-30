"""Read-only terminal evidence projection, without source replay or score execution."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import parse_decision
from nurse_rcl.v2.live.ops_agent_state import OpsAgentResult
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.report_evidence_schemas import (
    ReportEvidenceFact,
    ReportEvidenceOmission,
    ReportEvidenceValue,
)


class OpsRankRow(StrictFrozenModel):
    rank: int = Field(ge=1)
    candidate_service_key: str
    quantized_total_score: str
    local_score_component: float
    propagation_score_component: float
    local_evidence_status: str


class OpsReportFact(StrictFrozenModel):
    fact: ReportEvidenceFact
    role: Literal["formal_record", "supplementary"]
    evidence_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    source_path: str
    action_index: int | None = Field(default=None, ge=1, le=8)
    origin: str
    reused_core_evidence_ids: tuple[str, ...] = ()


class OpsReportHeader(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-report-context/1", "nurse.v2.ops-report-context/2"] = (
        "nurse.v2.ops-report-context/1"
    )
    profile: Literal["v2-ops-isolated/1", "v2-ops-isolated/2"] = "v2-ops-isolated/1"
    opaque_case_id: str
    case_snapshot_id: SHA256
    core_result_digest: SHA256
    ranking: tuple[OpsRankRow, ...] = Field(min_length=1)
    formal_top_k: tuple[str, ...]
    mandatory_fact_refs: tuple[str, ...]
    formal_detail_omissions: tuple[ReportEvidenceOmission, ...]
    investigation: tuple[OpsInvestigationEvidence, ...] = Field(max_length=8)
    formal_projection_available: bool
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"]
    investigation_failure_type: str | None
    model_trace_unverified: tuple[dict[str, Any], ...] = Field(max_length=8)


class OpsReportContext(OpsReportHeader):
    facts: tuple[OpsReportFact, ...]

    @property
    def context_sha256(self) -> str:
        return sha256_hex(encode_json(self.model_dump(mode="json"), REPORT_LIMITS))


def _targets(value: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                v
                for k, v in value.items()
                if k
                in {
                    "candidate_service_key",
                    "source_candidate",
                    "target_candidate",
                    "caller_service_key",
                    "callee_service_key",
                    "source_service_key",
                    "target_service_key",
                }
                and type(v) is str
            }
        )
    )


def _field_unit(name: str, unit: str | None) -> str | None:
    if name.endswith("_count") or name in {"hop_count", "rank"}:
        return "count"
    if name.endswith("_ns"):
        return "unix_nanoseconds" if "unix" in name else "nanoseconds"
    if name.endswith("_seconds"):
        return "seconds"
    if name.endswith("_fraction") or name in {"coverage", "quality_factor", "path_quality"}:
        return "ratio"
    if name in {"value", "median", "mad", "minimum", "maximum"}:
        return unit
    # Unknown numeric quantities do not borrow a parent measurement's unit.
    return None


def _scalars(
    value: Any,
    *,
    path: str = "payload",
    targets: tuple[str, ...] = (),
    interval: TimeInterval | None = None,
    unit: str | None = None,
) -> Iterator[tuple[str, dict[str, Any], tuple[str, ...], TimeInterval | None, str | None]]:
    """Preserve every scalar and scalar-list, with exact record paths; no sampling."""
    if type(value) is dict:
        targets = _targets(value) or targets
        if type(value.get("interval")) is dict:
            interval = TimeInterval.model_validate(value["interval"])
        if "unit" in value:
            unit = value["unit"]
        scalars = {}
        for key, child in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
                raise ValueError("unregistered report field name")
            if type(child) not in (dict, list):
                scalars[key] = child
            elif type(child) is list and all(type(v) not in (dict, list) for v in child):
                scalars[key] = canonical_json_bytes(child).decode("utf-8")
        yield path, scalars, targets, interval, unit
        for key, child in value.items():
            if type(child) is dict or (
                type(child) is list and any(type(v) in (dict, list) for v in child)
            ):
                child_interval = interval
                if key in {"baseline", "fault"} and type(value.get(key + "_window")) is dict:
                    child_interval = TimeInterval.model_validate(value[key + "_window"])
                # These units come from the graph/node schemas, not field-name
                # guesses over user telemetry. In particular an edge nested in
                # a timing card must not inherit unix_nanoseconds.
                child_unit = {
                    "edge": None,
                    "edges": None,
                    "traffic": "count",
                    "error_count": "count",
                    "error_rate": "ratio",
                    "latency_p95_seconds": "seconds",
                    "coverage": "ratio",
                    "clock": None,
                    "interval": "unix_nanoseconds",
                    "baseline_window": "unix_nanoseconds",
                    "fault_window": "unix_nanoseconds",
                }.get(key, unit)
                yield from _scalars(
                    child,
                    path=path + "." + key,
                    targets=targets,
                    interval=child_interval,
                    unit=child_unit,
                )
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _scalars(
                child, path=f"{path}[{index}]", targets=targets, interval=interval, unit=unit
            )


def build_report_context(
    *,
    core: OpsCoreResult,
    sources: tuple[RepairMeasurementSource, ...],
    investigation: tuple[OpsInvestigationEvidence, ...],
    agent_result: OpsAgentResult | None = None,
    soft_targets: bool = False,
) -> OpsReportContext:
    """Trusted owner supplies admitted terminal core and saved tool receipts.

    This is not an API for model-authored cards. Hashes bind identity, not external
    authority; the public run owner additionally checks code/ledger and leakage.
    """
    from nurse_rcl.v2.live.normal_report import project_normal_evidence

    if type(core) is not OpsCoreResult:
        raise ValueError("complete terminal core required")
    core = OpsCoreResult.model_validate_json(core.model_dump_json())
    trace = (
        ()
        if agent_result is None
        else tuple(
            parse_decision(message["content"].encode("utf-8"), empirical=soft_targets)
            for message in agent_result.messages
            if message.get("role") == "assistant" and type(message.get("content")) is str
        )
    )
    if soft_targets:
        # Full model prose remains in the durable conversation, not repeated as
        # factual input. Validate every decision above before referencing it.
        trace = tuple(dict(
            decision_index=index, full_decision_sha256=sha256_hex(canonical_json_bytes(row)),
            action_count=len(row["actions"]), stopped=row["stop_reason"] is not None,
            unresolved_count=len(row["unresolved"]),
            omitted_detail="unverified model prose retained in complete conversation",
        ) for index, row in enumerate(trace, 1))
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"]
    if agent_result is None:
        investigation_status = "partial" if investigation else "disabled"
    elif agent_result.termination == "unavailable":
        investigation_status = "partial" if investigation else "unavailable"
    else:
        investigation_status = "completed"
    ranking = core.evidence.ranking
    candidates = set(ranking.candidate_service_keys)
    scores = {s.candidate_service_key: s for s in ranking.scores}
    projected = project_normal_evidence(core.evidence, sources=sources,
                                        ops_soft_targets=soft_targets)
    profile: Literal["v2-ops-isolated/1", "v2-ops-isolated/2"] = (
        "v2-ops-isolated/2" if soft_targets else "v2-ops-isolated/1"
    )
    facts = []
    mandatory: list[str] = []
    if projected.view is not None:
        for fact, binding in zip(projected.view.facts, projected.source_bindings, strict=True):
            if fact.kind == "limit":
                values = {v.name: v.value for v in fact.values}
                values.update(
                    profile=profile,
                    pre_ranking_semantics=False,
                    interpretation="core evaluation records; supplementary never scores",
                )
                fact = fact.model_copy(
                    update={
                        "values": tuple(
                            ReportEvidenceValue(name=k, value=v) for k, v in values.items()
                        ),
                        "status_codes": tuple(
                            sorted(
                                {
                                    "omitted_details_remain_in_complete_core"
                                    if "private_audit" in code
                                    else code
                                    for code in fact.status_codes
                                }
                            )
                        ),
                    }
                )
            facts.append(
                OpsReportFact(
                    fact=fact,
                    role="formal_record",
                    evidence_ids=(binding.source_id,),
                    source_refs=tuple(
                        dict.fromkeys((binding.source_sha256, *binding.same_source_ids))
                    ),
                    source_path=binding.source_kind,
                    origin="core_evaluation",
                )
            )
        mandatory.extend(projected.view.mandatory_fact_refs)
    indices = tuple(item.action_index for item in investigation)
    if len(set(indices)) != len(indices) or indices != tuple(sorted(indices)):
        raise ValueError("investigation receipt sequence differs")
    for item in investigation:
        node = item.evidence
        if node.case_snapshot_id != core.identity.case_snapshot_id:
            raise ValueError("report investigation snapshot differs")
        if node.tool_version == "ops-index-navigation/2":
            # A directory page records discovery, not measured system health.
            # Keep the complete receipt/action but never grant it fact aliases.
            continue
        body = json.loads(node.payload_json)
        parameters = json.loads(node.parameters_json)
        targets = _targets(body) or _targets(parameters)
        if not set(targets) <= candidates:
            raise ValueError("report investigation candidate differs")
        window = (
            None
            if node.window_ns is None
            else TimeInterval(
                start_unix_ns=node.window_ns[0],
                end_unix_ns=node.window_ns[1],
            )
        )
        for path, values, scoped, local_window, local_unit in _scalars(
            body,
            targets=targets,
            interval=window,
            unit=node.unit,
        ):
            if not set(scoped) <= candidates:
                raise ValueError("report nested candidate differs")
            metadata = dict(
                unit=local_unit,
                coverage=node.coverage,
                availability=node.availability,
                truth_value=node.truth_value if path == "payload" else "unknown",
                record_path=path,
            )
            metadata.update(values)
            metadata.update(
                {
                    name + "_unit": _field_unit(name, local_unit)
                    for name, value in values.items()
                    if type(value) in (int, float)
                    and len(name) <= 58
                    and name + "_unit" not in values
                }
            )
            fact = ReportEvidenceFact(
                fact_ref=f"f{len(facts) + 1}",
                kind="observation",
                target_service_keys=scoped,
                intervals=() if local_window is None else (local_window,),
                values=tuple(ReportEvidenceValue(name=k, value=v) for k, v in metadata.items()),
                status_codes=tuple(sorted(set(node.limitations))),
                allowed_claim_strength="frozen_observation",
                scope_basis="bound_observation",
            )
            facts.append(
                OpsReportFact(
                    fact=fact,
                    role="supplementary",
                    evidence_ids=node.evidence_ids,
                    source_refs=node.source_refs,
                    source_path=path,
                    action_index=item.action_index,
                    origin=item.origin,
                    reused_core_evidence_ids=item.reused_core_evidence_ids,
                )
            )
            mandatory.append(fact.fact_ref)
    result = OpsReportContext(
        schema_version="nurse.v2.ops-report-context/2" if soft_targets else (
            "nurse.v2.ops-report-context/1"
        ),
        profile=profile,
        opaque_case_id=core.identity.opaque_case_id,
        case_snapshot_id=core.identity.case_snapshot_id,
        core_result_digest=core.core_result_digest,
        formal_top_k=ranking.top_k_service_keys,
        ranking=tuple(
            OpsRankRow(
                rank=i,
                candidate_service_key=key,
                quantized_total_score=scores[key].quantized_total_score,
                local_score_component=scores[key].local_score_component,
                propagation_score_component=scores[key].propagation_score_component,
                local_evidence_status=scores[key].local_evidence_status,
            )
            for i, key in enumerate(ranking.ordered_service_keys, 1)
        ),
        facts=tuple(facts),
        mandatory_fact_refs=tuple(mandatory),
        investigation=investigation,
        formal_projection_available=projected.view is not None,
        investigation_status=investigation_status,
        investigation_failure_type=None if agent_result is None else agent_result.failure_type,
        model_trace_unverified=trace,
        formal_detail_omissions=() if projected.view is None else projected.view.omissions,
    )
    encode_json(result.model_dump(mode="json"), REPORT_LIMITS)
    return result
