"""D13 report supply: complete retained sources are not implicitly model-visible.

This module only projects already-admitted sources and delivery records. It
does not read telemetry, run tools, evaluate mechanisms, or invoke a provider.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_report import _fact_identity
from nurse_rcl.v2.live.ops_report_context import OpsReportContext, OpsReportFact, OpsReportHeader
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION, project_evidence_brief
from nurse_rcl.v2.live.ops_sre_comparable_briefs import project_comparable_brief
from nurse_rcl.v2.live.ops_sre_condition_briefs import project_condition_brief
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore, build_fact_store
from nurse_rcl.v2.live.ops_sre_focused_report_sources import (
    bind_retained_source as bind_retained_source,
)
from nurse_rcl.v2.live.ops_sre_focused_report_sources import (
    restore_retained_source as restore_retained_source,
)
from nurse_rcl.v2.live.ops_sre_navigation_failure import OpsNavigationFailureDetail
from nurse_rcl.v2.live.ops_sre_report_context import _iter_record_facts

BRIEF_SUPPLY_SCHEMA = "nurse.v2.sre-brief-supply/1"


def _brief_groups(
    row: dict[str, Any],
    *,
    condition_explanations: bool = False,
    comparable_evidence: bool = False,
) -> list[dict[str, Any]]:
    """No f aliases: each b has explicitly supplied scalar and list fields."""
    project = (
        project_comparable_brief
        if comparable_evidence
        else project_condition_brief
        if condition_explanations
        else project_evidence_brief
    )
    projected = project(row)
    if projected.get("representation") != BRIEF_REPRESENTATION:
        return []
    header = dict(
        source_record_sha256=projected["record_sha256"],
        card_id=row["card_id"],
        source_role=row["source_role"],
        source_refs=row.get("source_refs", []),
        target_service_keys=row.get("targets", []),
        window_ns=row.get("window_ns"),
        unit=row.get("unit"),
        limitations=row.get("limitations", []),
        expansion=projected["expansion"],
        allowed_step_refs=[],
        authority="condition_summary_not_hidden_leaf_or_independent_observation",
    )
    payload = projected["payload"]
    if comparable_evidence:
        header["mechanism_source_kind"] = projected["mechanism_source_kind"]
    if "formal_decision" in payload:
        records = [
            ("payload.formal_decision", "formal_mechanism_reference", payload["formal_decision"]),
            (
                "payload.supplementary_necessary_support",
                "supplementary_support",
                dict(
                    supplementary_necessary_support=payload["supplementary_necessary_support"],
                    score_authority=payload["score_authority"],
                ),
            ),
        ]
    else:
        records = [("payload.decision", "mechanism_decision", payload["decision"])]
    records.extend(
        (f"payload.conditions[{i}]", "mechanism_condition", condition)
        for i, condition in enumerate(projected["payload"]["conditions"])
    )
    groups = []
    for path, kind, values in records:
        fields = {k: v for k, v in values.items() if type(v) not in (list, dict)}
        lists = {k: v for k, v in values.items() if type(v) is list}
        if kind == "mechanism_condition":
            fields.update(
                {key + "_count": count for key, count in values["leaf_truth_counts"].items()}
            )
        if comparable_evidence:
            decision = payload.get("decision", payload.get("formal_decision"))
            fields.update(
                mechanism_key=decision["mechanism_key"],
                mechanism_definition_sha256=decision["mechanism_definition_sha256"],
                disposition_rule=payload["disposition_rule"],
                score_effect_rule=payload["score_effect_rule"],
            )
        groups.append(dict(header, source_path=path, granularity=kind, fields=fields, lists=lists))
    return groups


def _validate_source(
    source: dict[str, Any],
    store: NativeFactStore,
    *,
    action_index: int,
    candidates: set[str],
    focused: bool = False,
) -> None:
    row, refs = source["record"], source["fact_refs"]
    generated = list(
        _source_facts(row, action_index=action_index, candidates=candidates, focused=focused)
    )
    if len(refs) != len(set(refs)) or len(refs) != len(generated):
        raise ValueError("brief source fact closure differs")
    by_path = {item.source_path: item for item in generated}
    actual_paths = [store.resolve(ref).source_path for ref in refs]
    if (
        len(by_path) != len(generated)
        or len(set(actual_paths)) != len(refs)
        or (set(actual_paths) != set(by_path))
    ):
        raise ValueError("brief source path closure differs")
    for ref in refs:
        actual = store.resolve(ref)
        expected = by_path[actual.source_path]
        expected = expected.model_copy(
            update={"fact": expected.fact.model_copy(update={"fact_ref": ref})}
        )
        if _fact_identity(actual) != _fact_identity(expected):
            raise ValueError("brief source differs from complete local facts")


def build_brief_supply(
    *,
    core_result_digest: str,
    fact_store: NativeFactStore,
    formal_fact_refs: tuple[str, ...],
    sources: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    deliveries: list[dict[str, Any]],
    condition_explanations: bool = False,
    historical_briefs: bool = False,
    comparable_evidence: bool = False,
    focused: bool = False,
) -> dict[str, Any]:
    """Build a stable b/f permission set from actual successful deliveries.

    sources holds admitted full records and their complete local f references;
    deliveries holds exactly the row each step received (brief or exact).
    The caller binds case/snapshot and persists the complete fact store separately.
    """
    require_pin(core_result_digest)
    if comparable_evidence and (not condition_explanations or historical_briefs):
        raise ValueError("comparable supply requires its exact new condition protocol")
    project = (
        project_comparable_brief
        if comparable_evidence
        else project_condition_brief
        if condition_explanations
        else project_evidence_brief
    )
    if historical_briefs and not condition_explanations:
        raise ValueError("historical brief projection requires diagnostic condition supply")
    if type(fact_store) is not NativeFactStore:
        raise ValueError("brief supply requires native local facts")
    if len(set(formal_fact_refs)) != len(formal_fact_refs):
        raise ValueError("duplicate formal fact references")
    for ref in formal_fact_refs:
        if fact_store.resolve(ref).role != "formal_record":
            raise ValueError("formal supply references a supplementary source")
    if len(steps) > 8 or [s["step_ref"] for s in steps] != [
        f"s{i}" for i in range(1, len(steps) + 1)
    ]:
        raise ValueError("brief supply step order differs")
    by_step = {s["step_ref"]: s for s in steps}
    supplied_steps = [
        dict(
            step_ref=s["step_ref"],
            decision_index=s["decision_index"],
            action_index=s["action_index"],
            planned=s["planned"],
            executed={
                k: v
                for k, v in s["executed"].items()
                if k in {"status", "tool", "parameters", "failure_type", "origin"}
            },
            responded_to=s["responded_to"],
            response_state=s["response_state"],
            delivered_brief_refs=[],
            delivered_fact_refs=[],
        )
        for s in steps
    ]
    brief_step = {s["step_ref"]: s for s in supplied_steps}
    source_map: dict[str, dict[str, Any]] = {}
    source_steps: dict[str, list[str]] = {}
    for source in sources:
        identity = sha256_hex(canonical_json_bytes(source["record"]))
        if identity in source_map:
            raise ValueError("duplicate brief source")
        source_map[identity] = source
        source_steps[identity] = []
    validated_deliveries = []
    legacy_sources: set[str] = set()
    for delivered in deliveries:
        step_ref, identity = delivered["step_ref"], delivered["source_record_sha256"]
        if step_ref not in by_step or identity not in source_map:
            raise ValueError("brief delivery source or step differs")
        step = by_step[step_ref]
        if step["executed"]["status"] != "completed" or type(step["action_index"]) is not int:
            raise ValueError("unsuccessful step cannot supply evidence")
        row = delivered["record"]
        source = source_map[identity]["record"]
        projected = project(source)
        is_brief = row.get("representation") == BRIEF_REPRESENTATION
        if (
            historical_briefs
            and is_brief
            and row.get("schema_version") == "nurse.v2.sre-evidence-brief/1"
        ):
            projected = project_evidence_brief(source)
            legacy_sources.add(identity)
        if canonical_json_bytes(row) != canonical_json_bytes(projected if is_brief else source):
            raise ValueError("brief delivery differs from retained source")
        if step_ref not in source_steps[identity]:
            source_steps[identity].append(step_ref)
        validated_deliveries.append((delivered, is_brief))
    briefs: list[dict[str, Any]] = []
    grouped_refs = {}
    candidates = {
        target for item in fact_store.iter_facts() for target in item.fact.target_service_keys
    }
    for identity, source in source_map.items():
        seen_steps = source_steps[identity]
        if not seen_steps:
            continue  # Retained but not successfully delivered grants no supply.
        _validate_source(
            source,
            fact_store,
            candidates=candidates,
            action_index=min(by_step[s]["action_index"] for s in seen_steps),
            focused=focused,
        )
        refs = []
        for group in _brief_groups(
            source["record"],
            condition_explanations=condition_explanations and identity not in legacy_sources,
            comparable_evidence=comparable_evidence,
        ):
            ref = f"b{len(briefs) + 1}"
            briefs.append(dict(group, evidence_ref=ref, allowed_step_refs=list(seen_steps)))
            refs.append(ref)
        grouped_refs[identity] = refs
    provided_facts = set(formal_fact_refs)
    for delivered, is_brief in validated_deliveries:
        identity = delivered["source_record_sha256"]
        step = brief_step[delivered["step_ref"]]
        step["delivered_brief_refs"].extend(grouped_refs.get(identity, []))
        if not is_brief:
            refs = source_map[identity]["fact_refs"]
            step["delivered_fact_refs"].extend(refs)
            provided_facts.update(refs)
    for step in supplied_steps:
        for field in ("delivered_brief_refs", "delivered_fact_refs"):
            step[field] = list(dict.fromkeys(step[field]))
    ordered = [ref for ref in fact_store.refs() if ref in provided_facts]
    # Return a detached object: a caller cannot mutate the source ledger via it.
    return decode_json_object(
        canonical_json_bytes(
            dict(
                schema_version="nurse.v2.sre-brief-supply/3"
                if comparable_evidence
                else "nurse.v2.sre-brief-supply/2"
                if condition_explanations
                else BRIEF_SUPPLY_SCHEMA,
                core_result_digest=core_result_digest,
                briefs=briefs,
                provided_brief_refs=[b["evidence_ref"] for b in briefs],
                provided_fact_refs=ordered,
                formal_fact_refs=list(formal_fact_refs),
                facts=[fact_store.resolve(ref).model_dump(mode="json") for ref in ordered],
                steps=supplied_steps,
                semantics="only_supplied_b_fields_and_f_values_are_assertable; "
                "local_fact_store_does_not_grant_hidden_detail_permission",
            )
        ),
        REPORT_LIMITS,
    )


def _source_facts(
    row: dict[str, Any],
    *,
    action_index: int,
    candidates: set[str],
    focused: bool = False,
) -> Iterator[OpsReportFact]:
    if row.get("tool_version") == "sre-score-lineage/1":
        from nurse_rcl.v2.live.ops_sre_score_facts import score_record_facts

        yield from score_record_facts(row, action_index=action_index, candidates=candidates)
        return
    for item in _iter_record_facts(row, action_index=action_index, candidates=candidates):
        if row["source_role"] == "query_scope_metadata":
            item = item.model_copy(
                update={
                    "fact": item.fact.model_copy(
                        update=dict(
                            kind="uncertainty",
                            scope_basis="frozen_assessment",
                            allowed_claim_strength="ranking_inference",
                        )
                    )
                }
            )
        yield item
    if focused and "source_attribution" in row:
        # Attribution is the identity of this same delivered record, not another vote.
        from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceValue

        original = next(_iter_record_facts(row, action_index=action_index, candidates=candidates))

        def attributes(body: Any, path: str) -> Iterator[tuple[str, dict[str, Any]]]:
            # Cell names such as fields.unit are structure, not inherited unit declarations.
            if type(body) is dict:
                values = {
                    key: canonical_json_bytes(value).decode() if type(value) is list else value
                    for key, value in body.items()
                    if type(value) not in (dict, list)
                    or (type(value) is list and all(type(v) not in (dict, list) for v in value))
                }
                yield path, values
                for key, value in body.items():
                    if type(value) in (dict, list) and key not in values:
                        yield from attributes(value, path + "." + key)
            elif type(body) is list:
                for index, value in enumerate(body):
                    yield from attributes(value, f"{path}[{index}]")

        for path, values in attributes(row["source_attribution"], "source_attribution"):
            values.update(
                record_path=path,
                representation=row.get("representation", "full_semantic_record"),
                source_role=row["source_role"],
            )
            yield original.model_copy(
                update=dict(
                    source_path=path,
                    fact=original.fact.model_copy(
                        update={
                            "values": tuple(
                                ReportEvidenceValue(name=key, value=value)
                                for key, value in values.items()
                            ),
                            "status_codes": tuple(
                                sorted(
                                    {
                                        *original.fact.status_codes,
                                        "same_record_attribution_not_independent_evidence",
                                    }
                                )
                            ),
                        }
                    ),
                )
            )


class BriefReportFields(StrictFrozenModel):
    evidence: OpsReportHeader
    steps: tuple[dict[str, Any], ...] = Field(max_length=8)
    source_bindings: tuple[dict[str, Any], ...]
    delivery_bindings: tuple[dict[str, Any], ...]
    formal_fact_refs: tuple[str, ...]
    termination: Literal["model_stop", "decision_budget", "unavailable", "disabled"]
    action_seconds: tuple[float, ...] = Field(max_length=8)
    decision_seconds: tuple[float, ...] = Field(max_length=8)


class BriefReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/6"] = "nurse.v2.sre-report-context/6"
    profile: Literal["hermes-sre-tools/6"] = "hermes-sre-tools/6"
    # The older value is read-only saved-trajectory diagnostics; live owner rejects it.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/6", "nurse.v2.sre-investigation-input/7"
    ] = "nurse.v2.sre-investigation-input/7"


class ConditionReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/7"] = "nurse.v2.sre-report-context/7"
    profile: Literal["hermes-sre-tools/7"] = "hermes-sre-tools/7"
    # /7 is read-only D13 saved-material diagnostics. Live owners require /8.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/7", "nurse.v2.sre-investigation-input/8"
    ] = "nurse.v2.sre-investigation-input/8"


class RecoveryReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/8"] = "nurse.v2.sre-report-context/8"
    profile: Literal["hermes-sre-tools/8"] = "hermes-sre-tools/8"
    # /8 is saved D14 material, never eligible for the live report owner.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/8", "nurse.v2.sre-investigation-input/9"
    ] = "nurse.v2.sre-investigation-input/9"


class ComparableReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/9"] = "nurse.v2.sre-report-context/9"
    profile: Literal["hermes-sre-tools/9"] = "hermes-sre-tools/9"
    # /9 retains D15 supply/2 verbatim for offline diagnosis only. Both live
    # owners reject it before credential access; it is not a migrated session.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/9", "nurse.v2.sre-investigation-input/10"
    ] = "nurse.v2.sre-investigation-input/10"


class NavigationReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/10"] = "nurse.v2.sre-report-context/10"
    profile: Literal["hermes-sre-tools/10"] = "hermes-sre-tools/10"
    # /10 is saved D16a projection ONLY. Live owners require /11.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/10", "nurse.v2.sre-investigation-input/11"
    ] = "nurse.v2.sre-investigation-input/11"
    investigation_failure: OpsNavigationFailureDetail | None = None


class WorkbenchReportMetadata(BriefReportFields):
    schema_version: Literal["nurse.v2.sre-report-context/11"] = "nurse.v2.sre-report-context/11"
    profile: Literal["hermes-sre-tools/11"] = "hermes-sre-tools/11"
    # /11 is saved D17 pure projection only; both live owners require /12.
    investigation_input_schema: Literal[
        "nurse.v2.sre-investigation-input/11", "nurse.v2.sre-investigation-input/12"
    ] = "nurse.v2.sre-investigation-input/12"
    investigation_failure: OpsNavigationFailureDetail | None = None


class UsedSourceReportMetadata(WorkbenchReportMetadata):
    # The subclass intentionally narrows runtime Pydantic identity to the new version.
    schema_version: Literal["nurse.v2.sre-report-context/12"] = "nurse.v2.sre-report-context/12"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/12"] = "hermes-sre-tools/12"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/13"] = (
        "nurse.v2.sre-investigation-input/13"  # type: ignore[assignment]
    )


class ReceiptReuseReportMetadata(UsedSourceReportMetadata):
    schema_version: Literal["nurse.v2.sre-report-context/13"] = "nurse.v2.sre-report-context/13"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/13"] = "hermes-sre-tools/13"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/14"] = (
        "nurse.v2.sre-investigation-input/14"  # type: ignore[assignment]
    )


class BalancedReportMetadata(ReceiptReuseReportMetadata):
    schema_version: Literal["nurse.v2.sre-report-context/14"] = "nurse.v2.sre-report-context/14"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/14"] = "hermes-sre-tools/14"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/15"] = (
        "nurse.v2.sre-investigation-input/15"  # type: ignore[assignment]
    )


class ReplanningReportMetadata(BalancedReportMetadata):
    termination: Literal[  # type: ignore[assignment]
        "model_stop", "decision_budget", "unavailable", "disabled", "no_progress_after_correction"
    ]
    schema_version: Literal["nurse.v2.sre-report-context/15"] = "nurse.v2.sre-report-context/15"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/15"] = "hermes-sre-tools/15"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/16"] = (
        "nurse.v2.sre-investigation-input/16"  # type: ignore[assignment]
    )


class QuestionReportMetadata(ReplanningReportMetadata):
    schema_version: Literal["nurse.v2.sre-report-context/16"] = "nurse.v2.sre-report-context/16"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/16"] = "hermes-sre-tools/16"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/17"] = (
        "nurse.v2.sre-investigation-input/17"  # type: ignore[assignment]
    )
    question_state: dict[str, Any]
    diagnostic_semantics: dict[str, Any]


class ActionReportMetadata(QuestionReportMetadata):
    schema_version: Literal["nurse.v2.sre-report-context/17"] = "nurse.v2.sre-report-context/17"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/17"] = "hermes-sre-tools/17"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/18"] = (
        "nurse.v2.sre-investigation-input/18"  # type: ignore[assignment]
    )


class AtomicReportMetadata(ActionReportMetadata):
    """Report-only successor; never enables the investigation's disabled report slot."""

    schema_version: Literal["nurse.v2.sre-report-context/19"] = "nurse.v2.sre-report-context/19"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/23"] = "hermes-sre-tools/23"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/24"] = (
        "nurse.v2.sre-investigation-input/24"  # type: ignore[assignment]
    )
    termination: Literal[  # type: ignore[assignment]
        "model_stop",
        "decision_budget",
        "unavailable",
        "no_progress_after_correction",
        "stage_pause",
    ]
    source_identity: dict[str, Any]
    handoff_sha256: str


class FocusedReportMetadata(AtomicReportMetadata):
    """Explicit focused report source; never an old atomic-report source identity."""

    schema_version: Literal["nurse.v2.sre-report-context/20"] = "nurse.v2.sre-report-context/20"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/25"] = "hermes-sre-tools/25"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/26"] = (
        "nurse.v2.sre-investigation-input/26"  # type: ignore[assignment]
    )


@dataclass(frozen=True)
class BriefSreReportContext:
    metadata: (
        BriefReportMetadata
        | ConditionReportMetadata
        | RecoveryReportMetadata
        | ComparableReportMetadata
        | NavigationReportMetadata
        | WorkbenchReportMetadata
        | UsedSourceReportMetadata
        | ReceiptReuseReportMetadata
        | BalancedReportMetadata
        | ReplanningReportMetadata
        | QuestionReportMetadata
        | ActionReportMetadata
        | AtomicReportMetadata
        | FocusedReportMetadata
    )
    fact_store: NativeFactStore
    supply: dict[str, Any]  # Values for provided f refs live only in fact_store.

    @property
    def core_result_digest(self) -> str:
        return self.metadata.evidence.core_result_digest

    @property
    def context_sha256(self) -> str:
        return sha256_hex(encode_brief_report_storage(self))


def brief_report_supply(context: BriefSreReportContext) -> dict[str, Any]:
    """Caller already admitted context; do not replay all sources per claim."""
    supply = decode_json_object(encode_json(context.supply, REPORT_LIMITS), REPORT_LIMITS)
    if "facts" in supply:
        raise ValueError("brief retained supply duplicates exact facts")
    supply["facts"] = [
        context.fact_store.resolve(ref).model_dump(mode="json")
        for ref in supply["provided_fact_refs"]
    ]
    return supply


def _retained_supply(
    metadata: BriefReportMetadata
    | ConditionReportMetadata
    | RecoveryReportMetadata
    | ComparableReportMetadata
    | NavigationReportMetadata
    | WorkbenchReportMetadata
    | UsedSourceReportMetadata
    | ReceiptReuseReportMetadata
    | BalancedReportMetadata
    | ReplanningReportMetadata
    | QuestionReportMetadata
    | ActionReportMetadata
    | AtomicReportMetadata
    | FocusedReportMetadata,
    store: NativeFactStore,
) -> dict[str, Any]:
    comparable_container = isinstance(
        metadata, ComparableReportMetadata | NavigationReportMetadata | WorkbenchReportMetadata
    )
    comparable = isinstance(metadata, NavigationReportMetadata | WorkbenchReportMetadata) or (
        comparable_container
        and (metadata.investigation_input_schema == "nurse.v2.sre-investigation-input/10")
    )
    condition = (
        isinstance(metadata, ConditionReportMetadata | RecoveryReportMetadata)
        or comparable_container
    )
    diagnostic = (
        condition and metadata.investigation_input_schema == "nurse.v2.sre-investigation-input/7"
    )
    project = (
        project_comparable_brief
        if comparable
        else project_condition_brief
        if condition
        else project_evidence_brief
    )
    sources = [restore_retained_source(b, store) for b in metadata.source_bindings]
    by_digest = {
        b["record_sha256"]: s["record"]
        for b, s in zip(metadata.source_bindings, sources, strict=True)
    }
    deliveries = []
    seen: set[tuple[str, str, str]] = set()
    for binding in metadata.delivery_bindings:
        if set(binding) != {"step_ref", "source_record_sha256", "kind", "delivered_sha256"} or (
            binding["kind"]
            not in ({"brief", "exact", "legacy_brief"} if diagnostic else {"brief", "exact"})
        ):
            raise ValueError("brief delivery binding shape differs")
        identity = (binding["step_ref"], binding["source_record_sha256"], binding["kind"])
        if identity in seen or binding["source_record_sha256"] not in by_digest:
            raise ValueError("brief delivery source differs")
        seen.add(identity)
        source = by_digest[binding["source_record_sha256"]]
        row = (
            project_evidence_brief(source)
            if binding["kind"] == "legacy_brief"
            else project(source)
            if binding["kind"] == "brief"
            else source
        )
        if sha256_hex(canonical_json_bytes(row)) != binding["delivered_sha256"]:
            raise ValueError("brief delivery source digest differs")
        deliveries.append(
            dict(
                step_ref=binding["step_ref"],
                record=row,
                source_record_sha256=binding["source_record_sha256"],
            )
        )
    supply = build_brief_supply(
        core_result_digest=metadata.evidence.core_result_digest,
        fact_store=store,
        formal_fact_refs=metadata.formal_fact_refs,
        sources=sources,
        steps=list(metadata.steps),
        deliveries=deliveries,
        condition_explanations=condition,
        historical_briefs=diagnostic,
        comparable_evidence=comparable,
        focused=type(metadata) is FocusedReportMetadata,
    )
    if isinstance(metadata, QuestionReportMetadata):
        state, semantics = question_report_material(
            list(metadata.steps),
            [
                (
                    next(
                        s["action_index"] for s in metadata.steps if s["step_ref"] == d["step_ref"]
                    ),
                    d["record"],
                )
                for d in deliveries
            ],
            atomic=type(metadata) is AtomicReportMetadata,
            focused=type(metadata) is FocusedReportMetadata,
        )
        if metadata.question_state != state or metadata.diagnostic_semantics != semantics:
            raise ValueError("D26 report material differs from retained trajectory")
    supply.pop("facts")
    return supply


def question_report_material(
    steps: list[dict[str, Any]],
    deliveries: list[tuple[int, dict[str, Any]]],
    *,
    atomic: bool = False,
    focused: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state
    from nurse_rcl.v2.live.ops_sre_d26_semantics import semantics_for_deliveries

    memory = [
        s["planned"]
        for s in steps
        if ("investigation_plan" in s["planned"] or "annotation_admission" in s["planned"])
    ]
    if focused:
        from nurse_rcl.v2.live.ops_sre_focused_questions import project_focused_questions

        return project_focused_questions(memory), semantics_for_deliveries(deliveries)
    if atomic:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_questions import project_atomic_questions

        return project_atomic_questions(memory), semantics_for_deliveries(deliveries)
    return project_question_state(memory), semantics_for_deliveries(deliveries)


def _brief_storage_packet(context: BriefSreReportContext) -> dict[str, Any]:
    """Content snapshot only; not source admission and never a public bypass."""
    return dict(
        schema_version="nurse.v2.sre-report-storage/16"
        if type(context.metadata) is FocusedReportMetadata
        else "nurse.v2.sre-report-storage/15"
        if type(context.metadata) is AtomicReportMetadata
        else "nurse.v2.sre-report-storage/14"
        if isinstance(context.metadata, ActionReportMetadata)
        else "nurse.v2.sre-report-storage/13"
        if isinstance(context.metadata, QuestionReportMetadata)
        else "nurse.v2.sre-report-storage/12"
        if isinstance(context.metadata, ReplanningReportMetadata)
        else "nurse.v2.sre-report-storage/11"
        if isinstance(context.metadata, BalancedReportMetadata)
        else "nurse.v2.sre-report-storage/10"
        if isinstance(context.metadata, ReceiptReuseReportMetadata)
        else "nurse.v2.sre-report-storage/9"
        if isinstance(context.metadata, UsedSourceReportMetadata)
        else "nurse.v2.sre-report-storage/8"
        if isinstance(context.metadata, WorkbenchReportMetadata)
        else "nurse.v2.sre-report-storage/7"
        if isinstance(context.metadata, NavigationReportMetadata)
        else "nurse.v2.sre-report-storage/6"
        if isinstance(context.metadata, ComparableReportMetadata)
        else "nurse.v2.sre-report-storage/5"
        if isinstance(context.metadata, RecoveryReportMetadata)
        else "nurse.v2.sre-report-storage/4"
        if isinstance(context.metadata, ConditionReportMetadata)
        else "nurse.v2.sre-report-storage/3",
        metadata=context.metadata.model_dump(mode="json"),
        fact_records=context.fact_store.to_packet(),
        supply=context.supply,
    )


def _require_brief_unchanged(context: BriefSreReportContext, raw: bytes) -> None:
    if encode_json(_brief_storage_packet(context), REPORT_LIMITS) != raw:
        raise ValueError("brief admitted context changed during call")


def validate_brief_report(context: BriefSreReportContext) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_provider import _require_outbound
    from nurse_rcl.v2.live.ops_sre_report_bounds import _measure

    if (
        type(context) is not BriefSreReportContext
        or type(context.metadata)
        not in {
            BriefReportMetadata,
            ConditionReportMetadata,
            RecoveryReportMetadata,
            ComparableReportMetadata,
            NavigationReportMetadata,
            WorkbenchReportMetadata,
            UsedSourceReportMetadata,
            ReceiptReuseReportMetadata,
            BalancedReportMetadata,
            ReplanningReportMetadata,
            QuestionReportMetadata,
            ActionReportMetadata,
            AtomicReportMetadata,
            FocusedReportMetadata,
        }
        or (type(context.fact_store) is not NativeFactStore)
    ):
        raise ValueError("brief report context type differs")
    metadata = context.metadata.model_dump(mode="json")
    packet = _brief_storage_packet(context)
    _measure(packet, node_limit=100_000, boundary="storage_nodes")
    admitted = type(context.metadata).model_validate_json(
        encode_json(metadata, REPORT_LIMITS), strict=True
    )
    if type(admitted) is AtomicReportMetadata:
        from nurse_rcl.v2.live.ops_sre_atomic_report import validate_atomic_report_identity

        validate_atomic_report_identity(admitted)
    elif type(admitted) is FocusedReportMetadata:
        from nurse_rcl.v2.live.ops_sre_focused_report import validate_focused_report_identity

        validate_focused_report_identity(admitted)
    if type(admitted) is not FocusedReportMetadata and any(
        "schema_version" in binding for binding in admitted.source_bindings
    ):
        raise ValueError("legacy report cannot adopt focused attribution bindings")
    _require_outbound(
        packet, case=admitted.evidence.opaque_case_id, snapshot=admitted.evidence.case_snapshot_id
    )
    refs = context.fact_store.refs()
    grouped = [ref for b in admitted.source_bindings for ref in b["fact_refs"]]
    if (
        len(set(grouped)) != len(grouped)
        or set(grouped) & set(admitted.formal_fact_refs)
        or (
            set(grouped) | set(admitted.formal_fact_refs) != set(refs)
            or not set(admitted.evidence.mandatory_fact_refs) <= set(refs)
        )
    ):
        raise ValueError("brief report retained source closure differs")
    candidates = {r.candidate_service_key for r in admitted.evidence.ranking}
    for item in context.fact_store.iter_facts():
        if not set(item.fact.target_service_keys) <= candidates:
            raise ValueError("brief report retained candidate differs")
    if canonical_json_bytes(_retained_supply(admitted, context.fact_store)) != (
        canonical_json_bytes(context.supply)
    ):
        raise ValueError("brief report supply differs from retained source and delivery")
    return packet


def encode_brief_report_storage(context: BriefSreReportContext) -> bytes:
    return encode_json(validate_brief_report(context), REPORT_LIMITS)


def decode_brief_report_storage(raw: bytes) -> BriefSreReportContext:
    packet = decode_json_object(raw, REPORT_LIMITS)
    if set(packet) != {"schema_version", "metadata", "fact_records", "supply"} or (
        packet["schema_version"]
        not in {
            "nurse.v2.sre-report-storage/3",
            "nurse.v2.sre-report-storage/4",
            "nurse.v2.sre-report-storage/5",
            "nurse.v2.sre-report-storage/6",
            "nurse.v2.sre-report-storage/7",
            "nurse.v2.sre-report-storage/8",
            "nurse.v2.sre-report-storage/9",
            "nurse.v2.sre-report-storage/10",
            "nurse.v2.sre-report-storage/11",
            "nurse.v2.sre-report-storage/12",
            "nurse.v2.sre-report-storage/13",
            "nurse.v2.sre-report-storage/14",
            "nurse.v2.sre-report-storage/15",
            "nurse.v2.sre-report-storage/16",
        }
    ):
        raise ValueError("brief storage identity differs")
    metadata_type = (
        FocusedReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/16"
        else AtomicReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/15"
        else ActionReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/14"
        else QuestionReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/13"
        else ReplanningReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/12"
        else BalancedReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/11"
        else ReceiptReuseReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/10"
        else UsedSourceReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/9"
        else WorkbenchReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/8"
        else NavigationReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/7"
        else ComparableReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/6"
        else RecoveryReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/5"
        else ConditionReportMetadata
        if packet["schema_version"] == "nurse.v2.sre-report-storage/4"
        else BriefReportMetadata
    )
    context = BriefSreReportContext(
        metadata=metadata_type.model_validate_json(
            encode_json(packet["metadata"], REPORT_LIMITS), strict=True
        ),
        fact_store=NativeFactStore.from_packet(packet["fact_records"]),
        supply=packet["supply"],
    )
    if encode_brief_report_storage(context) != raw:
        raise ValueError("brief report storage is not canonical")
    return context


def _assemble_brief_sources(
    *,
    formal: OpsReportContext,
    steps: list[dict[str, Any]],
    records: tuple[OpsActionRecord, ...],
    retained_brief_source: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    focused: bool = False,
) -> tuple[
    NativeFactStore, list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, dict[str, Any]]]
]:
    """Shared pure source projection. Accessors must already have admitted their sources."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records, _receipt
    from nurse_rcl.v2.live.ops_sre_report_context import _candidate_scopes, _inspection_scope
    from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

    step_for_action = {s["action_index"]: s["step_ref"] for s in steps}
    candidates = {r.candidate_service_key for r in formal.ranking}
    sources: dict[str, dict[str, Any]] = {}
    deliveries: list[dict[str, Any]] = []
    semantic_deliveries: list[tuple[int, dict[str, Any]]] = []

    def facts() -> Iterator[OpsReportFact]:
        yield from formal.facts
        count = len(formal.facts)
        for record in records:
            _, view = _receipt(record)
            if record.result is None or record.status != "completed":
                continue
            if record.result.evidence.case_snapshot_id != formal.case_snapshot_id:
                raise ValueError("brief source receipt snapshot differs")
            rows = _delivered_records(working_evidence_view(view))
            scope = _inspection_scope(view, record, working=True, brief=True)
            if scope is not None:
                rows.append(scope)
                for row in _candidate_scopes(view, scope):
                    payload = dict(row["payload"])
                    route = payload.pop("expansion")
                    rows.append(
                        dict(
                            row,
                            payload=payload,
                            navigation=[dict(path=["expansion"], expansion=route)],
                        )
                    )
            for row in rows:
                semantic_deliveries.append((record.index, row))
                is_brief = row.get("representation") == BRIEF_REPRESENTATION
                source = retained_brief_source(view, row) if is_brief else row
                identity = sha256_hex(canonical_json_bytes(source))
                if identity not in sources:
                    refs = []
                    for item in _source_facts(
                        source,
                        action_index=record.index,
                        candidates=candidates,
                        focused=focused,
                    ):
                        count += 1
                        ref = f"f{count}"
                        refs.append(ref)
                        yield item.model_copy(
                            update={"fact": item.fact.model_copy(update={"fact_ref": ref})}
                        )
                    sources[identity] = dict(record=source, fact_refs=refs)
                deliveries.append(
                    dict(
                        step_ref=step_for_action[record.index],
                        source_record_sha256=identity,
                        kind="brief" if is_brief else "exact",
                        delivered_sha256=sha256_hex(canonical_json_bytes(row)),
                    )
                )

    store = build_fact_store(facts())
    return store, list(sources.values()), deliveries, semantic_deliveries


def assemble_brief_report_context(
    *,
    formal: OpsReportContext,
    decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    toolbox: Any,
    investigation_status: Literal["completed", "partial", "unavailable", "disabled"],
    termination: Literal[
        "model_stop", "decision_budget", "unavailable", "disabled", "no_progress_after_correction"
    ],
    failure_type: str | None = None,
    investigation_failure: OpsNavigationFailureDetail | None = None,
) -> BriefSreReportContext:
    """Read only admitted cards, never execute a tool or replay raw telemetry."""
    from nurse_rcl.v2.live.ops_sre_action_tools import OpsSreActionToolbox
    from nurse_rcl.v2.live.ops_sre_brief_tools import OpsSreBriefToolbox
    from nurse_rcl.v2.live.ops_sre_comparable_tools import OpsSreComparableToolbox
    from nurse_rcl.v2.live.ops_sre_condition_tools import OpsSreConditionToolbox
    from nurse_rcl.v2.live.ops_sre_context import (
        BRIEF_INPUT_SCHEMA,
        investigation_steps,
    )
    from nurse_rcl.v2.live.ops_sre_d21_tools import OpsSreUsedSourceToolbox
    from nurse_rcl.v2.live.ops_sre_d22_tools import OpsSreReceiptReuseToolbox
    from nurse_rcl.v2.live.ops_sre_d23_tools import OpsSreBalancedToolbox
    from nurse_rcl.v2.live.ops_sre_d25_tools import OpsSreReplanningToolbox
    from nurse_rcl.v2.live.ops_sre_d26_tools import OpsSreQuestionToolbox
    from nurse_rcl.v2.live.ops_sre_feedback_context import FEEDBACK_INPUT_SCHEMA
    from nurse_rcl.v2.live.ops_sre_navigation_tools import OpsSreNavigationToolbox
    from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsSreRecoveryToolbox
    from nurse_rcl.v2.live.ops_sre_workbench_tools import OpsSreWorkbenchToolbox

    if (
        type(toolbox)
        not in {
            OpsSreBriefToolbox,
            OpsSreConditionToolbox,
            OpsSreRecoveryToolbox,
            OpsSreComparableToolbox,
            OpsSreNavigationToolbox,
            OpsSreWorkbenchToolbox,
            OpsSreUsedSourceToolbox,
            OpsSreReceiptReuseToolbox,
            OpsSreBalancedToolbox,
            OpsSreReplanningToolbox,
            OpsSreQuestionToolbox,
            OpsSreActionToolbox,
        }
        or toolbox.case_snapshot_id != formal.case_snapshot_id
    ):
        raise ValueError("brief source toolbox identity differs")
    action_first = type(toolbox) is OpsSreActionToolbox
    question_led = type(toolbox) is OpsSreQuestionToolbox or action_first
    replanning = type(toolbox) is OpsSreReplanningToolbox or question_led
    balanced = type(toolbox) is OpsSreBalancedToolbox or replanning
    receipt_reuse = type(toolbox) is OpsSreReceiptReuseToolbox or balanced
    used_source = type(toolbox) is OpsSreUsedSourceToolbox or receipt_reuse
    workbench = type(toolbox) is OpsSreWorkbenchToolbox or used_source
    navigation = type(toolbox) is OpsSreNavigationToolbox or workbench
    comparable = type(toolbox) is OpsSreComparableToolbox or navigation
    recovery = type(toolbox) is OpsSreRecoveryToolbox
    condition = type(toolbox) is OpsSreConditionToolbox or recovery or comparable
    steps = investigation_steps(
        decisions=decisions,
        records=records,
        input_schema="nurse.v2.sre-investigation-input/18"
        if action_first
        else "nurse.v2.sre-investigation-input/17"
        if question_led
        else "nurse.v2.sre-investigation-input/16"
        if replanning
        else "nurse.v2.sre-investigation-input/15"
        if balanced
        else "nurse.v2.sre-investigation-input/14"
        if receipt_reuse
        else "nurse.v2.sre-investigation-input/13"
        if used_source
        else "nurse.v2.sre-investigation-input/12"
        if workbench
        else "nurse.v2.sre-investigation-input/11"
        if navigation
        else "nurse.v2.sre-investigation-input/10"
        if comparable
        else "nurse.v2.sre-investigation-input/9"
        if recovery
        else FEEDBACK_INPUT_SCHEMA
        if condition
        else BRIEF_INPUT_SCHEMA,
    )
    store, sources, deliveries, semantic_deliveries = _assemble_brief_sources(
        formal=formal,
        steps=steps,
        records=records,
        retained_brief_source=toolbox.retained_brief_source,
    )
    header = formal.model_dump(mode="json", exclude={"facts"})
    header.update(
        investigation_status=investigation_status, investigation_failure_type=failure_type
    )
    metadata_type = (
        ActionReportMetadata
        if action_first
        else QuestionReportMetadata
        if question_led
        else ReplanningReportMetadata
        if replanning
        else BalancedReportMetadata
        if balanced
        else ReceiptReuseReportMetadata
        if receipt_reuse
        else UsedSourceReportMetadata
        if used_source
        else WorkbenchReportMetadata
        if workbench
        else NavigationReportMetadata
        if navigation
        else ComparableReportMetadata
        if comparable
        else RecoveryReportMetadata
        if recovery
        else (ConditionReportMetadata if condition else BriefReportMetadata)
    )
    question_material = {}
    if question_led:
        state, meanings = question_report_material(steps, semantic_deliveries)
        question_material = dict(question_state=state, diagnostic_semantics=meanings)
    metadata = metadata_type.model_validate_json(
        canonical_json_bytes(
            dict(
                evidence=header,
                **question_material,
                steps=steps,
                source_bindings=[bind_retained_source(s, store) for s in sources],
                delivery_bindings=deliveries,
                formal_fact_refs=[f.fact.fact_ref for f in formal.facts],
                termination=termination,
                action_seconds=[r.elapsed_seconds for r in records],
                decision_seconds=[r.provider_seconds for r in decisions],
                **(
                    {
                        "investigation_failure": investigation_failure.model_dump(mode="json")
                        if investigation_failure
                        else None
                    }
                    if navigation
                    else {}
                ),
            )
        )
    )
    context = BriefSreReportContext(metadata, store, _retained_supply(metadata, store))
    validate_brief_report(context)
    return context
