"""Source-fixed metadata choices; expansion is explicit, not model claim repair.

No discovery, measurement, dispatch, owner restoration or provider activation.
Passing the original checks still does not establish an interpretation's quality.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_proposals import OpsRecommendationOutcome, Text
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_claims import StepRef, parse_comparable_report
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore

EXPRESSION_SCHEMA = "nurse.v2.atomic-report-expression-response/3"
_SECTIONS = (
    "executive_summary",
    "formal_interpretations",
    "step_findings",
    "final_synthesis",
    "limitations",
    "next_checks",
)


class LegacyExpressionSelection(StrictFrozenModel):
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    expression_ref: str = Field(pattern=r"^e[1-9][0-9]{0,5}$")
    surface_text: Text
    outcomes: tuple[OpsRecommendationOutcome, ...] | None


class LegacyExpressionReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.atomic-report-expression-response/1"]
    core_result_digest: SHA256
    executive_summary: tuple[LegacyExpressionSelection, ...]
    formal_interpretations: tuple[LegacyExpressionSelection, ...]
    step_findings: dict[StepRef, tuple[LegacyExpressionSelection, ...]] = Field(max_length=8)
    final_synthesis: tuple[LegacyExpressionSelection, ...]
    limitations: tuple[LegacyExpressionSelection, ...]
    next_checks: tuple[LegacyExpressionSelection, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        selections = [
            c for name in _SECTIONS if name != "step_findings" for c in getattr(self, name)
        ]
        selections.extend(c for claims in self.step_findings.values() for c in claims)
        ids = [c.claim_id for c in selections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate expression claim identity")
        return self


class ExpressionOutcome(OpsRecommendationOutcome):
    condition_effect: Literal[
        "necessary_condition_holds", "necessary_condition_fails", "optional_support_present",
        "optional_support_absent", "refuter_present", "refuter_absent", "condition_unknown",
    ]


_OUTCOME_TRUTHS = ("true", "false", "unknown")
CompleteExpressionOutcomes = Annotated[
    tuple[ExpressionOutcome, ...],
    Field(
        min_length=len(_OUTCOME_TRUTHS), max_length=len(_OUTCOME_TRUTHS),
        json_schema_extra={"allOf": [
            {"contains": {"properties": {"outcome": {"const": truth}}, "required": ["outcome"]}}
            for truth in _OUTCOME_TRUTHS
        ]},
    ),
]


def validate_expression_outcomes(outcomes: tuple[ExpressionOutcome, ...] | None) -> None:
    """Null or all three truths once, preserving the caller's original order/text."""
    if outcomes is not None and (
        len(outcomes) != len(_OUTCOME_TRUTHS)
        or {row.outcome for row in outcomes} != set(_OUTCOME_TRUTHS)
    ):
        raise ValueError("expression outcomes must keep the same three-valued subject")


class PriorExpressionSelection(LegacyExpressionSelection):
    outcomes: tuple[ExpressionOutcome, ...] | None


class PriorExpressionReportResponse(LegacyExpressionReportResponse):
    schema_version: Literal["nurse.v2.atomic-report-expression-response/2"]  # type: ignore[assignment]
    executive_summary: tuple[PriorExpressionSelection, ...]
    formal_interpretations: tuple[PriorExpressionSelection, ...]
    step_findings: dict[StepRef, tuple[PriorExpressionSelection, ...]] = Field(max_length=8)  # type: ignore[assignment]
    final_synthesis: tuple[PriorExpressionSelection, ...]
    limitations: tuple[PriorExpressionSelection, ...]
    next_checks: tuple[PriorExpressionSelection, ...]


class ExpressionSelection(LegacyExpressionSelection):
    outcomes: CompleteExpressionOutcomes | None

    @model_validator(mode="after")
    def complete_outcomes(self) -> Self:
        validate_expression_outcomes(self.outcomes)
        return self


class ExpressionReportResponse(LegacyExpressionReportResponse):
    schema_version: Literal["nurse.v2.atomic-report-expression-response/3"]  # type: ignore[assignment]
    executive_summary: tuple[ExpressionSelection, ...]
    formal_interpretations: tuple[ExpressionSelection, ...]
    step_findings: dict[StepRef, tuple[ExpressionSelection, ...]] = Field(max_length=8)  # type: ignore[assignment]
    final_synthesis: tuple[ExpressionSelection, ...]
    limitations: tuple[ExpressionSelection, ...]
    next_checks: tuple[ExpressionSelection, ...]


def parse_legacy_expression_response(raw: bytes) -> LegacyExpressionReportResponse:
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        return LegacyExpressionReportResponse.model_validate_json(
            encode_json(value, REPORT_LIMITS), strict=True)
    except ValidationError:
        raise ExchangeBoundaryError("sre_legacy_atomic_expression_schema") from None


def parse_prior_expression_response(raw: bytes) -> PriorExpressionReportResponse:
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        return PriorExpressionReportResponse.model_validate_json(
            encode_json(value, REPORT_LIMITS), strict=True)
    except ValidationError:
        raise ExchangeBoundaryError("sre_prior_atomic_expression_schema") from None


def parse_expression_response(raw: bytes) -> ExpressionReportResponse:
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        return ExpressionReportResponse.model_validate_json(
            encode_json(value, REPORT_LIMITS), strict=True
        )
    except ValidationError:
        raise ExchangeBoundaryError("sre_atomic_expression_schema") from None


def _facts(supply: dict[str, Any], store: NativeFactStore) -> list[OpsReportFact]:
    refs = supply["provided_fact_refs"]
    if len(refs) != len(set(refs)) or not set(refs) <= set(store.refs()):
        raise ValueError("expression source permission differs")
    return [store.resolve(ref) for ref in refs]


def _values(item: OpsReportFact) -> dict[str, Any]:
    return {v.name: v.value for v in item.fact.values}


def program_fact_rows(
    supply: dict[str, Any],
    store: NativeFactStore,
    *,
    step_ref: str,
) -> list[dict[str, Any]]:
    """Expose only this completed step's exact source facts, not interpretation."""
    step = next((s for s in supply["steps"] if s["step_ref"] == step_ref), None)
    if step is None:
        raise ValueError("program facts name an absent step")
    if step["executed"]["status"] != "completed":
        return []
    permitted = {r.fact.fact_ref: r for r in _facts(supply, store)}
    if not set(step["delivered_fact_refs"]) <= permitted.keys():
        raise ValueError("program facts exceed step delivery")
    return [
        dict(
            authority="program_source_fact_not_model_diagnosis",
            evidence_ref=ref,
            target_service_keys=list(permitted[ref].fact.target_service_keys),
            intervals=[i.model_dump(mode="json") for i in permitted[ref].fact.intervals],
            scope_basis=permitted[ref].fact.scope_basis,
            source_role=permitted[ref].role,
            source_path=permitted[ref].source_path,
            fields=_values(permitted[ref]),
        )
        for ref in step["delivered_fact_refs"]
    ]


def expression_catalog(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Finite, deterministic alternatives. No surface text or diagnosis is generated."""
    rows = _facts(supply, store)
    choices: list[dict[str, Any]] = []
    formal = set(supply["formal_fact_refs"])
    if not formal <= set(supply["provided_fact_refs"]):
        raise ValueError("expression formal permission differs")

    def add(items: list[OpsReportFact], operation: str, window: tuple[int, int] | None) -> None:
        refs = [r.fact.fact_ref for r in items]
        targets = list(dict.fromkeys(t for r in items for t in r.fact.target_service_keys))
        observation = operation == "describe_observation"
        advice = operation == "propose_verification"
        claim: dict[str, Any] = dict(
            statement_type="recommendation"
            if advice
            else "observation"
            if observation
            else "inference",
            operation=operation,
            evidence_refs=refs,
            target_service_keys=targets,
            interval=None
            if window is None
            else dict(start_unix_ns=window[0], end_unix_ns=window[1]),
            requested_claim_strength="conditional_recommendation"
            if advice
            else ("frozen_observation" if observation else "ranking_inference"),
            assertions=[
                dict(evidence_ref=r.fact.fact_ref, field=v.name, value=v.value)
                for r in items
                for v in r.fact.values
            ],
            recommendation=None,
        )
        sections = ["executive_summary", "final_synthesis"]
        steps = [
            s["step_ref"]
            for s in supply["steps"]
            if s["executed"]["status"] == "completed" and set(refs) <= set(s["delivered_fact_refs"])
        ]
        if steps:
            sections.append("step_findings")
        if set(refs) <= formal:
            sections.append("formal_interpretations")
        if operation == "explain_uncertainty":
            sections.append("limitations")
        if advice:
            subject = items[0].fact.fact_ref
            predicate = _values(items[0])["predicate_kind"]
            claim["recommendation"] = dict(
                action_class="read_only_diagnostic_check",
                status="proposed",
                precondition_refs=[subject],
                outcome_subject_ref=subject,
                verification_question=(
                    f"Recheck the supplied predicate {predicate} "
                    "in the bound target/window; does that predicate hold?"
                ),
                safety_limitations=[
                    "Proposed read-only check, not executed; "
                    "no production mutation or causal confirmation."
                ],
            )
            sections, steps = ["next_checks"], []
        choices.append(
            dict(
                expression_ref=f"e{len(choices) + 1}",
                sections=sections,
                step_refs=steps,
                claim=claim,
            )
        )

    grouped: dict[Any, list[OpsReportFact]] = defaultdict(list)
    paired: dict[Any, list[OpsReportFact]] = defaultdict(list)
    for row in rows:
        fact, fields = row.fact, _values(row)
        windows: list[tuple[int, int] | None] = [
            (i.start_unix_ns, i.end_unix_ns) for i in fact.intervals
        ] or [None]
        if fact.kind in {"ranking", "competition"} and row.role == "formal_record":
            add([row], "explain_priority" if fact.kind == "ranking" else "compare_candidates", None)
            continue
        for window in windows:
            if (
                fact.kind in {"observation", "predicate"}
                and fact.scope_basis == "bound_observation"
                and fact.allowed_claim_strength == "frozen_observation"
                and fact.target_service_keys
            ):
                add([row], "describe_observation", window)
            if fact.status_codes or any(
                v in ("unknown", "false", "no_record", "degraded") for v in fields.values()
            ):
                add([row], "explain_uncertainty", window)
            if fact.kind in {"observation", "predicate", "candidate_evidence", "hypothesis"}:
                grouped[(fact.target_service_keys, window)].append(row)
            if fact.kind == "candidate_evidence" or (
                fact.kind == "hypothesis" and type(fields.get("mechanism_key")) is str
            ):
                paired[
                    (
                        fact.kind if fact.kind == "hypothesis" else row.source_path,
                        window,
                        fields.get("mechanism_key"),
                        fields.get("mechanism_definition_sha256"),
                    )
                ].append(row)
            if (
                window is not None
                and fact.kind in {"observation", "predicate"}
                and type(fields.get("predicate_kind")) is str
                and fields.get("truth_value") in {"true", "false", "unknown"}
                and fact.target_service_keys
            ):
                add([row], "propose_verification", window)
    for (_, window), items in grouped.items():
        if len(items) > 1:
            add(items, "synthesize_evidence", window)
    for (_, window, _, _), items in paired.items():
        if len({i.fact.target_service_keys for i in items}) > 1:
            add(items, "synthesize_evidence", window)
    from nurse_rcl.v2.live.ops_sre_report_grounding import condition_boundary, scalar_field_scope

    boundaries = {
        r["expression_ref"]: condition_boundary(
            supply, store, r["claim"]["recommendation"]["outcome_subject_ref"],
            r["claim"]["interval"])
        for r in choices if r["claim"]["recommendation"] is not None
    }
    return dict(
        schema_version="nurse.v2.atomic-report-expressions/2",
        expressions=choices,
        field_scope=scalar_field_scope(supply, store),
        condition_boundaries=boundaries,
        authority="source_metadata_only_not_model_diagnosis",
        interpretation_task=(
            "Explain supplied necessary/refuting/optional/unknown "
            "conditions and what can or cannot distinguish candidates; accepted "
            "observable_contract_only is not causal confirmation. Empty advice is legal."
        ),
    )


def atomic_section_guide(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_section_guidance import workbench_section_guide

    guide = workbench_section_guide(supply, store)
    for section in ("executive_summary", "formal_interpretations", "final_synthesis"):
        guide[section].pop("expression_choices", None)
    for step in guide["step_findings"].values():
        step.pop("expression_choices", None)
    guide["atomic_expressions"] = wire_expression_catalog(supply, store)
    return guide


def wire_expression_catalog(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Send pointers, not another copy of every fact value/assertion tree."""
    catalog = expression_catalog(supply, store)
    from nurse_rcl.v2.live.ops_sre_report_grounding import condition_expression_refs

    # Names/types are metadata, never values at registered identifier positions.
    field_scope = {
        ref: [dict(field_name=name, json_type=kind) for name, kind in fields.items()]
        for ref, fields in catalog["field_scope"].items()
    }
    boundaries = deepcopy(catalog["condition_boundaries"])
    for boundary in boundaries.values():
        for mechanism in boundary["mechanisms"]:
            # Versioned runtime binds these public definitions. Repeating a digest
            # here creates a new H handle which a short-only reader cannot rebuild.
            del mechanism["mechanism_definition_sha256"]
    choices = []
    for row in catalog["expressions"]:
        claim = row["claim"]
        descriptor = {
            name: claim[name]
            for name in (
                "operation",
                "evidence_refs",
                "target_service_keys",
                "interval",
            )
        }
        advice = claim["recommendation"]
        if advice is not None:
            descriptor["outcome_subject_ref"] = advice["outcome_subject_ref"]
            descriptor["verification_question"] = advice["verification_question"]
            descriptor["condition_boundary_ref"] = row["expression_ref"]
        choices.append(
            dict(
                expression_ref=row["expression_ref"],
                sections=row["sections"],
                step_refs=row["step_refs"],
                **descriptor,
            )
        )
    return dict(
        catalog,
        expressions=choices,
        field_scope=field_scope,
        condition_boundaries=boundaries,
        condition_competition_choices=condition_expression_refs(
            catalog["expressions"], supply, store),
        assertions="all_original_scalar_fields_of_selected_source_facts; "
        "values_and_types_remain_in_fact_records_not_duplicated_here",
    )


def resolve_expression_response(
    raw: bytes,
    supply: dict[str, Any],
    store: NativeFactStore,
) -> bytes:
    """Expand a NEW native selection, retaining its text unchanged for original guards."""
    response = parse_expression_response(raw)
    if response.core_result_digest != supply["core_result_digest"]:
        raise ValueError("expression core source differs")
    by_ref = {r["expression_ref"]: r for r in expression_catalog(supply, store)["expressions"]}

    def expand(
        selection: ExpressionSelection, section: str, step: str | None = None
    ) -> dict[str, Any]:
        choice = by_ref.get(selection.expression_ref)
        if (
            choice is None
            or section not in choice["sections"]
            or (section == "step_findings" and step not in choice["step_refs"])
        ):
            raise ValueError("expression selection exceeds section/step permission")
        claim: dict[str, Any] = deepcopy(choice["claim"])
        claim.update(claim_id=selection.claim_id, surface_text=selection.surface_text)
        if claim["recommendation"] is None:
            if selection.outcomes is not None:
                raise ValueError("outcomes without a supplied predicate")
        else:
            advice = claim["recommendation"]
            outcomes = selection.outcomes
            validate_expression_outcomes(outcomes)
            if outcomes is not None:
                from nurse_rcl.v2.live.ops_sre_report_grounding import condition_boundary

                boundary = condition_boundary(
                    supply, store, advice["outcome_subject_ref"], claim["interval"])
                if boundary["role"] is None or any(
                    r.condition_effect != boundary["effects"].get(r.outcome) for r in outcomes
                ):
                    raise ValueError("expression condition effect differs from subject role")
            advice["possible_outcomes"] = [r.model_dump(mode="json", exclude={"condition_effect"})
                                           for r in outcomes or ()]
            if outcomes is None:
                advice["outcome_subject_ref"] = None
        return claim

    value: dict[str, Any] = dict(
        schema_version="nurse.v2.sre-report-response/4",
        core_result_digest=response.core_result_digest,
    )
    steps = {s["step_ref"] for s in supply["steps"]}
    if not response.step_findings.keys() <= steps:
        raise ValueError("expression names an absent step")
    for section in _SECTIONS:
        value[section] = (
            {
                step: [expand(c, section, step) for c in claims]
                for step, claims in response.step_findings.items()
            }
            if section == "step_findings"
            else [expand(c, section) for c in getattr(response, section)]
        )
    result = encode_json(value, REPORT_LIMITS)
    parse_comparable_report(result)
    return result
