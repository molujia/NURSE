# ruff: noqa: RUF001 -- Chinese model guidance
"""D17 reference locations, not another fact table or claim repair layer."""

from __future__ import annotations

from typing import Any

from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore

_USEFUL_FIELDS = (
    "truth_value", "disposition", "supplementary_necessary_support",
    "used_in_aggregation", "family_used_in_aggregation", "family_representative",
    "pool_selected", "selected_in_pool", "selected", "total_score_gap",
    "event_evidence_gap", "propagation_gap", "evidence_tied", "total_score", "rank",
    "family_strength", "local_strength", "unknown_term_count", "measured_value",
    "false_count", "unknown_count", "value",
)


def _field(values: dict[str, Any]) -> str | None:
    """Name an actually supplied scalar; its value remains in the original tree."""
    return next(
        (name for name in _USEFUL_FIELDS if name in values
         and type(values[name]) in (str, int, float, bool, type(None))),
        None,
    )


def _shape(
    operation: str,
    refs: list[str],
    targets: list[str],
    fields: list[str],
    *,
    interval: dict[str, int] | None = None,
) -> dict[str, Any]:
    return dict(
        operation=operation,
        statement_type="inference",
        requested_claim_strength="ranking_inference",
        evidence_refs=refs,
        target_service_keys=targets,
        interval=interval,
        assertion_fields=[dict(evidence_ref=r, field=f) for r, f in zip(refs, fields, strict=True)],
        authority="structure_only_fill_exact_values_and_own_judgment_then_check",
    )


def _formal_shapes(formal: dict[str, Any], store: NativeFactStore) -> list[dict[str, Any]]:
    choices = []
    for kind, operation in (("ranking", "explain_priority"),
                            ("competition", "compare_candidates")):
        for ref in formal[kind + "_refs"]:
            fact = store.resolve(ref).fact
            field = _field({v.name: v.value for v in fact.values})
            if field is not None and len(fact.target_service_keys) == (
                1 if kind == "ranking" else 2
            ):
                choices.append(_shape(operation, [ref], list(fact.target_service_keys), [field]))
                break
    return choices


def _step_shapes(step: dict[str, Any], briefs: dict[str, dict[str, Any]],
                 store: NativeFactStore) -> list[dict[str, Any]]:
    if step["executed"]["status"] != "completed":
        return []
    choices: list[dict[str, Any]] = []
    eligible_briefs = [briefs[r] for r in step["delivered_brief_refs"]
                       if r in briefs and _field(briefs[r]["fields"]) is not None]
    for left in eligible_briefs:
        other_brief = next((r for r in eligible_briefs if r is not left
                            and r["target_service_keys"] == left["target_service_keys"]
                            and r["window_ns"] == left["window_ns"]), None)
        if other_brief is None or not left["target_service_keys"]:
            continue
        left_field, right_field = _field(left["fields"]), _field(other_brief["fields"])
        assert left_field is not None and right_field is not None
        window = left["window_ns"]
        choices.append(_shape(
            "synthesize_evidence", [left["evidence_ref"], other_brief["evidence_ref"]],
            list(left["target_service_keys"]),
            [left_field, right_field],
            interval=(dict(start_unix_ns=window[0], end_unix_ns=window[1])
                      if window is not None else None),
        ))
        break
    if not choices:
        for brief in eligible_briefs:
            fields = brief["fields"]
            field = _field(fields)
            assert field is not None
            if not brief["target_service_keys"] or not (
                fields.get("truth_value") in {"false", "unknown"}
                or fields.get("unknown_count", 0) > 0
                or fields.get("false_count", 0) > 0
                or brief["limitations"]
            ):
                continue
            window = brief["window_ns"]
            choices.append(_shape(
                "explain_uncertainty", [brief["evidence_ref"]],
                list(brief["target_service_keys"]), [field],
                interval=(dict(start_unix_ns=window[0], end_unix_ns=window[1])
                          if window is not None else None),
            ))
            break
    selected: list[tuple[str, tuple[str, ...], str]] = []
    for ref in step["delivered_fact_refs"]:
        fact = store.resolve(ref).fact
        field = _field({v.name: v.value for v in fact.values})
        if fact.kind == "candidate_evidence" and fact.target_service_keys and field is not None:
            selected.append((ref, fact.target_service_keys, field))
    for score_fact in selected:
        other_fact = next((r for r in selected
                           if r[0] != score_fact[0] and r[1] == score_fact[1]), None)
        if other_fact is not None:
            choices.append(_shape(
                "synthesize_evidence", [score_fact[0], other_fact[0]],
                list(score_fact[1]), [score_fact[2], other_fact[2]],
            ))
            break
    return choices[:2]


def _cross_candidate_shape(
    supply: dict[str, Any], formal: dict[str, Any], store: NativeFactStore,
) -> list[dict[str, Any]]:
    """Join one formal gap to two *delivered* score summaries, without rescoring."""
    delivered = dict.fromkeys(
        ref for step in supply["steps"] if step["executed"]["status"] == "completed"
        for ref in step["delivered_fact_refs"]
    )
    ranked = []
    for ref in formal["ranking_refs"]:
        fact = store.resolve(ref).fact
        values = {v.name: v.value for v in fact.values}
        if len(fact.target_service_keys) == 1 and type(values.get("rank")) is int:
            ranked.append((values["rank"], fact.target_service_keys[0]))
    top_two = tuple(target for _, target in sorted(ranked)[:2])
    comparisons = list(formal["competition_refs"])
    comparisons.sort(
        key=lambda ref: (
            tuple(store.resolve(ref).fact.target_service_keys) != top_two,
            formal["competition_refs"].index(ref),
        )
    )
    for competition_ref in comparisons:
        comparison = store.resolve(competition_ref).fact
        targets = comparison.target_service_keys
        gap_field = _field({v.name: v.value for v in comparison.values})
        if len(targets) != 2 or gap_field is None:
            continue
        sides: list[tuple[str, str]] = []
        for target in targets:
            found = None
            for ref in delivered:
                item = store.resolve(ref)
                fact = item.fact
                if (fact.kind == "candidate_evidence"
                        and item.source_path == "payload.score"
                        and fact.target_service_keys == (target,)):
                    field = _field({v.name: v.value for v in fact.values})
                    if field is not None:
                        found = (ref, field)
                        break
            if found is None:
                break
            sides.append(found)
        if len(sides) == 2:
            return [_shape(
                "synthesize_evidence", [competition_ref, *(ref for ref, _ in sides)],
                list(targets), [gap_field, *(field for _, field in sides)],
            )]
    return []


def section_reference_guide(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Caller admits supply; all locations point to its unchanged permission sets."""
    formal = {}
    for kind in ("ranking", "competition"):
        formal[kind + "_refs"] = [
            r
            for r in supply["formal_fact_refs"]
            if r in supply["provided_fact_refs"]
            and store.resolve(r).role == "formal_record"
            and store.resolve(r).fact.kind == kind
        ]
    return dict(
        schema_version="nurse.v2.sre-section-reference-guide/1",
        formal_interpretations=formal,
        step_findings=[
            dict(
                step_ref=s["step_ref"],
                fact_refs_location=f"supply.steps[{i}].delivered_fact_refs",
                brief_refs_location=f"supply.steps[{i}].delivered_brief_refs",
            )
            for i, s in enumerate(supply["steps"])
        ],
        final_synthesis=dict(
            fact_refs_location="supply.provided_fact_refs",
            brief_refs_location="supply.provided_brief_refs",
            rule="事后综合保留各来源角色；不得倒灌早期步骤。",
        ),
        missing_observation_precondition="只有未读明细问题时保留实际公开未决项，不构造条件预测或无前提建议。",
        authority="guide_only_no_new_fact_or_step_permission",
    )


def workbench_section_guide(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Explicit section choices over actual supply, never candidate combinations."""
    provided = set(supply["provided_fact_refs"])
    if not provided <= set(store.refs()) or not set(supply["formal_fact_refs"]) <= provided:
        raise ValueError("workbench guide supply differs")
    briefs_by_ref = {b["evidence_ref"]: b for b in supply["briefs"]}
    if (len(briefs_by_ref) != len(supply["briefs"])
            or set(briefs_by_ref) != set(supply["provided_brief_refs"])):
        raise ValueError("workbench guide brief supply differs")
    sections = {}
    for step in supply["steps"]:
        refs, briefs = step["delivered_fact_refs"], step["delivered_brief_refs"]
        if not set(refs) <= provided or not set(briefs) <= set(supply["provided_brief_refs"]):
            raise ValueError("workbench step guide supply differs")
        sections[step["step_ref"]] = dict(
            fact_refs=list(refs),
            brief_refs=list(briefs),
            expression_choices=_step_shapes(step, briefs_by_ref, store),
            rule="only_this_step_delivery; no_global_formal_fallback",
        )
    formal = section_reference_guide(supply, store)["formal_interpretations"]
    formal["expression_choices"] = _formal_shapes(formal, store)
    observed = []
    for ref in supply["provided_fact_refs"]:
        fact = store.resolve(ref).fact
        if (fact.kind in {"observation", "predicate"}
                and fact.scope_basis == "bound_observation"
                and fact.target_service_keys):
            observed.append(ref)
            if len(observed) == 3:
                break
    return dict(
        schema_version="nurse.v2.sre-section-reference-guide/3",
        operations=dict(
            explain_priority=dict(target_count=1, source_kind="formal_ranking", interval="null"),
            compare_candidates=dict(
                target_count=2,
                source_kind="formal_competition",
                target_order="source_order",
                interval="null",
            ),
            synthesize_evidence=dict(
                min_refs=2,
                source_kind="actual_supplied",
                target_order="retain_all_source_targets",
                interval="original_scope_checks",
            ),
            compare_mechanisms=dict(
                source_kind="same_mechanism_definition_window_and_category",
                rule="same_family_is_not_same_signal_or_mechanism",
            ),
            explain_uncertainty=dict(source_kind="actual_adverse_unknown_or_limit"),
            propose_verification=dict(
                source_kind="valid_observed_precondition",
                rule="otherwise_keep_unresolved; never_invent_outcomes",
            ),
        ),
        executive_summary=dict(formal, other_refs_location="supply.provided_fact_refs"),
        formal_interpretations=formal,
        step_findings=sections,
        final_synthesis=dict(
            fact_refs_location="supply.provided_fact_refs",
            brief_refs_location="supply.provided_brief_refs",
            expression_choices=_cross_candidate_shape(supply, formal, store),
            rule="preserve_roles_no_backfill",
        ),
        limitations=dict(refs_location="supply", rule="numeric_unknown_is_not_unread"),
        next_checks=dict(
            refs_location="supply",
            rule="保留原观测前提；分支主体仅本次实供、同目标同窗口的单一三值谓词或条件；否则空分支或保留未决。",
            observed_precondition_refs=observed,
            fallback=(
                "empty_branch_or_unresolved" if not observed else "check_exact_scope_and_window"
            ),
        ),
        field_types_and_targets_location="claim_scope_guide.fact_layouts and field_layouts",
        exact_windows_location="fact_records original fact intervals or brief window_ns",
        authority="expression_guidance_only_original_response_4_checker_is_authoritative",
    )


# Executable illustrative contract, checked with a separate synthetic fixture.
# These refs/values are examples only, never copied into a real case's claims.
REPORT_CONTRACT_EXAMPLES = [
    {
        "formal_interpretations": [
            dict(
                claim_id="example_comparison",
                statement_type="inference",
                operation="compare_candidates",
                evidence_refs=["f2"],
                target_service_keys=["svc-a", "svc-b"],
                interval=None,
                requested_claim_strength="ranking_inference",
                assertions=[dict(evidence_ref="f2", field="measured_value", value=123.456)],
                recommendation=None,
                surface_text="依据正式比较字段作有限比较，不作因果确认。",
            )
        ]
    },
    {
        "step_findings": {
            "s1": [
                dict(
                    claim_id="example_step",
                    statement_type="inference",
                    operation="synthesize_evidence",
                    evidence_refs=["f3", "f4"],
                    target_service_keys=["svc-a"],
                    interval=dict(start_unix_ns=10, end_unix_ns=20),
                    requested_claim_strength="ranking_inference",
                    assertions=[
                        dict(evidence_ref=r, field="measured_value", value=123.456)
                        for r in ("f3", "f4")
                    ],
                    recommendation=None,
                    surface_text="综合本步实际交付的两个评分字段，不宣称读过下级明细。",
                )
            ]
        }
    },
    {"next_checks": []},
]
