"""Versioned report-only request contract; no provider/credential activation."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY


def atomic_report_policy() -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_exchange import action_report_policy
    from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import ExpressionReportResponse

    previous = action_report_policy()
    system = previous["system"].replace("sre-model-view/14", "sre-model-view/23") + (
        "\nThis is a separately bound report-only handoff from atomic/M3 investigation. "
        "Only supplied facts and fields are assertable. Preserve original targets, values, "
        "types, units, baseline/fault windows, observed-record versus request-population "
        "scope and coverage. Zero, null, unknown, unread and a rejected note are distinct. "
        "question_state uses the original explicit atomic lifecycle; summaries and notes "
        "are agent statements, never measured facts. stage_pause is a program stage limit, "
        "not a model stop, convergence or completion of fault investigation. Report-time "
        "synthesis of the last receipt does not mean the investigation model consumed it. "
        "Keep real unresolved questions and unchecked scope; do not repair model history.\n"
        "Minimal report discipline (no new evidence or permissions):\n"
        "Every numeric value written in surface_text needs a cited exact field assertion "
        "with its ORIGINAL JSON type. For example a supplied string quantized_total_score "
        "must be asserted as that exact string, not merely its rank. A fact citation alone "
        "does not assert all its values. Prefer the program's exact ranking table over "
        "repeating scores and pairwise score differences.\n"
        "Describe baseline and fault as separate observation claims, each with its own "
        "actual interval and only matching-window sources. A single fault interval cannot "
        "cover baseline sources. Do not erase interval constraints to pass validation.\n"
        "Legal operation -> statement_type / requested_claim_strength combinations: "
        "describe_observation -> observation / frozen_observation; "
        "synthesize_evidence -> inference / ranking_inference; "
        "explain_priority, compare_candidates, explain_uncertainty -> inference / "
        "ranking_inference; propose_verification -> recommendation / "
        "conditional_recommendation. These are existing permissions, not upgrades.\n"
        "A verification outcome_subject_ref must be a single supplied predicate_kind and "
        "truth_value fact, or one supplied mechanism_condition, with the exact target "
        "and interval. All true/false/unknown branches address that same predicate. "
        "Metric summaries and collections of predicates are NOT single predicates. "
        "When no suitable subject is supplied, leave the question unresolved and omit "
        "the recommendation; do not invent predicates or force a next-check count.\n"
        "Prioritize supplied necessary, refuting, optional and unknown conditions and "
        "their AND/OR meaning when explaining competing candidates. Explain what "
        "supports, weakens, fails to distinguish, or remains untested. Template accepted "
        "with observable_contract_only is not causal confirmation. Do not count leaves "
        "as votes, infer health from zero/unknown, or invent a root cause.\n"
        "OUTPUT OVERRIDE for this separately versioned atomic handoff: use only "
        "nurse.v2.atomic-report-expression-response/3, NOT the earlier response/4 "
        "examples. Select expression_ref from section_reference_guide.atomic_expressions. "
        "Each choice fixes source refs, original typed assertions, targets, interval, "
        "operation and strength. Its wire descriptor is an index only: original "
        "values and types remain in fact_records; all selected source scalar fields "
        "are asserted exactly during expansion. Output only claim_id, expression_ref, "
        "surface_text and "
        "outcomes; never repeat or change frozen metadata. Choose only an allowed "
        "section and step. The program expands that selection explicitly into the "
        "existing checked claim format; it does not correct model text or diagnose. "
        "outcomes must be null except for a selected proposed verification; there they "
        "must be null or exactly three interpretations, one true, one false and one unknown, "
        "in any order, for the SAME fixed predicate and fixed recheck question. A current unknown "
        "fact does not permit an unknown-only branch list. Never fill or omit a branch inside "
        "a nonnull list. No matching question: omit advice, leave unresolved; "
        "do not relabel an episode question as a pressure predicate. "
        "Program facts will be displayed separately with original windows, units and "
        "coverage, even when model findings are absent or blocked. They are not model "
        "diagnoses and must not be presented as your conclusions. Use the supplied "
        "same-window condition combinations for a short support/refutation/unknown "
        "and candidate-distinguishability explanation, not repeated rank differences. "
        "Do not claim that an offered choice or mechanical acceptance proves causality. "
        "The shared field_scope index names the exact fields available under each f ref; "
        "a summary ref does not cite its children. Use an existing same-window combination "
        "when your text needs those children; never borrow a baseline value into a fault choice. "
        "For each nonempty outcome include condition_effect from that expression's "
        "condition_boundary_ref. This is only the effect on the registered predicate role, "
        "not acceptance of its mechanism: false refuting removes that refuter, while "
        "necessary unknown/not_supplied remains unresolved. Optional absence cannot fail a "
        "necessary condition. Preserve those limits in your own interpretation. "
        "When actual condition combinations exist, final_synthesis should use one for a "
        "short support/refutation/unknown and distinguishability handoff. A ranking-only "
        "handoff remains structurally uncovered; no automatic retry or invented conclusion follows."
    )
    pinned = dict(
        protocol="nurse.v2.sre-report-exchange/23",
        system=system,
        output_schema=canonical_json_bytes(ExpressionReportResponse.model_json_schema()).decode(),
        output_policy=OUTPUT_POLICY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return dict(
        pinned,
        profile="hermes-sre-tools/23",
        schema="nurse.v2.sre-report-input/22",
        prompt_sha256=sha256_hex(canonical_json_bytes(pinned)),
    )
