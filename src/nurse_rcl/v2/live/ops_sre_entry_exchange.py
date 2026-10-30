"""Entry-only action policy; retains the original SRE purposes and evidence limits."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_sre_action_exchange import (
    ACTION_SKILL_GUIDANCE,
    ACTION_SYSTEM,
    action_common_guidance,
)

ENTRY_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/20"
ENTRY_GUIDANCE = (
    "Choose one current action_entries entry using {entry_ref,question}, or one typed "
    "{navigation,question}. Do not reconstruct skill/tool/parameters or mix branches. "
    "Entries are source-bound navigation, not delivered evidence; omission is not absence. "
    "Entry menu order is not formal rank. Describe the actual selected targets and purpose, "
    "not an assumed top-ranked subject. Read each entry's delivery and supply_contract: "
    "complete_same_record means this exact full record was already supplied at the listed "
    "actions; repeating it provides no new record or independent support. inspect_sources "
    "aggregates the selected complete cards, not a deeper source read; it cannot fill "
    "coverage or windows missing from those cards. An unread, partial, brief or changed "
    "record is not proof of complete reuse. Choose a legitimate new check, justified reuse "
    "or a bounded stop consistent with these limits and the latest real feedback. "
    "Navigation kind case uses cursor; candidate uses candidate_service_key/focus/cursor; "
    "candidate_pair uses left_candidate_service_key/right_candidate_service_key/focus/cursor; "
    "dependency uses caller_service_key/callee_service_key/cursor; catalog uses "
    "catalog_kind/filters/cursor. Use issued service names, original focus and exact cursors. "
    "Every navigation costs one action and never performs a second implicit check. "
    "Inspect actual symptoms and competing mechanisms; a score is not fault evidence. "
    "A prepared summary is not an episode measurement. Preserve baseline/fault windows, "
    "units, coverage and episode status. Missing, mismatched and unread sources remain gaps. "
    "latest_result_update and an explicit update of latest_action_question_id concern the "
    "same stored action question: use answered/answered only if that precise question is "
    "answered, with remaining_question=null; keep diagnostic unknown in unresolved, not "
    "as an unanswered version of the same read question. If requested values or windows "
    "remain missing, use partial with open/blocked/deferred and a specific remaining gap. "
    "Other question IDs have their own scopes. Do not rename an old question or infer an "
    "answer merely from complete delivery. Keep the stored read question's original scope: "
    "only missing parts of that original request belong in its remaining gap. A broader "
    "diagnostic question belongs in unresolved; when selecting a check for it, write a "
    "new action question with an explicit public relation to the prior question. Do not "
    "invent question IDs or parent links. Keep question and discriminating_value within "
    "the selected supply_contract and limits. For a symptom check, state a specific symptom "
    "hypothesis, the "
    "observable contrast and how each outcome changes the next check, not source identity. "
    "Cross-candidate symptoms test only their corresponding scoped symptom hypotheses; "
    "different metrics are not interchangeable negative controls. Zero errors with an "
    "unknown denominator or coverage is not covered absence. "
    "In discriminating_value and actual feedback, "
    "unknown is a gap, not refutation; missing, unread or mismatched observations do not "
    "weaken a hypothesis. Observed absence weakens only with adequate coverage and a "
    "matched subject, window and unit for that hypothesis. Measured symptoms alone do not "
    "establish a fault source or distinguish source from affected service: that requires "
    "mechanism and direction or timing evidence. Preserve known numbers and unresolved "
    "conditions without turning unknown into false. "
    "A rejected selection is a paid action refusal, not permission for a quality redraw."
)
ENTRY_SYMPTOM_GUIDANCE = (
    "Question: What actually worsened for the selected subject, metric and windows versus "
    "baseline? Use a current symptom entry, or registered measurement only with real IDs. "
    "Keep baseline/fault values, units, coverage and episode status together; test a stated "
    "symptom hypothesis within the selected supply. Feedback: degradation supports that "
    "symptom; a covered matched contrast can weaken only that symptom hypothesis, not "
    "identify or refute a fault source. Cross-candidate comparisons require matching "
    "metric, units, windows and coverage with explicit subjects; caller latency is not "
    "callee self-time. Unknown, unread and mismatched remain separate gaps. Reuse known "
    "results and choose the next useful mechanism or direction/timing check without "
    "claiming this symptom read already performs it."
)
ENTRY_SYSTEM = (
    ACTION_SYSTEM.replace(
        "Each action has skill_id/tool/parameters/question; "
        "question is the public operational question.",
        "Each action is exactly {entry_ref,question} or {navigation,question}; "
        "question is the public operational question.",
    ).replace(
        "visible card_id.", "a current entry_ref; this is unchecked-action handoff, never evidence."
    )
    + "\n"
    + ENTRY_GUIDANCE
)
ENTRY_SKILL_GUIDANCE = {
    node: (ENTRY_SYMPTOM_GUIDANCE if node == "symptoms" else guidance) + " " + ENTRY_GUIDANCE
    for node, guidance in ACTION_SKILL_GUIDANCE.items()
}


def entry_common_guidance() -> dict[str, str]:
    text = action_common_guidance()["guidance"] + " " + ENTRY_GUIDANCE
    return dict(guidance=text, content_sha256=sha256_hex(text.encode()))


def entry_decision_policy() -> dict[str, Any]:
    pinned = dict(
        protocol=ENTRY_DECISION_PROTOCOL,
        system=ENTRY_SYSTEM,
        output_policy=OUTPUT_POLICY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return dict(
        protocol=ENTRY_DECISION_PROTOCOL,
        system=ENTRY_SYSTEM,
        profile="hermes-sre-tools/19",
        schema="nurse.v2.sre-investigation-input/20",
        guidance=entry_common_guidance(),
        prompt_sha256=sha256_hex(canonical_json_bytes(pinned)),
    )
