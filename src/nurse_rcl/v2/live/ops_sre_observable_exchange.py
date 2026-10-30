"""Observable-purpose declarations are operational scope, never fault evidence."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_sre_action_decision import (
    ACTION_FIELDS,
    NOTE_FIELDS,
    QUESTION_UPDATE_FIELDS,
    QUESTION_UPDATE_STATES,
    STOP_BASES,
)
from nurse_rcl.v2.live.ops_sre_d23_decision import CONTENT_RELATIONS
from nurse_rcl.v2.live.ops_sre_entry_exchange import (
    ENTRY_GUIDANCE,
    ENTRY_SKILL_GUIDANCE,
    ENTRY_SYSTEM,
)
from nurse_rcl.v2.live.ops_sre_feedback_context import ANSWER_STATUSES, LATEST_RESULT_FIELDS
from nurse_rcl.v2.live.ops_sre_observable_scope import OBSERVABLE_INPUT_SCHEMA, OBSERVABLE_TOOLS

OBSERVABLE_PROTOCOL = "nurse.v2.sre-decision-exchange/21"
OBSERVABLE_SELECTION_GUIDANCE = (
    "Choose the next useful check by a specific unresolved distinction in the currently "
    "supplied material, not by a symptom-card quota: there is no required number of symptom "
    "cards before investigation can proceed. You may keep an unconfirmed working explanation "
    "in unresolved as a question to test; ground it in currently visible material, not as an "
    "observed fact. If its definition, registered condition or observation prerequisite is "
    "unread, choose a targeted prerequisite read and explain what is missing and how obtaining "
    "or not obtaining it changes the next step; do not invent an unseen condition. Directory, "
    "definition and scoring reads do not themselves distinguish competing explanations. "
    "If another symptom read is useful, explain how its subject, modality or field fills "
    "that particular gap rather than merely adding another candidate or card. Describe how "
    "the obtainable values or statuses would change the next check; unknown, null and missing "
    "do not refute an explanation. Keep question, observable_test and discriminating_value "
    "within the selected entry's immediate contract and supply; why_now carries the broader "
    "purpose, not a claim that later mechanism, direction or timing checks have occurred. "
    "Diagnostic coverage adequacy or a unique root cause need not be established before "
    "posing a question; stronger conclusions still require the original evidence limits. "
    "If no available legal check can advance the distinction, explain that limit and retain "
    "unresolved questions and unchecked scope in an honest limited stop. Keep prerequisite "
    "acquisition, a discriminating check and later result use separate; this guidance is "
    "not a mandatory stage, tool, candidate or rank order, and is not evidence or programmatic "
    "semantic approval."
)
OBSERVABLE_ENTRY_GUIDANCE = ENTRY_GUIDANCE.replace(
    "For a symptom check, state a specific symptom hypothesis, the observable contrast "
    "and how each outcome changes the next check, not source identity.",
    "For a summary read, the action question and discriminating_value ask only what the "
    "selected metric and requested facets actually supply. Keep a broader symptom or "
    "mechanism hypothesis in unresolved and explain why this narrow observation is useful; "
    "do not turn the read question into a saturation or service-failure threshold test.",
)
OBSERVABLE_GUIDANCE = (
    OBSERVABLE_ENTRY_GUIDANCE
    + " For every record read, including paid_episode, discriminating_value describes only "
    "the immediately obtainable registered values or status markers and how known, "
    "unknown, null or missing outcomes choose a later check. why_now may explain the "
    "broader investigation motive, but explicitly name the mechanism, dependency or "
    "timing check still needed; the present read neither performs that check nor "
    "supports or refutes source identity on its own. For error_summary, recorded "
    "coverage may be unknown or null: the purpose is to read the coverage marker, "
    "not determine coverage adequacy. For paid_episode, the purpose is to read episode "
    "contents and qualifiers, not decide source versus affected service. The same "
    "limits apply to both the action question and discriminating_value. Do not "
    "promise diagnostic adequacy or causal discrimination from a record read. "
    + " Each non-null investigation_note additionally has observable_test. "
    "For a concrete current entry declare exactly {entry_ref,objective_kind,metric_name,"
    "requested_facets}, matching its observable_contract. requested_facets is a nonempty "
    "unique subset of allowed_facets. For a concrete current entry, "
    "note.target_service_keys must exactly match the selected entry.observable_contract."
    "binding.targets: this is the current action subject, not an older question or "
    "reference subject. Historical question_updates retain each stored question's "
    "immutable text, targets and observable_scope, which may differ from the current "
    "note targets. Past basis_refs do not rebind either subject or scope; a legal past "
    "reference can concern a different subject. Do not copy an old question's targets "
    "into the current note merely because updating that question. "
    "For typed navigation or stopping observable_test is null. These are registered supply "
    "types, not known values, evidence or automatic answers. Wrong entry, metric or facet "
    "rejects only the note; the legal action remains and there is no quality redraw. "
    "Each question_update additionally has remaining_facets: a unique subset of that "
    "stored question's immutable requested_facets, or null for an unscoped question. "
    "answered requires explicit empty remaining_facets and consistent latest_result_update; "
    "complete delivery or null values never answer it automatically. Missing original "
    "coverage or episode status belongs to those facets. Broader mechanisms or another "
    "subject belong in unresolved or a new action question, not an expanded old scope. "
    "For each original facet distinguish a supplied known value, a supplied unknown or "
    "unmeasured marker, and not supplied. not_measured_by_summary is a supplied status, "
    "not an observed episode and not an unread field. A supplied native unit is not a "
    "missing unit merely because capacity normalization is unavailable. Do not leave "
    "all facets remaining solely because a broader diagnosis is unknown; make each "
    "remaining facet and remaining_gap reflect that original observation request. "
    "The model must explicitly update the question; no programmatic auto-answer. "
    "Keep native question and remaining_gap prose; structure cannot certify its semantics. "
    "A CPU median increase or decrease describes that statistic only; it neither proves "
    "nor refutes "
    "saturation without capacity-normalized units, thresholds, episodes and mechanism "
    "conditions. Error fractions do not supply a service-failure threshold. Missing these "
    "remains unknown, not false. Select any useful candidate, including a lower-ranked one, "
    "but justify the actual selection: use formal rank or score only when supplied in "
    "the candidate directory. Menu order is not formal rank or proof of a high score. "
    "answer_status describes feedback; question_updates.state describes question lifecycle. "
    "For incomplete observations, answer_status=partial may pair with state=open, blocked "
    "or deferred, never state=partial. For an explicitly answered narrow read use "
    "answer_status=answered, remaining_question=null, state=answered, remaining_gap='', "
    "remaining_facets=[]; broader diagnostic unknown remains in unresolved. Always emit "
    "all eight top-level fields: stop_basis is null while continuing, never omitted. "
    "Before emitting public_summary, latest_result_update.summary or why_now, perform "
    "a factual self-check against the current supplied records, not earlier prose. "
    "factual_reading_aid is a presentation of those same records, not new evidence or "
    "semantic approval. Its time_windows separates exact_window_ns (Unix ns) from "
    "exact_unix_seconds (Unix seconds); never label Unix seconds as ns. Either retain "
    "the original integer ns endpoints with their unit, or explicitly label the exact "
    "seconds conversion as Unix seconds. Do not shorten, round or change units silently; "
    "timestamp endpoints and duration are distinct quantities, and an absent named "
    "baseline/fault window cannot be supplied from the generic interval. "
    "The aid covers registered summary window fields only; absence of an aid does not "
    "erase an original supplied field or certify that it was not delivered. "
    "Its latest_question_fields is a registered field-delivery display for the original "
    "narrow question, with no automatic question update. Delivery and diagnostic "
    "adequacy are separate dimensions: supplied_unknown is not not_supplied. For example, "
    "when a question asks what coverage is recorded and the supplied sampling_coverage "
    "field is 'unknown', that marker supplies the recorded state; it does not leave a "
    "missing coverage field in that record-read question. diagnostic adequacy remains "
    "a separate unresolved question. Explicit null is supplied_null, not an omitted key; "
    "supplied_value is the original value, not a validity or causal assessment. Each "
    "registered path is shown independently, including both denominator sides; "
    "not_supplied means that particular path is absent, not that the whole facet or "
    "record was unread. Decide and explicitly emit any question update yourself, "
    "keeping the original scope and question text. This field aid only registers "
    "ops-symptoms/1 error_summary and the latest matched immutable question; an empty "
    "list for other records does not erase their supplied values or answer their questions. "
    "For index/2 paired with factual_reading_aid/4, question_delivery_fields retains "
    "registered path-presence states for matching original questions, including older "
    "non-latest reads. It contains no values or automatic answers. A remaining_facets "
    "entry whose every necessary registered path is supplied_value, supplied_unknown "
    "or supplied_null causes atomic note rejection remaining_facets_already_supplied; "
    "an actually absent denominator side may remain. Rejection preserves native text "
    "and does not cancel an independent legal action or buy a correction. Keep broader "
    "diagnostic uncertainty separate; empty remaining does not automatically answer. "
    "The candidate_directory contains published node-card counts, not measured dependency "
    "edges, support votes or observed causal relations. A dependency-node card count is not "
    "topology degree or a count of unique neighbors, callees or edges; it is not a measured "
    "failure count either. Do not relabel that quantity as dependency count, even while "
    "leaving topology versus failures unresolved. Name the published node-card unit when "
    "using it as a navigation reason. Judgment counts are recorded "
    "occurrences, not independent evidence. Smaller supplied rank numbers precede larger "
    "ones; menu order and card count are not rank. If you make a best/highest/leading "
    "comparison within a stated subset, check every supplied candidate in that stated "
    "subset against the original scoring records. Missing ranks or partial publication "
    "do not license a global extremum. Otherwise give a limited reason about the actual "
    "selected record and its useful next observation; a non-leading candidate remains "
    "legal. Do not call a candidate highest merely to justify selecting it. "
    "An unread record is not yet a matched symptom observation. Before claiming a "
    "cross-subject symptom comparison, check the actual supplied modality, statistic, "
    "observation role, unit, named windows, denominators and coverage; keep missing or "
    "unknown qualifiers unresolved and do not equate log severity with request outcome "
    "or client spans with server failures. The same record kind or unit alone does not "
    "establish comparability. Registered caller/callee navigation does not establish "
    "symptom propagation, matching observations or causal direction. Reading the next "
    "record is legal preliminary exploration, but do not promise a matched comparison "
    "before inspecting those qualifiers, or turn zero with unknown coverage into absence. "
    "The program "
    "does not elect a candidate, fix prose or buy a quality retry; accepted structure "
    "still requires factual semantic review. Its note_references.proposal_refs shows "
    "copyable native basis_refs selectors: exactly action_index and card_id, at most "
    "16, for both investigation_note and each question_update. Historical basis_refs "
    "with record_sha256 and delivery_kind are read-only program bindings, not the "
    "proposal format; never copy those extra fields into a native proposal. The "
    "bounded display omits no legal permission: an omitted past selector still needs "
    "the original unique-delivery binder. Selecting a reference neither answers a "
    "question nor supplies new evidence."
    + " "
    + OBSERVABLE_SELECTION_GUIDANCE
)
OBSERVABLE_SYMPTOM_GUIDANCE = (
    "Question: What values and availability markers are actually recorded for the selected "
    "subject, metric and baseline/fault windows? For a summary, request its original "
    "values, unit, coverage and episode status; for error_summary request its error count, "
    "denominators, fractions and coverage, not CPU. The narrow question does not test an "
    "unsupported threshold or a mechanism. Feedback: retain each supplied known value "
    "and unknown/unmeasured marker, distinguish missing delivery from a diagnostic gap. "
    "A covered matched observation can inform a separate hypothesis only within its "
    "real evidence limits. Choose the next useful check or honest stop yourself by its "
    "concrete unresolved gap using the common selection guidance, with no symptom-card quota; this "
    "read does not already perform that check, establish a source or confirm saturation."
)
OBSERVABLE_SKILL_GUIDANCE = {
    node: (
        OBSERVABLE_SYMPTOM_GUIDANCE
        if node == "symptoms"
        else text.removesuffix(" " + ENTRY_GUIDANCE)
    )
    + " Apply common_skill_guidance.guidance supplied with this context "
    "(readable wire: context.history.content.common_skill_guidance); "
    "its limits apply to this skill."
    for node, text in ENTRY_SKILL_GUIDANCE.items()
}


def observable_response_contract() -> dict[str, Any]:
    """Finite shape guidance shares parser constants; never generates an agent response.

    This is prompt content, not a provider response_format or a semantic validator.
    Native responses still pass the original strict parser and independent note gate.
    """
    shape = dict(
        authority="response_shape_guidance_not_evidence_or_provider_schema",
        exact_fields=sorted(ACTION_FIELDS),
        always_include_even_when_null=[
            "stop_reason",
            "stop_basis",
            "latest_result_update",
            "investigation_note",
        ],
        stop_basis=[*sorted(STOP_BASES), None],
        latest_result_update=dict(
            exact_fields=sorted(LATEST_RESULT_FIELDS | {"content_relation"}),
            answer_status=sorted(ANSWER_STATUSES),
            content_relation=sorted(CONTENT_RELATIONS),
            first_turn=None,
            question_scope=(
                "latest actual action's supplied original question; when "
                "context.history.content.question_state.latest_action_question_id is "
                "non-null, retain its immutable text, target_service_keys and "
                "observable_scope, not the new action or global diagnosis. Without a "
                "stored question binding, retain only the supplied original question; "
                "do not invent an ID or scope."
            ),
            answer_status_scope=(
                "feedback answer extent for that original question, not delivery "
                "completeness, record availability/truth_value or diagnostic adequacy; "
                "model judgment, never automatic question update"
            ),
            remaining_question_scope=(
                "only unanswered parts of the original request; when explicitly "
                "answered use null. Broader diagnostic unknown belongs in unresolved "
                "or a legitimate new action question with an explicit public relation, "
                "not an expanded old scope."
            ),
            explicit_latest_question_update=dict(
                applies_when=(
                    "non-null latest_action_question_id and latest_result_update, and "
                    "a non-null note explicitly updates that same stored question_id"
                ),
                answered_equivalence=(
                    "answer_status == 'answered' iff same question_update.state == "
                    "'answered'; necessary consistency only, not semantic approval"
                ),
                otherwise=(
                    "Other question IDs retain their own scope. Without that explicit "
                    "update, with null note or null latest binding, feedback has no "
                    "lifecycle write authority; never auto-answer or fabricate an ID/scope."
                ),
            ),
        ),
        investigation_note=dict(
            exact_fields=sorted(NOTE_FIELDS | {"observable_test"}),
            target_service_keys=dict(
                concrete_current_entry=(
                    "exactly selected entry.observable_contract.binding.targets; "
                    "current action subject, not an older question or reference subject"
                ),
                typed_navigation_or_stop=(
                    "retain existing target admission rules; observable_test is null"
                ),
            ),
            basis_refs=dict(
                exact_fields=["action_index", "card_id"],
                max_items=16,
                subject_scope=(
                    "legal past records may have a different subject; references never "
                    "rebind current targets or an old question scope"
                ),
            ),
            question_updates=dict(
                exact_fields=sorted(QUESTION_UPDATE_FIELDS | {"remaining_facets"}),
                subject_scope=(
                    "retain each stored immutable question text, target_service_keys and "
                    "observable_scope; may differ from current note targets"
                ),
                basis_refs=dict(
                    exact_fields=["action_index", "card_id"],
                    max_items=16,
                    subject_scope=(
                        "legal past records may have a different subject; references never "
                        "rebind current targets or an old question scope"
                    ),
                ),
                state=sorted(QUESTION_UPDATE_STATES),
                state_scope=(
                    "lifecycle of this stored question, not the feedback status domain; "
                    "partial is not a lifecycle state. For explicitly answered scoped "
                    "questions remaining_facets=[]; for unscoped questions retain "
                    "remaining_facets=null. Other question IDs keep their own scope; "
                    "only an explicit latest-question update uses the same-question "
                    "answered equivalence in latest_result_update."
                ),
                remaining_gap_scope=(
                    "only unanswered parts of this original question; when explicitly "
                    "answered use the empty string. Keep broader diagnostic unknown in "
                    "unresolved or a legitimate new action question with an explicit "
                    "public relation; do not expand or rename the stored question."
                ),
                remaining_facets="subset of stored requested_facets; [] when explicitly answered",
            ),
        ),
    )
    shape["continue"] = dict(stop_reason=None, stop_basis=None)
    return shape


OBSERVABLE_SYSTEM = (
    ENTRY_SYSTEM.removesuffix("\n" + ENTRY_GUIDANCE)
    + "\n"
    + OBSERVABLE_GUIDANCE
    + "\nResponse shape contract (not an answer):\n"
    + canonical_json_bytes(observable_response_contract()).decode()
)


def observable_common_guidance() -> dict[str, str]:
    return dict(
        guidance=OBSERVABLE_GUIDANCE, content_sha256=sha256_hex(OBSERVABLE_GUIDANCE.encode())
    )


def observable_decision_policy() -> dict[str, Any]:
    pinned = dict(
        protocol=OBSERVABLE_PROTOCOL,
        system=OBSERVABLE_SYSTEM,
        output_policy=OUTPUT_POLICY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return dict(
        protocol=OBSERVABLE_PROTOCOL,
        system=OBSERVABLE_SYSTEM,
        profile=OBSERVABLE_TOOLS,
        schema=OBSERVABLE_INPUT_SCHEMA,
        guidance=observable_common_guidance(),
        prompt_sha256=sha256_hex(canonical_json_bytes(pinned)),
    )
