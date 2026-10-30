"""Public SRE memory and steps projected from the existing conversation/ledger.

No parallel fact store, source replay, scoring, or model-written observations.
Only already delivered receipts can support an at-the-time response link.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS, is_stop, parse_decision
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
from nurse_rcl.v2.live.ops_context_view import (
    INVESTIGATION_BYTES,
    DiagnosedInputProjection,
    OpsInputProjection,
    WireBuilder,
    _pack,
)
from nurse_rcl.v2.live.ops_failure import failure_detail
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedActionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_sre_encoding import SEMANTIC_ENCODING, compact, expand
from nurse_rcl.v2.live.ops_sre_overview import overview_source_ids
from nurse_rcl.v2.live.ops_sre_skills import sre_skill_index
from nurse_rcl.v2.live.ops_sre_tools import SRE_PROFILE, SRE_TOOL_PARAMETERS
from nurse_rcl.v2.live.ops_sre_views import critical_record
from nurse_rcl.v2.live.ops_tools import card_id

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_source_checkpoint import ScoreSource

SRE_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/5"
WORKING_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/6"
BRIEF_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/7"
FEEDBACK_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/8"
RECOVERY_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/9"
COMPARABLE_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/10"
NAVIGATION_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/11"
WORKBENCH_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/12"
USED_SOURCE_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/13"
RECEIPT_REUSE_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/14"
BALANCED_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/15"
REPLANNING_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/16"
QUESTION_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/17"
ACTION_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/18"
DIRECTED_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/19"
ENTRY_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/20"


def parse_sre_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    value = decode_json_object(raw, CONTEXT_LIMITS)
    if set(value) != {"actions", "stop_reason", "public_summary", "unresolved", "responds_to"}:
        raise ExchangeBoundaryError("sre_decision_shape")
    parse_decision(
        canonical_json_bytes({k: v for k, v in value.items() if k != "responds_to"}), empirical=True
    )
    links = value["responds_to"]
    if type(links) is not list or len(links) > 8:
        raise ExchangeBoundaryError("sre_response_links")
    seen = set()
    available = (
        {}
        if context is None
        else {r["action_index"]: set(r["visible_card_ids"]) for r in context["actions"]}
    )
    for link in links:
        if (
            type(link) is not dict
            or set(link) != {"action_index", "card_ids"}
            or (
                type(link["action_index"]) is not int
                or not 1 <= link["action_index"] <= 8
                or link["action_index"] in seen
                or type(link["card_ids"]) is not list
                or any(type(key) is not str for key in link["card_ids"])
                or len(set(link["card_ids"])) != len(link["card_ids"])
            )
        ):
            raise ExchangeBoundaryError("sre_response_links")
        seen.add(link["action_index"])
        if context is not None and (
            link["action_index"] not in available
            or not set(link["card_ids"]) <= available[link["action_index"]]
        ):
            raise ExchangeBoundaryError("sre_response_not_previously_visible")
    return value


def _text(value: str, limit: int = 1024) -> dict[str, Any]:
    # Input-only excerpt, never an output rejection or hidden reasoning request.
    return dict(
        text=value[:limit],
        omitted_characters=max(0, len(value) - limit),
        complete_text_sha256=sha256_hex(value.encode()),
        omission_note="full public text retained in original conversation",
    )


def _memory(
    row: OpsDecisionRecord, value: dict[str, Any], *, references: dict[str, Any] | None = None
) -> dict[str, Any]:
    parameters = SRE_TOOL_PARAMETERS
    if json.loads(row.context_json).get("schema_version") in {
        WORKING_INPUT_SCHEMA,
        BRIEF_INPUT_SCHEMA,
        FEEDBACK_INPUT_SCHEMA,
        RECOVERY_INPUT_SCHEMA,
        COMPARABLE_INPUT_SCHEMA,
        NAVIGATION_INPUT_SCHEMA,
        WORKBENCH_INPUT_SCHEMA,
        REPLANNING_INPUT_SCHEMA,
        QUESTION_INPUT_SCHEMA,
        ACTION_INPUT_SCHEMA,
        DIRECTED_INPUT_SCHEMA,
    }:
        from nurse_rcl.v2.live.ops_sre_working_tools import WORKING_TOOL_PARAMETERS

        parameters = WORKING_TOOL_PARAMETERS
    if json.loads(row.context_json).get("schema_version") == DIRECTED_INPUT_SCHEMA:
        from nurse_rcl.v2.live.ops_sre_directed_tools import OpsSreDirectedToolbox

        parameters = OpsSreDirectedToolbox.tool_parameters
    requested = None
    if (
        json.loads(row.context_json).get("schema_version")
        in {
            ENTRY_INPUT_SCHEMA,
            "nurse.v2.sre-investigation-input/24",
            "nurse.v2.sre-investigation-input/26",
        }
        and len(value["actions"]) == 1
    ):
        # Native navigation retains short names. It is audit text, not another
        # source-name issuance position; actual original parameters come from
        # the separate ledger receipt. Never issue "S1" as a new service object.
        requested = dict(
            native_selection_json=canonical_json_bytes(value["actions"][0]).decode(),
            authority="unverified_proposal_not_execution",
        )
    if len(value["actions"]) == 1:
        proposal = value["actions"][0]
        if (
            type(proposal) is dict
            and type(proposal.get("tool")) is str
            and (
                proposal["tool"] in parameters
                and type(proposal.get("skill_id")) is str
                and type(proposal.get("parameters")) is dict
                and set(proposal["parameters"]) == set(parameters[proposal["tool"]])
            )
        ):
            requested = dict(
                tool=proposal["tool"],
                skill_id=proposal["skill_id"],
                parameters=proposal["parameters"],
                authority="unverified_proposal_not_execution",
            )
    result = dict(
        role="agent_statement",
        decision_index=row.index,
        action_index=row.action_index,
        requested_action=requested,
        requested_action_note="null means stop or unregistered shape; original retained",
        questions=[
            _text(a["question"])
            for a in value["actions"]
            if type(a) is dict and type(a.get("question")) is str
        ],
        public_summary=_text(value["public_summary"]),
        unresolved=[_text(s) for s in value["unresolved"]],
        stop_reason=None if value["stop_reason"] is None else _text(value["stop_reason"]),
        responds_to=value["responds_to"],
        meaning="unverified_public_intent_not_tool_fact_or_hidden_reasoning",
    )
    if "investigation_question" in value:
        from nurse_rcl.v2.live.ops_sre_brief_context import stop_diagnostics

        context = json.loads(row.context_json)
        result.update(
            investigation_question=value["investigation_question"],
            stop_basis=value["stop_basis"],
            stop_diagnostics=stop_diagnostics(
                value, decision_index=row.index, actions_used=len(context["actions"])
            ),
        )
    if "investigation_plan" in value:
        result.update(
            investigation_plan=value["investigation_plan"],
            stop_assessment=value["stop_assessment"],
            stop_basis=value["stop_basis"],
        )
    if "latest_result_update" in value:
        result["latest_result_update"] = value["latest_result_update"]
    if "investigation_note" in value:
        from nurse_rcl.v2.live.ops_sre_action_decision import note_memory

        result.update(
            note_memory(value, context=json.loads(row.context_json), references=references)
        )
    return result


def _semantic_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    view = expand(view)
    rows = (
        []
        if view.get("page", {}).get("unit") == "candidate_overview"
        else list(view.get("page", {}).get("records", []))
    )
    ids = [r["card_id"] for r in rows]
    critical = view.get("critical_record_ids", [])
    if (
        type(critical) is not list
        or any(type(k) is not str for k in critical)
        or (len(set(critical)) != len(critical) or not set(critical) <= set(ids))
    ):
        raise ValueError("SRE critical record reference differs")
    if "critical_records" in view or len(ids) != len(set(ids)):
        raise ValueError("SRE repeated or legacy record bodies")
    if "semantic_record" in view:
        rows.append(view["semantic_record"])
    return list({r["card_id"]: r for r in rows}.values())


def _delivered_records(view: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _semantic_rows(view)
    if (
        not rows
        and "node" in view
        and "card_id" in view
        and "truth_value" in view
        and (view.get("truth_value") in {"false", "unknown"} or view.get("limitations"))
    ):
        # Non-payload sections expose a header and a partial page. Retain exactly
        # that delivered view, not a fresh full-card read or invented predicates.
        rows.append(
            dict(
                card_id=view["card_id"],
                node=view["node"],
                source_role="delivered_partial_card",
                truth_value=view["truth_value"],
                representation="delivered_section_not_full_judgment",
                payload=dict(delivered_view=view),
                limitations=view.get("limitations", []),
            )
        )
    return rows


def _receipt(record: OpsActionRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, OpsIndexedActionRecord):
        raise ValueError("SRE requires saved model-side receipts")
    row: dict[str, Any] = dict(
        action_index=record.index,
        status=record.status,
        failure_type=record.failure_type,
        visible_card_ids=[],
        input_sha256=record.input_sha256,
    )
    if record.action is not None:
        row.update(
            tool=record.action.tool,
            skill_id=record.action.skill_id,
            parameters=json.loads(record.action.parameters_json),
        )
    if record.result is None:
        if record.status not in {"reserved", "interrupted"}:
            raise ValueError("SRE action receipt missing")
        return row, dict(
            status="interrupted",
            action_index=record.index,
            limitations=["completion_unknown_no_refund", "no_factual_conclusion"],
        )
    if record.model_view_json is None or record.result.action_index != record.index:
        raise ValueError("SRE view/action binding differs")
    view = json.loads(record.model_view_json)
    if view.get("action_index") != record.index or view.get("origin") != record.result.origin:
        raise ValueError("SRE feedback binding differs")
    ids = {r["card_id"] for r in _semantic_rows(view)}
    ids.update(overview_source_ids(expand(view)))
    # Discovery grants a visible ID, not observation/health authority.
    ids.update(
        r["card_id"]
        for r in view.get("page", {}).get("items", [])
        if type(r) is dict and type(r.get("card_id")) is str
    )
    if "card_id" in view:
        ids.add(view["card_id"])
    row.update(
        visible_card_ids=sorted(ids),
        receipt_card_id=card_id(record.result.evidence),
        origin=record.result.origin,
        reused_core_evidence_ids=record.result.reused_core_evidence_ids,
        publication_prefix=view.get("publication_prefix"),
        meaning="action_record_not_independent_evidence_or_formal_scoring",
    )
    return row, view


def investigation_steps(
    *,
    decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    input_schema: str = SRE_INPUT_SCHEMA,
    references: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project, do not repair or embellish, the saved order and response links."""
    actions = {r.index: r for r in records}
    if list(actions) != list(range(1, len(records) + 1)):
        raise ValueError("SRE action sequence differs")
    steps: list[dict[str, Any]] = []
    by_action: dict[int, dict[str, Any]] = {}
    declared = 0
    for index, row in enumerate(decisions, 1):
        if row.index != index:
            raise ValueError("SRE decision sequence differs")
        context = json.loads(row.context_json)
        if (
            input_schema
            not in {
                SRE_INPUT_SCHEMA,
                WORKING_INPUT_SCHEMA,
                BRIEF_INPUT_SCHEMA,
                FEEDBACK_INPUT_SCHEMA,
                RECOVERY_INPUT_SCHEMA,
                COMPARABLE_INPUT_SCHEMA,
                NAVIGATION_INPUT_SCHEMA,
                WORKBENCH_INPUT_SCHEMA,
                USED_SOURCE_INPUT_SCHEMA,
                RECEIPT_REUSE_INPUT_SCHEMA,
                BALANCED_INPUT_SCHEMA,
                REPLANNING_INPUT_SCHEMA,
                QUESTION_INPUT_SCHEMA,
                ACTION_INPUT_SCHEMA,
                DIRECTED_INPUT_SCHEMA,
                ENTRY_INPUT_SCHEMA,
                "nurse.v2.sre-investigation-input/21",
                "nurse.v2.sre-investigation-input/22",
                "nurse.v2.sre-investigation-input/23",
                "nurse.v2.sre-investigation-input/24",
                "nurse.v2.sre-investigation-input/26",
            }
            or (context.get("schema_version") != input_schema)
            or (context.get("decision_index") != index)
        ):
            raise ValueError("SRE history context identity differs")
        parser = parse_sre_decision
        if input_schema == "nurse.v2.sre-investigation-input/26":
            from nurse_rcl.v2.live.ops_sre_focused_decision import parse_focused_decision

            parser = parse_focused_decision
        if input_schema == "nurse.v2.sre-investigation-input/24":
            from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import parse_atomic_decision

            parser = parse_atomic_decision
        if input_schema == "nurse.v2.sre-investigation-input/21":
            from nurse_rcl.v2.live.ops_sre_entry_decision import parse_observable_decision

            parser = parse_observable_decision
        if input_schema == "nurse.v2.sre-investigation-input/23":
            from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                parse_typed_purpose_decision,
            )

            parser = parse_typed_purpose_decision
        if input_schema == "nurse.v2.sre-investigation-input/22":
            from nurse_rcl.v2.live.ops_sre_single_owner_decision import parse_single_owner_decision

            parser = parse_single_owner_decision
        if input_schema == BRIEF_INPUT_SCHEMA:
            from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

            parser = parse_brief_decision
        if input_schema in {
            FEEDBACK_INPUT_SCHEMA,
            RECOVERY_INPUT_SCHEMA,
            COMPARABLE_INPUT_SCHEMA,
            NAVIGATION_INPUT_SCHEMA,
            WORKBENCH_INPUT_SCHEMA,
            USED_SOURCE_INPUT_SCHEMA,
            RECEIPT_REUSE_INPUT_SCHEMA,
        }:
            from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

            parser = parse_feedback_decision
        if input_schema in {BALANCED_INPUT_SCHEMA, REPLANNING_INPUT_SCHEMA}:
            from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

            parser = parse_balanced_decision
        if input_schema in {QUESTION_INPUT_SCHEMA, ACTION_INPUT_SCHEMA, DIRECTED_INPUT_SCHEMA}:
            from nurse_rcl.v2.live.ops_sre_d26_decision import parse_question_decision

            parser = parse_question_decision
        if input_schema == ENTRY_INPUT_SCHEMA:
            from nurse_rcl.v2.live.ops_sre_entry_decision import parse_entry_decision

            parser = parse_entry_decision
        value = (
            None
            if row.response_json is None
            else parser(row.response_json.encode(), context=context)
        )
        if value is not None:
            for link in value["responds_to"]:
                if link["action_index"] not in by_action:
                    raise ValueError("SRE response lacks a prior planned action")
                prior = by_action[link["action_index"]]
                prior["responded_to"].append(
                    dict(
                        decision_index=index,
                        role="agent_statement",
                        card_ids=link["card_ids"],
                        public_summary=_text(value["public_summary"]),
                        unresolved=[_text(s) for s in value["unresolved"]],
                    )
                )
                prior["response_state"] = "recorded"
                update = value.get("latest_result_update")
                if update is not None and update["action_index"] == link["action_index"]:
                    prior["responded_to"][-1]["latest_result_update"] = update
        executed: dict[str, Any] = dict(status="not_dispatched", visible_card_ids=[])
        if row.action_index is not None:
            declared += 1
            if row.action_index != declared or value is None or is_stop(value):
                raise ValueError("SRE declared action sequence differs")
            actual = actions.get(declared)
            if actual is not None:
                if actual.input_sha256 != sha256_hex(canonical_json_bytes(value["actions"])):
                    raise ValueError("SRE action does not match its plan")
                executed, _ = _receipt(actual)
        elif value is not None and not is_stop(value):
            if input_schema not in {
                REPLANNING_INPUT_SCHEMA,
                QUESTION_INPUT_SCHEMA,
                ACTION_INPUT_SCHEMA,
                DIRECTED_INPUT_SCHEMA,
                ENTRY_INPUT_SCHEMA,
                "nurse.v2.sre-investigation-input/24",
                "nurse.v2.sre-investigation-input/26",
            } or (getattr(row, "blocked_read_proof", None) is None):
                raise ValueError("SRE non-stop decision lacks an action index")
            from nurse_rcl.v2.live.ops_sre_d25_progress import validate_blocked_read

            if input_schema == ENTRY_INPUT_SCHEMA:
                from nurse_rcl.v2.live.ops_sre_entry_decision import (
                    validate_blocked_entry_read as validate_blocked_read,
                )

            if input_schema == "nurse.v2.sre-investigation-input/24":
                from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
                    validate_atomic_blocked_entry_read as validate_blocked_read,
                )

            if input_schema == "nurse.v2.sre-investigation-input/26":
                from nurse_rcl.v2.live.ops_sre_focused_decision import (
                    validate_focused_blocked_entry_read as validate_blocked_read,
                )

            validate_blocked_read(value, row.blocked_read_proof, context)  # type: ignore[attr-defined]
        known = context["actions"]
        published = context["capabilities"].get("published_node_summaries", [])
        step = dict(
            step_ref=f"s{index}",
            decision_index=index,
            action_index=row.action_index,
            planned=dict(role="agent_statement", status="awaiting_response")
            if value is None
            else _memory(row, value, references=references),
            known_at_decision=dict(
                publication_prefixes=[
                    a["publication_prefix"]
                    for a in known
                    if a.get("publication_prefix") is not None
                ],
                visible_card_ids=sorted(
                    {key for a in known for key in a["visible_card_ids"]}
                    | {p["card_id"] for p in published}
                ),
                core_publication_summaries=published,
                context_sha256=sha256_hex(row.context_json.encode()),
            ),
            executed=executed,
            responded_to=[],
            response_state="not_recorded",
            report_synthesis="later_interpretation_requires_fact_refs",
            core_actions="independent_not_agent_scheduled",
        )
        steps.append(step)
        if row.action_index is not None:
            by_action[row.action_index] = step
    if len(records) > declared or declared > len(records) + 1:
        raise ValueError("SRE conversation/ledger count differs")
    return steps


def sre_investigation_input(
    *,
    capabilities: dict[str, Any],
    records: tuple[OpsActionRecord, ...],
    decisions: tuple[OpsDecisionRecord, ...],
    decision_index: int,
    make_wire: WireBuilder,
    navigation_source: ScoreSource | None = None,
    role_source_resolver: Callable[[str], dict[str, Any]] | None = None,
    entry_references: dict[str, Any] | None = None,
    entry_case_id: str | None = None,
) -> OpsInputProjection:
    from nurse_rcl.v2.live.ops_sre_brief_tools import BRIEF_PROFILE
    from nurse_rcl.v2.live.ops_sre_feedback_context import CONDITION_PROFILE, latest_result_context
    from nurse_rcl.v2.live.ops_sre_working_memory import working_history
    from nurse_rcl.v2.live.ops_sre_working_tools import WORKING_PROFILE

    profile = capabilities.get("investigation_profile_id")
    working = profile == WORKING_PROFILE
    focused = profile == "hermes-sre-tools/25"
    atomic_selector = profile == "hermes-sre-tools/23" or focused
    typed_purpose = profile == "hermes-sre-tools/22"
    single_owner = profile == "hermes-sre-tools/21"
    observable = (
        profile == "hermes-sre-tools/20" or single_owner or typed_purpose or atomic_selector
    )
    entry = profile == "hermes-sre-tools/19" or observable
    directed = profile == "hermes-sre-tools/18" or entry
    action_first = profile in {"hermes-sre-tools/17", "hermes-sre-tools/18"} or entry
    question_led = (
        profile in {"hermes-sre-tools/16", "hermes-sre-tools/17", "hermes-sre-tools/18"} or entry
    )
    replanning = profile == "hermes-sre-tools/15" or question_led
    balanced = profile == "hermes-sre-tools/14" or replanning
    receipt_reuse = profile == "hermes-sre-tools/13" or balanced
    used_source = profile == "hermes-sre-tools/12" or receipt_reuse
    workbench = profile == "hermes-sre-tools/11" or used_source
    navigation = profile == "hermes-sre-tools/10" or workbench
    comparable = profile == "hermes-sre-tools/9" or navigation
    recovery = profile == "hermes-sre-tools/8" or comparable
    condition = profile == CONDITION_PROFILE or recovery
    brief = profile in {BRIEF_PROFILE, CONDITION_PROFILE} or recovery
    schema = (
        "nurse.v2.sre-investigation-input/26"
        if focused
        else "nurse.v2.sre-investigation-input/24"
        if atomic_selector
        else "nurse.v2.sre-investigation-input/23"
        if typed_purpose
        else "nurse.v2.sre-investigation-input/22"
        if single_owner
        else "nurse.v2.sre-investigation-input/21"
        if observable
        else ENTRY_INPUT_SCHEMA
        if entry
        else DIRECTED_INPUT_SCHEMA
        if directed
        else ACTION_INPUT_SCHEMA
        if action_first
        else QUESTION_INPUT_SCHEMA
        if question_led
        else REPLANNING_INPUT_SCHEMA
        if replanning
        else BALANCED_INPUT_SCHEMA
        if balanced
        else RECEIPT_REUSE_INPUT_SCHEMA
        if receipt_reuse
        else USED_SOURCE_INPUT_SCHEMA
        if used_source
        else WORKBENCH_INPUT_SCHEMA
        if workbench
        else NAVIGATION_INPUT_SCHEMA
        if navigation
        else COMPARABLE_INPUT_SCHEMA
        if comparable
        else RECOVERY_INPUT_SCHEMA
        if recovery
        else FEEDBACK_INPUT_SCHEMA
        if condition
        else BRIEF_INPUT_SCHEMA
        if brief
        else WORKING_INPUT_SCHEMA
        if working
        else SRE_INPUT_SCHEMA
    )
    if profile not in {
        SRE_PROFILE,
        WORKING_PROFILE,
        BRIEF_PROFILE,
        CONDITION_PROFILE,
        "hermes-sre-tools/8",
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
        "hermes-sre-tools/18",
        "hermes-sre-tools/19",
        "hermes-sre-tools/20",
        "hermes-sre-tools/21",
        "hermes-sre-tools/22",
        "hermes-sre-tools/23",
        "hermes-sre-tools/25",
    } or (
        capabilities.get("semantic_encoding") != SEMANTIC_ENCODING
        or type(decision_index) is not int
        or not 1 <= decision_index <= 8
        or len(decisions) != decision_index - 1
        or len(records) >= decision_index
    ):
        raise ValueError("SRE investigation context identity differs")
    investigation_steps(
        decisions=decisions, records=records, input_schema=schema, references=entry_references
    )
    snapshot = capabilities["case_snapshot_id"]
    actions, previous, overviews = [], None, []
    critical: dict[str, dict[str, Any]] = {}
    latest_ids: set[str] = set()
    history = []
    for record in records:
        row, previous = _receipt(record)
        if record.result is not None and record.result.evidence.case_snapshot_id != snapshot:
            raise ValueError("SRE receipt snapshot differs")
        if previous.get("status") == "resource_limit":
            return OpsInputProjection(
                status="resource_limit", scope="tool_view", target_bytes=INVESTIGATION_BYTES
            )
        actions.append(row)
        if working or brief:
            history.append(dict(action_index=record.index, status=record.status, view=previous))
            continue
        latest_ids = set()
        for semantic in _delivered_records(previous):
            if critical_record(semantic) or semantic.get("representation") == (
                "delivered_section_not_full_judgment"
            ):
                key = semantic["card_id"]
                latest_ids.add(key)
                saved = critical.setdefault(
                    key, dict(record=semantic, action_indices=[], record_action_index=record.index)
                )
                # A later partial view must not downgrade delivered full semantics.
                partial = {"overview_not_full_judgment", "delivered_section_not_full_judgment"}
                if semantic.get("representation") not in partial:
                    saved["record"] = semantic
                    saved["record_action_index"] = record.index
                saved["action_indices"].append(record.index)
        if "critical_overview" in previous:
            expanded = expand(previous)
            summaries = (
                expanded["page"]["records"]
                if expanded.get("page", {}).get("unit") == "candidate_overview"
                else expanded.get("candidate_overviews", [])
            )
            overviews.append(
                dict(
                    action_index=record.index,
                    publication_prefix=previous["publication_prefix"],
                    overview=previous["critical_overview"],
                    candidate_overviews=summaries,
                )
            )
    parser = parse_sre_decision
    if brief:
        from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

        parser = parse_brief_decision
    if condition:
        from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

        parser = parse_feedback_decision
    if balanced:
        from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

        parser = parse_balanced_decision
    if question_led:
        from nurse_rcl.v2.live.ops_sre_d26_decision import parse_question_decision

        parser = parse_question_decision
    if entry:
        from nurse_rcl.v2.live.ops_sre_entry_decision import parse_entry_decision

        parser = parse_entry_decision
        if atomic_selector:
            from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import parse_atomic_decision

            parser = parse_atomic_decision
            if focused:
                from nurse_rcl.v2.live.ops_sre_focused_decision import parse_focused_decision

                parser = parse_focused_decision
        elif typed_purpose:
            from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                parse_typed_purpose_decision,
            )

            parser = parse_typed_purpose_decision
        elif single_owner:
            from nurse_rcl.v2.live.ops_sre_single_owner_decision import parse_single_owner_decision

            parser = parse_single_owner_decision
        elif observable:
            from nurse_rcl.v2.live.ops_sre_entry_decision import parse_observable_decision

            parser = parse_observable_decision
    memory = [
        _memory(
            row,
            parser(row.response_json.encode(), context=json.loads(row.context_json)),
            references=entry_references,
        )
        for row in decisions
        if row.response_json is not None
    ]
    evidence = []
    for key, item in critical.items():
        evidence_entry = dict(
            card_id=key,
            action_indices=item["action_indices"],
            record_action_index=item["record_action_index"],
        )
        # Previous receipt is included verbatim, so avoid sending those exact rows twice.
        last = next((r for r in _delivered_records(previous or {}) if r["card_id"] == key), None)
        if key in latest_ids and last == item["record"]:
            evidence_entry["record_location"] = (
                "previous_feedback semantic record with this card_id"
            )
        else:
            evidence_entry["record"] = compact(item["record"])
        evidence.append(evidence_entry)
    capabilities = dict(capabilities)
    routes = capabilities.pop("investigation_routes")
    view = dict(
        schema_version=schema,
        case_snapshot_id=snapshot,
        decision_index=decision_index,
        remaining_actions=8 - len(records),
        remaining_decisions=9 - decision_index,
        capabilities=capabilities,
        skills=sre_skill_index(profile=str(profile)),
        actions=actions,
        previous_feedback=previous,
        critical_evidence=evidence,
        investigation_routes=routes,
        prior_response_target=None
        if not actions
        else dict(
            action_index=actions[-1]["action_index"], card_ids=actions[-1]["visible_card_ids"]
        ),
        unexpanded_critical_overviews=overviews,
        public_working_memory=memory,
        omitted_detail=[
            "ordinary past tool detail remains in original receipts",
            "agent_statement is not a fact; excerpts are explicitly marked",
            "core autonomous work is not agent-scheduled investigation",
        ],
    )
    try:
        if brief:
            from nurse_rcl.v2.live.ops_sre_brief_context import (
                brief_working_history,
                decision_budget,
            )
            from nurse_rcl.v2.live.ops_sre_skills import brief_common_guidance

            view.update(brief_working_history(history, condition_explanations=condition))
            view["budget"] = decision_budget(
                decision_index=decision_index, actions_used=len(records)
            )
            view.pop("remaining_actions")
            view.pop("remaining_decisions")
            view.pop("unexpanded_critical_overviews")
            view["common_skill_guidance"] = brief_common_guidance()
            if condition:
                from nurse_rcl.v2.live.ops_sre_skills import condition_common_guidance

                view["common_skill_guidance"] = condition_common_guidance()
                view["latest_result"] = latest_result_context(
                    actions, memory, view["delivery_feedback"]
                )
                if recovery:
                    from nurse_rcl.v2.live.ops_sre_skills import recovery_common_guidance

                    view["common_skill_guidance"] = recovery_common_guidance()
                    if comparable:
                        from nurse_rcl.v2.live.ops_sre_skills import comparable_common_guidance

                        view["common_skill_guidance"] = comparable_common_guidance()
                    if navigation:
                        from nurse_rcl.v2.live.ops_sre_delivered_navigation import (
                            verify_navigation_source,
                        )
                        from nurse_rcl.v2.live.ops_sre_skills import navigation_common_guidance

                        view["common_skill_guidance"] = navigation_common_guidance()
                        view["delivered_navigation"] = verify_navigation_source(
                            history, source=navigation_source
                        )
                    if view["latest_result"] is not None:
                        view["latest_result"]["recovery_location"] = (
                            "investigation_routes.recovery_feedback"
                        )
            view["omitted_detail"].append(
                "briefs do not supply hidden leaves; exact details require explicit actions"
            )
        if working:
            view.update(working_history(history))
            view["omitted_detail"].append(
                "historical_evidence_summary retains condition judgments and caveats, "
                "not every leaf; full historical details require an explicit registered reread"
            )
        if workbench:
            from nurse_rcl.v2.live.ops_sre_skills import (
                balanced_common_guidance,
                replanning_common_guidance,
                used_source_common_guidance,
                workbench_common_guidance,
            )
            from nurse_rcl.v2.live.ops_sre_workbench import (
                delivered_record_index,
                select_public_memory,
            )

            view["delivered_record_index"] = delivered_record_index(history)
            if replanning:
                from nurse_rcl.v2.live.ops_sre_d25_progress import select_replanning_memory

                view["public_working_memory"] = select_replanning_memory(memory)
            else:
                view["public_working_memory"] = select_public_memory(memory)
            view["common_skill_guidance"] = (
                replanning_common_guidance()
                if replanning
                else balanced_common_guidance()
                if balanced
                else used_source_common_guidance()
                if used_source
                else workbench_common_guidance()
            )
            if used_source:
                from nurse_rcl.v2.live.ops_sre_d21_workbench import post_action_state

                view["post_action_state"] = post_action_state(
                    actions,
                    view["delivered_record_index"],
                    actions_available_now=view["budget"]["actions_available_now"],
                )
                if balanced:
                    from nurse_rcl.v2.live.ops_sre_d23_workbench import (
                        build_investigation_workbench,
                    )

                    builder = build_investigation_workbench
                    field = "investigation_workbench"
                    if replanning:
                        from nurse_rcl.v2.live.ops_sre_d25_progress import (
                            build_investigation_progress,
                        )

                        builder = build_investigation_progress
                        field = "investigation_progress"
                    view[field] = builder(
                        actions,
                        history,
                        view["delivered_record_index"],
                        view["post_action_state"],
                        role_source_resolver=role_source_resolver,
                    )
            if replanning:
                view["omitted_detail"].append(
                    "D25 lossy input selection: prior public_summary/unresolved omitted "
                    "from decision memory; questions and receipt links kept, "
                    "original conversation retained"
                )
            view["omitted_detail"].append(
                "D18 lossy input selection: older agent summaries omitted, "
                "original conversation retained"
            )
        if question_led:
            from nurse_rcl.v2.live.ops_sre_atomic_selector_questions import project_atomic_questions
            from nurse_rcl.v2.live.ops_sre_d26_exchange import question_common_guidance
            from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state
            from nurse_rcl.v2.live.ops_sre_d26_semantics import diagnostic_semantics
            from nurse_rcl.v2.live.ops_sre_observable_scope import project_observable_questions
            from nurse_rcl.v2.live.ops_sre_typed_purpose_questions import project_typed_questions

            view["question_state"] = (
                project_atomic_questions(memory)
                if atomic_selector
                else project_typed_questions(memory)
                if typed_purpose
                else project_observable_questions(memory)
                if observable
                else project_question_state(memory, entry_field="entry_ref" if entry else "card_id")
            )
            view["diagnostic_semantics"] = diagnostic_semantics(history)
            view["common_skill_guidance"] = question_common_guidance()
            if action_first:
                from nurse_rcl.v2.live.ops_sre_action_exchange import action_common_guidance

                view["common_skill_guidance"] = action_common_guidance()
            if directed:
                from nurse_rcl.v2.live.ops_sre_directed_exchange import directed_common_guidance

                view["common_skill_guidance"] = directed_common_guidance()
            if entry:
                from nurse_rcl.v2.live.ops_sre_entry_exchange import entry_common_guidance

                view["common_skill_guidance"] = entry_common_guidance()
                if atomic_selector:
                    from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import (
                        atomic_common_guidance,
                    )

                    view["common_skill_guidance"] = atomic_common_guidance()
                elif typed_purpose:
                    from nurse_rcl.v2.live.ops_sre_typed_purpose_exchange import (
                        typed_purpose_common_guidance,
                    )

                    view["common_skill_guidance"] = typed_purpose_common_guidance()
                elif single_owner:
                    from nurse_rcl.v2.live.ops_sre_single_owner_exchange import (
                        single_owner_common_guidance,
                    )

                    view["common_skill_guidance"] = single_owner_common_guidance()
                elif observable:
                    from nurse_rcl.v2.live.ops_sre_observable_exchange import (
                        observable_common_guidance,
                    )

                    view["common_skill_guidance"] = observable_common_guidance()
        if entry:
            from nurse_rcl.v2.live.ops_sre_action_entries import (
                compile_action_entries,
                entry_public_scope,
            )
            from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
            from nurse_rcl.v2.live.ops_sre_references import encode_references

            if observable:
                # Same original projection, now available before entries and the aid.
                view["delivered_record_index"] = delivered_record_index(
                    history, question_state=view["question_state"]
                )

            if entry_references is None or entry_case_id is None:
                raise ValueError("entry input requires owner references and case")
            book = SreActionReferenceBook.from_state(
                entry_references, snapshot=snapshot, run_id=entry_references["run_id"]
            )
            encode_references(
                expand(view), book=book, visible_at=decision_index, case_id=entry_case_id
            )
            view["action_entries"] = compile_action_entries(
                dict(view["capabilities"], investigation_routes=view["investigation_routes"]),
                book=book,
                request_index=decision_index,
                case_id=entry_case_id,
                public_scope=entry_public_scope(view),
                delivered_index=view["delivered_record_index"],
                _observable=observable,
                _single_owner=single_owner,
                _typed_purpose=typed_purpose,
                _atomic_selector=atomic_selector and not focused,
                _focused=focused,
            )
            entry_references.clear()
            entry_references.update(book.to_state())
        if observable:
            from nurse_rcl.v2.live.ops_sre_factual_reading import factual_reading_aid

            view["factual_reading_aid"] = factual_reading_aid(
                expand(view), schema_version="nurse.v2.sre-factual-reading-aid/4"
            )
        expand(view)  # The whole semantic context, not just its compact wire, is bounded.
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        if working or brief:
            return DiagnosedInputProjection(
                status="resource_limit",
                scope="expanded_context",
                target_bytes=INVESTIGATION_BYTES,
                failure_detail=failure_detail("investigation_input", exc),
            )
        return OpsInputProjection(
            status="resource_limit", scope="expanded_context", target_bytes=INVESTIGATION_BYTES
        )
    return _pack(view, make_wire, INVESTIGATION_BYTES, soft_targets=True, diagnose=working or brief)
