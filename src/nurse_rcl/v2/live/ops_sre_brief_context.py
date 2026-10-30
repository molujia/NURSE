"""D13 public questions, exact delivery memory and unambiguous action budget."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_tools import brief_delivery_feedback
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION, project_evidence_brief
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

BRIEF_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/7"


def decision_budget(*, decision_index: int, actions_used: int) -> dict[str, int]:
    if type(decision_index) is not int or not 1 <= decision_index <= 8:
        raise ValueError("decision index differs")
    if type(actions_used) is not int or not 0 <= actions_used < decision_index:
        raise ValueError("action count differs")
    return dict(
        decision_index=decision_index,
        actions_available_now=8 - actions_used,
        decisions_after_this=8 - decision_index,
    )


def parse_brief_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision

    value = decode_json_object(raw, CONTEXT_LIMITS)
    extra = {"investigation_question", "stop_basis"}
    base_fields = {"actions", "stop_reason", "public_summary", "unresolved", "responds_to"}
    if set(value) != base_fields | extra:
        raise ExchangeBoundaryError("sre_brief_decision_shape")
    parse_sre_decision(
        canonical_json_bytes({k: v for k, v in value.items() if k not in extra}), context=context
    )
    basis = value["stop_basis"]
    if basis is not None and (
        type(basis) is not str
        or basis
        not in {
            "no_useful_check",
            "budget_exhausted",
            "unavailable",
        }
    ):
        raise ExchangeBoundaryError("sre_stop_basis")
    question = value["investigation_question"]
    if question is None:
        if value["actions"]:
            raise ExchangeBoundaryError("sre_investigation_question_missing")
        return value
    if type(question) is not dict or set(question) != {
        "target_service_keys",
        "card_ids",
        "discriminating_value",
        "why_now",
    }:
        raise ExchangeBoundaryError("sre_investigation_question_shape")
    for field in ("discriminating_value", "why_now"):
        if type(question[field]) is not str or not question[field].strip():
            raise ExchangeBoundaryError("sre_investigation_question_shape")
    for field in ("target_service_keys", "card_ids"):
        items = question[field]
        if (
            type(items) is not list
            or any(type(k) is not str for k in items)
            or (len(set(items)) != len(items))
        ):
            raise ExchangeBoundaryError("sre_investigation_question_shape")
    if context is not None:
        from nurse_rcl.v2.live.ops_sre_delivered_navigation import navigation_targets

        capabilities = context["capabilities"]
        candidates = {
            c["candidate_service_key"]
            for c in capabilities.get("investigation_situation", {}).get("candidates", [])
        }
        visible = {key for r in context["actions"] for key in r["visible_card_ids"]}
        visible.update(navigation_targets(context))
        visible.update(r["card_id"] for r in capabilities.get("published_node_summaries", []))
        visible.update(
            key
            for route in context.get("investigation_routes", {}).get("entries", [])
            for key in route.get("card_ids", [])
        )
        visible.update(
            r["card_id"]
            for c in capabilities.get("investigation_situation", {}).get("candidates", [])
            for r in c.get("scoring", {}).get("records", [])
        )
        if not set(question["target_service_keys"]) <= candidates or (
            not set(question["card_ids"]) <= visible
        ):
            raise ExchangeBoundaryError("sre_question_not_previously_visible")
    return value


def stop_diagnostics(value: dict[str, Any], *, decision_index: int, actions_used: int) -> list[str]:
    budget = decision_budget(decision_index=decision_index, actions_used=actions_used)
    if (
        not value["actions"]
        and value["stop_basis"] == "budget_exhausted"
        and (budget["actions_available_now"] > 0)
    ):
        return ["model_claimed_budget_exhausted_with_action_available"]
    return []


def brief_working_history(
    history: list[dict[str, Any]], *, condition_explanations: bool = False
) -> dict[str, Any]:
    """Share bodies, not source/step identities. Never promote a brief to full."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    project = project_evidence_brief
    if condition_explanations:
        from nurse_rcl.v2.live.ops_sre_condition_briefs import project_condition_brief

        project = project_condition_brief

    if len(history) > 8 or [r["action_index"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("brief history action sequence differs")
    retained: dict[str, dict[str, Any]] = {}
    bodies: dict[str, dict[str, Any]] = {}
    overview_deliveries = []
    last: set[str] = set()
    for saved in history:
        last = set()
        index = saved["action_index"]
        view = working_evidence_view(expand(saved["view"]))
        if saved["status"] != "completed" or view.get("status") == "resource_limit":
            continue
        for row in _delivered_records(view):
            identity = sha256_hex(canonical_json_bytes(row))
            rep = row.get("representation")
            kind = (
                "brief"
                if rep == BRIEF_REPRESENTATION
                else "partial"
                if rep in {"overview_not_full_judgment", "delivered_section_not_full_judgment"}
                else "complete"
            )
            entry = retained.setdefault(
                identity,
                dict(
                    card_id=row["card_id"],
                    record_sha256=identity,
                    delivery_kind=kind,
                    record=row if kind != "complete" else project(row),
                    action_indices=[],
                    record_action_index=index,
                ),
            )
            entry["action_indices"].append(index)
            last.add(identity)
        if "critical_overview" in view:
            rows = (
                view["page"]["records"]
                if view.get("page", {}).get("unit") == ("candidate_overview")
                else view.get("candidate_overviews", [])
            )
            body = dict(overview=view["critical_overview"], candidate_overviews=rows)
            identity = sha256_hex(canonical_json_bytes(body))
            bodies.setdefault(identity, body)
            overview_deliveries.append(
                dict(
                    body_sha256=identity,
                    action_index=index,
                    publication_prefix=view["publication_prefix"],
                    scope=view.get("scope", []),
                    parameters=view.get("parameters", {}),
                )
            )
    evidence = []
    for key, entry in retained.items():
        projected = dict(entry)
        if key in last:
            projected.pop("record")
            projected["record_location"] = "previous_feedback record with this exact digest"
        evidence.append(projected)
    return dict(
        critical_evidence=evidence,
        overview_bodies=bodies,
        overview_deliveries=overview_deliveries,
        delivery_feedback=brief_delivery_feedback(history),
    )
