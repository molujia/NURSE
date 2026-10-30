"""D15 navigation attached to this delivery, never retrofitted to old receipts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedActionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_routes import investigation_routes, pending_pages, route_binding

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsSreRecoveryToolbox

RECOVERY_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/9"


def _history(records: tuple[OpsActionRecord, ...]) -> list[dict[str, Any]]:
    if len(records) > 8:
        raise ValueError("SRE recovery history exceeds action budget")
    return [
        dict(
            action_index=r.index,
            tool=r.action.tool,
            parameters=json.loads(r.action.parameters_json),
            status=r.status,
            view=expand(json.loads(r.model_view_json)) if r.model_view_json else {},
        )
        for r in records
        if r.action is not None and isinstance(r, OpsIndexedActionRecord)
    ]


def _identity(row: dict[str, Any]) -> bytes:
    return canonical_json_bytes(dict(tool=row["tool"], parameters=row["parameters"]))


def recovery_routes(
    toolbox: OpsSreRecoveryToolbox,
    records: tuple[OpsActionRecord, ...],
) -> dict[str, Any]:
    history = _history(records)
    latest = history[-1] if history and history[-1]["action_index"] == records[-1].index else None
    feedback: dict[str, Any] | None = None
    options: list[dict[str, Any]] = []
    if latest is not None and latest["status"] in {"rejected", "failed"}:
        diagnostic = latest["view"].get("parameter_feedback", {}).get("diagnostic")
        if diagnostic is not None:
            reason = diagnostic["reason_code"]
            current_pin = latest["view"].get("route_binding_sha256")
            prior = []
            for row in history[:-1]:
                if _identity(row) != _identity(latest):
                    continue
                previous_pin = row["view"].get("route_binding_sha256")
                prior.append(
                    dict(
                        action_index=row["action_index"],
                        status=row["status"],
                        source_binding="unknown"
                        if None in (current_pin, previous_pin)
                        else "same"
                        if previous_pin == current_pin
                        else "changed",
                    )
                )
            feedback = dict(
                action_index=latest["action_index"],
                diagnostic_location="previous_feedback.parameter_feedback.diagnostic",
                route_bindings=[],
                prior_same_requests=prior,
                unchanged_request_will_not_fix_parameters=reason
                in {"card_kind_mismatch", "invalid_section", "invalid_parameters"},
                authority="navigation_not_evidence",
            )
            # Only already-delivered metadata may explain a resource-specific error.
            contract = latest["view"].get("parameter_feedback", {}).get("object_contract")
            if contract and reason in {
                "card_kind_mismatch",
                "invalid_section",
                "request_binding_mismatch",
            }:
                key = contract["card_id"]
                # Reuse the original exact endpoints; never fill from later publication.
                if toolbox.object_contract(key) == contract:
                    options.append(
                        dict(
                            tool="read_node",
                            parameters=dict(card_id=key, section="payload", cursor=None),
                        )
                    )
                    targets = contract["targets"]
                    if len(targets) == 1:
                        options.append(
                            dict(
                                tool="inspect_candidate",
                                parameters=dict(
                                    candidate_service_key=targets[0],
                                    focus="mechanisms",
                                    cursor=None,
                                ),
                            )
                        )
            if reason not in {"reference_not_available", "unclassified_failure"}:
                pages = pending_pages(history, lambda t, p: route_binding(toolbox, t, p))
                options.extend(p for p in pages if p["tool"] == "compare_candidates")
                options = options[:3]
    routes = investigation_routes(toolbox, records, recovery_options=tuple(options))
    routes["schema_version"] = "nurse.v2.sre-investigation-routes/3"
    if feedback is not None:
        identities = {_identity(option) for option in options}
        # Bind only actually delivered entries, after original round-robin and omission.
        feedback["route_bindings"] = list(
            dict.fromkeys(
                e["binding_sha256"] for e in routes["entries"] if _identity(e) in identities
            )
        )
    routes["recovery_feedback"] = feedback
    return routes


def validate_recovery_delivery(context: dict[str, Any]) -> None:
    """Check wire association structure; it cannot grant tool or fact access."""
    routes = context["investigation_routes"]
    if routes.get("schema_version") != "nurse.v2.sre-investigation-routes/3" or (
        len(routes["entries"]) > 16
    ):
        raise ValueError("recovery route identity differs")
    feedback = routes.get("recovery_feedback")
    if feedback is None:
        return
    if not context["actions"] or feedback["action_index"] != context["actions"][-1]["action_index"]:
        raise ValueError("recovery action differs")
    if feedback["diagnostic_location"] != "previous_feedback.parameter_feedback.diagnostic":
        raise ValueError("recovery diagnostic location differs")
    pins = feedback["route_bindings"]
    if (
        len(pins) > 3
        or len(set(pins)) != len(pins)
        or not set(pins) <= {e["binding_sha256"] for e in routes["entries"]}
    ):
        raise ValueError("recovery route references are not delivered")
