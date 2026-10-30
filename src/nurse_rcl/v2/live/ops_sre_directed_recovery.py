"""Directed plural-card diagnostics and same-scope optional symptom navigation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail, failure_detail
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_sre_encoding import expand, serialize_sre_view
from nurse_rcl.v2.live.ops_sre_recovery_context import recovery_routes
from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsRecoveryActionRecord
from nurse_rcl.v2.live.ops_sre_routes import investigation_routes

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_directed_tools import OpsSreDirectedToolbox


def _visible_contracts(
    toolbox: OpsSreDirectedToolbox, record: OpsActionRecord,
) -> tuple[list[dict[str, Any]], list[str]] | None:
    if (not isinstance(record, OpsRecoveryActionRecord) or record.status != "rejected"
        or record.failure_detail is not None or record.action is None
        or record.action.tool != "inspect_sources" or record.rejection is None
        or record.rejection.reason_code != "card_kind_mismatch"):
        return None
    params = json.loads(record.action.parameters_json)
    if type(params) is not dict or set(params) != {"card_ids", "candidate_service_keys"}:
        return None
    keys, scope = params["card_ids"], params["candidate_service_keys"]
    if any(type(values) is not list or not 1 <= len(values) <= 2
           or any(type(value) is not str for value in values)
           or len(values) != len(set(values)) for values in (keys, scope)):
        return None
    if any(not toolbox._visible("service", c) or c not in toolbox._by_candidate for c in scope):
        return None
    contracts = []
    for key in keys:
        contract = toolbox.object_contract(key)
        if (contract is None or len(contract["targets"]) != 1
            or not set(contract["targets"]) <= set(scope)):
            return None  # Never fill unseen endpoints, other scopes or future cards.
        contracts.append(contract)
    return contracts, scope


def directed_rejection_metadata(
    toolbox: OpsSreDirectedToolbox, record: OpsActionRecord,
) -> OpsActionRecord:
    selected = _visible_contracts(toolbox, record)
    if (not isinstance(record, OpsRecoveryActionRecord)
        or selected is None or record.model_view_json is None):
        return record
    view = expand(json.loads(record.model_view_json))
    if view.get("status") == "resource_limit":
        return record
    feedback = view.setdefault("parameter_feedback", {})
    feedback["object_contracts"] = selected[0]
    feedback["source_selection_guidance"] = (
        "These card types are navigation, not observations. Discover issued symptom sources "
        "for the original candidate scope with inspect_candidate or compare_candidates "
        "focus=symptoms; scoring cards cannot be read by inspect_sources. Recovery routes "
        "are optional and do not execute or answer the investigation question."
    )
    raw = serialize_sre_view(view, diagnose=True)
    projected = json.loads(raw)
    updates: dict[str, Any] = {"model_view_json": raw}
    if projected.get("status") == "resource_limit":
        updates["failure_detail"] = (
            OpsFailureDetail.model_validate(projected["failure_detail"])
            if "failure_detail" in projected
            else failure_detail("investigation_tool", ValueError("view_resource_limit"))
        )
    return record.model_copy(update=updates)


def directed_recovery_routes(
    toolbox: OpsSreDirectedToolbox, records: tuple[OpsActionRecord, ...],
) -> dict[str, Any]:
    routes = recovery_routes(toolbox, records)
    if not records or not isinstance(records[-1], OpsRecoveryActionRecord):
        return routes
    latest = records[-1]
    if latest.model_view_json is None:
        return routes
    selected = _visible_contracts(toolbox, latest)
    if selected is None:
        return routes
    assert latest.model_view_json is not None
    view = expand(json.loads(latest.model_view_json))
    contracts, scope = selected
    if (view.get("status") == "resource_limit" or view.get("parameter_feedback", {}).get(
        "object_contracts") != contracts):
        return routes  # Do not retrofit metadata into any historical receipt.
    options = [dict(tool="inspect_candidate", parameters=dict(
        candidate_service_key=c, focus="symptoms", cursor=None,
    )) for c in scope]
    if len(scope) == 2:
        options.append(dict(tool="compare_candidates", parameters=dict(
            left_candidate_service_key=scope[0], right_candidate_service_key=scope[1],
            focus="symptoms", cursor=None,
        )))
    feedback = routes["recovery_feedback"]
    assert feedback is not None
    routes = investigation_routes(toolbox, records, recovery_options=tuple(options))
    routes["schema_version"] = "nurse.v2.sre-investigation-routes/3"
    identities = {canonical_json_bytes(o) for o in options}
    feedback["route_bindings"] = list(dict.fromkeys(
        entry["binding_sha256"] for entry in routes["entries"]
        if canonical_json_bytes(dict(tool=entry["tool"], parameters=entry["parameters"]))
        in identities
    ))
    routes["recovery_feedback"] = feedback
    return routes
