"""Bounded navigational options over admitted metadata; never executes a tool."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedActionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_focused_tools import OpsSreFocusedToolbox
    from nurse_rcl.v2.live.ops_sre_tools import OpsSreToolbox

CATEGORIES = ("current", "scoring", "dependency_timing", "catalog")


def pending_pages(
    rows: list[dict[str, Any]],
    binding: Callable[[str, dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Project admitted historical metadata only; never fetch or execute a page.

    Keeping this small projection separate also permits read-only diagnosis of
    historical receipts without loading a full preparation or promoting its run.
    """
    if len(rows) > 8:
        raise ValueError("SRE route history exceeds action budget")
    completed = {
        r["view"].get("route_binding_sha256")
        for r in rows
        if r["status"] == "completed" and r["view"].get("route_binding_sha256")
    }
    following = []
    seen: set[str] = set()
    for row in reversed(rows):
        params, view = row["parameters"], row["view"]
        if row["status"] != "completed" or "cursor" not in params:
            continue
        cursor = view.get("page", {}).get("next_cursor", view.get("next_cursor"))
        if cursor is None:
            continue
        parameters = dict(params, cursor=cursor)
        pin = binding(row["tool"], parameters)
        if pin not in completed and pin not in seen:
            seen.add(pin)
            following.append(
                dict(
                    tool=row["tool"],
                    parameters=parameters,
                    continuation_from_action_index=row["action_index"],
                )
            )
    return following


def route_binding(toolbox: OpsSreToolbox, tool: str, params: dict[str, Any]) -> str:
    """Capture the exact semantic source before dispatch, not merely the tool name.

    This is a receipt hint, not an alternate admission check. Execution continues
    through the existing registry/ledger. Source hashes include windows and versions.
    """
    if tool not in toolbox.tool_parameters or set(params) != set(toolbox.tool_parameters[tool]):
        toolbox._reject_tool("invalid_parameters", "route parameters differ")
    if tool == "inspect_sources" and toolbox.profile_id in {
        "hermes-sre-tools/18",
        "hermes-sre-tools/19",
        "hermes-sre-tools/20",
        "hermes-sre-tools/21",
        "hermes-sre-tools/22",
        "hermes-sre-tools/23",
        "hermes-sre-tools/25",
    }:
        from nurse_rcl.v2.live.ops_sre_directed_sources import directed_source_binding

        keys = params["card_ids"]
        scope = params["candidate_service_keys"]
        if any(
            type(values) is not list
            or not 1 <= len(values) <= 2
            or any(type(value) is not str for value in values)
            or len(values) != len(set(values))
            for values in (keys, scope)
        ):
            toolbox._reject_tool("invalid_parameters", "directed route bounds differ")
        toolbox._refresh_semantic()
        if any(key not in toolbox._semantic for key in keys) or any(
            candidate not in toolbox._by_candidate for candidate in scope
        ):
            toolbox._reject_tool("reference_not_available", "directed route source differs")
        rows = [toolbox._semantic[key] for key in keys]
        if any(
            row["node"] != "symptoms"
            or len(row["targets"]) != 1
            or not set(row["targets"]) <= set(scope)
            for row in rows
        ):
            toolbox._reject_tool("invalid_parameters", "directed route scope differs")
        if toolbox.profile_id == "hermes-sre-tools/25":
            from nurse_rcl.v2.live.ops_sre_focused_sources import focused_source_binding

            return focused_source_binding(toolbox.case_snapshot_id, params, rows)
        return directed_source_binding(toolbox.case_snapshot_id, params, rows)
    if (
        tool == "browse_index"
        and params.get("kind") == "evidence"
        and toolbox.profile_id == "hermes-sre-tools/25"
    ):
        focused_toolbox = cast("OpsSreFocusedToolbox", toolbox)
        return focused_toolbox._catalog().binding(
            filters=params["filters"], cursor=params["cursor"]
        )
    source: Any
    if tool in toolbox.inspection_parameters:
        if params["cursor"] is None:
            scope = toolbox._scope(tool, params)
            source = toolbox._pin(
                toolbox._current_prefix(tool, scope, focus=toolbox._focus(tool, params))
            )
        else:
            try:
                source = toolbox._inspection_cursors[params["cursor"]]["prefix"]
            except (KeyError, TypeError):
                toolbox._reject_tool(
                    "request_binding_mismatch",
                    "SRE route uses an unregistered parameter or cursor",
                )
    elif tool == "measure_signal":
        request = toolbox._requests[params["request_key"]]
        core = toolbox._core_steps.get(request.request_key)
        source = dict(
            source_sha256=request.source_sha256,
            measurement_sha256=None if core is None else core.measurement.measurement_sha256,
            result_sha256=None if core is None else core.feedback.result_sha256,
        )
    elif tool in {"read_node", "check_mechanism"}:
        key = params["card_id"]
        if type(key) is not str or key not in toolbox._cards:
            toolbox._reject_tool("reference_not_available", "route card not admitted")
        source = dict(card_id=key, published_by_core=key in toolbox._core_cards)
        if tool == "check_mechanism":
            source["feedback"] = sorted(f.result_sha256 for f in toolbox._feedback.values())
    elif tool == "compare_timing":
        source = dict(
            graph=toolbox._data.graph.graph_pair.model_dump(mode="json"),
            clock=toolbox._data.clock.model_dump(mode="json"),
        )
    else:
        index = toolbox._index
        kind = params["kind"]
        index._filters(kind, params["filters"])
        cursor = params["cursor"]
        source = (
            index._cursors[cursor]["prefix"]
            if cursor
            else index._digest(dict(kind=kind, ids=sorted(index._rows[kind])))
        )
    return sha256_hex(
        canonical_json_bytes(
            dict(
                tool=tool,
                parameters=params,
                source=source,
                snapshot=toolbox.case_snapshot_id,
                policy="sre-route-binding/1",
            )
        )
    )


def investigation_routes(
    toolbox: OpsSreToolbox,
    records: tuple[OpsActionRecord, ...],
    *,
    recovery_options: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_encoding import expand

    if len(records) > 8:
        raise ValueError("SRE route history exceeds action budget")
    if len(recovery_options) > 3:
        raise ValueError("SRE recovery options exceed bound")
    toolbox._refresh_semantic()
    queues: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    seen: dict[str, set[bytes]] = {c: set() for c in CATEGORIES}

    def add(
        category: str,
        tool: str,
        parameters: dict[str, Any],
        *,
        scope: list[str] | None = None,
        card_ids: list[str] | None = None,
        source_refs: list[str] | None = None,
        ranks: list[int] | None = None,
        sort_key: tuple[Any, ...] = (),
        continuation_from: int | None = None,
    ) -> None:
        parameters = toolbox._route_parameters(tool, parameters)
        if set(parameters) != set(toolbox.tool_parameters[tool]):
            raise ValueError("route parameter fields differ")
        if tool == "measure_signal":
            key = parameters["request_key"]
            request = toolbox._requests.get(key)
            if (
                request is None
                or not request.candidate_service_key
                or (key in toolbox._feedback and key not in toolbox._core_steps)
            ):
                return  # Same admission domain as _measure; read obtained cards instead.
        identity = canonical_json_bytes(dict(tool=tool, parameters=parameters))
        if identity in seen[category]:
            return
        seen[category].add(identity)
        item: dict[str, Any] = dict(
            tool=tool,
            parameters=parameters,
            scope=scope or [],
            skill_id=next(s for s, tools in toolbox.skill_tools.items() if tool in tools),
            card_ids=card_ids or [],
            source_refs=source_refs or [],
            availability="available",
            category=category,
            sort_key=(*sort_key, identity),
        )
        if ranks is not None:
            item["existing_ranks"] = ranks
        if continuation_from is not None:
            item["continuation_from_action_index"] = continuation_from
        queues[category].append(item)

    history = []
    for record in records:
        if record.action is None:
            continue
        params = json.loads(record.action.parameters_json)
        view = (
            {}
            if not isinstance(record, OpsIndexedActionRecord) or (record.model_view_json is None)
            else expand(json.loads(record.model_view_json))
        )
        history.append((record, params, view))
    page_history = [
        dict(
            action_index=r.index,
            tool=r.action.tool,
            status=r.status,
            binding_sha256=v.get("route_binding_sha256"),
        )
        for r, p, v in history
        if r.action is not None and "cursor" in p
    ]
    # Actual returned continuations survive intervening tools. Binding includes
    # the issued cursor and frozen prefix; a later publication is not this read.
    history_rows = [
        dict(action_index=r.index, tool=r.action.tool, parameters=p, status=r.status, view=v)
        for r, p, v in history
        if r.action is not None
    ]
    for index, option in enumerate(recovery_options):
        add(
            "current",
            option["tool"],
            option["parameters"],
            sort_key=(-1, index),
            continuation_from=option.get("continuation_from_action_index"),
        )
    for page in pending_pages(
        history_rows, lambda tool, params: route_binding(toolbox, tool, params)
    ):
        index = page["continuation_from_action_index"]
        add(
            "current",
            page["tool"],
            page["parameters"],
            sort_key=(0, -index),
            continuation_from=index,
        )
    if history:
        record, params, last = history[-1]
        assert record.action is not None
        tool = record.action.tool
        if record.status == "completed":
            add("current", tool, params, sort_key=(1,))
            for item in last.get("next_checks", []):
                add("current", item["tool"], item["parameters"], sort_key=(2,))
            for row in last.get("page", {}).get("records", []):
                expansion = row.get("expansion")
                if expansion:
                    add("current", expansion["tool"], expansion["parameters"], sort_key=(2,))
    else:
        add("current", "inspect_case", dict(cursor=None), sort_key=(0,))

    scores = []
    for key in sorted(toolbox._core_cards):
        card = toolbox._cards[key]
        if card.node != "scoring":
            continue
        body = json.loads(card.payload_json)
        candidate = json.loads(card.parameters_json).get("candidate_service_key")
        rank = body.get("rank")
        if type(rank) is not int or candidate not in toolbox._by_candidate:
            continue
        scores.append((rank, candidate, key, card.source_refs))
    scores.sort()
    by_source: dict[tuple[str, ...], list[tuple[Any, ...]]] = {}
    for rank, candidate, key, refs in scores:
        by_source.setdefault(refs, []).append((rank, candidate, key))
        add(
            "scoring",
            "inspect_candidate",
            dict(candidate_service_key=candidate, cursor=None),
            scope=[candidate],
            card_ids=[key],
            source_refs=list(refs),
            ranks=[rank],
            sort_key=(rank, candidate, 1),
        )
    complete = [
        refs
        for refs, rows in by_source.items()
        if len(rows) == len(toolbox._by_candidate)
        and {r[1] for r in rows} == set(toolbox._by_candidate)
        and [r[0] for r in rows] == list(range(1, len(rows) + 1))
    ]
    for refs, rows in sorted(by_source.items()):
        leading = {r[0]: r for r in rows if r[0] in (1, 2)}
        if set(leading) == {1, 2}:
            left, right = leading[1], leading[2]
            add(
                "scoring",
                "compare_candidates",
                dict(
                    left_candidate_service_key=left[1],
                    right_candidate_service_key=right[1],
                    cursor=None,
                ),
                scope=[left[1], right[1]],
                card_ids=[left[2], right[2]],
                source_refs=list(refs),
                ranks=[1, 2],
                sort_key=(1, left[1], 0),
            )

    onsets: dict[str, list[str]] = {}
    for row in toolbox._onset_rows():
        onsets.setdefault(row["candidate_service_key"], []).append(row["onset_id"])
    for key, row in sorted(toolbox._semantic.items()):
        edge = row["payload"].get("edge")
        if not edge or edge.get("observation_state") != "observed":
            continue
        caller, callee = edge["caller_service_key"], edge["callee_service_key"]
        if caller not in toolbox._by_candidate or callee not in toolbox._by_candidate:
            continue
        add(
            "dependency_timing",
            "inspect_dependency",
            dict(caller_service_key=caller, callee_service_key=callee, cursor=None),
            scope=[caller, callee],
            card_ids=[key],
        )
        for source in sorted(onsets.get(caller, [])):
            for target in sorted(onsets.get(callee, [])):
                add(
                    "dependency_timing",
                    "compare_timing",
                    dict(source_onset_id=source, target_onset_id=target),
                    scope=[caller, callee],
                    card_ids=[key],
                )
    for candidate in sorted(toolbox._by_candidate):
        add(
            "catalog",
            "inspect_candidate",
            dict(candidate_service_key=candidate, cursor=None),
            scope=[candidate],
        )
    for key, request in sorted(toolbox._requests.items()):
        if request.candidate_service_key and (
            key not in toolbox._feedback or key in toolbox._core_steps
        ):
            add(
                "catalog",
                "measure_signal",
                dict(request_key=key),
                scope=[request.candidate_service_key],
                source_refs=[request.source_sha256],
            )
    for queue in queues.values():
        queue.sort(key=lambda r: r["sort_key"])
    selected: list[dict[str, Any]] = []
    for index in range(16):
        for category in CATEGORIES:
            if index < len(queues[category]) and len(selected) < 16:
                selected.append(queues[category][index])
    for item in selected:
        item.pop("sort_key")
        tool, params = item["tool"], item["parameters"]
        pin = route_binding(toolbox, tool, params)
        prior = [
            (r, v)
            for r, p, v in history
            if r.action is not None and r.action.tool == tool and p == params
        ]
        item.update(
            binding_sha256=pin,
            execution_history=[
                dict(action_index=r.index, status=r.status)
                for r, v in prior
                if v.get("route_binding_sha256") == pin
            ],
            new_publication=any(v.get("route_binding_sha256") not in (None, pin) for _, v in prior),
            core_reusable=tool == "measure_signal" and params["request_key"] in toolbox._core_steps,
        )
    selected_ids = {id(item) for item in selected}
    omitted_continuations = [
        item["continuation_from_action_index"]
        for item in queues["current"]
        if "continuation_from_action_index" in item and id(item) not in selected_ids
    ]
    return dict(
        schema_version="nurse.v2.sre-investigation-routes/2",
        selection="category_round_robin/1",
        entries=selected,
        page_history=page_history,
        omitted_continuation_actions=omitted_continuations,
        categories={
            c: dict(
                total=len(queues[c]),
                shown=sum(e["category"] == c for e in selected),
                omitted=len(queues[c]) - sum(e["category"] == c for e in selected),
            )
            for c in CATEGORIES
        },
        scoring_publication=dict(
            status="complete" if complete else "partial" if scores else "not_published",
            visible_count=len(scores),
            candidate_count=len(toolbox._by_candidate),
            source_count=len(by_source),
        ),
        navigation=[
            dict(
                tool="browse_index",
                skill_id=next(s for s in toolbox.skill_tools if s.startswith("sre-boundary/")),
                parameters=dict(kind=k, filters={}, cursor=None),
            )
            for k in ("candidates", "signals", "nodes", "onsets")
        ],
        legal_options_do_not_require_continuation=True,
        meaning="legal options only; omitted is not absent; "
        "no diagnostic value or causality implied",
    )
