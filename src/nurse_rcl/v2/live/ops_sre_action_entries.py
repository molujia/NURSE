"""Compile only supplied, issued navigation metadata; no reader or execution API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_directed_tools import OpsSreDirectedToolbox
from nurse_rcl.v2.live.ops_sre_references import FIELD_KINDS, VIEW_LIMITS

ENTRY_SCHEMA = "nurse.v2.sre-action-entries/1"
ENTRY_PROFILE = "v2-ops-sre/9"
ENTRY_TOOLS = "hermes-sre-tools/19"
ENTRY_SKILLS = {
    s.removesuffix("/14") + "/15": tools for s, tools in OpsSreDirectedToolbox.skill_tools.items()
}
NAVIGATION_KINDS = ["case", "candidate", "candidate_pair", "dependency", "catalog"]
DELIVERY_AUTHORITY = "delivery_metadata_not_diagnostic_evidence"


def _delivery_rows(
    index: dict[str, Any] | None, request_index: int, *, observable: bool = False
) -> list[dict[str, Any]]:
    """Check the owner-supplied metadata and its past-only action boundary."""
    if index is None:
        return []
    if (
        type(index) is not dict
        or set(index)
        != {"schema_version", "authority", "latest_action_index", "latest_record_refs", "records"}
        or index["schema_version"]
        not in (
            {"nurse.v2.sre-delivered-record-index/1", "nurse.v2.sre-delivered-record-index/2"}
            if observable
            else {"nurse.v2.sre-delivered-record-index/1"}
        )
        or index["authority"] != DELIVERY_AUTHORITY
        or type(index["records"]) is not list
        or type(index["latest_record_refs"]) is not list
    ):
        raise ValueError("entry delivery index differs")
    latest = index["latest_action_index"]
    if latest is not None and (type(latest) is not int or not 0 < latest < request_index):
        raise ValueError("entry delivery action boundary differs")
    digests: set[str] = set()
    for row in index["records"]:
        if type(row) is not dict:
            raise ValueError("entry delivery record differs")
        require_pin(row.get("record_sha256"))
        indices = row.get("action_indices")
        if (
            latest is None
            or type(indices) is not list
            or not indices
            or any(type(i) is not int or not 0 < i <= latest for i in indices)
            or indices != sorted(set(indices))
            or row.get("delivery_kind") not in {"complete", "partial", "brief"}
            or any(type(row.get(k)) is not str for k in ("card_id", "node", "source_role"))
            or type(row.get("targets")) is not list
            or any(type(s) is not str for s in row["targets"])
            or row["record_sha256"] in digests
        ):
            raise ValueError("entry delivery record boundary differs")
        digests.add(row["record_sha256"])
    refs = index["latest_record_refs"]
    if any(type(r) is not str or r not in digests for r in refs) or len(refs) != len(set(refs)):
        raise ValueError("entry delivery latest records differ")
    return index["records"]


def _supply_context(
    row: dict[str, Any], deliveries: list[dict[str, Any]], *, complete_read: bool, aggregate: bool
) -> dict[str, Any]:
    """Bind prior full-record delivery, not a claim about deeper source access."""
    prior = [d for d in deliveries if d["card_id"] == row["card_id"]]
    same = [
        d
        for d in prior
        if complete_read
        and d["delivery_kind"] == "complete"
        and all(d.get(k) == row.get(k) for k in ("record_sha256", "node", "source_role", "targets"))
    ]
    state = (
        "complete_same_record"
        if same
        else "unread"
        if not prior
        else "partial_or_brief_only"
        if all(d["delivery_kind"] != "complete" for d in prior)
        else "different_record_or_representation"
    )
    return dict(
        delivery=dict(
            authority=DELIVERY_AUTHORITY,
            state=state,
            prior_action_indices=sorted({i for d in same for i in d["action_indices"]}),
            expected_record_relation="no_new_record" if same else "not_proven",
        ),
        supply_contract=(
            "complete_selected_cards_aggregation" if aggregate else "selected_card_record_read"
        ),
        supply_limits=[
            "not_deeper_source_read",
            "cannot_fill_unsupplied_coverage_or_windows",
            "not_independent_evidence",
        ],
    )


def _issued(value: Any, kind: str, book: SreActionReferenceBook, index: int) -> bool:
    if type(value) is not str:
        return False
    handle = book._originals.get((kind, value))
    if handle is None:
        return False
    try:
        return book.resolve(handle, kind, visible_at=index) == value
    except ValueError:
        return False


def _issued_fields(value: Any, book: SreActionReferenceBook, index: int) -> bool:
    if type(value) is list:
        return all(_issued_fields(v, book, index) for v in value)
    if type(value) is not dict:
        return True
    for key, child in value.items():
        if key in FIELD_KINDS and child is not None:
            values = child if type(child) is list else [child]
            if not all(_issued(v, FIELD_KINDS[key], book, index) for v in values):
                return False
        elif not _issued_fields(child, book, index):
            return False
    return True


def entry_public_scope(view: dict[str, Any]) -> tuple[str, ...]:
    """Only admitted public task scope or the latest real receipt, never a diagnosis."""
    state = view.get("question_state", {})
    current = state.get("current_question_id")
    for question in state.get("questions", []):
        if question.get("question_id") == current and question.get("state") in {
            "open",
            "blocked",
            "deferred",
        }:
            return tuple(question["target_service_keys"])
    latest = view.get("previous_feedback")
    if type(latest) is dict and type(latest.get("scope")) is list:
        return tuple(s for s in latest["scope"] if type(s) is str)
    return ()


def compile_action_entries(
    capabilities: dict[str, Any],
    *,
    book: SreActionReferenceBook,
    request_index: int,
    case_id: str,
    public_scope: tuple[str, ...] = (),
    delivered_index: dict[str, Any] | None = None,
    _observable: bool = False,
    _single_owner: bool = False,
    _typed_purpose: bool = False,
    _atomic_selector: bool = False,
    _focused: bool = False,
) -> dict[str, Any]:
    """A transactional finite menu, not authority to skip the original tool guards.

    Inputs are original-name metadata already admitted to this request's outbound
    view. Ordinary names must be issued first. No hidden source is issued here.
    """
    from nurse_rcl.v2.live.ops_provider import _require_outbound
    from nurse_rcl.v2.live.ops_sre_observable_scope import (
        OBSERVABLE_ENTRY_SCHEMA,
        OBSERVABLE_PROFILE,
        OBSERVABLE_TOOLS,
        observable_contract,
    )

    if sum((_single_owner, _typed_purpose, _atomic_selector, _focused)) > 1:
        raise ValueError("mixed entry protocol identities")
    if _single_owner:
        from nurse_rcl.v2.live.ops_sre_single_owner_decision import (
            SINGLE_OWNER_ENTRY_SCHEMA,
            SINGLE_OWNER_PROFILE,
            SINGLE_OWNER_TOOLS,
        )

        OBSERVABLE_ENTRY_SCHEMA = SINGLE_OWNER_ENTRY_SCHEMA
        OBSERVABLE_PROFILE = SINGLE_OWNER_PROFILE
        OBSERVABLE_TOOLS = SINGLE_OWNER_TOOLS
        _observable = True

    if _atomic_selector:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
            ATOMIC_ENTRY_SCHEMA,
            ATOMIC_PROFILE,
            ATOMIC_TOOLS,
        )

        OBSERVABLE_ENTRY_SCHEMA, OBSERVABLE_PROFILE, OBSERVABLE_TOOLS = (
            ATOMIC_ENTRY_SCHEMA,
            ATOMIC_PROFILE,
            ATOMIC_TOOLS,
        )
        _observable = True

    if _focused:
        from nurse_rcl.v2.live.ops_sre_focused_decision import (
            FOCUSED_ENTRY_SCHEMA,
            FOCUSED_PROFILE,
            FOCUSED_TOOLS,
        )

        OBSERVABLE_ENTRY_SCHEMA, OBSERVABLE_PROFILE, OBSERVABLE_TOOLS = (
            FOCUSED_ENTRY_SCHEMA,
            FOCUSED_PROFILE,
            FOCUSED_TOOLS,
        )
        _observable = True

    if _typed_purpose:
        from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
            TYPED_ENTRY_SCHEMA,
            TYPED_PROFILE,
            TYPED_TOOLS,
        )

        if _single_owner:
            raise ValueError("mixed entry protocol identities")
        OBSERVABLE_ENTRY_SCHEMA, OBSERVABLE_PROFILE, OBSERVABLE_TOOLS = (
            TYPED_ENTRY_SCHEMA,
            TYPED_PROFILE,
            TYPED_TOOLS,
        )
        _observable = True

    if type(book) is not SreActionReferenceBook:
        raise ValueError("action reference identity differs")
    encode_json(capabilities, VIEW_LIMITS)
    _require_outbound(capabilities, case=case_id, snapshot=book.snapshot)
    if delivered_index is not None:
        encode_json(delivered_index, VIEW_LIMITS)
        _require_outbound(delivered_index, case=case_id, snapshot=book.snapshot)
    deliveries = _delivery_rows(delivered_index, request_index, observable=_observable)
    prefix = book.source_prefix(visible_at=request_index)
    working = type(book).from_state(book.to_state(), snapshot=book.snapshot, run_id=book.run_id)
    priority = {s for s in public_scope if _issued(s, "service", working, request_index)}
    queues: dict[str, list[dict[str, Any]]] = {
        k: []
        for k in (("focused",) if _focused else ())
        + ("symptoms", "current", "scoring", "dependency_timing", "catalog")
    }
    seen: set[str] = set()
    # Normalize only registered supplier identities. The observable profile
    # publishes /16 routes; the legacy compiler must retain its /14-only domain.
    route_skills = {s.removesuffix("/15") + "/14": s for s in ENTRY_SKILLS}
    if _observable:
        route_skills.update({s.removesuffix("/15") + "/16": s for s in ENTRY_SKILLS})
    if _typed_purpose:
        route_skills.update({s.removesuffix("/15") + "/18": s for s in ENTRY_SKILLS})
    if _atomic_selector:
        route_skills.update({s.removesuffix("/15") + "/19": s for s in ENTRY_SKILLS})
    if _focused:
        route_skills.update({s.removesuffix("/15") + "/20": s for s in ENTRY_SKILLS})

    def add(
        category: str,
        skill: str,
        tool: str,
        params: Any,
        binding: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        if (
            category not in queues
            or skill not in ENTRY_SKILLS
            or tool not in ENTRY_SKILLS[skill]
            or type(params) is not dict
            or set(params) != set(OpsSreDirectedToolbox.tool_parameters.get(tool, ()))
            or not _issued_fields(
                dict(parameters=params, binding=binding, context=context), working, request_index
            )
        ):
            return
        action = dict(
            skill_id=skill.removesuffix("/15") + "/20"
            if _focused
            else skill.removesuffix("/15") + "/19"
            if _atomic_selector
            else skill.removesuffix("/15") + "/18"
            if _typed_purpose
            else skill.removesuffix("/15") + "/16"
            if _observable
            else skill,
            tool=tool,
            parameters=deepcopy(params),
        )
        identity = dict(
            profile_id=OBSERVABLE_PROFILE if _observable else ENTRY_PROFILE,
            tool_profile_id=OBSERVABLE_TOOLS if _observable else ENTRY_TOOLS,
            run_id=book.run_id,
            snapshot=book.snapshot,
            source_prefix=prefix,
            action=action,
            source_binding=deepcopy(binding),
            source_context=deepcopy(context),
        )
        if _observable:
            identity["observable_contract"] = observable_contract(context, binding, prefix)
        entry_id = sha256_hex(canonical_json_bytes(identity))
        if entry_id not in seen:
            seen.add(entry_id)
            queues[category].append(dict(entry_id=entry_id, **identity))

    if _focused:
        # A completed catalog page's exact actions enter this next frozen table
        # before generic queues. Directory metadata itself is never a delivery.
        for row in capabilities.get("evidence_catalog_targets", []):
            for item in row.get("actions", []):
                tool = item.get("tool")
                skill = next((s for s, tools in ENTRY_SKILLS.items() if tool in tools), None)
                if skill is None:
                    continue
                context = {
                    k: deepcopy(row[k])
                    for k in (
                        "node",
                        "targets",
                        "source_role",
                        "signal_family",
                        "metric_name",
                        "request_key",
                        "source_refs",
                        "representation",
                        "record_kind",
                        "record_version",
                    )
                    if k in row
                }
                binding = dict(
                    record_sha256=row["record_sha256"],
                    card_id=row["card_id"],
                    source_refs=row.get("source_refs", []),
                )
                if "binding_sha256" in item:
                    require_pin(item["binding_sha256"])
                    binding = dict(route_binding_sha256=item["binding_sha256"])
                    context = deepcopy(item["source_context"])
                add("focused", skill, tool, item.get("parameters"), binding, context)

    for row in capabilities.get("directed_sources", {}).get("entries", []):
        params = row.get("entry", {}).get("parameters", {})
        if (
            row.get("node") != "symptoms"
            or row.get("entry", {}).get("tool") != "inspect_sources"
            or type(row.get("targets")) is not list
            or len(row["targets"]) != 1
            or params != dict(card_ids=[row.get("card_id")], candidate_service_keys=row["targets"])
            or not _issued_fields(row, working, request_index)
        ):
            continue
        try:
            require_pin(row.get("record_sha256"))
        except (ValueError, TypeError):
            continue
        context = {
            k: deepcopy(row[k])
            for k in (
                "node",
                "targets",
                "source_role",
                "signal_family",
                "metric_name",
                "request_key",
                "source_refs",
                "representation",
            )
            if k in row
        }
        binding = dict(
            record_sha256=row["record_sha256"],
            card_id=row["card_id"],
            source_refs=deepcopy(row.get("source_refs", [])),
        )
        if _observable:
            context.update(
                record_kind=row.get("record_kind"), record_version=row.get("record_version")
            )
        add(
            "symptoms",
            "sre-symptoms/15",
            "inspect_sources",
            params,
            binding,
            dict(context, **_supply_context(row, deliveries, complete_read=True, aggregate=True)),
        )
        complete = row.get("complete_entry", {})
        if (
            complete.get("tool") == "read_node"
            and complete.get("parameters", {}).get("card_id") == row["card_id"]
        ):
            add(
                "symptoms",
                "sre-symptoms/15",
                "read_node",
                complete["parameters"],
                binding,
                dict(
                    context,
                    **_supply_context(
                        row,
                        deliveries,
                        complete_read=(
                            complete["parameters"].get("section") == "payload"
                            and complete["parameters"].get("cursor") is None
                        ),
                        aggregate=False,
                    ),
                ),
            )

    for row in capabilities.get("investigation_routes", {}).get("entries", []):
        # Untyped direct routes cannot turn scoring/other cards into symptoms.
        if row.get("tool") == "inspect_sources":
            continue
        pin = row.get("binding_sha256")
        try:
            require_pin(pin)
        except (ValueError, TypeError):
            continue
        supplied_skill = row.get("skill_id")
        if type(supplied_skill) is not str or supplied_skill not in route_skills:
            continue
        skill = route_skills[supplied_skill]
        context = {
            k: deepcopy(row[k])
            for k in (
                "category",
                "scope",
                "card_ids",
                "source_refs",
                "availability",
                "execution_history",
                "new_publication",
                "core_reusable",
                "continuation_from_action_index",
            )
            if k in row
        }
        if row.get("availability") != "available":
            continue
        add(
            row.get("category"),
            skill,
            row.get("tool"),
            row.get("parameters"),
            dict(route_binding_sha256=pin),
            context,
        )

    for name, queue in queues.items():
        if name == "focused":
            continue
        queue.sort(
            key=lambda e: (
                not bool(
                    priority
                    & set(e["source_context"].get("targets", e["source_context"].get("scope", [])))
                ),
                e["entry_id"],
            )
        )
    selected = [
        *queues.get("focused", []),
        *[
            queue[i]
            for i in range(max((len(q) for k, q in queues.items() if k != "focused"), default=0))
            for name, queue in queues.items()
            if name != "focused"
            if i < len(queue)
        ],
    ][:16]
    entries = [
        dict(entry_ref=working.issue("action", e["entry_id"], visible_at=request_index), **e)
        for e in selected
    ]
    total = sum(map(len, queues.values()))
    table = dict(
        schema_version=OBSERVABLE_ENTRY_SCHEMA if _observable else ENTRY_SCHEMA,
        source_prefix=prefix,
        authority="navigation_not_delivered_evidence",
        shown=len(entries),
        total=total,
        omitted=total - len(entries),
        navigation_kinds=list(NAVIGATION_KINDS),
        entries=entries,
    )
    encode_json(table, VIEW_LIMITS)
    _require_outbound(table, case=case_id, snapshot=book.snapshot)
    working.to_state()
    book._rows, book._originals = working._rows, working._originals
    book._handles, book._counts = working._handles, working._counts
    return table
