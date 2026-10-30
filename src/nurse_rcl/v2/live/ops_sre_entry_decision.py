"""Selection shape errors and action-semantic rejections are separate boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_sre_action_decision import _parse_action_decision, admit_note
from nurse_rcl.v2.live.ops_sre_action_entries import (
    ENTRY_PROFILE,
    ENTRY_SCHEMA,
    ENTRY_SKILLS,
    ENTRY_TOOLS,
    NAVIGATION_KINDS,
)
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_directed_tools import OpsSreDirectedToolbox
from nurse_rcl.v2.live.ops_sre_references import VIEW_LIMITS

ENTRY_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/20"
NAV_FIELDS = {
    "case": {"kind", "cursor"},
    "candidate": {"kind", "candidate_service_key", "focus", "cursor"},
    "candidate_pair": {
        "kind",
        "left_candidate_service_key",
        "right_candidate_service_key",
        "focus",
        "cursor",
    },
    "dependency": {"kind", "caller_service_key", "callee_service_key", "cursor"},
    "catalog": {"kind", "catalog_kind", "filters", "cursor"},
}
NAV_TOOLS = dict(
    case="inspect_case",
    candidate="inspect_candidate",
    candidate_pair="compare_candidates",
    dependency="inspect_dependency",
    catalog="browse_index",
)


def selection_shape(value: Any) -> None:
    error = "sre_entry_selection_shape"
    if (
        type(value) is not dict
        or set(value)
        not in (
            {"entry_ref", "question"},
            {"navigation", "question"},
        )
        or type(value["question"]) is not str
        or not value["question"].strip()
    ):
        raise ExchangeBoundaryError(error)
    if "entry_ref" in value:
        if type(value["entry_ref"]) is not str:
            raise ExchangeBoundaryError(error)
        return
    nav = value["navigation"]
    if (
        type(nav) is not dict
        or type(nav.get("kind")) is not str
        or nav["kind"] not in NAV_FIELDS
        or set(nav) != NAV_FIELDS[nav["kind"]]
    ):
        raise ExchangeBoundaryError(error)
    for key, child in nav.items():
        if key == "cursor":
            if child is not None and type(child) is not str:
                raise ExchangeBoundaryError(error)
        elif key == "filters":
            if type(child) is not dict or any(
                type(k) is not str or type(v) is not str for k, v in child.items()
            ):
                raise ExchangeBoundaryError(error)
        elif type(child) is not str:
            raise ExchangeBoundaryError(error)


def parse_entry_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if context is None or context.get("schema_version") != ENTRY_INPUT_SCHEMA:
        raise ExchangeBoundaryError("sre_entry_context_required")
    value = _parse_action_decision(raw, context=context, input_schema=ENTRY_INPUT_SCHEMA)
    if len(value["actions"]) > 1:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    for action in value["actions"]:
        selection_shape(action)
    return value


def parse_observable_decision(
    raw: bytes, *, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_observable_scope import OBSERVABLE_INPUT_SCHEMA

    if context is None or context.get("schema_version") != OBSERVABLE_INPUT_SCHEMA:
        raise ExchangeBoundaryError("observable_context_required")
    value = _parse_action_decision(raw, context=context, input_schema=OBSERVABLE_INPUT_SCHEMA)
    if len(value["actions"]) > 1:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    for action in value["actions"]:
        selection_shape(action)
    return value


def admit_entry_note(value: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    if context.get("schema_version") != ENTRY_INPUT_SCHEMA:
        raise ExchangeBoundaryError("sre_entry_context_required")
    return admit_note(value, context=context)


def validate_action_entries(
    table: dict[str, Any],
    *,
    book: SreActionReferenceBook,
    request_index: int,
    _observable: bool = False,
    _single_owner: bool = False,
    _typed_purpose: bool = False,
    _atomic_selector: bool = False,
    _focused: bool = False,
) -> None:
    """Validate the frozen original-name table, never remap via a future toolbox."""
    encode_json(table, VIEW_LIMITS)
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
    skills = (
        {k.removesuffix("/15") + "/20": v for k, v in ENTRY_SKILLS.items()}
        if _focused
        else {k.removesuffix("/15") + "/19": v for k, v in ENTRY_SKILLS.items()}
        if _atomic_selector
        else {k.removesuffix("/15") + "/18": v for k, v in ENTRY_SKILLS.items()}
        if _typed_purpose
        else {k.removesuffix("/15") + "/16": v for k, v in ENTRY_SKILLS.items()}
        if _observable
        else ENTRY_SKILLS
    )
    if (
        type(book) is not SreActionReferenceBook
        or type(table) is not dict
        or set(table)
        != {
            "schema_version",
            "source_prefix",
            "authority",
            "shown",
            "total",
            "omitted",
            "navigation_kinds",
            "entries",
        }
        or table["schema_version"] != (OBSERVABLE_ENTRY_SCHEMA if _observable else ENTRY_SCHEMA)
        or table["source_prefix"] != book.source_prefix(visible_at=request_index)
        or table["authority"] != "navigation_not_delivered_evidence"
        or table["navigation_kinds"] != NAVIGATION_KINDS
        or type(table["entries"]) is not list
        or not 0 <= len(table["entries"]) <= 16
        or any(type(table[k]) is not int for k in ("shown", "total", "omitted"))
        or table["shown"] != len(table["entries"])
        or table["omitted"] < 0
        or table["total"] != table["shown"] + table["omitted"]
    ):
        raise ValueError("action entry table binding differs")
    seen = set()
    for entry in table["entries"]:
        if type(entry) is not dict or set(entry) != (
            {
                "entry_ref",
                "entry_id",
                "profile_id",
                "tool_profile_id",
                "run_id",
                "snapshot",
                "source_prefix",
                "action",
                "source_binding",
                "source_context",
            }
            | ({"observable_contract"} if _observable else set())
        ):
            raise ValueError("action entry binding differs")
        identity = {k: v for k, v in entry.items() if k not in {"entry_ref", "entry_id"}}
        action = entry["action"]
        if (
            entry["profile_id"] != (OBSERVABLE_PROFILE if _observable else ENTRY_PROFILE)
            or entry["tool_profile_id"] != (OBSERVABLE_TOOLS if _observable else ENTRY_TOOLS)
            or entry["run_id"] != book.run_id
            or entry["snapshot"] != book.snapshot
            or entry["source_prefix"] != table["source_prefix"]
            or entry["entry_id"] != sha256_hex(canonical_json_bytes(identity))
            or type(action) is not dict
            or set(action) != {"skill_id", "tool", "parameters"}
            or action["skill_id"] not in skills
            or action["tool"] not in skills[action["skill_id"]]
            or type(action["parameters"]) is not dict
            or set(action["parameters"])
            != set(OpsSreDirectedToolbox.tool_parameters[action["tool"]])
            or type(entry["entry_ref"]) is not str
            or entry["entry_ref"] in seen
        ):
            raise ValueError("action entry binding differs")
        if _observable and entry["observable_contract"] != observable_contract(
            entry["source_context"],
            entry["source_binding"],
            table["source_prefix"],
        ):
            raise ValueError("action entry observable binding differs")
        try:
            if (
                book.resolve(entry["entry_ref"], "action", visible_at=request_index)
                != entry["entry_id"]
            ):
                raise ValueError("different action")
        except ValueError as exc:
            raise ValueError("action entry reference binding differs") from exc
        seen.add(entry["entry_ref"])


def resolve_entry_selection(
    value: Any,
    *,
    table: dict[str, Any],
    book: SreActionReferenceBook,
    request_index: int,
    _observable: bool = False,
    _single_owner: bool = False,
    _typed_purpose: bool = False,
    _atomic_selector: bool = False,
    _focused: bool = False,
) -> dict[str, Any]:
    """Pure precheck. Tool source/scope/cursor checks still run at durable admission."""
    validate_action_entries(
        table,
        book=book,
        request_index=request_index,
        _observable=_observable,
        _single_owner=_single_owner,
        _typed_purpose=_typed_purpose,
        _atomic_selector=_atomic_selector,
        _focused=_focused,
    )
    _observable = _observable or _single_owner or _typed_purpose or _atomic_selector or _focused
    if _focused:
        from nurse_rcl.v2.live.ops_sre_focused_decision import selection_shape as focused_shape

        focused_shape(value)
    elif _atomic_selector:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
            selection_shape as atomic_shape,
        )

        atomic_shape(value)
    elif _typed_purpose:
        from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import selection_shape as typed_shape

        typed_shape(value)
    else:
        selection_shape(value)

    def rejected(code: str) -> dict[str, Any]:
        return dict(status="rejected", reason_code=code, action=None)

    concrete = type(value) is str if _atomic_selector or _focused else "entry_ref" in value
    if concrete:
        reference = value if _atomic_selector or _focused else value["entry_ref"]
        entry = next((e for e in table["entries"] if e["entry_ref"] == reference), None)
        if entry is None:
            return rejected("entry_not_available")
        action = deepcopy(entry["action"])
    else:
        nav = value["navigation"]
        kind = nav["kind"]
        params = {k: deepcopy(v) for k, v in nav.items() if k != "kind"}
        if params.get("focus", "all") not in {"all", "symptoms", "mechanisms", "scoring"}:
            return rejected("invalid_navigation")
        fields = [k for k in params if k.endswith("service_key")]
        try:
            for field in fields:
                params[field] = book.resolve(params[field], "service", visible_at=request_index)
            if kind == "catalog":
                from nurse_rcl.v2.live.ops_index import _FILTERS

                params["kind"] = params.pop("catalog_kind")
                if _focused and params["kind"] == "evidence":
                    from nurse_rcl.v2.live.ops_sre_focused_catalog import FILTERS

                    filters = params["filters"]
                    if not set(filters) <= FILTERS or "candidate_service_keys" not in filters:
                        return rejected("invalid_navigation")
                    filters["candidate_service_keys"] = [
                        book.resolve(c, "service", visible_at=request_index)
                        for c in filters["candidate_service_keys"]
                    ]
                elif (
                    params["kind"] not in _FILTERS
                    or not set(params["filters"]) <= _FILTERS[params["kind"]]
                ):
                    return rejected("invalid_navigation")
                if "candidate_service_key" in params["filters"]:
                    params["filters"]["candidate_service_key"] = book.resolve(
                        params["filters"]["candidate_service_key"],
                        "service",
                        visible_at=request_index,
                    )
        except ValueError:
            return rejected("navigation_reference_not_available")
        if len(fields) == 2 and params[fields[0]] == params[fields[1]]:
            return rejected("invalid_navigation")
        skill = "sre-dependency/15" if kind == "dependency" else "sre-boundary/15"
        if kind == "candidate_pair":
            skill = "sre-symptoms/15"
        if _focused:
            skill = skill.removesuffix("/15") + "/20"
        elif _atomic_selector:
            skill = skill.removesuffix("/15") + "/19"
        elif _typed_purpose:
            skill = skill.removesuffix("/15") + "/18"
        elif _observable:
            skill = skill.removesuffix("/15") + "/16"
        action = dict(skill_id=skill, tool=NAV_TOOLS[kind], parameters=params)
    return dict(
        status="accepted",
        reason_code=None,
        action=dict(
            action,
            question=None
            if (_atomic_selector or _focused) and concrete
            else value.get("question")
            if _typed_purpose
            else value["question"],
        ),
    )


def canonical_entry_decision(
    value: dict[str, Any],
    *,
    table: dict[str, Any],
    book: SreActionReferenceBook,
    request_index: int,
) -> dict[str, Any] | None:
    """Only legal selections can reach the unchanged complete-read proof logic."""
    if type(value.get("actions")) is not list or len(value["actions"]) != 1:
        return None
    resolved = resolve_entry_selection(
        value["actions"][0],
        table=table,
        book=book,
        request_index=request_index,
        _observable=table.get("schema_version") == "nurse.v2.sre-action-entries/2",
        _single_owner=table.get("schema_version") == "nurse.v2.sre-action-entries/3",
        _typed_purpose=table.get("schema_version") == "nurse.v2.sre-action-entries/4",
        _atomic_selector=table.get("schema_version") == "nurse.v2.sre-action-entries/5",
        _focused=table.get("schema_version") == "nurse.v2.sre-action-entries/6",
    )
    if resolved["status"] != "accepted":
        return None
    return dict(deepcopy(value), actions=[resolved["action"]])


def exact_entry_delivered_read(
    value: dict[str, Any],
    *,
    table: dict[str, Any],
    book: SreActionReferenceBook,
    request_index: int,
    records: Any,
    toolbox: Any,
) -> dict[str, Any] | None:
    from nurse_rcl.v2.live.ops_sre_d21_reuse import exact_delivered_ledger_read

    canonical = canonical_entry_decision(value, table=table, book=book, request_index=request_index)
    return (
        None
        if canonical is None
        else exact_delivered_ledger_read(canonical, records, toolbox=toolbox)
    )


def exact_entry_replanning_read(
    value: dict[str, Any],
    records: Any,
    *,
    toolbox: Any,
    decisions: Any,
    context: dict[str, Any],
    book: SreActionReferenceBook,
) -> dict[str, Any] | None:
    from nurse_rcl.v2.live.ops_sre_d25_progress import exact_replanning_read

    canonical = canonical_entry_decision(
        value, table=context["action_entries"], book=book, request_index=context["decision_index"]
    )
    return (
        None
        if canonical is None
        else exact_replanning_read(
            canonical,
            records,
            toolbox=toolbox,
            decisions=decisions,
            context=context,
        )
    )


def validate_blocked_entry_read(
    value: dict[str, Any], proof: dict[str, Any], context: dict[str, Any]
) -> None:
    """Check a historical frozen selection; the original proof retains fact authority."""
    from nurse_rcl.v2.live.ops_sre_d25_progress import validate_blocked_read

    actions = value.get("actions")
    if type(actions) is not list or len(actions) != 1:
        raise ValueError("blocked entry selection differs")
    selection_shape(actions[0])
    matches = [
        e
        for e in context["action_entries"]["entries"]
        if e["entry_ref"] == actions[0].get("entry_ref")
    ]
    if len(matches) != 1:
        raise ValueError("blocked entry is not frozen in this request")
    validate_blocked_read(
        dict(value, actions=[dict(matches[0]["action"], question=actions[0]["question"])]),
        proof,
        context,
    )
