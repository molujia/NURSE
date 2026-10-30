"""Three fixed overview routes, without diagnostic aliases or source replays."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_report_context import OpsReportFact

OVERVIEW = "candidate_overview_not_full_judgment"
_ENTRY_FIELDS = {"card_id", "targets", "source_role", "source_refs", "window_ns", "expansion"}
_PARENT = re.compile(r"payload\.(mechanisms\.states|scoring\.records)\[(0|[1-9][0-9]*)\]")


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError("SRE candidate navigation differs")


def _route(*, candidate: str | None = None, card: str | None = None,
           working: bool = False) -> dict[str, Any]:
    if candidate is not None:
        _require(type(candidate) is str and bool(candidate))
        parameters: dict[str, Any] = dict(candidate_service_key=candidate, cursor=None)
        if working:
            parameters["focus"] = "all"
        return dict(tool="inspect_candidate", parameters=parameters)
    _require(type(card) is str and bool(card))
    return dict(tool="read_node", parameters=dict(card_id=card, section="payload", cursor=None))


def split_candidate_navigation(
    payload: dict[str, Any], *, working: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Input was resource-admitted; preserve it, including every non-route field."""
    _require(type(payload) is dict and set(payload) == {
        "candidate_service_key", "source_role", "representation", "record_counts",
        "mechanisms", "scoring", "expansion", "meaning",
    })
    _require(payload["representation"] == OVERVIEW and
             payload["source_role"] == "candidate_overview_metadata")
    candidate = payload["candidate_service_key"]
    expected = _route(candidate=candidate, working=working)
    _require(canonical_json_bytes(payload["expansion"]) == canonical_json_bytes(expected))
    for section, name, fields in (
        ("mechanisms", "states", _ENTRY_FIELDS | {
            "mechanism_key", "disposition", "truth_value", "record_count",
            "additional_source_count", "judgment_status_counts", "judgment_status_by_group"}),
        ("scoring", "records", _ENTRY_FIELDS | {"rank"}),
    ):
        part = payload[section]
        _require(type(part) is dict and set(part) == {"status", name}
                 and type(part[name]) is list)
        for parent in part[name]:
            _require(type(parent) is dict and set(parent) == fields)
            expected = _route(card=parent["card_id"])
            _require(canonical_json_bytes(parent["expansion"]) == canonical_json_bytes(expected))
    body = deepcopy(payload)
    slots: list[tuple[dict[str, Any], list[Any]]] = [(body, ["expansion"])]
    slots += [(state, ["mechanisms", "states", i, "expansion"])
              for i, state in enumerate(body["mechanisms"]["states"])]
    slots += [(score, ["scoring", "records", i, "expansion"])
              for i, score in enumerate(body["scoring"]["records"])]
    navigation = [dict(path=path, expansion=parent.pop("expansion")) for parent, path in slots]
    restored = deepcopy(body)
    for entry in navigation:
        path = entry["path"]
        parent = restored if path == ["expansion"] else restored[path[0]][path[1]][path[2]]
        parent["expansion"] = deepcopy(entry["expansion"])
    _require(canonical_json_bytes(restored) == canonical_json_bytes(payload))
    return body, navigation


def validate_navigation_group(
    group: dict[str, Any], facts: tuple[OpsReportFact, ...], *, working: bool = False,
) -> None:
    """Check exact paths against this group's delivered parents, not other cards.

    Source authenticity remains the owner's admitted receipt/context binding.
    Matching these fields never authorizes reading or asserting an unseen card.
    """
    if group.get("representation") != OVERVIEW:
        _require("navigation" not in group)
        return
    _require(group.get("source_role") == "query_scope_metadata" and group.get("card_id") is None)
    paths = {f.source_path: f for f in facts}
    _require(len(paths) == len(facts) and "payload" in paths)
    root = paths["payload"]
    values = {v.name: v.value for v in root.fact.values}
    candidate = values.get("candidate_service_key")
    _require(type(candidate) is str and root.fact.target_service_keys == (candidate,))
    expected: list[dict[str, Any]] = [dict(path=["expansion"],
                                         expansion=_route(candidate=cast(str, candidate),
                                                          working=working))]
    children: dict[str, dict[int, OpsReportFact]] = {"mechanisms.states": {}, "scoring.records": {}}
    for path, fact in paths.items():
        _require(not path.startswith("payload.expansion") and not re.match(
            r"payload\.(?:mechanisms\.states|scoring\.records)\[[0-9]+\]\.expansion(?:\.|$)", path))
        match = _PARENT.fullmatch(path)
        if match is not None:
            children[match[1]][int(match[2])] = fact
    for prefix, indexed in children.items():
        _require(sorted(indexed) == list(range(len(indexed))))
        for index in sorted(indexed):
            parent = {v.name: v.value for v in indexed[index].fact.values}
            card = parent.get("card_id")
            expected.append(dict(path=[*prefix.split("."), index, "expansion"],
                                 expansion=_route(card=cast(str | None, card))))
    navigation = group.get("navigation")
    _require(type(navigation) is list and
             canonical_json_bytes(navigation) == canonical_json_bytes(expected))
