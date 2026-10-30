"""D17 navigation from delivered registered score records, not target evidence.

Projection is not source authentication. The owner must independently verify
the delivered records against its admitted source; the provider checks only the
frozen request structure and at-the-time delivery associations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_encoding import expand

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_source_checkpoint import ScoreSource

NAVIGATION_SCHEMA = "nurse.v2.sre-delivered-navigation/1"
NAVIGATION_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/11"
SOURCE_VERSION = "sre-score-lineage/1"
_PATHS: dict[str, dict[str, tuple[str, str | None]]] = {
    "score_lineage_summary": {
        "family_details": ("score_family", "primary_family"),
        "propagation_detail": ("score_propagation", None),
    },
    "score_family": {"term_details": ("score_term", "term_id")},
    "score_propagation": {"path_details": ("score_path", "path_id")},
}


def _text(value: Any) -> bool:
    return type(value) is str and bool(value) and len(value) <= 2048


def _links(row: dict[str, Any]) -> list[dict[str, Any]]:
    body = row.get("payload", {})
    kind = body.get("kind")
    if kind not in _PATHS:
        return []
    if (
        row.get("tool_version") != SOURCE_VERSION
        or row.get("node") != "scoring"
        or row.get("published_by_core") is not True
        or row.get("source_role") != "core_published"
        or row.get("representation") is not None
        or row.get("targets") != [body.get("candidate_service_key")]
        or not _text(body.get("candidate_service_key"))
        or not _text(row.get("card_id"))
    ):
        raise ValueError("delivered navigation source differs")
    links = []
    for field, (target_kind, identity) in _PATHS[kind].items():
        if field not in body:
            continue  # Missing fields do not imply hidden links.
        values = body[field] if identity else [body[field]]
        if type(values) is not list:
            raise ValueError("delivered navigation path shape differs")
        for index, item in enumerate(values):
            if type(item) is not dict or not _text(item.get("card_id")):
                raise ValueError("delivered navigation target differs")
            path: list[str | int] = ["payload", field]
            if identity:
                path.append(index)
            path.append("card_id")
            link = dict(card_id=item["card_id"], kind=target_kind, source_path=path)
            if identity:
                if not _text(item.get(identity)):
                    raise ValueError("delivered navigation identity differs")
                link[identity] = item[identity]
            if kind == "score_family":
                family = body.get("family", {}).get("primary_family")
                if not _text(family):
                    raise ValueError("delivered navigation family differs")
                link["primary_family"] = family
            links.append(link)
    return links


def delivered_navigation(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Project only delivered bodies. No registry scan, target read or fact aliases."""
    from nurse_rcl.v2.live.ops_sre_context import _semantic_rows

    if len(history) > 8 or [r["action_index"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("delivered navigation action sequence differs")
    sources: dict[str, dict[str, Any]] = {}
    for saved in history:
        if saved["status"] != "completed":
            continue
        view = expand(saved["view"])
        if view.get("status") == "resource_limit":
            continue
        for row in _semantic_rows(view):
            links = _links(row)
            if not links:
                continue
            digest = sha256_hex(canonical_json_bytes(row))
            source = sources.setdefault(
                digest,
                dict(
                    source=dict(
                        card_id=row["card_id"],
                        record_sha256=digest,
                        kind=row["payload"]["kind"],
                        candidate_service_key=row["targets"][0],
                    ),
                    links=links,
                    delivery_action_indices=[],
                ),
            )
            if saved["action_index"] not in source["delivery_action_indices"]:
                source["delivery_action_indices"].append(saved["action_index"])
    result = dict(schema_version=NAVIGATION_SCHEMA, sources=list(sources.values()))
    encode_json(result, CONTEXT_LIMITS)
    return result


def navigation_targets(context: dict[str, Any]) -> set[str]:
    """Validate provider-visible structure; source authenticity belongs to owner."""
    if context.get("schema_version") not in {
        NAVIGATION_INPUT_SCHEMA,
        "nurse.v2.sre-investigation-input/12",
        "nurse.v2.sre-investigation-input/13",
        "nurse.v2.sre-investigation-input/14",
        "nurse.v2.sre-investigation-input/15",
        "nurse.v2.sre-investigation-input/16",
        "nurse.v2.sre-investigation-input/17",
        "nurse.v2.sre-investigation-input/18",
    }:
        return set()
    error = "sre_delivered_navigation"
    nav = context.get("delivered_navigation")
    if (
        type(nav) is not dict
        or set(nav) != {"schema_version", "sources"}
        or (nav["schema_version"] != NAVIGATION_SCHEMA or type(nav["sources"]) is not list)
    ):
        raise ExchangeBoundaryError(error)
    encode_json(nav, CONTEXT_LIMITS)
    actions = {r["action_index"]: r for r in context["actions"]}
    candidates = {
        r["candidate_service_key"]
        for r in context["capabilities"].get("investigation_situation", {}).get("candidates", [])
    }
    result: set[str] = set()
    seen = set()
    for entry in nav["sources"]:
        if type(entry) is not dict or set(entry) != {"source", "links", "delivery_action_indices"}:
            raise ExchangeBoundaryError(error)
        src, indices = entry["source"], entry["delivery_action_indices"]
        if type(src) is not dict or set(src) != {
            "card_id",
            "record_sha256",
            "kind",
            "candidate_service_key",
        }:
            raise ExchangeBoundaryError(error)
        if (
            not all(_text(v) for v in src.values())
            or src["kind"] not in _PATHS
            or (src["candidate_service_key"] not in candidates or src["record_sha256"] in seen)
        ):
            raise ExchangeBoundaryError(error)
        seen.add(src["record_sha256"])
        if (
            type(indices) is not list
            or not indices
            or len(indices) > 8
            or (
                any(type(i) is not int or i not in actions for i in indices)
                or sorted(set(indices)) != indices
                or any(
                    i >= context["decision_index"]
                    or actions[i]["status"] != "completed"
                    or src["card_id"] not in actions[i]["visible_card_ids"]
                    for i in indices
                )
            )
        ):
            raise ExchangeBoundaryError(error)
        if type(entry["links"]) is not list or not entry["links"]:
            raise ExchangeBoundaryError(error)
        paths = set()
        for link in entry["links"]:
            if type(link) is not dict:
                raise ExchangeBoundaryError(error)
            path = link.get("source_path")
            if (
                type(path) is not list
                or len(path) not in {3, 4}
                or (
                    path[0] != "payload"
                    or type(path[1]) is not str
                    or path[1] not in _PATHS[src["kind"]]
                    or path[-1] != "card_id"
                )
            ):
                raise ExchangeBoundaryError(error)
            target_kind, identity = _PATHS[src["kind"]][path[1]]
            keys = {"card_id", "kind", "source_path"} | ({identity} if identity else set())
            if src["kind"] == "score_family":
                keys.add("primary_family")
            if (
                set(link) != keys
                or link["kind"] != target_kind
                or (
                    not all(_text(v) for k, v in link.items() if k != "source_path")
                    or len(path) != (4 if identity else 3)
                    or (identity and (type(path[2]) is not int or path[2] < 0))
                    or tuple(path) in paths
                )
            ):
                raise ExchangeBoundaryError(error)
            paths.add(tuple(path))
            result.add(link["card_id"])
    return result


def verify_navigation_source(
    history: list[dict[str, Any]],
    *,
    source: ScoreSource | None,
) -> dict[str, Any]:
    """Owner-only exact source check, after ledger receipt/time admission.

    A terminal score source is atomically published as one immutable card set.
    An authentic delivered source record therefore names only that publication's
    children. Target metadata is checked here; no target becomes supplied facts.
    """
    import json

    from nurse_rcl.v2.live.ops_sre_views import semantic_record

    result = delivered_navigation(history)
    if not result["sources"]:
        return result
    if source is None:
        raise ValueError("delivered navigation lacks admitted score source")
    snapshot = source.descriptor["core_identity"]["case_snapshot_id"]
    for entry in result["sources"]:
        src = entry["source"]
        card = source.cards.get(src["card_id"])
        if (
            card is None
            or card.case_snapshot_id != snapshot
            or (
                sha256_hex(canonical_json_bytes(semantic_record(card, published=True)))
                != src["record_sha256"]
            )
        ):
            raise ValueError("delivered navigation source record differs")
        for link in entry["links"]:
            target = source.cards.get(link["card_id"])
            if (
                target is None
                or target.case_snapshot_id != snapshot
                or (
                    target.tool_version != SOURCE_VERSION
                    or json.loads(target.parameters_json).get("candidate_service_key")
                    != src["candidate_service_key"]
                )
            ):
                raise ValueError("delivered navigation target identity differs")
            body = json.loads(target.payload_json)
            if body.get("kind") != link["kind"]:
                raise ValueError("delivered navigation target kind differs")
            identity = {
                "score_family": ("family", "primary_family"),
                "score_term": ("term", "term_id"),
                "score_path": ("path", "path_id"),
            }.get(link["kind"])
            if identity and body[identity[0]][identity[1]] != link[identity[1]]:
                raise ValueError("delivered navigation target identity differs")
    return result
