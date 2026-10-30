"""Flat, same-request publication and tool-contract sharing in fixed regions.

Publication definitions are literals, not recursive references. Skill prose and
all feedback are preserved. No previous provider request is assumed available.
"""

from __future__ import annotations

import json
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

HISTORY_SCHEMA = "nurse.v2.sre-history-view/1"


def _steps(view: dict[str, Any]) -> list[dict[str, Any]]:
    rows = view.get("steps", [])
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("SRE history steps differ")
    return rows


def _contracts(view: dict[str, Any]) -> dict[str, Any]:
    capabilities = view.get("capabilities", {})
    if type(capabilities) is not dict:
        raise ValueError("SRE capabilities shape differs")
    value = capabilities.get("parameter_contracts", {})
    if type(value) is not dict or any(type(v) is not dict for v in value.values()):
        raise ValueError("SRE parameter contracts differ")
    return value


def _skills(view: dict[str, Any]) -> list[dict[str, Any]]:
    rows = view.get("skills", [])
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("SRE skills shape differs")
    return rows


def _nodes(value: Any) -> int:
    """Count an already preflighted JSON value, including dictionary keys."""
    if type(value) is dict:
        return 1 + len(value) + sum(_nodes(child) for child in value.values())
    if type(value) is list:
        return 1 + sum(_nodes(child) for child in value)
    return 1


def pack_history_view(
    view: dict[str, Any], *, limits: ExchangeLimits = REPORT_LIMITS,
) -> dict[str, Any]:
    content = json.loads(encode_json(view, limits))
    publications: list[list[dict[str, Any]]] = []
    publication_ids: dict[bytes, int] = {}
    for step in _steps(content):
        known = step.get("known_at_decision")
        if type(known) is not dict or "publication_ref" in known:
            raise ValueError("SRE historical publication shape differs")
        published = known.pop("core_publication_summaries", None)
        if type(published) is not list or any(type(p) is not dict for p in published):
            raise ValueError("SRE publication must be a literal list")
        key = canonical_json_bytes(published)
        if key not in publication_ids:
            publication_ids[key] = len(publications)
            publications.append(published)
        known["publication_ref"] = publication_ids[key]
    rules: list[Any] = []
    rule_ids: dict[bytes, int] = {}
    for params in _contracts(content).values():
        for name, rule in params.items():
            # Existing contracts include enumeration arrays and structured
            # filters as well as text. Store each literal intact, never coerce.
            key = canonical_json_bytes(rule)
            if key not in rule_ids:
                rule_ids[key] = len(rules)
                rules.append(rule)
            params[name] = rule_ids[key]
    skills = _skills(content)
    common = {} if not skills else {k: skills[0][k] for k in skills[0] if all(
        k in row and canonical_json_bytes(row[k]) == canonical_json_bytes(skills[0][k])
        for row in skills[1:])}
    # Identity and tool permissions remain explicit even in a one-skill fixture.
    for name in ("skill_id", "node", "tools", "guidance", "content_sha256"):
        common.pop(name, None)
    for row in skills:
        for name in common:
            row.pop(name)
    result = dict(schema_version=HISTORY_SCHEMA, content=content, publications=publications,
                  parameter_rules=rules, skill_common=common)
    encode_json(result, limits)
    return result


def unpack_history_view(
    packet: dict[str, Any], *, limits: ExchangeLimits = REPORT_LIMITS,
) -> dict[str, Any]:
    # The preflight bounds before serializing/cloning. Reused definitions below
    # share memory; the final encode preflights their full expanded JSON size.
    source = json.loads(encode_json(packet, limits))
    if type(source) is not dict or set(source) != {
        "schema_version", "content", "publications", "parameter_rules", "skill_common",
    } or source["schema_version"] != HISTORY_SCHEMA or type(source["content"]) is not dict:
        raise ValueError("SRE history packet shape differs")
    content, publications = source["content"], source["publications"]
    rules, common = source["parameter_rules"], source["skill_common"]
    if type(publications) is not list or any(type(p) is not list or
        any(type(row) is not dict for row in p) for p in publications
    ) or (len({canonical_json_bytes(p) for p in publications}) != len(publications)
          or type(rules) is not list
          or len({canonical_json_bytes(r) for r in rules}) != len(rules)
          or type(common) is not dict):
        raise ValueError("SRE shared definitions differ")
    used_publications: set[int] = set()
    for step in _steps(content):
        known = step.get("known_at_decision")
        if type(known) is not dict or "core_publication_summaries" in known:
            raise ValueError("SRE historical publication collision")
        index = known.pop("publication_ref", None)
        if type(index) is not int or not 0 <= index < len(publications):
            raise ValueError("SRE publication reference differs")
        used_publications.add(index)
        known["core_publication_summaries"] = publications[index]
    used_rules: set[int] = set()
    for params in _contracts(content).values():
        for name, index in params.items():
            if type(index) is not int or not 0 <= index < len(rules):
                raise ValueError("SRE parameter rule reference differs")
            used_rules.add(index)
            params[name] = rules[index]
    skills = _skills(content)
    if (common and not skills) or common.keys() & {
        "skill_id", "node", "tools", "guidance", "content_sha256",
    }:
        raise ValueError("SRE skill identity cannot be implicit")
    for row in skills:
        if row.keys() & common.keys():
            raise ValueError("SRE skill common field collision")
    # Unlike publication/rule references, updating each skill allocates dict
    # slots. Bound the complete expansion BEFORE making any of these copies.
    size = len(encode_json(content, limits))
    common_size = len(encode_json(common, limits)) - 2
    extra_bytes = sum(common_size + int(bool(row)) for row in skills) if common else 0
    extra_nodes = len(skills) * (_nodes(common) - 1)
    if size + extra_bytes > limits.max_bytes or _nodes(content) + extra_nodes > limits.max_nodes:
        raise ExchangeBoundaryError("json_limit")
    for row in skills:
        row.update(common)
    if used_publications != set(range(len(publications))) or used_rules != set(range(len(rules))):
        raise ValueError("SRE unused definition or reference differs")
    encode_json(content, limits)
    return dict(content)
