"""D12 native source-local trees with bounded, single-alias fact access.

Build incrementally into shared defaults. No flat fact collection is constructed
or cached. The alias index is explicit and counted as part of the native packet.
Old /1 codecs retain their old aggregate expansion boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_fact_view import _META, _SOURCE, _count, _parts
from nurse_rcl.v2.live.ops_sre_report_bounds import _check, _measure

NATIVE_FACT_SCHEMA = "nurse.v2.sre-fact-records/2"


class _NativeBudget:
    """Account for edits to the actual shared JSON, including every index entry."""

    def __init__(self, packet: dict[str, Any]) -> None:
        self.nodes = _count(packet)
        self.bytes = len(canonical_json_bytes(packet))

    def check(self) -> None:
        _check("facts_nodes", self.nodes, 100_000)
        _check("logical_bytes", self.bytes, REPORT_LIMITS.max_bytes)

    def put(self, target: dict[str, Any], key: str, value: Any) -> None:
        if key in target:
            self.nodes += _count(value) - _count(target[key])
            self.bytes += len(canonical_json_bytes(value)) - len(canonical_json_bytes(target[key]))
        else:
            self.nodes += 1 + _count(value)
            self.bytes += len(canonical_json_bytes(key)) + 1 + len(canonical_json_bytes(value))
            self.bytes += int(bool(target))
        self.check()
        target[key] = value

    def remove(self, target: dict[str, Any], key: str) -> Any:
        value = target.pop(key)
        self.nodes -= 1 + _count(value)
        self.bytes -= len(canonical_json_bytes(key)) + 1 + len(canonical_json_bytes(value))
        self.bytes -= int(bool(target))
        return value

    def append(self, target: list[Any], value: Any) -> None:
        self.nodes += _count(value)
        self.bytes += len(canonical_json_bytes(value)) + int(bool(target))
        self.check()
        target.append(value)


def _factor(
    container: dict[str, Any], metadata: dict[str, Any], values: dict[str, Any],
    budget: _NativeBudget, *, root: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Demote only noncommon defaults one level, never across source trees."""
    remaining = []
    for label, field, incoming in (("fact_defaults", "metadata", metadata),
                                   ("value_defaults", "values", values)):
        shared = container[label]
        changed = [key for key, value in shared.items() if key not in incoming or
                   canonical_json_bytes(value) != canonical_json_bytes(incoming[key])]
        for key in changed:
            old = budget.remove(shared, key)
            for child in container["tree" if root else "children"]:
                budget.put(child[label], key, old)
            if "fact" in container:
                body = container["fact"]
                if field not in body:
                    budget.put(body, field, {})
                budget.put(body[field], key, old)
        remaining.append({key: value for key, value in incoming.items() if key not in shared})
    return remaining[0], remaining[1]


def build_fact_store(facts: Iterable[OpsReportFact]) -> NativeFactStore:
    """Consume once; retain source trees and alias locations, not fact wrappers."""
    packet: dict[str, Any] = dict(schema_version=NATIVE_FACT_SCHEMA,
                                fact_order=[], records=[], alias_index={})
    budget = _NativeBudget(packet)
    latest: dict[bytes, tuple[int, dict[str, Any], set[str]]] = {}
    for item in facts:
        if type(item) is not OpsReportFact:
            raise ValueError("SRE native fact requires admitted typed object")
        # OpsReportFact's older non-strict serializer turns bool into int before
        # JSON validation. Inspect this process index before invoking it.
        if item.action_index is not None and type(item.action_index) is not int:
            raise ValueError("SRE native fact action type differs")
        raw = encode_json(item.model_dump(mode="json"), REPORT_LIMITS)
        source = OpsReportFact.model_validate_json(raw, strict=True).model_dump(mode="json")
        metadata = source.pop("fact")
        path = source.pop("source_path")
        segments = _parts(path)
        ref = metadata.pop("fact_ref")
        if ref in packet["alias_index"]:
            raise ValueError("SRE duplicate native fact alias")
        values = {v["name"]: v["value"] for v in metadata.pop("values")}
        mirror = values.get("record_path") == path
        if mirror:
            del values["record_path"]
        identity = canonical_json_bytes(source)
        group = latest.get(identity)
        if group is None or path in group[2]:
            record: dict[str, Any] = dict(source=source, fact_defaults=dict(metadata),
                value_defaults=dict(values), mirror_record_paths=False, tree=[])
            group = len(packet["records"]), record, set()
            budget.append(packet["records"], record)
            latest[identity] = group
        position, record, paths = group
        paths.add(path)
        metadata, values = _factor(record, metadata, values, budget, root=True)
        branch = record["tree"]
        for segment in segments:
            node = next((n for n in branch if n["at"] == segment), None)
            if node is None:
                node = dict(at=segment, fact_defaults=dict(metadata),
                            value_defaults=dict(values), children=[])
                budget.append(branch, node)
            metadata, values = _factor(node, metadata, values, budget)
            branch = node["children"]
        body = dict(ref=ref, mirror_record_path=mirror)
        if metadata:
            body["metadata"] = metadata
        if values:
            body["values"] = values
        assert node is not None  # _parts rejects an empty path.
        budget.put(node, "fact", body)
        budget.append(packet["fact_order"], ref)
        budget.put(packet["alias_index"], ref, [position, path])
    # Exact final measurement checks incremental accounting, bytes and depth.
    if budget.nodes != _count(packet) or budget.bytes != len(canonical_json_bytes(packet)):
        raise ValueError("SRE native budget accounting differs")
    return NativeFactStore.from_packet(packet)


def _defaults(
    node: dict[str, Any], parent_meta: dict[str, Any], parent_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta, values = node.get("fact_defaults", {}), node.get("value_defaults", {})
    if type(meta) is not dict or not meta.keys() <= _META or meta.keys() & parent_meta.keys() or (
        type(values) is not dict or values.keys() & parent_values.keys()
    ):
        raise ValueError("SRE native default collision")
    return {**parent_meta, **meta}, {**parent_values, **values}


def _fact(
    record: dict[str, Any], node: dict[str, Any], path: str,
    inherited_meta: dict[str, Any], inherited_values: dict[str, Any],
) -> OpsReportFact:
    row = node["fact"]
    if type(row) is not dict or "ref" not in row or not row.keys() <= {
        "ref", "metadata", "values", "mirror_record_path",
    }:
        raise ValueError("SRE native tree fact shape differs")
    meta, values = row.get("metadata", {}), row.get("values", {})
    if type(meta) is not dict or not meta.keys() <= _META or (
        meta.keys() & inherited_meta.keys() or type(values) is not dict
        or values.keys() & inherited_values.keys()
    ):
        raise ValueError("SRE native fact field collision")
    mirror = row.get("mirror_record_path", record["mirror_record_paths"])
    if type(mirror) is not bool:
        raise ValueError("SRE native path mirror type differs")
    all_values = {**inherited_values, **values}
    if mirror:
        if "record_path" in all_values:
            raise ValueError("SRE native path mirror collision")
        all_values["record_path"] = path
    raw = dict(record["source"], source_path=path, fact=dict(
        **inherited_meta, **meta, fact_ref=row["ref"],
        values=[dict(name=k, value=v) for k, v in sorted(all_values.items())]))
    return OpsReportFact.model_validate_json(encode_json(raw, REPORT_LIMITS), strict=True)


class NativeFactStore:
    """An owned, admitted packet. Access materializes one typed fact at a time."""

    __slots__ = ("_packet",)

    def __init__(self, packet: dict[str, Any]) -> None:
        self._packet = json.loads(encode_json(packet, REPORT_LIMITS))
        self._validate()

    @classmethod
    def from_packet(cls, packet: dict[str, Any]) -> NativeFactStore:
        return cls(packet)

    def to_packet(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(canonical_json_bytes(self._packet)))

    def refs(self) -> tuple[str, ...]:
        return tuple(self._packet["fact_order"])

    def measure(self) -> dict[str, int]:
        nodes, size, depth = _measure(self._packet, node_limit=100_000, boundary="facts_nodes")
        return dict(fact_nodes=nodes, logical_bytes=size, logical_depth=depth)

    def resolve(self, ref: str) -> OpsReportFact:
        position, path = self._packet["alias_index"][ref]
        record = self._packet["records"][position]
        meta, values = record["fact_defaults"], record["value_defaults"]
        branch = record["tree"]
        for segment in _parts(path):
            node = next(n for n in branch if n["at"] == segment)
            meta, values = _defaults(node, meta, values)
            branch = node.get("children", [])
        return _fact(record, node, path, meta, values)

    def iter_facts(self) -> Iterator[OpsReportFact]:
        for ref in self._packet["fact_order"]:
            yield self.resolve(ref)

    def _validate(self) -> None:
        packet = self._packet
        if type(packet) is not dict or set(packet) != {
            "schema_version", "fact_order", "records", "alias_index",
        } or packet["schema_version"] != NATIVE_FACT_SCHEMA or (
            type(packet["fact_order"]) is not list or type(packet["records"]) is not list
            or type(packet["alias_index"]) is not dict
            or any(type(ref) is not str for ref in packet["fact_order"])
            or len(set(packet["fact_order"])) != len(packet["fact_order"])
            or set(packet["fact_order"]) != packet["alias_index"].keys()
        ):
            raise ValueError("SRE native fact packet shape differs")
        self.measure()
        aliases = set(packet["fact_order"])
        seen: set[str] = set()
        for position, record in enumerate(packet["records"]):
            if type(record) is not dict or set(record) != {
                "source", "fact_defaults", "value_defaults", "mirror_record_paths", "tree",
            } or (type(record["source"]) is not dict or set(record["source"]) != _SOURCE
                  or type(record["mirror_record_paths"]) is not bool
                  or type(record["tree"]) is not list or not record["tree"]):
                raise ValueError("SRE native source shape differs")
            root_meta, root_values = _defaults(record, {}, {})
            pending = [(n, "", root_meta, root_values) for n in reversed(record["tree"])]
            paths: set[str] = set()
            initial = len(seen)
            while pending:
                node, parent, parent_meta, parent_values = pending.pop()
                if type(node) is not dict or "at" not in node or not node.keys() <= {
                    "at", "children", "fact", "fact_defaults", "value_defaults",
                } or type(node.get("children", [])) is not list:
                    raise ValueError("SRE native tree node differs")
                if len(_parts(node["at"])) != 1:
                    raise ValueError("SRE native segment differs")
                path = node["at"] if not parent else parent + "." + node["at"]
                _parts(path)
                if path in paths:
                    raise ValueError("SRE duplicate native path")
                paths.add(path)
                meta, values = _defaults(node, parent_meta, parent_values)
                pending.extend((child, path, meta, values)
                               for child in reversed(node.get("children", [])))
                if "fact" not in node:
                    if not node.get("children"):
                        raise ValueError("SRE empty native leaf")
                    continue
                fact = _fact(record, node, path, meta, values)
                ref = fact.fact.fact_ref
                if ref in seen or ref not in aliases or (
                    not set(fact.fact.related_fact_refs) <= aliases
                ):
                    raise ValueError("SRE native alias closure differs")
                location = packet["alias_index"][ref]
                if type(location) is not list or len(location) != 2 or (
                    type(location[0]) is not int or type(location[1]) is not str
                    or location != [position, path]
                ):
                    raise ValueError("SRE native alias index differs")
                seen.add(ref)
            if len(seen) == initial:
                raise ValueError("SRE native source has no facts")
        if seen != aliases:
            raise ValueError("SRE native fact order closure differs")
