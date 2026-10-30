"""Exact report facts in source-local trees, not LLM summaries or new evidence.

Each record has exact shared source/fact/value fields. Tree segments concatenate
with dots to the original source_path. A node's fact retains its original alias;
mirror_record_path means the named scalar equals this exact reconstructed path.
No arbitrary references, expressions or schema-inferred scalar values exist.
"""

from __future__ import annotations

import re
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

FACT_VIEW_SCHEMA = "nurse.v2.sre-fact-records/1"
_SEGMENT = re.compile(r"[a-z][a-z0-9_]*(?:\[(?:0|[1-9][0-9]{0,5})\])*")
_META = {"kind", "target_service_keys", "intervals", "status_codes", "related_fact_refs",
         "allowed_claim_strength", "scope_basis"}
_SOURCE = {"role", "evidence_ids", "source_refs", "action_index", "origin",
           "reused_core_evidence_ids"}


def _parts(path: str) -> list[str]:
    if type(path) is not str or len(path) > 4096:
        raise ValueError("SRE fact path differs")
    parts = path.split(".")
    if len(parts) > 24 or any(_SEGMENT.fullmatch(p) is None for p in parts):
        raise ValueError("SRE fact path differs")
    return parts


def _common(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {key: first[key] for key in sorted(first) if all(
        key in row and canonical_json_bytes(row[key]) == canonical_json_bytes(first[key])
        for row in rows[1:])}


def _share_subtree(node: dict[str, Any]) -> None:
    rows = []
    pending = [node]
    while pending:
        current = pending.pop()
        if "fact" in current:
            rows.append(current["fact"])
        pending.extend(current["children"])
    if len(rows) > 1:
        for field, label in (("metadata", "fact_defaults"), ("values", "value_defaults")):
            shared = _common([row.get(field, {}) for row in rows])
            if shared:
                node[label] = shared
                for row in rows:
                    rest = {k: v for k, v in row[field].items() if k not in shared}
                    if rest:
                        row[field] = rest
                    else:
                        row.pop(field)
    for child in node["children"]:
        _share_subtree(child)
    if not node["children"]:
        node.pop("children")


def _count(value: Any) -> int:
    # Input and individual fact have already passed the depth/cycle boundary.
    if type(value) is dict:
        return 1 + len(value) + sum(_count(v) for v in value.values())
    if type(value) is list:
        return 1 + sum(_count(v) for v in value)
    return 1


class _Budget:
    def __init__(self) -> None:
        self.nodes, self.bytes = 1, 2

    def add(self, value: dict[str, Any]) -> bytes:
        raw = encode_json(value, REPORT_LIMITS)
        self.nodes += _count(value)
        self.bytes += len(raw) + 1
        if self.nodes > REPORT_LIMITS.max_nodes or self.bytes > REPORT_LIMITS.max_bytes:
            raise ExchangeBoundaryError("json_limit")
        return raw


def pack_fact_records(facts: tuple[OpsReportFact, ...]) -> dict[str, Any]:
    """Partition on exact provenance, then share only exact common fields."""
    buckets: list[dict[str, Any]] = []
    latest: dict[bytes, dict[str, Any]] = {}
    refs: list[str] = []
    seen_refs: set[str] = set()
    budget = _Budget()
    for item in facts:
        if type(item) is not OpsReportFact:
            raise ValueError("SRE fact requires admitted typed object")
        raw = budget.add(item.model_dump(mode="json"))
        # Revalidate model_copy updates rather than trusting the wrapper type.
        source = OpsReportFact.model_validate_json(raw).model_dump(mode="json")
        fact = source.pop("fact")
        path = source.pop("source_path")
        _parts(path)
        ref = fact.pop("fact_ref")
        if ref in seen_refs:
            raise ValueError("SRE duplicate fact alias")
        seen_refs.add(ref)
        refs.append(ref)
        values = {v["name"]: v["value"] for v in fact.pop("values")}
        mirror = values.get("record_path") == path
        if mirror:
            del values["record_path"]
        identity = canonical_json_bytes(source)
        group = latest.get(identity)
        # Same provenance/path can belong to different actual tool records.
        # Start another tree, never fuse observations or overwrite their values.
        if group is None or path in group["paths"]:
            group = dict(source=source, rows=[], paths=set())
            buckets.append(group)
            latest[identity] = group
        group["paths"].add(path)
        group["rows"].append(dict(ref=ref, path=path, metadata=fact,
                                  values=values, mirror=mirror))
    for group in buckets:
        if any(not set(row["metadata"]["related_fact_refs"]) <= seen_refs
               for row in group["rows"]):
            raise ValueError("SRE fact alias closure differs")
    records = []
    for group in buckets:
        rows = group["rows"]
        meta = _common([r["metadata"] for r in rows])
        values = _common([r["values"] for r in rows])
        mirror_all = all(r["mirror"] for r in rows)
        roots: list[dict[str, Any]] = []
        locations: dict[str, dict[str, Any]] = {}
        for row in rows:
            branch = roots
            prefix = ""
            for segment in _parts(row["path"]):
                prefix = segment if not prefix else prefix + "." + segment
                if prefix not in locations:
                    node: dict[str, Any] = dict(at=segment, children=[])
                    locations[prefix] = node
                    branch.append(node)
                node = locations[prefix]
                branch = node["children"]
            payload: dict[str, Any] = dict(ref=row["ref"])
            rest_meta = {k: v for k, v in row["metadata"].items() if k not in meta}
            rest_values = {k: v for k, v in row["values"].items() if k not in values}
            if rest_meta:
                payload["metadata"] = rest_meta
            if rest_values:
                payload["values"] = rest_values
            if row["mirror"] and not mirror_all:
                payload["mirror_record_path"] = True
            node["fact"] = payload
        for root in roots:
            _share_subtree(root)
        records.append(dict(source=group["source"], fact_defaults=meta,
                            value_defaults=values, mirror_record_paths=mirror_all, tree=roots))
    result = dict(schema_version=FACT_VIEW_SCHEMA, fact_order=refs, records=records)
    encode_json(result, REPORT_LIMITS)
    return result


def unpack_fact_records(packet: dict[str, Any]) -> tuple[OpsReportFact, ...]:
    """Bound the packet and each expanded fact before adding it to the result."""
    encode_json(packet, REPORT_LIMITS)
    if type(packet) is not dict or set(packet) != {"schema_version", "fact_order", "records"} or (
        packet["schema_version"] != FACT_VIEW_SCHEMA or type(packet["records"]) is not list
        or type(packet["fact_order"]) is not list
        or any(type(r) is not str for r in packet["fact_order"])
        or len(set(packet["fact_order"])) != len(packet["fact_order"])
    ):
        raise ValueError("SRE fact record packet shape differs")
    by_ref: dict[str, OpsReportFact] = {}
    budget = _Budget()
    for record in packet["records"]:
        if type(record) is not dict or set(record) != {
            "source", "fact_defaults", "value_defaults", "mirror_record_paths", "tree",
        } or (type(record["source"]) is not dict or set(record["source"]) != _SOURCE
              or type(record["fact_defaults"]) is not dict
              or not record["fact_defaults"].keys() <= _META
              or type(record["value_defaults"]) is not dict
              or type(record["mirror_record_paths"]) is not bool
              or type(record["tree"]) is not list or not record["tree"]):
            raise ValueError("SRE source record shape differs")
        seen_paths: set[str] = set()
        start_count = len(by_ref)
        pending = [(n, "", record["fact_defaults"], record["value_defaults"])
                   for n in reversed(record["tree"])]
        while pending:
            node, parent, parent_meta, parent_values = pending.pop()
            if type(node) is not dict or "at" not in node or (
                not node.keys() <= {"at", "children", "fact", "fact_defaults", "value_defaults"}
                or type(node["at"]) is not str or _SEGMENT.fullmatch(node["at"]) is None
                or type(node.get("children", [])) is not list
            ):
                raise ValueError("SRE fact tree node differs")
            local_meta, local_values = node.get("fact_defaults", {}), node.get("value_defaults", {})
            if type(local_meta) is not dict or not local_meta.keys() <= _META or (
                local_meta.keys() & parent_meta.keys() or type(local_values) is not dict
                or local_values.keys() & parent_values.keys()
            ):
                raise ValueError("SRE subtree default collision")
            inherited_meta = {**parent_meta, **local_meta}
            inherited_values = {**parent_values, **local_values}
            path = node["at"] if not parent else parent + "." + node["at"]
            _parts(path)
            if path in seen_paths:
                raise ValueError("SRE duplicate fact tree path")
            seen_paths.add(path)
            pending.extend((child, path, inherited_meta, inherited_values)
                           for child in reversed(node.get("children", [])))
            if "fact" not in node:
                if not node.get("children"):
                    raise ValueError("SRE empty fact tree leaf")
                continue
            row = node["fact"]
            if type(row) is not dict or "ref" not in row or not row.keys() <= {
                "ref", "metadata", "values", "mirror_record_path",
            }:
                raise ValueError("SRE tree fact shape differs")
            meta, values = row.get("metadata", {}), row.get("values", {})
            if type(meta) is not dict or not meta.keys() <= _META or (
                meta.keys() & inherited_meta.keys()
                or type(values) is not dict or values.keys() & inherited_values.keys()
            ):
                raise ValueError("SRE fact common field collision")
            mirror = row.get("mirror_record_path", record["mirror_record_paths"])
            if type(mirror) is not bool:
                raise ValueError("SRE path mirror type differs")
            all_values = {**inherited_values, **values}
            if mirror:
                if "record_path" in all_values:
                    raise ValueError("SRE path mirror collision")
                all_values["record_path"] = path
            raw_fact = dict(record["source"], source_path=path, fact=dict(
                **inherited_meta, **meta, fact_ref=row["ref"],
                values=[dict(name=k, value=v) for k, v in sorted(all_values.items())]))
            fact = OpsReportFact.model_validate_json(budget.add(raw_fact))
            if fact.fact.fact_ref in by_ref:
                raise ValueError("SRE duplicate fact alias")
            by_ref[fact.fact.fact_ref] = fact
        if len(by_ref) == start_count:
            raise ValueError("SRE source record has no facts")
    if set(packet["fact_order"]) != set(by_ref) or any(
        not set(f.fact.related_fact_refs) <= by_ref.keys() for f in by_ref.values()
    ):
        raise ValueError("SRE fact reference closure differs")
    return tuple(by_ref[ref] for ref in packet["fact_order"])
