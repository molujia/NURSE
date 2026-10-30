"""Local source layouts over existing native facts, not a second payload tree.

Only fixed object/list structure and original field names are retained here.
Values are read from the same source-local f records. There are no expressions,
cross-source inheritance, file references or model-visible lookup permissions.
"""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore


def bind_retained_source(source: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    row, refs = source["record"], source["fact_refs"]
    by_path = {store.resolve(ref).source_path: ref for ref in refs}
    if len(by_path) != len(refs):
        raise ValueError("retained source paths differ")

    def layout(body: Any, path: str) -> Any:
        if type(body) is list:
            return ["array", [layout(v, f"{path}[{i}]") for i, v in enumerate(body)]]
        if type(body) is not dict:
            # _scalars does not turn scalar members of a mixed list into facts.
            return ["literal", body]
        ref = by_path[path]
        values = {v.name: v.value for v in store.resolve(ref).fact.values}
        scalar: list[str] = []
        arrays: list[str] = []
        children: dict[str, Any] = {}
        overrides: dict[str, Any] = {}
        for name, value in body.items():
            is_array = type(value) is list and all(type(v) not in (dict, list) for v in value)
            if type(value) not in (dict, list) or is_array:
                (arrays if is_array else scalar).append(name)
                exact = canonical_json_bytes(value).decode() if is_array else value
                if name not in values or canonical_json_bytes(values[name]) != canonical_json_bytes(
                    exact
                ):
                    # The existing fact projection adds authoritative header fields.
                    # Preserve a colliding original field explicitly, never change that fact.
                    overrides[name] = value
            else:
                children[name] = layout(value, path + "." + name)
        return ["object", ref, scalar, arrays, children, overrides]

    result = dict(
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        header={k: v for k, v in row.items() if k != "payload"},
        fact_refs=refs,
        payload_layout=layout(row["payload"], "payload"),
    )
    return decode_json_object(encode_json(result, REPORT_LIMITS), REPORT_LIMITS)


def restore_retained_source(binding: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    encode_json(binding, REPORT_LIMITS)
    if set(binding) != {"record_sha256", "header", "fact_refs", "payload_layout"} or (
        type(binding["header"]) is not dict or "payload" in binding["header"]
    ):
        raise ValueError("retained source binding shape differs")
    refs = binding["fact_refs"]
    if type(refs) is not list or len(set(refs)) != len(refs) or not set(refs) <= set(store.refs()):
        raise ValueError("retained source references differ")
    seen: set[str] = set()

    def restore(node: Any, path: str) -> Any:
        if type(node) is not list or not node:
            raise ValueError("retained source layout differs")
        if node[0] == "literal" and len(node) == 2 and type(node[1]) not in (dict, list):
            return node[1]
        if node[0] == "array" and len(node) == 2 and type(node[1]) is list:
            return [restore(v, f"{path}[{i}]") for i, v in enumerate(node[1])]
        if len(node) != 6 or node[0] != "object":
            raise ValueError("retained source layout differs")
        _, ref, scalar, arrays, children, overrides = node
        if (
            ref not in refs
            or ref in seen
            or type(scalar) is not list
            or type(arrays) is not list
            or (type(children) is not dict or type(overrides) is not dict)
        ):
            raise ValueError("retained source object differs")
        names = [*scalar, *arrays, *children]
        if (
            any(type(k) is not str for k in names)
            or len(set(names)) != len(names)
            or (not overrides.keys() <= set(scalar + arrays))
        ):
            raise ValueError("retained source fields differ")
        seen.add(ref)
        item = store.resolve(ref)
        if item.source_path != path:
            raise ValueError("retained source path differs")
        values = {v.name: v.value for v in item.fact.values}
        result = {}
        for name in scalar + arrays:
            if name in overrides:
                value = overrides[name]
            elif name not in values:
                raise ValueError("retained source field missing")
            elif name in arrays:
                raw = values[name]
                if type(raw) is not str:
                    raise ValueError("retained source array differs")
                value = decode_json_object(('{"items":' + raw + "}").encode(), REPORT_LIMITS)[
                    "items"
                ]
            else:
                value = values[name]
            if name in arrays and (
                type(value) is not list or any(type(v) in (list, dict) for v in value)
            ):
                raise ValueError("retained source scalar array differs")
            if name in scalar and type(value) in (list, dict):
                raise ValueError("retained source scalar differs")
            result[name] = value
        for name, child in children.items():
            result[name] = restore(child, path + "." + name)
        return result

    row = dict(binding["header"], payload=restore(binding["payload_layout"], "payload"))
    # Non-payload overview counts have one separate native fact.
    extra = set(refs) - seen
    if any(store.resolve(ref).source_path != "judgment_status_counts" for ref in extra):
        raise ValueError("retained source fact closure differs")
    if sha256_hex(encode_json(row, REPORT_LIMITS)) != binding["record_sha256"]:
        raise ValueError("retained source digest differs")
    return dict(record=row, fact_refs=list(refs))
