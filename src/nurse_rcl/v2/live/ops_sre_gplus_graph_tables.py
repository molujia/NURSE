"""Flat lossless G+ /2 tables; references are only bounded node indexes.

No recursive substitutions or fact/permission inference. The serialized packet
keeps the original JSON limits. The expansion is linear in its bounded rows;
it is not fed through an unrelated expanded-JSON representation quota.
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

TABLE_GRAPH_SCHEMA = "nurse.v2.sre-gplus-evidence-dag/2"
CHOICE_COLUMNS = [
    "expression_ref",
    "section",
    "step_ref",
    "leaf_node_index",
    "closure_admitted",
    "admission_reason",
]
# Separate in-memory codec work bound, never an increased outbound JSON limit.
# Exact repeated literals/edges must not amplify a small packet without bound.
GRAPH_WORK_BYTES = 64 * 1024**2


def _small_scalar(value: Any) -> bool:
    return (
        type(value) in {str, int, float, bool, type(None)}
        and len(canonical_json_bytes(value)) <= 256
    )


def _source_values(nodes: list[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    values = {}
    for node in nodes:
        if node["kind"] != "context":
            continue
        data = node["data"]
        ref = data["evidence_ref"]
        source = data["source"]["fact"]
        if source["fact_ref"] != ref:
            raise ValueError("graph context/source reference differs")
        for value in source["values"]:
            key = (ref, value["name"])
            if key in values:
                raise ValueError("duplicate graph source field")
            values[key] = value["value"]
    return values


def pack_graph_tables(graph: dict[str, Any]) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_gplus_graph import GRAPH_SCHEMA, validate_gplus_graph

    if graph["schema_version"] != GRAPH_SCHEMA:
        raise ValueError("table codec requires expanded /1 graph")
    validate_gplus_graph(graph)
    node_ids = [node["id"] for node in graph["nodes"]]
    indexes = {key: index for index, key in enumerate(node_ids)}
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in graph["nodes"]:
        groups[(item["kind"], tuple(sorted(item["data"])))].append(item)
    tables = []
    expressions: list[dict[str, Any]] = []
    expression_ids: dict[bytes, int] = {}
    source_values = _source_values(graph["nodes"])

    def expression_ref(value: dict[str, Any]) -> int:
        key = canonical_json_bytes(value)
        if key not in expression_ids:
            expression_ids[key] = len(expressions)
            compact = deepcopy(value)
            assertions = compact["claim"].pop("assertions")
            fields = []
            for assertion in assertions:
                pair = (assertion["evidence_ref"], assertion["field"])
                if (
                    set(assertion) != {"evidence_ref", "field", "value"}
                    or pair not in source_values
                    or (
                        canonical_json_bytes(assertion["value"])
                        != canonical_json_bytes(source_values[pair])
                    )
                ):
                    raise ValueError("expression assertion differs from exact source field")
                fields.append(list(pair))
            compact["claim"]["assertion_fields"] = fields
            expressions.append(compact)
        return expression_ids[key]

    for (kind, keys), nodes in sorted(groups.items()):
        first = nodes[0]["data"]
        common = {
            key: first[key]
            for key in keys
            if len(key) <= 128
            and _small_scalar(first[key])
            and all(
                canonical_json_bytes(node["data"][key]) == canonical_json_bytes(first[key])
                for node in nodes[1:]
            )
        }
        literal_columns = [key for key in keys if key not in common]
        columns = [
            "expression_index" if kind == "permission" and key == "expression" else key
            for key in literal_columns
        ]
        tables.append(
            dict(
                kind=kind,
                columns=columns,
                common=deepcopy(common),
                rows=[
                    [
                        indexes[node["id"]],
                        *[
                            expression_ref(node["data"][key])
                            if kind == "permission" and key == "expression"
                            else deepcopy(node["data"][key])
                            for key in literal_columns
                        ],
                    ]
                    for node in nodes
                ],
            )
        )
    edge_groups: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for edge in graph["edges"]:
        edge_groups[edge["relation"]][indexes[edge["from"]]].append(indexes[edge["to"]])
    adjacency_groups = []
    for relation, adjacency in sorted(edge_groups.items()):
        targets: list[list[int]] = []
        target_ids: dict[tuple[int, ...], int] = {}
        rows = []
        for parent, children in sorted(adjacency.items()):
            key = tuple(sorted(children))
            if key not in target_ids:
                target_ids[key] = len(targets)
                targets.append(list(key))
            rows.append([parent, target_ids[key]])
        adjacency_groups.append(dict(relation=relation, mandatory=True, targets=targets, rows=rows))
    result = dict(
        schema_version=TABLE_GRAPH_SCHEMA,
        expanded_sha256=sha256_hex(canonical_json_bytes(graph)),
        node_ids=node_ids,
        node_tables=tables,
        expression_pool=expressions,
        edge_groups=adjacency_groups,
        choice_table=dict(
            columns=CHOICE_COLUMNS,
            rows=[
                [
                    row["expression_ref"],
                    row["section"],
                    row["step_ref"],
                    indexes[row["leaf_id"]],
                    row["closure_admitted"],
                    row["admission_reason"],
                ]
                for row in graph["choices"]
            ],
        ),
        metadata=deepcopy(
            {
                key: value
                for key, value in graph.items()
                if key not in {"schema_version", "nodes", "edges", "choices"}
            }
        ),
    )
    encode_json(result, REPORT_LIMITS)
    return result


def expand_graph_tables(packet: dict[str, Any]) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_gplus_graph import GRAPH_SCHEMA, KINDS, RELATIONS

    encode_json(packet, REPORT_LIMITS)
    if (
        type(packet) is not dict
        or set(packet)
        != {
            "schema_version",
            "expanded_sha256",
            "node_ids",
            "node_tables",
            "edge_groups",
            "choice_table",
            "metadata",
            "expression_pool",
        }
        or packet["schema_version"] != TABLE_GRAPH_SCHEMA
    ):
        raise ValueError("table graph schema differs")
    require_pin(packet["expanded_sha256"])
    ids, tables, groups, choices, metadata = (
        packet[name]
        for name in ("node_ids", "node_tables", "edge_groups", "choice_table", "metadata")
    )
    if (
        type(ids) is not list
        or any(
            type(key) is not str
            or re.fullmatch(
                r"(?:source|context|field|permission|leaf):[0-9a-f]{64}",
                key,
            )
            is None
            for key in ids
        )
        or len(set(ids)) != len(ids)
        or type(tables) is not list
        or type(groups) is not list
        or type(metadata) is not dict
        or set(metadata)
        != {
            "catalog_sha256",
            "core_result_digest",
            "upstream_limit",
            "optional_support",
            "authority",
        }
    ):
        raise ValueError("table graph bounded definitions differ")
    expressions = packet["expression_pool"]
    if (
        type(expressions) is not list
        or any(type(row) is not dict for row in expressions)
        or (len({canonical_json_bytes(row) for row in expressions}) != len(expressions))
    ):
        raise ValueError("table graph expression pool differs")
    used_expressions = set()
    work_bytes = len(canonical_json_bytes(metadata))

    def reserve(amount: int) -> None:
        nonlocal work_bytes
        work_bytes += amount
        if work_bytes > GRAPH_WORK_BYTES:
            raise ExchangeBoundaryError(
                "json_limit",
                boundary="logical_bytes",
                observed=work_bytes,
                limit=GRAPH_WORK_BYTES,
                observed_is_lower_bound=True,
            )

    def index(value: Any) -> int:
        if type(value) is not int or not 0 <= value < len(ids):
            raise ValueError("table graph node reference differs")
        return value

    nodes: dict[int, dict[str, Any]] = {}
    permission_refs = []
    for table in tables:
        if type(table) is not dict or set(table) != {"kind", "columns", "common", "rows"}:
            raise ValueError("node table shape differs")
        kind, columns, common, rows = (table[key] for key in ("kind", "columns", "common", "rows"))
        if (
            type(kind) is not str
            or kind not in KINDS
            or type(columns) is not list
            or any(type(key) is not str for key in columns)
            or columns != sorted(set(columns))
            or type(common) is not dict
            or len(common) > 16
            or any(len(key) > 128 or not _small_scalar(value) for key, value in common.items())
            or set(columns) & common.keys()
            or type(rows) is not list
            or not rows
        ):
            raise ValueError("node table columns differ")
        for row in rows:
            if type(row) is not list or len(row) != len(columns) + 1:
                raise ValueError("node table row differs")
            number = index(row[0])
            if number in nodes:
                raise ValueError("duplicate node definition")
            data = deepcopy(dict(common, **dict(zip(columns, row[1:], strict=True))))
            reserve(256 + len(canonical_json_bytes(data)))
            if kind == "permission":
                ref = data.pop("expression_index", None)
                if "expression" in data or type(ref) is not int or not 0 <= ref < len(expressions):
                    raise ValueError("table graph expression reference differs")
                used_expressions.add(ref)
                permission_refs.append((data, ref))
            elif "expression_index" in data:
                raise ValueError("expression reference on wrong node kind")
            nodes[number] = dict(id=ids[number], kind=kind, data=data)
    if set(nodes) != set(range(len(ids))):
        raise ValueError("unused or missing node definition")
    if used_expressions != set(range(len(expressions))):
        raise ValueError("unused expression definition")
    source_values = _source_values(list(nodes.values()))
    # One bounded serialization per actual source field, never one per reference.
    source_value_sizes = {
        pair: len(canonical_json_bytes(value)) for pair, value in source_values.items()
    }
    expanded_expressions = []
    expression_sizes = []
    for expression in expressions:
        if set(expression) != {"expression_ref", "sections", "step_refs", "claim"} or (
            type(expression["claim"]) is not dict
            or "assertions" in expression["claim"]
            or type(expression["claim"].get("assertion_fields")) is not list
        ):
            raise ValueError("graph expression field reference shape differs")
        claim = dict(expression["claim"])
        fields = claim.pop("assertion_fields")
        assertions, cost = [], len(canonical_json_bytes(expression))
        reserve(cost)
        for pair in fields:
            if (
                type(pair) is not list
                or len(pair) != 2
                or any(type(p) is not str for p in pair)
                or tuple(pair) not in source_values
            ):
                raise ValueError("graph assertion source field missing")
            increment = 64 + source_value_sizes[tuple(pair)]
            reserve(increment)
            cost += increment
        for ref, field in fields:
            assertions.append(
                dict(evidence_ref=ref, field=field, value=source_values[(ref, field)])
            )
        claim["assertions"] = assertions
        expanded_expressions.append(dict(expression, claim=claim))
        expression_sizes.append(cost)
    for data, ref in permission_refs:
        reserve(expression_sizes[ref])
        # Shared complete literal, no per-permission copy or recursive reference.
        data["expression"] = expanded_expressions[ref]
    edges: list[dict[str, Any]] = []
    relations = set()
    for group in groups:
        if type(group) is not dict or set(group) != {"relation", "mandatory", "targets", "rows"}:
            raise ValueError("edge table shape differs")
        relation = group["relation"]
        if (
            type(relation) is not str
            or relation not in RELATIONS
            or relation in relations
            or group["mandatory"] is not True
            or type(group["targets"]) is not list
            or type(group["rows"]) is not list
            or not group["rows"]
        ):
            raise ValueError("edge table relation differs")
        relations.add(relation)
        target_rows = group["targets"]
        for target_row in target_rows:
            if (
                type(target_row) is not list
                or not target_row
                or any(type(n) is not int or not 0 <= n < len(ids) for n in target_row)
                or target_row != sorted(set(target_row))
            ):
                raise ValueError("edge table targets differ")
        used_targets, used_parents = set(), set()
        for pair in group["rows"]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("edge table pair differs")
            parent, target = index(pair[0]), pair[1]
            if (
                type(target) is not int
                or not 0 <= target < len(target_rows)
                or parent in used_parents
            ):
                raise ValueError("edge table adjacency reference differs")
            used_targets.add(target)
            used_parents.add(parent)
            reserve(512 * len(target_rows[target]))
            edges.extend(
                dict(**{"from": ids[parent], "to": ids[child]}, relation=relation, mandatory=True)
                for child in target_rows[target]
            )
        if used_targets != set(range(len(target_rows))):
            raise ValueError("unused adjacency definition")
    if (
        type(choices) is not dict
        or set(choices) != {"columns", "rows"}
        or choices["columns"] != CHOICE_COLUMNS
        or type(choices["rows"]) is not list
    ):
        raise ValueError("choice table shape differs")
    expanded_choices = []
    for row in choices["rows"]:
        if type(row) is not list or len(row) != len(CHOICE_COLUMNS):
            raise ValueError("choice table row differs")
        expanded_choices.append(
            dict(
                expression_ref=row[0],
                section=row[1],
                step_ref=row[2],
                leaf_id=ids[index(row[3])],
                closure_admitted=row[4],
                admission_reason=row[5],
            )
        )
    graph = dict(
        deepcopy(metadata),
        schema_version=GRAPH_SCHEMA,
        nodes=[nodes[number] for number in range(len(ids))],
        edges=sorted(edges, key=canonical_json_bytes),
        choices=expanded_choices,
    )
    if sha256_hex(canonical_json_bytes(graph)) != packet["expanded_sha256"]:
        raise ValueError("table graph complete expansion differs")
    # Canonical packing excludes alternate column layouts, unused tables or
    # redundant representations. Structural graph validation follows in caller.
    if canonical_json_bytes(pack_graph_tables(graph)) != canonical_json_bytes(packet):
        raise ValueError("table graph is not canonical")
    return graph
