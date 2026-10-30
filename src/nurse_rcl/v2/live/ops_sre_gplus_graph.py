"""Normal G+ source/field/permission DAG, independent of research cases.

Graph edges expose possible bases, never entailment or independent votes.
Mandatory source constraints are never cut to satisfy a closure budget.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore

GRAPH_SCHEMA = "nurse.v2.sre-gplus-evidence-dag/1"
RELATIONS = {
    "derived_from",
    "field_unit_of",
    "scope_constraint",
    "section_permission",
    "available_basis",
}
KINDS = {"source", "context", "field", "permission", "leaf"}


class ClosureBudgetError(ValueError):
    """An otherwise legal choice has too many mandatory upstream nodes."""


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _node_id(kind: str, identity: Any) -> str:
    return kind + ":" + _digest(identity)


def build_gplus_graph(
    catalog: dict[str, Any],
    *,
    representation: str = "expanded",
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_gplus_graph_tables import GRAPH_WORK_BYTES

    if representation not in {"expanded", "tables", "auto"}:
        raise ValueError("unknown G+ graph representation")
    encode_json(catalog, REPORT_LIMITS)
    require_pin(catalog["core_result_digest"])
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[bytes, dict[str, Any]] = {}
    choices = []
    fields_by_ref: dict[str, list[str]] = {}
    work_bytes = 0

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

    def node(key: str, kind: str, data: dict[str, Any]) -> str:
        literal = dict(id=key, kind=kind, data=data)
        if key in nodes:
            if canonical_json_bytes(nodes[key]) != canonical_json_bytes(literal):
                raise ValueError("conflicting G+ source node")
            return key
        reserve(256 + len(canonical_json_bytes(data)))
        value = dict(id=key, kind=kind, data=deepcopy(data))
        nodes[key] = value
        return key

    def edge(parent: str, child: str, relation: str) -> None:
        value = dict(**{"from": parent, "to": child}, relation=relation, mandatory=True)
        key = canonical_json_bytes(value)
        if key not in edges:
            reserve(512)
            edges[key] = value

    for ref, row in catalog["sources"].items():
        source = row["source"]
        if row["source_sha256"] != _digest(source) or source["fact"]["fact_ref"] != ref:
            raise ValueError("G+ exact source version differs")
        identity = {
            k: source[k]
            for k in ("source_path", "origin", "evidence_ids", "source_refs", "action_index")
        }
        parent = node(
            _node_id("source", identity), "source", dict(identity=identity, independent_votes=False)
        )
        context = node(
            _node_id("context", ref),
            "context",
            dict(
                evidence_ref=ref,
                source=source,
                record_unit=row["record_unit"],
                shared_source=parent,
                independent_votes=False,
            ),
        )
        edge(parent, context, "derived_from")
        fields_by_ref[ref] = []
        for field in row["fields"]:
            name = field["field"]
            field_id = node(
                _node_id("field", [ref, name]), "field", dict(evidence_ref=ref, **field)
            )
            if field_id in fields_by_ref[ref]:
                raise ValueError("duplicate G+ source field")
            fields_by_ref[ref].append(field_id)
            edge(context, field_id, "scope_constraint")
            if field["unit"]["state"] != "not_supplied":
                edge(_node_id("field", [ref, field["unit"]["field"]]), field_id, "field_unit_of")
    seen_expressions = set()
    for row in catalog["rows"]:
        choice = row["expression"]
        ref = choice["expression_ref"]
        if ref in seen_expressions:
            raise ValueError("duplicate G+ expression")
        seen_expressions.add(ref)
        required = choice["claim"]["evidence_refs"]
        if len(required) != len(set(required)) or not set(required) <= fields_by_ref.keys():
            raise ValueError("G+ expression exceeds exact sources")
        for section in choice["sections"]:
            steps = choice["step_refs"] if section == "step_findings" else [None]
            for step in steps:
                choice_identity = [ref, section, step]
                permission = node(
                    _node_id("permission", choice_identity),
                    "permission",
                    dict(
                        expression=choice,
                        section=section,
                        step_ref=step,
                        condition_boundary=catalog["condition_boundaries"].get(ref),
                        source_scope_sha256=row["source_scope_sha256"],
                        row_version=row["row_version"],
                    ),
                )
                leaf = node(
                    _node_id("leaf", choice_identity),
                    "leaf",
                    dict(
                        expression_ref=ref,
                        section=section,
                        step_ref=step,
                        role="available_choice_not_a_conclusion",
                    ),
                )
                edge(permission, leaf, "section_permission")
                for source_ref in required:
                    for field_id in fields_by_ref[source_ref]:
                        edge(field_id, leaf, "available_basis")
                choices.append(
                    dict(expression_ref=ref, section=section, step_ref=step, leaf_id=leaf)
                )
    graph = dict(
        schema_version=GRAPH_SCHEMA,
        catalog_sha256=_digest(catalog),
        core_result_digest=catalog["core_result_digest"],
        nodes=sorted(nodes.values(), key=lambda n: n["id"]),
        edges=sorted(edges.values(), key=canonical_json_bytes),
        choices=choices,
        upstream_limit=dict(
            optional_ancestors=12,
            optional_hops=3,
            total_nodes_per_leaf=64,
            mandatory_constraints_are_never_trimmed=True,
        ),
        optional_support="none_supplied_no_invented_ancestors",
        authority="program_source_constraints_not_semantic_certification",
    )
    index = validate_gplus_graph(graph)
    parents = _parents(graph, index)
    for choice in choices:
        try:
            _closure(graph, index, parents, choice["leaf_id"])
            choice.update(closure_admitted=True, admission_reason="complete_mandatory_closure")
        except ClosureBudgetError:
            choice.update(
                closure_admitted=False, admission_reason="mandatory_closure_over64_no_trimming"
            )
    if representation != "tables":
        try:
            encode_json(graph, REPORT_LIMITS)
            return graph
        except ExchangeBoundaryError as exc:
            if representation != "auto" or exc.code != "json_limit":
                raise
    from nurse_rcl.v2.live.ops_sre_gplus_graph_tables import pack_graph_tables

    return pack_graph_tables(graph)


def expand_gplus_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Explicit version reader; no recursive references or permission inheritance."""
    from nurse_rcl.v2.live.ops_sre_gplus_graph_tables import TABLE_GRAPH_SCHEMA, expand_graph_tables

    if graph.get("schema_version") == TABLE_GRAPH_SCHEMA:
        return expand_graph_tables(graph)
    if graph.get("schema_version") != GRAPH_SCHEMA or set(graph) != {
        "schema_version",
        "catalog_sha256",
        "core_result_digest",
        "nodes",
        "edges",
        "choices",
        "upstream_limit",
        "optional_support",
        "authority",
    }:
        raise ValueError("G+ graph schema differs")
    return graph


def gplus_graph_facts(graph: dict[str, Any], *, ordered_refs: list[str]) -> NativeFactStore:
    """Reconstruct each exact fact once from complete graph context sources.

    The /2 wire shares these sources instead of carrying a duplicate fact store.
    This is a codec inverse, not inference, source discovery or omitted evidence.
    """
    from nurse_rcl.v2.live.ops_report_context import OpsReportFact
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store

    logical = expand_gplus_graph(graph)
    sources = {}
    for node in logical["nodes"]:
        if node["kind"] == "context":
            data = node["data"]
            ref = data["evidence_ref"]
            if ref in sources or data["source"]["fact"]["fact_ref"] != ref:
                raise ValueError("G+ shared fact source differs")
            sources[ref] = data["source"]
    if (
        type(ordered_refs) is not list
        or len(set(ordered_refs)) != len(ordered_refs)
        or set(ordered_refs) != sources.keys()
    ):
        raise ValueError("G+ shared fact source order differs")
    return build_fact_store(
        [
            OpsReportFact.model_validate_json(encode_json(sources[ref], REPORT_LIMITS), strict=True)
            for ref in ordered_refs
        ]
    )


def validate_gplus_graph(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    graph = expand_gplus_graph(graph)
    if graph["schema_version"] != GRAPH_SCHEMA:
        raise ValueError("G+ graph schema differs")
    nodes = {n["id"]: n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]) or any(n["kind"] not in KINDS for n in nodes.values()):
        raise ValueError("duplicate or untyped G+ node")
    indegree = dict.fromkeys(nodes, 0)
    children: dict[str, list[str]] = {key: [] for key in nodes}
    seen = set()
    for edge in graph["edges"]:
        if (
            set(edge) != {"from", "to", "relation", "mandatory"}
            or edge["from"] not in nodes
            or edge["to"] not in nodes
            or edge["relation"] not in RELATIONS
            or edge["mandatory"] is not True
            or canonical_json_bytes(edge) in seen
            or nodes[edge["from"]]["kind"] == "leaf"
        ):
            raise ValueError("dangling, duplicate or non-sink G+ edge")
        seen.add(canonical_json_bytes(edge))
        indegree[edge["to"]] += 1
        children[edge["from"]].append(edge["to"])
    queue = deque(key for key, degree in indegree.items() if not degree)
    visited = 0
    while queue:
        visited += 1
        for child in children[queue.popleft()]:
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if visited != len(nodes):
        raise ValueError("cyclic G+ graph")
    leaves = [c["leaf_id"] for c in graph["choices"]]
    if len(leaves) != len(set(leaves)) or set(leaves) != {
        k for k, n in nodes.items() if n["kind"] == "leaf"
    }:
        raise ValueError("G+ choices differ from sink leaves")
    return nodes


def _parents(graph: dict[str, Any], nodes: dict[str, Any]) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {key: [] for key in nodes}
    for edge in graph["edges"]:
        parents[edge["to"]].append(edge["from"])
    return parents


def _closure(
    graph: dict[str, Any], nodes: dict[str, Any], parents: dict[str, list[str]], leaf_id: str
) -> dict[str, Any]:
    if leaf_id not in nodes or nodes[leaf_id]["kind"] != "leaf":
        raise ValueError("G+ closure requires a choice leaf")
    included, pending = {leaf_id}, [leaf_id]
    while pending:
        for parent in parents[pending.pop()]:
            if parent not in included:
                included.add(parent)
                pending.append(parent)
    if len(included) > 64:
        raise ClosureBudgetError("G+ mandatory closure over64; never trim constraints")
    return dict(
        leaf_id=leaf_id,
        mandatory_complete=True,
        optional_ancestor_count=0,
        optional_truncated=False,
        nodes=[deepcopy(nodes[k]) for k in sorted(included)],
        edges=[
            deepcopy(e) for e in graph["edges"] if e["from"] in included and e["to"] in included
        ],
    )


def gplus_closure(graph: dict[str, Any], leaf_id: str) -> dict[str, Any]:
    graph = expand_gplus_graph(graph)
    nodes = validate_gplus_graph(graph)
    return _closure(graph, nodes, _parents(graph, nodes), leaf_id)
