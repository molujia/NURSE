"""Lossless scientific display with private, context-bound short references.

This codec has no source or outbound authority. The normal owner supplies its
admitted fixed context. Only identities are private; every scientific field of
each displayed record is decoded and compared at the representation boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_semantic import NormalSemanticResponse, parse_normal_semantic_response
from nurse_rcl.v2.live.normal_semantic_context import NormalSemanticContext

WIRE_VERSION = "nurse.v2.normal-semantic-wire/1"
# Codec ceiling only, not a registered model input allowance. Legacy unchanged.
WIRE_LIMITS = ExchangeLimits(max_bytes=1_048_576, max_nodes=100_000)
AliasKind = Literal["candidate", "evidence", "request", "need"]
SIGNAL_COLUMNS = (
    "reference_id", "candidate", "metric_name", "family", "modality", "unit",
    "baseline_summary", "fault_summary", "evidence_state", "quality_flags",
    "formal_scoring_eligible", "lifecycle_meaning", "lifecycle_zero_count",
    "lifecycle_eligible", "fact_set",
)
SUMMARY_COLUMNS = (
    "record_state", "valid_point_count", "coverage_fraction", "median", "mad",
    "minimum", "maximum", "event_count",
)
FACT_COLUMNS = ("predicate_kind", "measurement_field", "truth_value", "reason_codes")
NEED_COLUMNS = (
    "mechanism_key", "predicate_kind", "requirement", "queryable", "initial_tier",
    "reason_codes",
)
ERROR_COLUMNS = (
    "reference_id", "candidate_service_key", "modality", "span_kind", "statistic",
    "severity_basis", "baseline_known_denominator", "fault_known_denominator",
    "baseline_fraction", "fault_fraction", "observed_error_count", "sampling_coverage",
    "fact_set", "limitations",
)
EDGE_COLUMNS = (
    "reference_id", "window", "caller", "callee", "relation_type", "direction",
    "allows_directed_propagation", "observation_state",
)


class SemanticAlias(StrictFrozenModel):
    kind: AliasKind
    original: str
    alias: str


class SemanticWirePackage(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-wire-package/1"] = (
        "nurse.v2.normal-semantic-wire-package/1"
    )
    source_context_sha256: SHA256
    wire_json: str
    wire_sha256: SHA256
    aliases: tuple[SemanticAlias, ...]


class _Table:
    def __init__(self) -> None:
        self.rows: list[Any] = []
        self.index: dict[bytes, int] = {}

    def add(self, row: Any) -> int:
        key = canonical_json_bytes(row)
        if key not in self.index:
            self.index[key] = len(self.rows)
            self.rows.append(row)
        return self.index[key]


def _encode(context: NormalSemanticContext) -> SemanticWirePackage:
    """Representation only. No telemetry reconstruction or source revalidation."""
    aliases: list[SemanticAlias] = []
    by_original: dict[tuple[AliasKind, str], str] = {}
    counts: dict[str, int] = {}

    def alias(kind: AliasKind, original: str, prefix: str) -> str:
        key = (kind, original)
        if key not in by_original:
            value = prefix + str(counts.get(prefix, 0))
            counts[prefix] = counts.get(prefix, 0) + 1
            by_original[key] = value
            aliases.append(SemanticAlias(kind=kind, original=original, alias=value))
        return by_original[key]

    candidates = [[alias("candidate", c, "C"), c] for c in context.candidate_service_keys]
    summaries, facts, needs = _Table(), _Table(), _Table()

    def fact_set(values: list[dict[str, Any]]) -> int:
        return facts.add([[f[k] for k in FACT_COLUMNS] for f in values])

    signals = []
    signal_refs: dict[tuple[str, str, str | None], str] = {}
    for signal in context.signals:
        value = signal.model_dump(mode="json")
        ref = alias("evidence", signal.reference_id, "S")
        key = (signal.source_sha256, signal.observation_id, signal.candidate_service_key)
        if key in signal_refs:
            raise ValueError("duplicate semantic signal identity")
        signal_refs[key] = ref
        signals.append([
            ref, by_original["candidate", signal.candidate_service_key], signal.metric_name,
            signal.family, signal.modality, signal.unit,
            summaries.add([value["baseline"][k] for k in SUMMARY_COLUMNS]),
            summaries.add([value["fault"][k] for k in SUMMARY_COLUMNS]),
            signal.evidence_state, list(signal.quality_flags), signal.formal_scoring_eligible,
            signal.lifecycle_meaning, signal.lifecycle_zero_count, signal.lifecycle_eligible,
            fact_set(value["facts"]),
        ])
    errors = []
    for error in context.errors:
        row = error.model_dump(mode="json")
        row["reference_id"] = alias("evidence", error.reference_id, "E")
        row["candidate_service_key"] = by_original["candidate", error.candidate_service_key]
        row["fact_set"] = fact_set(row.pop("facts"))
        errors.append([row[k] for k in ERROR_COLUMNS])
    edges = []
    for edge in context.edges:
        row = edge.model_dump(mode="json")
        row["reference_id"] = alias("evidence", edge.reference_id, "G")
        # Keep endpoint service names readable, including non-candidate endpoints.
        edges.append([row[k] for k in EDGE_COLUMNS])
    requests = []
    for request in context.requests:
        request_source = (
            request.source_sha256, request.observation_id, request.candidate_service_key,
        )
        if request_source not in signal_refs:
            raise ValueError("semantic request lacks its displayed source")
        definitions = []
        for need in request.needs:
            row = need.model_dump(mode="json")
            definitions.append([alias("need", need.need_key, "N"),
                                needs.add([row[k] for k in NEED_COLUMNS])])
        requests.append([
            alias("request", request.request_key, "R"), signal_refs[request_source], definitions,
        ])
    wire = dict(
        schema_version=WIRE_VERSION, opaque_case_id=context.opaque_case_id,
        scope=context.scope, score_authority=False, omitted_signal_count=0,
        candidates=candidates,
        focus=[by_original["candidate", c] for c in context.focus_service_keys],
        baseline_window=context.baseline_window.model_dump(mode="json"),
        fault_window=context.fault_window.model_dump(mode="json"),
        signal_columns=SIGNAL_COLUMNS, signals=signals,
        summary_columns=SUMMARY_COLUMNS, summaries=summaries.rows,
        fact_columns=FACT_COLUMNS, fact_sets=facts.rows,
        need_columns=NEED_COLUMNS, need_definitions=needs.rows,
        request_columns=("request_key", "signal_reference", "needs"),
        request_need_columns=("need_key", "definition_index"), requests=requests,
        error_columns=ERROR_COLUMNS, errors=errors, edge_columns=EDGE_COLUMNS, edges=edges,
        graph_limitations=context.graph_limitations,
    )
    raw = encode_json(wire, WIRE_LIMITS)
    return SemanticWirePackage(
        source_context_sha256=context.context_sha256, wire_json=raw.decode(),
        wire_sha256=sha256_hex(raw), aliases=tuple(aliases),
    )


def build_semantic_wire(context: NormalSemanticContext) -> SemanticWirePackage:
    if type(context) is not NormalSemanticContext or _seal(context, "context_sha256") != context:
        raise ValueError("semantic context identity differs")
    return _encode(context)


def restore_semantic_context(
    package: SemanticWirePackage, context: NormalSemanticContext,
) -> NormalSemanticContext:
    """Decode public facts; join only private IDs; compare to owner's context."""
    if type(package) is not SemanticWirePackage or type(context) is not NormalSemanticContext:
        raise ValueError("invalid semantic representation type")
    # A deterministic display comparison rejects extra fields, bool/int aliasing,
    # dropped rows and re-signed private bindings. It is not full source replay.
    expected = build_semantic_wire(context)
    if canonical_json_bytes(package.model_dump(mode="json")) != canonical_json_bytes(
        expected.model_dump(mode="json"),
    ):
        raise ValueError("semantic representation differs from owned context")
    wire = decode_json_object(package.wire_json.encode(), WIRE_LIMITS)
    refs = {(a.kind, a.alias): a.original for a in package.aliases}
    value = context.model_dump(mode="json")
    value["candidate_service_keys"] = [row[1] for row in wire["candidates"]]
    value["focus_service_keys"] = [refs["candidate", c] for c in wire["focus"]]
    for key in ("opaque_case_id", "scope", "score_authority", "baseline_window",
                "fault_window", "graph_limitations"):
        value[key] = wire[key]

    def facts(index: int) -> list[dict[str, Any]]:
        return [dict(zip(FACT_COLUMNS, row, strict=True)) for row in wire["fact_sets"][index]]

    signals = []
    for row, private in zip(wire["signals"], context.signals, strict=True):
        item = dict(zip(SIGNAL_COLUMNS, row, strict=True))
        item["reference_id"] = refs["evidence", item["reference_id"]]
        item["candidate_service_key"] = refs["candidate", item.pop("candidate")]
        for side in ("baseline", "fault"):
            item[side] = dict(zip(SUMMARY_COLUMNS, wire["summaries"][item.pop(side + "_summary")],
                                  strict=True))
        item["facts"] = facts(item.pop("fact_set"))
        item.update(source_sha256=private.source_sha256, observation_id=private.observation_id)
        signals.append(item)
    value["signals"] = signals
    errors = []
    for row in wire["errors"]:
        item = dict(zip(ERROR_COLUMNS, row, strict=True))
        item["reference_id"] = refs["evidence", item["reference_id"]]
        item["candidate_service_key"] = refs["candidate", item["candidate_service_key"]]
        item["facts"] = facts(item.pop("fact_set"))
        errors.append(item)
    value["errors"] = errors
    value["edges"] = [dict(zip(EDGE_COLUMNS, [refs["evidence", row[0]], *row[1:]], strict=True))
                      for row in wire["edges"]]
    requests = []
    signals_by_ref = {s.reference_id: s for s in context.signals}
    for row, private_request in zip(wire["requests"], context.requests, strict=True):
        signal = signals_by_ref[refs["evidence", row[1]]]
        item = private_request.model_dump(mode="json")
        item.update(request_key=refs["request", row[0]], source_sha256=signal.source_sha256,
                    observation_id=signal.observation_id,
                    candidate_service_key=signal.candidate_service_key)
        item["needs"] = [dict(
            zip(NEED_COLUMNS, wire["need_definitions"][definition], strict=True),
            need_key=refs["need", ref], condition_key=need.condition_key,
        ) for (ref, definition), need in zip(row[2], private_request.needs, strict=True)]
        requests.append(item)
    value["requests"] = requests
    if canonical_json_bytes(value) != canonical_json_bytes(context.model_dump(mode="json")):
        raise ValueError("decoded semantic facts differ")
    return NormalSemanticContext.model_validate_json(canonical_json_bytes(value))


def expand_semantic_response(
    response: NormalSemanticResponse, package: SemanticWirePackage,
) -> NormalSemanticResponse:
    """Unknown/wrong-kind/long IDs remain semantic rejections, never retries.

    The caller must first bind this package to its owned context using restore.
    Strings in rationales are not interpreted as executable references.
    """
    return _expand_response_aliases(response, package.aliases)


def _expand_response_aliases(
    response: NormalSemanticResponse, aliases: tuple[SemanticAlias, ...],
) -> NormalSemanticResponse:
    """Reference syntax only; caller separately binds the owning representation."""
    refs = {(a.kind, a.alias): a.original for a in aliases}
    originals = {a.original for a in aliases}

    def resolve(kind: AliasKind, key: str) -> str:
        if (kind, key) in refs:
            return refs[kind, key]
        # A marker must be outside the actual identity universe, not merely
        # unlikely to collide with an unusual but valid telemetry service name.
        for nonce in range(len(originals) + 1):
            marker = "unresolved_" + sha256_hex(canonical_json_bytes([kind, key, nonce]))
            if marker not in originals:
                return marker
        raise AssertionError("finite missing-reference marker space exhausted")

    value = response.model_dump(mode="json")
    for hypothesis in value["hypotheses"]:
        hypothesis["candidate_service_key"] = resolve(
            "candidate", hypothesis["candidate_service_key"],
        )
        hypothesis["evidence_refs"] = [resolve("evidence", r) for r in hypothesis["evidence_refs"]]
    for query in value["queries"]:
        if query["kind"] == "card":
            query["candidate_service_key"] = resolve("candidate", query["candidate_service_key"])
        else:
            query["request_key"] = resolve("request", query["request_key"])
            query["need_keys"] = [resolve("need", n) for n in query["need_keys"]]
    return parse_normal_semantic_response(canonical_json_bytes(value))
