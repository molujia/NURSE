"""Normal G+ projection over original complete source and outbound admission.

Shared source admission is independent of typed saved views. This wire has one
independent G+ system/schema pair only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import _validate_payload, prepare_exchange
from nurse_rcl.v2.live.ops_context_view import (
    REPORT_BYTES,
    DiagnosedInputProjection,
    OpsInputProjection,
)
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail, failure_detail
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_report import (
    BriefSreReportContext,
    _require_brief_unchanged,
    validate_brief_report,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_gplus_contract import (
    GPLUS_INPUT,
    GPLUS_INPUT_TABLES,
    GPLUS_PROTOCOL,
    GPLUS_VARIANT,
    GPLUS_VIEW,
    GPLUS_VIEW_TABLES,
    gplus_report_policy,
)
from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, gplus_graph_facts
from nurse_rcl.v2.live.ops_sre_gplus_graph_tables import TABLE_GRAPH_SCHEMA
from nurse_rcl.v2.live.ops_sre_history_view import unpack_history_view
from nurse_rcl.v2.live.ops_sre_native_report import NATIVE_REPORT_INPUT_LIMITS
from nurse_rcl.v2.live.ops_sre_typed_report import (
    _admit_report_source,
    _report_source_parts,
)
from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding


def _request(body: dict[str, Any]) -> bytes:
    policy = gplus_report_policy()
    exchange = prepare_exchange(
        model=ARK_MODEL,
        max_output_tokens=policy["max_tokens"],
        limits=REPORT_LIMITS,
        messages=[
            dict(role="system", content=policy["system"]),
            dict(role="user", content=encode_json(body, REPORT_LIMITS).decode()),
        ],
    )
    return encode_json(
        dict(
            decode_json_object(exchange.payload_bytes, REPORT_LIMITS), thinking={"type": "disabled"}
        ),
        REPORT_LIMITS,
    )


def gplus_report_body(
    raw: bytes, *, references: dict[str, Any] | None = None
) -> tuple[dict[str, Any], NativeFactStore]:
    if type(raw) is not bytes:
        raise TypeError("G+ wire requires exact bytes")
    payload = decode_json_object(raw, REPORT_LIMITS)
    base = {k: v for k, v in payload.items() if k != "thinking"}
    messages = payload.get("messages")
    policy = gplus_report_policy()
    if (
        payload.get("thinking") != {"type": "disabled"}
        or _validate_payload(base) is not None
        or base["model"] != ARK_MODEL
        or base["max_tokens"] != policy["max_tokens"]
        or base["temperature"] != 0
        or encode_json(payload, REPORT_LIMITS) != raw
        or type(messages) is not list
        or len(messages) != 2
        or messages[0] != dict(role="system", content=policy["system"])
        or messages[1].get("role") != "user"
    ):
        raise ValueError("G+ single active contract differs")
    body = decode_json_object(messages[1]["content"].encode(), REPORT_LIMITS)
    if (
        set(body)
        != {
            "schema_version",
            "protocol",
            "context",
            "context_sha256",
            "output_schema",
            "opaque_case_id",
            "case_snapshot_id",
        }
        or body["schema_version"] not in {GPLUS_INPUT, GPLUS_INPUT_TABLES}
        or body["protocol"] != GPLUS_PROTOCOL
        or body["output_schema"] != policy["output_schema"]
    ):
        raise ValueError("G+ input contract differs")
    require_pin(body["context_sha256"])
    require_pin(body["case_snapshot_id"])
    packet = body["context"]
    tables = type(packet) is dict and packet.get("schema_version") == GPLUS_VIEW_TABLES
    if (
        type(packet) is not dict
        or set(packet)
        != (
            {"schema_version", "kind", "reference_prefix", "history", "graph"}
            | (set() if tables else {"fact_records"})
        )
        or packet["schema_version"] not in {GPLUS_VIEW, GPLUS_VIEW_TABLES}
        or packet["kind"] != "report"
    ):
        raise ValueError("G+ model graph packet differs")
    encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    metadata = unpack_history_view(packet["history"], limits=NATIVE_REPORT_INPUT_LIMITS)
    store = (
        gplus_graph_facts(packet["graph"], ordered_refs=metadata["supply"]["provided_fact_refs"])
        if tables
        else NativeFactStore.from_packet(packet["fact_records"])
    )
    if body["schema_version"] != (GPLUS_INPUT_TABLES if tables else GPLUS_INPUT) or (
        tables != (packet["graph"].get("schema_version") == TABLE_GRAPH_SCHEMA)
    ):
        raise ValueError("G+ versioned graph representation differs")
    representation = "tables" if tables else "expanded"
    graph = build_gplus_graph(
        typed_expression_catalog(metadata["supply"], store),
        representation=representation,
    )
    if canonical_json_bytes(graph) != canonical_json_bytes(packet["graph"]):
        raise ValueError("G+ graph differs from complete actual source")
    # Exactly the same source/alias/outbound checks, without constructing a
    # second unused typed wire or its expanded saved catalog.
    admitted, original_store = _report_source_parts(
        packet,
        case=body["opaque_case_id"],
        snapshot=body["case_snapshot_id"],
        references=references,
        fact_store=store,
    )
    return dict(
        body,
        context=admitted,
        graph_sha256=sha256_hex(canonical_json_bytes(graph)),
        graph_representation=representation,
    ), original_store


def project_gplus_report(
    context: BriefSreReportContext,
    *,
    references: dict[str, Any],
    graph_representation: str = "auto",
) -> OpsInputProjection:
    working = deepcopy(references)
    retained = validate_brief_report(context)
    header = context.metadata.evidence
    if not header.formal_projection_available:
        return DiagnosedInputProjection(
            status="resource_limit",
            scope="formal_projection",
            target_bytes=REPORT_BYTES,
            failure_detail=OpsFailureDetail(
                stage="report_projection", failure_type="FormalProjectionUnavailable"
            ),
        )
    graph = None
    try:
        metadata, store, packet, working = _admit_report_source(
            context,
            retained=retained,
            references=working,
        )
        short_metadata = unpack_history_view(packet["history"], limits=NATIVE_REPORT_INPUT_LIMITS)
        short_store = NativeFactStore.from_packet(packet["fact_records"])
        graph = build_gplus_graph(
            typed_expression_catalog(short_metadata["supply"], short_store),
            representation=graph_representation,
        )
        tables = graph["schema_version"] == TABLE_GRAPH_SCHEMA
        if tables:
            shared = gplus_graph_facts(
                graph, ordered_refs=short_metadata["supply"]["provided_fact_refs"]
            )
            if canonical_json_bytes(shared.to_packet()) != canonical_json_bytes(
                short_store.to_packet()
            ):
                raise ValueError("G+ shared source fact codec differs")
            del packet["fact_records"]
        packet.update(schema_version=GPLUS_VIEW_TABLES if tables else GPLUS_VIEW, graph=graph)
        body = dict(
            schema_version=GPLUS_INPUT_TABLES if tables else GPLUS_INPUT,
            protocol=GPLUS_PROTOCOL,
            output_schema=gplus_report_policy()["output_schema"],
            context=packet,
            context_sha256=sha256_hex(encode_json(retained, REPORT_LIMITS)),
            opaque_case_id=header.opaque_case_id,
            case_snapshot_id=header.case_snapshot_id,
        )
        raw = _request(body)
        decoded, decoded_store = gplus_report_body(raw, references=working)
        if canonical_json_bytes(decoded["context"]) != canonical_json_bytes(
            metadata
        ) or canonical_json_bytes(decoded_store.to_packet()) != canonical_json_bytes(
            store.to_packet()
        ):
            raise ValueError("G+ actual source/book roundtrip differs")
        saved = encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        if (
            graph_representation == "auto"
            and graph is not None
            and (graph["schema_version"] != TABLE_GRAPH_SCHEMA)
        ):
            # Pure deterministic re-encoding, not another model call or source
            # selection. Also cover a legal graph whose whole packet overflows.
            return project_gplus_report(
                context, references=references, graph_representation="tables"
            )
        return DiagnosedInputProjection(
            status="resource_limit",
            target_bytes=REPORT_BYTES,
            failure_detail=failure_detail("report_projection", exc),
        )
    _require_brief_unchanged(context, encode_json(retained, REPORT_LIMITS))
    references.clear()
    references.update(working)
    return OpsInputProjection(
        status="ready",
        view_json=saved.decode(),
        wire_bytes=raw,
        provided_fact_refs=store.refs(),
        omitted_fact_refs=tuple(
            ref for ref in context.fact_store.refs() if ref not in store.refs()
        ),
        target_bytes=REPORT_BYTES,
        target_exceeded=len(raw) > REPORT_BYTES,
        expansion_reasons=("required_complete_gplus_closure",) if len(raw) > REPORT_BYTES else (),
    )


def bind_gplus_report(raw: bytes) -> LiveRequestBinding:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    body, _ = gplus_report_body(raw)
    header = body["context"]["evidence"]
    return LiveRequestBinding(
        opaque_case_id=body["opaque_case_id"],
        variant=GPLUS_VARIANT,
        phase="report",
        input_sha256=header["core_result_digest"],
        context_sha256=body["context_sha256"],
        prompt_schema_sha256=gplus_report_policy()["prompt_sha256"],
        request_sha256=sha256_hex(raw),
    )
