"""Explicit opt-in normal report projection. No send mode or default routing.

The existing source admission and complete source metadata stay authoritative.
Typed companions are an additional complete view, never permission inheritance.
"""

from __future__ import annotations

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
from nurse_rcl.v2.live.ops_failure import failure_detail
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_exchange import _validate_report_parts, brief_wire_parts
from nurse_rcl.v2.live.ops_sre_brief_report import (
    BriefSreReportContext,
    _require_brief_unchanged,
    validate_brief_report,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_history_view import pack_history_view, unpack_history_view
from nurse_rcl.v2.live.ops_sre_model_view import _book, _prefix
from nurse_rcl.v2.live.ops_sre_native_report import NATIVE_REPORT_INPUT_LIMITS
from nurse_rcl.v2.live.ops_sre_references import _translate
from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog
from nurse_rcl.v2.live.ops_sre_typed_report_contract import (
    TYPED_REPORT_INPUT,
    TYPED_REPORT_PROTOCOL,
    TYPED_REPORT_VIEW,
    typed_policy_for_output_schema,
    typed_report_policy,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding


def _wire_catalog(supply: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    """Build from the SAME short native store; model never has to copy versions."""
    catalog = typed_expression_catalog(supply, store)
    return dict(
        schema_version=catalog["schema_version"],
        expressions=[row["expression"] for row in catalog["rows"]],
        sources={
            ref: {k: v for k, v in source.items() if k != "source_sha256"}
            for ref, source in catalog["sources"].items()
        },
        field_scope=catalog["field_scope"],
        condition_boundaries=catalog["condition_boundaries"],
        authority=catalog["authority"],
    )


def _request(body: dict[str, Any]) -> bytes:
    policy = typed_policy_for_output_schema(body.get("output_schema"))
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


def _admit_report_source(
    context: BriefSreReportContext, *, retained: dict[str, Any], references: dict[str, Any]
) -> tuple[dict[str, Any], NativeFactStore, dict[str, Any], dict[str, Any]]:
    """Common source admission, independent of typed/G+ output representation.

    Return complete original metadata/store, a lossless short-reference packet,
    and the updated private book. No caller-owned book is mutated here.
    """
    _require_brief_unchanged(context, encode_json(retained, REPORT_LIMITS))
    metadata, store = brief_wire_parts(context)
    _validate_report_parts(metadata, store)
    header = context.metadata.evidence
    book = _book(references, header.case_snapshot_id)
    short = _translate(
        dict(metadata=metadata, fact_records=store.to_packet()),
        book=book,
        visible_at=9,
        case_id=header.opaque_case_id,
        encode=True,
        limits=NATIVE_REPORT_INPUT_LIMITS,
        native_facts=True,
        brief_fields=True,
    )
    prefix, _ = _prefix(book, 9)
    packet = dict(
        kind="report",
        reference_prefix=prefix,
        history=pack_history_view(short["metadata"], limits=NATIVE_REPORT_INPUT_LIMITS),
        fact_records=short["fact_records"],
    )
    return metadata, store, packet, book.to_state()


def _report_source_parts(
    packet: dict[str, Any],
    *,
    case: str,
    snapshot: str,
    references: dict[str, Any] | None,
    fact_store: NativeFactStore | None = None,
) -> tuple[dict[str, Any], NativeFactStore]:
    """Exact shared packet/source/alias/outbound checks, never permission inference."""
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    require_pin(snapshot)
    encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    prefix = packet["reference_prefix"]
    if (
        type(prefix) is not dict
        or set(prefix) != {"run_id", "visible_at", "entry_count", "state_sha256"}
        or type(prefix["visible_at"]) is not int
        or prefix["visible_at"] != 9
        or type(prefix["entry_count"]) is not int
        or not 0 <= prefix["entry_count"] <= 4096
    ):
        raise ValueError("report reference prefix differs")
    require_pin(prefix["run_id"])
    require_pin(prefix["state_sha256"])
    metadata = unpack_history_view(packet["history"], limits=NATIVE_REPORT_INPUT_LIMITS)
    store = (
        NativeFactStore.from_packet(packet["fact_records"]) if fact_store is None else fact_store
    )
    _require_outbound(
        dict(metadata=metadata, fact_records=store.to_packet()),
        case=case,
        snapshot=snapshot,
    )
    _validate_report_parts(metadata, store, short_refs=True)
    if references is not None:
        expected, book = _prefix(_book(references, snapshot), 9)
        if prefix != expected:
            raise ValueError("report reference binding differs")
        original = _translate(
            dict(metadata=metadata, fact_records=store.to_packet()),
            book=book,
            visible_at=9,
            case_id=case,
            encode=False,
            limits=NATIVE_REPORT_INPUT_LIMITS,
            native_facts=True,
            brief_fields=True,
        )
        metadata = original["metadata"]
        store = NativeFactStore.from_packet(original["fact_records"])
        _validate_report_parts(metadata, store)
    header = metadata["evidence"]
    if (header["opaque_case_id"], header["case_snapshot_id"]) != (case, snapshot):
        raise ValueError("report case identity differs")
    return metadata, store


def typed_report_body(
    raw: bytes, *, references: dict[str, Any] | None = None
) -> tuple[dict[str, Any], NativeFactStore]:
    """Validate the new exact wire; with trusted book, resolve original metadata.

    Without references this is only short wire-domain validation, not source
    admission or proof that context_sha256 is a trusted external source pin.
    """
    if type(raw) is not bytes:
        raise TypeError("typed report wire requires exact bytes")
    payload = decode_json_object(raw, REPORT_LIMITS)
    base = {k: v for k, v in payload.items() if k != "thinking"}
    if (
        payload.get("thinking") != {"type": "disabled"}
        or _validate_payload(base) is not None
        or base["model"] != ARK_MODEL
        or base["temperature"] != 0
        or encode_json(payload, REPORT_LIMITS) != raw
    ):
        raise ValueError("typed report wire settings differ")
    messages = base["messages"]
    if (
        len(messages) != 2
        or messages[1]["role"] != "user"
    ):
        raise ValueError("typed report active contract differs")
    body = decode_json_object(messages[1]["content"].encode(), REPORT_LIMITS)
    policy = typed_policy_for_output_schema(body.get("output_schema"))
    if base["max_tokens"] != policy["max_tokens"] or messages[0] != dict(
        role="system", content=policy["system"]
    ):
        raise ValueError("typed report active contract differs")
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
        or body["schema_version"] != TYPED_REPORT_INPUT
        or body["protocol"] != TYPED_REPORT_PROTOCOL
        or body["output_schema"] != policy["output_schema"]
    ):
        raise ValueError("typed report input contract differs")
    require_pin(body["context_sha256"])
    require_pin(body["case_snapshot_id"])
    packet = body["context"]
    encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    if (
        set(packet)
        != {
            "schema_version",
            "kind",
            "reference_prefix",
            "history",
            "fact_records",
            "typed_catalog",
        }
        or packet["schema_version"] != TYPED_REPORT_VIEW
        or packet["kind"] != "report"
    ):
        raise ValueError("typed report model view differs")
    metadata, store = _report_source_parts(
        packet,
        case=body["opaque_case_id"],
        snapshot=body["case_snapshot_id"],
        references=None,
    )
    if canonical_json_bytes(packet["typed_catalog"]) != canonical_json_bytes(
        _wire_catalog(metadata["supply"], store)
    ):
        raise ValueError("typed report full source catalog differs")
    if references is not None:
        metadata, store = _report_source_parts(
            packet,
            case=body["opaque_case_id"],
            snapshot=body["case_snapshot_id"],
            references=references,
        )
    header = metadata["evidence"]
    if (header["opaque_case_id"], header["case_snapshot_id"]) != (
        body["opaque_case_id"],
        body["case_snapshot_id"],
    ):
        raise ValueError("typed report case identity differs")
    return dict(
        body, context=metadata,
        response_schema=policy["response_schema"]["properties"]["schema_version"]["const"],
    ), store


def project_typed_report(
    context: BriefSreReportContext, *, references: dict[str, Any]
) -> OpsInputProjection:
    """Public source-admitting projection, never an old-owner restore or send."""
    retained = validate_brief_report(context)
    return _project_admitted_typed_report(context, retained=retained, references=references)


def _project_admitted_typed_report(
    context: BriefSreReportContext,
    *,
    retained: dict[str, Any],
    references: dict[str, Any],
) -> OpsInputProjection:
    """Owner-only path after this invocation's unchanged source admission."""
    _require_brief_unchanged(context, encode_json(retained, REPORT_LIMITS))
    header = context.metadata.evidence
    if not header.formal_projection_available:
        return OpsInputProjection(
            status="resource_limit", scope="formal_projection", target_bytes=REPORT_BYTES
        )
    try:
        metadata, store, source_packet, updated = _admit_report_source(
            context,
            retained=retained,
            references=references,
        )
        short_metadata = unpack_history_view(
            source_packet["history"],
            limits=NATIVE_REPORT_INPUT_LIMITS,
        )
        short_store = NativeFactStore.from_packet(source_packet["fact_records"])
        packet = dict(
            source_packet,
            schema_version=TYPED_REPORT_VIEW,
            typed_catalog=_wire_catalog(short_metadata["supply"], short_store),
        )
        encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
        raw = _request(
            dict(
                schema_version=TYPED_REPORT_INPUT,
                protocol=TYPED_REPORT_PROTOCOL,
                context=packet,
                context_sha256=sha256_hex(encode_json(retained, REPORT_LIMITS)),
                output_schema=typed_report_policy()["output_schema"],
                opaque_case_id=header.opaque_case_id,
                case_snapshot_id=header.case_snapshot_id,
            )
        )
        actual, actual_store = typed_report_body(raw, references=updated)
        if canonical_json_bytes(actual["context"]) != canonical_json_bytes(
            metadata
        ) or canonical_json_bytes(actual_store.to_packet()) != canonical_json_bytes(
            store.to_packet()
        ):
            raise ValueError("typed report actual-book roundtrip differs")
        saved = encode_json(
            dict(
                metadata=metadata,
                fact_records=store.to_packet(),
                typed_catalog=typed_expression_catalog(metadata["supply"], store),
            ),
            NATIVE_REPORT_INPUT_LIMITS,
        )
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        return DiagnosedInputProjection(
            status="resource_limit",
            target_bytes=REPORT_BYTES,
            failure_detail=failure_detail("report_projection", exc),
        )
    _require_brief_unchanged(context, encode_json(retained, REPORT_LIMITS))
    references.clear()
    references.update(updated)
    exceeded = len(raw) > REPORT_BYTES
    return OpsInputProjection(
        status="ready",
        view_json=saved.decode(),
        wire_bytes=raw,
        provided_fact_refs=store.refs(),
        omitted_fact_refs=tuple(
            ref for ref in context.fact_store.refs() if ref not in store.refs()
        ),
        target_bytes=REPORT_BYTES,
        target_exceeded=exceeded,
        expansion_reasons=("required_semantics_or_complete_envelope",) if exceeded else (),
    )


def bind_typed_report(raw: bytes) -> LiveRequestBinding:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    body, _ = typed_report_body(raw)
    header = body["context"]["evidence"]
    return LiveRequestBinding(
        opaque_case_id=header["opaque_case_id"],
        variant="v2-ops-sre-typed-report/1",
        phase="report",
        input_sha256=header["core_result_digest"],
        context_sha256=body["context_sha256"],
        prompt_schema_sha256=typed_policy_for_output_schema(body["output_schema"])["prompt_sha256"],
        request_sha256=sha256_hex(raw),
    )
