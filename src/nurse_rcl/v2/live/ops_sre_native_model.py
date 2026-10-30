"""D12 report-only model packet: complete native tree, bounded metadata, local IDs."""

from __future__ import annotations

from typing import Any

from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_history_view import pack_history_view, unpack_history_view
from nurse_rcl.v2.live.ops_sre_model_view import _book, _prefix
from nurse_rcl.v2.live.ops_sre_native_report import NATIVE_REPORT_INPUT_LIMITS
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook, _translate
from nurse_rcl.v2.live.ops_sre_report_bounds import _measure

NATIVE_MODEL_SCHEMA = "nurse.v2.sre-model-view/3"


def _parts(metadata: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    if "fact_records" in metadata or "fact_tables" in metadata or "facts" in metadata:
        raise ValueError("SRE native model fact position differs")
    view = dict(metadata=metadata, fact_records=store.to_packet())
    _measure(view, node_limit=100_000, boundary="report_input_nodes")
    encode_json(view, NATIVE_REPORT_INPUT_LIMITS)
    return view


def pack_native_model_view(
    metadata: dict[str, Any], *, store: NativeFactStore, references: dict[str, Any],
    case_id: str, snapshot: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a new book only after the complete native packet succeeds."""
    book = _book(references, snapshot)
    value = _parts(metadata, store)
    short = _translate(value, book=book, visible_at=9, case_id=case_id,
        encode=True, limits=NATIVE_REPORT_INPUT_LIMITS, native_facts=True)
    prefix, _ = _prefix(book, 9)
    packet = dict(schema_version=NATIVE_MODEL_SCHEMA, kind="report", reference_prefix=prefix,
        history=pack_history_view(short["metadata"], limits=NATIVE_REPORT_INPUT_LIMITS),
        fact_records=short["fact_records"])
    encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    return packet, book.to_state()


def native_model_book(
    packet: dict[str, Any], *, references: dict[str, Any] | None, snapshot: str,
) -> SreReferenceBook | None:
    encode_json(packet, NATIVE_REPORT_INPUT_LIMITS)
    if set(packet) != {"schema_version", "kind", "reference_prefix", "history", "fact_records"} or (
        packet["schema_version"] != NATIVE_MODEL_SCHEMA or packet["kind"] != "report"
    ):
        raise ValueError("SRE native model identity differs")
    prefix = packet["reference_prefix"]
    if type(prefix) is not dict or set(prefix) != {
        "run_id", "visible_at", "entry_count", "state_sha256",
    } or type(prefix["visible_at"]) is not int or prefix["visible_at"] != 9 or (
        type(prefix["entry_count"]) is not int or not 0 <= prefix["entry_count"] <= 4096
    ):
        raise ValueError("SRE native reference prefix differs")
    require_pin(prefix["run_id"])
    require_pin(prefix["state_sha256"])
    if references is None:
        return None  # Transport cannot grant provenance; owner uses the full local book.
    expected, book = _prefix(_book(references, snapshot), 9)
    if prefix != expected:
        raise ValueError("SRE native reference prefix binding differs")
    return book


def unpack_native_model_view(
    packet: dict[str, Any], *, references: dict[str, Any] | None, case_id: str, snapshot: str,
) -> tuple[dict[str, Any], NativeFactStore]:
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    book = native_model_book(packet, references=references, snapshot=snapshot)
    metadata = unpack_history_view(packet["history"], limits=NATIVE_REPORT_INPUT_LIMITS)
    store = NativeFactStore.from_packet(packet["fact_records"])
    value = _parts(metadata, store)
    _require_outbound(value, case=case_id, snapshot=snapshot)
    if book is not None:
        value = _translate(value, book=book, visible_at=9, case_id=case_id, encode=False,
                           limits=NATIVE_REPORT_INPUT_LIMITS, native_facts=True)
        metadata = value["metadata"]
        store = NativeFactStore.from_packet(value["fact_records"])
        _parts(metadata, store)
    return metadata, store
