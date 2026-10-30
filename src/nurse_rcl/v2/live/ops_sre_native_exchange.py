"""Complete D12 report request and pure binding; never reads keys or sends HTTP."""

# ruff: noqa: RUF001 -- Chinese model contract

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import _validate_payload, prepare_exchange
from nurse_rcl.v2.live.ops_context_view import (
    REPORT_BYTES,
    DiagnosedInputProjection,
    OpsInputProjection,
)
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY, SRE_VARIANT
from nurse_rcl.v2.live.ops_failure import failure_detail
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_claim_guidance import claim_scope_guide
from nurse_rcl.v2.live.ops_sre_evidence_routing import workflow_evidence_index
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_frozen_language import FROZEN_RANKING_POLICY
from nurse_rcl.v2.live.ops_sre_model_view import REPORT_MODEL_VIEW_GUIDANCE
from nurse_rcl.v2.live.ops_sre_native_model import pack_native_model_view, unpack_native_model_view
from nurse_rcl.v2.live.ops_sre_native_report import (
    NATIVE_REPORT_INPUT_LIMITS,
    NativeReportMetadata,
    NativeSreReportContext,
    validate_native_report,
)
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report_exchange import SRE_REPORT_OUTPUT_SCHEMA, SRE_REPORT_SYSTEM

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

NATIVE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/8"
NATIVE_REPORT_INPUT_SCHEMA = "nurse.v2.sre-report-input/7"
NATIVE_GUIDANCE = REPORT_MODEL_VIEW_GUIDANCE.replace("sre-model-view/2", "sre-model-view/3") + """
本版完整事实在context.fact_records，schema为nurse.v2.sre-fact-records/2。
fact_order保持原事实顺序；alias_index仅索引每个f别名的[record下标,source_path]，
不是另一份证据。按固定来源树及局部默认值解释全部事实，不需要生成全表副本。
history仅共享元信息；history.content.evidence包含唯一正式排名、core_result_digest、
案例／快照和状态。steps、record_groups及指南在history.content中。
investigation_input_schema说明实际调查来源版本：旧版保存材料仅可做离线诊断，
不能冒称新版执行或新增检查。交付差异是过程元信息，不是独立观测或支持票数。
"""
NATIVE_REPORT_SYSTEM = SRE_REPORT_SYSTEM.replace(REPORT_MODEL_VIEW_GUIDANCE, NATIVE_GUIDANCE)
NATIVE_REPORT_PROMPT_SHA256 = sha256_hex(encode_json(dict(protocol=NATIVE_REPORT_PROTOCOL,
    system=NATIVE_REPORT_SYSTEM, output_schema=SRE_REPORT_OUTPUT_SCHEMA,
    output_policy=OUTPUT_POLICY, max_tokens=MAX_OUTPUT_TOKENS), REPORT_LIMITS))
_GUIDES = {"claim_scope_guide", "workflow_evidence_index", "frozen_ranking_policy"}


def native_report_projection(
    context: NativeSreReportContext, *, references: dict[str, Any] | None = None,
) -> OpsInputProjection:
    validated = validate_native_report(context)
    header = context.metadata.evidence
    if not header.formal_projection_available:
        return OpsInputProjection(status="resource_limit", scope="formal_projection",
                                  target_bytes=REPORT_BYTES)
    pin = sha256_hex(encode_json(validated, REPORT_LIMITS))
    local = references if references is not None else SreReferenceBook(
        snapshot=header.case_snapshot_id,
        run_id=sha256_hex(b"detached-native-preview" + header.case_snapshot_id.encode())).to_state()
    try:
        metadata = dict(validated["metadata"], schema_version=NATIVE_REPORT_INPUT_SCHEMA,
            claim_scope_guide=claim_scope_guide(context.fact_store.iter_facts()),
            workflow_evidence_index=workflow_evidence_index(
                context.metadata.record_groups, context.metadata.steps),
            frozen_ranking_policy=FROZEN_RANKING_POLICY)
        packet, updated = pack_native_model_view(metadata, store=context.fact_store,
            references=local, case_id=header.opaque_case_id, snapshot=header.case_snapshot_id)
        body = dict(protocol=NATIVE_REPORT_PROTOCOL, context=packet, context_sha256=pin,
            output_schema=SRE_REPORT_OUTPUT_SCHEMA, opaque_case_id=header.opaque_case_id,
            case_snapshot_id=header.case_snapshot_id)
        # Check decoded body as well as actual escaped outer request.
        content = encode_json(body, REPORT_LIMITS).decode()
        exchange = prepare_exchange(model=ARK_MODEL, max_output_tokens=MAX_OUTPUT_TOKENS,
            limits=REPORT_LIMITS, messages=[dict(role="system", content=NATIVE_REPORT_SYSTEM),
                                          dict(role="user", content=content)])
        payload = dict(decode_json_object(exchange.payload_bytes, REPORT_LIMITS),
                       thinking={"type": "disabled"})
        wire = encode_json(payload, REPORT_LIMITS)
        recovered, store = native_report_body(wire, references=updated)
        # Native representation remains native in the owner's saved input view.
        view = dict(metadata=recovered["context"], fact_records=store.to_packet())
        raw = encode_json(view, NATIVE_REPORT_INPUT_LIMITS)
        if canonical_json_bytes(recovered["context"]) != canonical_json_bytes(metadata) or (
            canonical_json_bytes(store.to_packet()) != canonical_json_bytes(
                context.fact_store.to_packet())
        ):
            raise ValueError("SRE native model roundtrip differs")
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        return DiagnosedInputProjection(status="resource_limit", target_bytes=REPORT_BYTES,
            failure_detail=failure_detail("report_projection", exc))
    if references is not None:
        references.clear()
        references.update(updated)
    exceeded = len(wire) > REPORT_BYTES
    return OpsInputProjection(status="ready", view_json=raw.decode(), wire_bytes=wire,
        provided_fact_refs=context.fact_store.refs(), target_bytes=REPORT_BYTES,
        target_exceeded=exceeded,
        expansion_reasons=("required_semantics_or_complete_envelope",) if exceeded else ())


def native_report_body(
    raw: bytes, *, references: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], NativeFactStore]:
    from nurse_rcl.v2.dynamic_verification.safety import require_pin

    payload = decode_json_object(raw, REPORT_LIMITS)
    base = {k: v for k, v in payload.items() if k != "thinking"}
    if payload.get("thinking") != {"type": "disabled"} or _validate_payload(base) is not None or (
        base["model"] != ARK_MODEL or base["max_tokens"] != MAX_OUTPUT_TOKENS
        or base["temperature"] != 0 or encode_json(payload, REPORT_LIMITS) != raw
    ):
        raise ExchangeBoundaryError("sre_native_wire_settings")
    messages = base["messages"]
    if len(messages) != 2 or messages[0] != {"role": "system", "content": NATIVE_REPORT_SYSTEM} or (
        messages[1]["role"] != "user"
    ):
        raise ExchangeBoundaryError("sre_native_wire_messages")
    body = decode_json_object(messages[1]["content"].encode(), REPORT_LIMITS)
    if set(body) != {"protocol", "context", "context_sha256", "output_schema",
                     "opaque_case_id", "case_snapshot_id"} or (
        body["protocol"] != NATIVE_REPORT_PROTOCOL
        or body["output_schema"] != SRE_REPORT_OUTPUT_SCHEMA
        or type(body["context"]) is not dict
    ):
        raise ExchangeBoundaryError("sre_native_wire_body")
    metadata, store = unpack_native_model_view(body["context"], references=references,
        case_id=body["opaque_case_id"], snapshot=body["case_snapshot_id"])
    if set(metadata) != set(NativeReportMetadata.model_fields) | _GUIDES or (
        metadata["schema_version"] != NATIVE_REPORT_INPUT_SCHEMA
        or metadata["frozen_ranking_policy"] != FROZEN_RANKING_POLICY
    ):
        raise ExchangeBoundaryError("sre_native_wire_context")
    fields = {k: v for k, v in metadata.items() if k not in _GUIDES}
    fields["schema_version"] = "nurse.v2.sre-report-context/5"
    admitted = NativeReportMetadata.model_validate_json(
        encode_json(fields, NATIVE_REPORT_INPUT_LIMITS), strict=True)
    if (admitted.evidence.opaque_case_id, admitted.evidence.case_snapshot_id) != (
        body["opaque_case_id"], body["case_snapshot_id"],
    ):
        raise ExchangeBoundaryError("sre_native_wire_case")
    require_pin(body["context_sha256"])
    if metadata["claim_scope_guide"] != claim_scope_guide(store.iter_facts()) or (
        metadata["workflow_evidence_index"] != workflow_evidence_index(
            metadata["record_groups"], metadata["steps"])
    ):
        raise ExchangeBoundaryError("sre_native_wire_guide")
    return dict(body, context=metadata), store


def bind_native_report(raw: bytes) -> LiveRequestBinding:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    body, _ = native_report_body(raw)
    header = body["context"]["evidence"]
    return LiveRequestBinding(opaque_case_id=header["opaque_case_id"], variant=SRE_VARIANT,
        phase="report", input_sha256=header["core_result_digest"],
        context_sha256=body["context_sha256"], prompt_schema_sha256=NATIVE_REPORT_PROMPT_SHA256,
        request_sha256=sha256_hex(raw))
