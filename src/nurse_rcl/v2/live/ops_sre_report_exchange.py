"""Offline SRE report wire projection. No transport, key read or live authority."""

# ruff: noqa: RUF001 -- Chinese prompt punctuation

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import _validate_payload, prepare_exchange
from nurse_rcl.v2.live.ops_context_view import (
    OpsInputProjection,
    expand_report_facts,
    report_input,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
    SRE_REPORT_PROTOCOL,
    SRE_VARIANT,
)
from nurse_rcl.v2.live.ops_report_exchange import EMPIRICAL_SYSTEM
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_claim_guidance import claim_scope_guide
from nurse_rcl.v2.live.ops_sre_evidence_routing import workflow_evidence_index
from nurse_rcl.v2.live.ops_sre_frozen_language import FROZEN_RANKING_POLICY
from nurse_rcl.v2.live.ops_sre_model_view import (
    REPORT_MODEL_VIEW_GUIDANCE,
    pack_model_view,
    unpack_model_view,
)
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report import OpsSreReportResponse
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding
SRE_ORGANIZATION_EXAMPLE = dict(
    summary_claim_ids=["c_later"], finding_claim_ids=["c_observed"],
    workflow_steps=[dict(step_ref="s1", observation_claim_ids=["c_observed"],
                         synthesis_claim_ids=["c_later"])],
    unresolved_claim_ids=["c_next"],
)
SRE_REPORT_SYSTEM = EMPIRICAL_SYSTEM + """
本协议输出 nurse.v2.sre-report-response/1，使用中文先汇报当前判断与保留意见，
再说明关键依据、竞争解释、实际调查和未决问题。不要重复逐项抄写字段。
总结不要逐项重抄 Top-5 或长小数；精确表格由代码保留，解释真正有区分力的依据和限制。
操作范围契约（不是任意目标列表）：
explain_priority 恰好1个目标，formal_grounds，必须含该目标ranking事实，interval=null。
compare_candidates 恰好2个不同目标，formal_grounds，competition事实必须为同序同一对，
interval=null。不能将多组候选对合并成五目标claim；多对象说明请分别生成合法claims。
claim_scope_guide 仅索引本次实际提供的事实；entries给出ranking/competition及精确目标；
source_fact_ref
及fact_refs不授予额外权限。field_layouts给可断言字段名与原JSON类型，fact_layouts将本次
实供事实映射到布局、kind、scope_basis与目标。只能断言fact.values中存在的字段；
status_codes不是同名bool字段。truth_value的"false"是字符串，不可输出布尔false。
verification_precondition_kind仅说明种类可能适用，不代表目标、时点或强度已经通过。
query_scope_metadata及候选概览是范围元信息，不是健康观测或建议的观测前提；缺少合法
前提可以留为未知，不凑建议。范围共享身份不是可执行卡片，deliveries保留实际回执。
record_groups.navigation仅保存实际概览交付的检查入口及原路径，不是诊断事实。
看到入口不等于已经执行或取得完整卡片证据，不能将其用作assertion或建议观测前提。
技术结论仍只引用实供fact_refs；已执行动作由真实步骤证明，未展开内容不能回填早期步骤。
原entries中的source_fact_ref
指向原来源事实，不是新证据。同项只提示范围相容，不豁免全部原有事实与步骤检查。
describe_observation 用 observation/frozen_observation 和匹配窗口的bound_observation事实。
synthesize_evidence 至少两条事实；explain_uncertainty 需要实际未知／局限依据，两者均为
inference/ranking_inference。explain_priority和compare_candidates也用此推断类型／强度。
propose_verification与conditional_mitigation 用recommendation/conditional_recommendation，
conditional_next_steps分区，保留合法前提；后者还需真实mitigation依据，不凭建议制造干预。
未来补查只能改变调查中的机制解释支持或不确定性，不会改变冻结正式结果；不要写
“排名可能下降／分数提高／排名区分度增强”，也不给出另一套专家排名。末尾免责声明
不能抵消前文矛盾。可说明“该结果增强对机制解释的支持，正式排名保持不变”，但仍须
有真实前提与合法作用域，不把该示例当作已观察结果。
同一陈述可以复用 claim_id，不必为每一步重复历史总结；重点汇报反馈新增或改变判断之处。
下一步建议尽量指出已有合法工具、目标和待区分结果；前提不足时如实留为缺口。
summary_claim_ids、finding_claim_ids、unresolved_claim_ids 仅组织 claims 的 ID。
claims 的每个 claim_id（包括 conditional_next_steps 建议）必须至少出现在一个组织列表；
所有组织列表引用的并集恰好等于 claims 的ID集合，不得漏接或悬空。未决／下一步建议
可放 unresolved_claim_ids；它不代替 fact_refs 和 recommendation.precondition_refs。
workflow_steps 每项为 step_ref、observation_claim_ids、synthesis_claim_ids；
observation_claim_ids 的每个事实必须在对应步骤 observed_fact_refs 中，且该步骤成功。
synthesis_claim_ids 是报告阶段事后综合，可以使用晚到事实，但不能说早先已知。
一个 claim 混用了早晚事实时，不能把它挂作早期 observation；可明确写成事后综合，
或拆成分别有依据的陈述。不要仅移动ID却保留“该步骤当时已查明”的错误文字。
导航步骤可使用空 observation_claim_ids，不为每一步凑技术结论。
所有技术结论必须经 claims 的同一断言合约。s1 等步骤只能证明过程，不能代替 f1 等事实。
无自由正文或 Markdown 字段。没有被引用的 claim 不要输出。历史公开目的不等于事实。
步骤未执行、被拒绝、被中断和未记录回应须照实保留；不要补写理由或伪造干预。
record_groups 按原记录绑定事实，record_path 保留 AND/OR 和必要／反证条件所在层级。
workflow_evidence_index 按 record_group_index 给出精确记录的合法 observation_step_refs；
fact_refs_location 指向原记录的事实列表，source_location 和 reuse_actions_location 保留来源。
跨记录 claim 必须取合法步骤的交集；没有共同步骤只能明确做报告综合。数值相同不等于
来源相同，不可把旧回执或晚到事实挂成本步观测。成功工具返回 unknown 也是合法观测。
proposed 建议的 possible_outcomes.interpretation 可用“若检查确认……则……”描述未来结果；
这不表示已经确认，不得夹带“已修复／已完成／已确认根因”等既成事实或因果结论。
fact_tables 按精确来源和字段布局共享 common；仍按原规则逐行还原，每个别名独立保留。
表格分组只减少重复字段，不代表多份独立证据，不改变事实作用域或步骤关联。
overview_not_full_judgment 是概览，不能当完整判断；计数是元信息，不是独立支持。
同源卡片或核心复用不是独立证据，core_published 不是 formal_used。
正式排名、精确分数及别名由代码展示；解释为何支持以及为何仍有限制。
精确类型例：原值 "0.25" 必须仍是字符串，不能写成数字 0.25；例值不是提供的事实。
单目标 claim 只引用覆盖该目标的事实。双目标事实要求同时保留两端，不能塞进单目标解释。
compare_candidates 需要 formal_grounds 的 competition 事实及其同序两端，不能用两条
补充观测自制正式排名。建议 precondition_refs 必须引用实际前提且包含于 fact_refs；
propose_verification 使用 proposed/read_only_diagnostic_check，提供 true/false/unknown
三种可能结果及安全限制；前提只覆盖一端时应拆分建议，不替另一端补前提。
每个 precondition_refs 的事实 kind 必须是 observation/predicate/hypothesis，且每个前提
都覆盖建议的全部目标；ranking、uncertainty 或机制未知数量不能替代这些观测前提。
缺合格前提时保留有限结论／未知，不强行编造建议；建议或补查不能改变冻结的正式排名。
describe_observation 的全部 fact_refs 必须是 scope_basis=bound_observation，且窗口匹配；
不能混入 frozen_assessment 概览计数或全案限制。范围判断／缺口应按适用的推断操作单列。
断言只复制该事实实际存在的字段、原值与类型；同一目标的一条阴性信号不能排除整个服务。
说明实际检查的意义和下一步，不要求凑足检查次数。机械检查不建立语义真值。
下面只是组织示意，不是本案证据、步骤或必须生成的claims：假设自行生成了c_observed、
c_later、c_next三个合法claim，c_observed的全部事实在实际s1已交付，c_later明确写作
事后综合，c_next是有合法观测前提的未执行建议，可这样组织。不要复制示例名称充当证据，
只使用当前请求的实际步骤与事实，并按实际claims和意义组织，不机械套用此示例：
""" + canonical_json_bytes(SRE_ORGANIZATION_EXAMPLE).decode() + REPORT_MODEL_VIEW_GUIDANCE
SRE_REPORT_OUTPUT_SCHEMA = canonical_json_bytes(OpsSreReportResponse.model_json_schema()).decode()
SRE_REPORT_PROMPT_SHA256 = sha256_hex(encode_json(dict(protocol=SRE_REPORT_PROTOCOL,
    system=SRE_REPORT_SYSTEM, output_schema=SRE_REPORT_OUTPUT_SCHEMA,
    output_policy=OUTPUT_POLICY, max_tokens=MAX_OUTPUT_TOKENS), REPORT_LIMITS))


def sre_report_projection(
    context: OpsSreReportContext, *, references: dict[str, Any] | None = None,
) -> OpsInputProjection:
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    if context.profile != "hermes-sre-tools/4":
        raise ValueError("SRE report projection profile differs")
    pin = context.context_sha256
    local = references if references is not None else SreReferenceBook(
        snapshot=context.evidence.case_snapshot_id,
        run_id=sha256_hex(b"detached-preview" +
                         context.evidence.case_snapshot_id.encode())).to_state()
    updated: dict[str, Any] | None = None

    def wire(original: dict[str, Any]) -> dict[str, Any]:
        nonlocal updated
        view = dict(original, schema_version="nurse.v2.sre-report-input/6",
            profile=context.profile, steps=context.steps, record_groups=context.record_groups,
            claim_scope_guide=claim_scope_guide(expand_report_facts(original)),
            frozen_ranking_policy=FROZEN_RANKING_POLICY,
            workflow_evidence_index=workflow_evidence_index(context.record_groups, context.steps),
            critical_fact_refs=context.critical_fact_refs,
            unexpanded_critical_overviews=context.unexpanded_critical_overviews,
            termination=context.termination)
        # Compact column names are data until decoded. Scan the expanded facts,
        # as well as every new public-memory field. No blanket whitelist.
        outbound = {k: v for k, v in view.items() if k != "fact_tables"}
        outbound["facts"] = [f.model_dump(mode="json") for f in expand_report_facts(view)]
        _require_outbound(outbound, case=context.evidence.opaque_case_id,
                          snapshot=context.evidence.case_snapshot_id)
        packet, updated = pack_model_view(view, references=local,
            case_id=context.evidence.opaque_case_id, snapshot=context.evidence.case_snapshot_id,
            index=9, report=True)
        body = dict(protocol=SRE_REPORT_PROTOCOL, context=packet, context_sha256=pin,
                    output_schema=SRE_REPORT_OUTPUT_SCHEMA,
                    opaque_case_id=context.evidence.opaque_case_id,
                    case_snapshot_id=context.evidence.case_snapshot_id)
        exchange = prepare_exchange(model=ARK_MODEL, max_output_tokens=MAX_OUTPUT_TOKENS,
            limits=REPORT_LIMITS, messages=[dict(role="system", content=SRE_REPORT_SYSTEM),
                dict(role="user", content=encode_json(body, REPORT_LIMITS).decode())])
        return dict(decode_json_object(exchange.payload_bytes, REPORT_LIMITS),
                    thinking={"type": "disabled"})

    result = report_input(context=context.evidence, make_wire=wire, soft_targets=True,
                          group_by_provenance=True)
    if result.wire_bytes is None:
        return result
    # Optional candidates may have been tried but not selected. Rebuild only
    # the selected view from the unchanged initial book, binding its exact map
    # without retaining every speculative candidate's full reference table.
    assert result.view_json is not None
    selected = wire(decode_json_object(result.view_json.encode(), REPORT_LIMITS))
    if encode_json(selected, REPORT_LIMITS) != result.wire_bytes:
        raise ValueError("SRE selected report wire differs")
    assert updated is not None
    actual = sre_report_body(result.wire_bytes, references=updated)["context"]
    if references is not None:
        references.clear()
        references.update(updated)
    return replace(result, view_json=encode_json(actual, REPORT_LIMITS).decode())


def sre_report_body(
    raw: bytes, *, references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate protocol/transport identity; source admission belongs to the owner."""
    payload = decode_json_object(raw, REPORT_LIMITS)
    base = {k: v for k, v in payload.items() if k != "thinking"}
    if payload.get("thinking") != {"type": "disabled"} or _validate_payload(base) is not None or (
        base["model"] != ARK_MODEL or base["max_tokens"] != MAX_OUTPUT_TOKENS
        or base["temperature"] != 0 or encode_json(payload, REPORT_LIMITS) != raw
    ):
        raise ExchangeBoundaryError("sre_report_wire_settings")
    messages = base["messages"]
    if len(messages) != 2 or messages[0] != {"role": "system", "content": SRE_REPORT_SYSTEM} or (
        messages[1]["role"] != "user"
    ):
        raise ExchangeBoundaryError("sre_report_wire_messages")
    body = decode_json_object(messages[1]["content"].encode(), REPORT_LIMITS)
    if set(body) != {"protocol", "context", "context_sha256", "output_schema",
                     "opaque_case_id", "case_snapshot_id"} or (
        body["protocol"] != SRE_REPORT_PROTOCOL
        or body["output_schema"] != SRE_REPORT_OUTPUT_SCHEMA or type(body["context"]) is not dict
    ):
        raise ExchangeBoundaryError("sre_report_wire_body")
    view = unpack_model_view(body["context"], references=references,
        case_id=body["opaque_case_id"], snapshot=body["case_snapshot_id"], index=9, report=True)
    body["context"] = view
    if (view.get("opaque_case_id"), view.get("case_snapshot_id")) != (
        body["opaque_case_id"], body["case_snapshot_id"],
    ):
        raise ExchangeBoundaryError("sre_report_case_identity")
    if view.get("schema_version") != "nurse.v2.sre-report-input/6" or (
        view.get("profile") != "hermes-sre-tools/4"
        or view.get("frozen_ranking_policy") != FROZEN_RANKING_POLICY
        or any(type(view.get(k)) is not list for k in (
            "fact_tables", "steps", "record_groups", "critical_fact_refs",
            "unexpanded_critical_overviews"))
    ):
        raise ExchangeBoundaryError("sre_report_wire_context")
    require_pin(body["context_sha256"])
    require_pin(view["core_result_digest"])
    require_pin(view["case_snapshot_id"])
    expand_report_facts(view)
    if view.get("claim_scope_guide") != claim_scope_guide(expand_report_facts(view)):
        raise ExchangeBoundaryError("sre_report_scope_guide")
    if view.get("workflow_evidence_index") != workflow_evidence_index(
        view["record_groups"], view["steps"],
    ):
        raise ExchangeBoundaryError("sre_report_workflow_guide")
    return body


def bind_sre_report(raw: bytes) -> LiveRequestBinding:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    body = sre_report_body(raw)
    view = body["context"]
    return LiveRequestBinding(opaque_case_id=view["opaque_case_id"], variant=SRE_VARIANT,
        phase="report", input_sha256=view["core_result_digest"],
        context_sha256=body["context_sha256"], prompt_schema_sha256=SRE_REPORT_PROMPT_SHA256,
        request_sha256=sha256_hex(raw))
