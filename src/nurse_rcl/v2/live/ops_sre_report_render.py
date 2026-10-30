"""Chinese, summary-first report; precise RCL and process remain code-owned."""

# ruff: noqa: RUF001 -- Chinese report punctuation

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_render import _text
from nurse_rcl.v2.live.ops_report_schema import OpsClaimCheck
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_native_checks import check_native_sre_report
from nurse_rcl.v2.live.ops_sre_native_report import (
    NativeFactLookup,
    NativeReportMetadata,
    NativeSreReportContext,
)
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report import OpsSreReportResponse, check_sre_report
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext

RENDER_POLICY = "sre-readable-pair/2"

_LIMIT_TEXT = {
    "collection_coverage_unknown": "采集覆盖程度未知",
    "no_confirmed_causality": "现有证据未确认因果",
    "omitted_details_remain_in_complete_core": "省略细节仍保存在完整核心结果",
    "pre_ranking_semantics_disabled": "解释不得改变正式排名",
    "service_call_not_confirmed_failure": "观测到调用关系不等于确认依赖故障",
    "missing_paid_latency_symptom": "缺少实际测得的延迟症状",
    "necessary_predicate_unknown": "必要条件仍有未知项",
    "all_necessary_conditions_true_event_role_binding_required":
        "必要条件成立仍须核对事件角色绑定",
    "absolute_resolution_unresolved": "绝对时间分辨率尚未确定",
    "episode_support_not_causal_confirmation": "异常时段提供支持但不确认因果",
    "overview_not_full_judgment": "当前材料为概览，不是完整机制判断",
    "observable_contract_not_causal_confirmation": "可观测机制规则不是因果证明",
    "unmeasured_conditions_remain_unknown": "尚未测量的条件仍为未知",
    "delivered_section_not_full_judgment": "仅交付卡片的部分内容，不能代替完整判断",
    "onset_interval_not_causal_order": "起点区间本身不能证明因果先后",
    "query_scope_not_health_observation": "查询范围与覆盖信息不是系统健康观测",
    "aggregation_not_independent_support": "聚合信息不是新增独立支持",
    "publication_not_formal_use": "已发布记录不等于已用于正式评分",
    "summary_cannot_prove_episode_or_normality": "摘要不能证明完整异常时段或正常状态",
    "next_checks_are_options_not_a_mandatory_sequence": "可用检查是选项，不是固定流程",
    "caller_latency_not_callee_self_time": "调用方感知延迟不是被调用方自耗时",
    "call_direction_not_symptom_direction_or_causality": "调用方向不等于症状传播方向或因果",
    "not_measured": "尚未测量",
    "missing_source_mechanism_witness": "缺少源端完整机制见证",
}

# Only these exact codes are presentation-wide usage rules. Unknown codes and
# incident-specific missing observations remain concrete caveats in the main text.
_GENERAL_RULES = frozenset({
    "aggregation_not_independent_support", "publication_not_formal_use",
    "next_checks_are_options_not_a_mandatory_sequence", "pre_ranking_semantics_disabled",
    "no_confirmed_causality", "omitted_details_remain_in_complete_core",
    "episode_support_not_causal_confirmation", "overview_not_full_judgment",
    "observable_contract_not_causal_confirmation", "unmeasured_conditions_remain_unknown",
    "delivered_section_not_full_judgment", "onset_interval_not_causal_order",
    "query_scope_not_health_observation", "summary_cannot_prove_episode_or_normality",
    "caller_latency_not_callee_self_time", "call_direction_not_symptom_direction_or_causality",
    "candidate_overview_not_full_judgment",
})
_LIMIT_TEXT["candidate_overview_not_full_judgment"] = "候选状态摘要不是完整机制核验"


def _query_scope_lines(facts: Mapping[str, OpsReportFact]) -> list[str]:
    lines = ["### 查询范围（不是健康判断）", "",
        "以下只描述实际查询窗口、候选域和摘要发布范围；未读目录不表示窗口未知，"
        "也不能由范围记录的 unknown 推断服务健康状态。", ""]
    boundaries: dict[str, list[str]] = {}
    scope_refs = []
    for item in facts.values():
        if item.origin != "query_scope_metadata":
            continue
        scope_refs.append(item.fact.fact_ref)
        values = {v.name: v.value for v in item.fact.values}
        if item.source_path.endswith(("baseline_window", "fault_window")) and (
            "start_unix_ns" in values and "end_unix_ns" in values
        ):
            label = "基线窗口" if item.source_path.endswith("baseline_window") else "故障窗口"
            text = f"{label}：{values['start_unix_ns']}–{values['end_unix_ns']} ns"
        elif "candidate_count" in values:
            text = (f"准入候选：{values['candidate_count']}；缺模态："
                    f"{values.get('missing_modalities', '未列出')}")
        else:
            continue
        boundaries.setdefault(text, []).append(item.fact.fact_ref)
    if not scope_refs:
        return []
    for text, refs in boundaries.items():
        lines += ["- " + _text(text) + " [" + _refs(refs) + "]", ""]
    lines += [f"共 {len(scope_refs)} 项范围／候选摘要元信息，逐动作交付和完整内容见附件；"
              "这些不是新增观测或建议所需的测量前提。 ["
              + _refs(scope_refs) + "]", ""]
    return lines


def _refs(refs: list[str] | tuple[str, ...]) -> str:
    """Compress only contiguous aliases; no citation or contrary record is dropped."""
    numbers = sorted({int(r[1:]) for r in refs})
    spans: list[tuple[int, int]] = []
    for number in numbers:
        if spans and number == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], number)
        else:
            spans.append((number, number))
    return ", ".join(f"f{a}" if a == b else f"f{a}–f{b}" for a, b in spans)


def _condition_qualifiers(
    context: OpsSreReportContext | NativeReportMetadata,
    facts: Mapping[str, OpsReportFact],
) -> dict[str, str]:
    """Describe recorded ancestors; never recompute an AND/OR or upgrade a leaf."""
    qualifiers = {}
    for group in context.record_groups:
        # Retain paths and condition descriptions, not a second expanded universe.
        paths = []
        parents = []
        for ref in group["fact_refs"]:
            item = facts[ref]
            paths.append((ref, item.source_path))
            values = {v.name: v.value for v in item.fact.values}
            if "condition_id" not in values and "formal_condition_id" not in values:
                continue
            name = values.get("condition_id", values.get("formal_condition_id"))
            role = {"necessary": "必要条件", "refuting": "反证条件"}.get(
                str(values.get("requirement")), "条件")
            operator = str(values.get("operator", "组合规则见原记录")).upper()
            truth = {"true": "成立", "false": "不成立", "unknown": "未知"}.get(
                str(values.get("truth_value")), "未记录")
            parents.append((item.source_path, f"{role} {name}（{operator}，上层判断{truth}）"))
        parents.sort(key=lambda row: len(row[0]))
        for ref, path in paths:
            ancestors = [text for parent, text in parents if path.startswith(parent + ".")]
            if ancestors:
                qualifiers[ref] = " → ".join(ancestors)
    return qualifiers


def service_aliases(keys: tuple[str, ...], references: dict[str, Any] | None = None,
                    *, snapshot: str | None = None) -> dict[str, str]:
    """Use the actual same-case book; missing handles are not guessed or issued."""
    handles = {}
    if references is not None:
        reader = (SreActionReferenceBook
                  if references.get("schema_version") == "nurse.v2.sre-reference-book/2"
                  else SreReferenceBook)
        book = reader.from_state(references, snapshot=snapshot or references["snapshot"],
                                           run_id=references["run_id"])
        handles = {original: handle for kind, handle, original, _ in book.to_state()["entries"]
                   if kind == "service"}
    return {key: "服务" + handles[key] if key in handles else
            "未映射服务（" + key + "）" for key in keys}


def _excerpt(row: dict[str, Any]) -> str:
    suffix = f"（摘录，另省略 {row['omitted_characters']} 字符）" if row.get(
        "omitted_characters") else ""
    return _text(row["text"]) + suffix


def _render_sre_material(
    *, context: OpsSreReportContext | NativeSreReportContext,
    response: OpsSreReportResponse | None,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
    accounting: dict[str, Any] | None = None,
    references: dict[str, Any] | None = None,
) -> str:
    native = context if isinstance(context, NativeSreReportContext) else None
    details: OpsSreReportContext | NativeReportMetadata
    facts: Mapping[str, OpsReportFact]
    if isinstance(context, NativeSreReportContext):
        details = context.metadata
        facts = NativeFactLookup(context.fact_store)
    else:
        details = context
        facts = {f.fact.fact_ref: f for f in context.evidence.facts}
    evidence = details.evidence
    aliases = service_aliases(tuple(r.candidate_service_key for r in evidence.ranking),
                              references, snapshot=evidence.case_snapshot_id)
    checks: tuple[OpsClaimCheck, ...]
    if response is None:
        checks = ()
    elif native:
        checks = check_native_sre_report(response, context=native,
            provided_fact_refs=provided_fact_refs, provided_step_refs=provided_step_refs)
    else:
        assert isinstance(context, OpsSreReportContext)
        checks = check_sre_report(response, context=context,
            provided_fact_refs=provided_fact_refs, provided_step_refs=provided_step_refs)
    by_id = {c.proposal.claim.claim_id: c for c in checks}
    accepted = [c for c in checks if c.surface_status != "blocked"]
    lines = ["# NURSE 运维调查报告", "", "## 当前判断", ""]
    top = evidence.ranking[0]
    lines += [f"确定性小模型将{aliases[top.candidate_service_key]}列为正式首位，"
        "精确分数及完整排名见证据附件。", "",
        "分数不是根因概率；首位不等于因果已确认，并列破除不代表证据优势。", "",
        "核心自主执行查询与评分；智能体编排补充调查及解释，未改变正式排名。", "",
        "解释状态：尚待人工语义核对与用户认可；机械检查不代表语义正确。", ""]

    shown: set[str] = set()

    def claims(ids: tuple[str, ...]) -> None:
        for key in ids:
            checked = by_id[key]
            if checked.surface_status == "blocked":
                continue
            claim = checked.proposal.claim
            if key in shown:
                continue
            shown.add(key)
            text = claim.surface_text
            lines.extend(["- " + _text(text) + " [" + ", ".join(claim.fact_refs) + "]", ""])
            if claim.recommendation is not None:
                advice = claim.recommendation
                lines.extend(["  条件建议（未执行）：前提 "
                              + ", ".join(advice.precondition_refs), ""])
                for outcome in advice.possible_outcomes:
                    lines.extend(["  - " + _text(outcome.outcome) + "："
                                  + _text(outcome.interpretation), ""])
                lines.extend(["  安全限制："
                              + "；".join(_text(t) for t in advice.safety_limitations), ""])

    if not accepted:
        lines += ["确定性替代摘要：没有可呈现的模型陈述，不计模型解释成功。", ""]
    elif all(re.search(r"\bf\d+[.:]|truth_value\s*=|quantized_total_score\s*=",
                       c.proposal.claim.surface_text) for c in accepted):
        lines += ["解释质量提示：正文可能退化为字段转述；结构合格不等于完成了运维解释。", ""]
    if response:
        claims(response.summary_claim_ids)
    lines += ["## 关键依据与竞争解释", ""]
    if response:
        claims(response.finding_claim_ids)
    lines += ["## 关键局限", "", "以下为代码保留的反证／未知，模型省略也不会消失；"
              "同源记录不是多份独立支持。同类局限合并展示，引用仍覆盖全部原记录。", ""]
    caveats: dict[tuple[Any, ...], list[str]] = {}
    general: dict[str, list[str]] = {}
    general_targets: dict[str, set[str]] = {}
    qualifiers = _condition_qualifiers(details, facts)
    critical = set(details.critical_fact_refs)
    for item in facts.values():
        fact = item.fact
        if fact.fact_ref not in critical:
            continue
        values = {v.name: v.value for v in fact.values}
        meaning = []
        qualifier = qualifiers.get(fact.fact_ref)
        scope_only = item.origin == "query_scope_metadata"
        condition = values.get("condition_id", values.get("formal_condition_id",
                               values.get("predicate_kind")))
        if condition:
            own = {"necessary": "必要条件", "refuting": "反证条件"}.get(
                str(values.get("requirement")), "条件") + " " + str(condition)
            qualifier = (qualifier + " → " if qualifier else "") + own
        if not scope_only:
            for field, label in (("truth_value", "子项" if qualifier else "判断"),
                                 ("support_value", "支持条件")):
                value = values.get(field)
                if value in {"false", "unknown"}:
                    meaning.append((qualifier + "：" if qualifier else "")
                                   + label + ("不成立" if value == "false" else "未知"))
            if values.get("requirement") == "refuting" and values.get("truth_value") == "true":
                meaning.append((qualifier + "：" if qualifier else "") + "有效反证成立")
        for code in fact.status_codes:
            if code in _GENERAL_RULES:
                general.setdefault(code, []).append(fact.fact_ref)
                general_targets.setdefault(code, set()).update(fact.target_service_keys)
            else:
                meaning.append(_LIMIT_TEXT.get(code, "另有约束“" + code + "”（见对应原记录）"))
        if not meaning and not fact.status_codes and not scope_only:
            meaning.append("存在范围或覆盖局限，详见对应事实")
        for message in meaning:
            windows = tuple((i.start_unix_ns, i.end_unix_ns) for i in fact.intervals)
            key = ("查询范围限制" if scope_only else
                   "正式评估" if item.role == "formal_record" else "补充调查", message,
                   item.origin or item.role, fact.target_service_keys, windows,
                   item.source_refs, item.evidence_ids, item.source_path)
            caveats.setdefault(key, []).append(fact.fact_ref)
    for (role, message, _origin, scope, windows, _sources, _ids, path), refs in caveats.items():
        targets = "、".join(aliases[k] for k in scope) or "全案边界"
        interval = "；窗口 " + "、".join(f"{a}–{b} ns" for a, b in windows) if windows else ""
        lines += ["- " + _text(f"{role}：{message}（{targets}{interval}；{path}）")
                  + " [" + _refs(refs) + "]", ""]
    lines += _query_scope_lines(facts)
    if general:
        lines += ["### 通用使用边界（不是个案反证）", ""]
        for code, refs in general.items():
            target_set = general_targets[code]
            scoped = f"{len(target_set)} 个服务" if target_set else "无单服务目标的范围记录"
            lines += ["- " + _LIMIT_TEXT[code] + f"。适用 {len(refs)} 项记录／{scoped}；"
                      "逐记录范围映射见附件。 ["
                      + _refs(refs) + "]", ""]
    for overview_entry in details.unexpanded_critical_overviews:
        overview = overview_entry["overview"]
        lines += [f"- 动作 {overview_entry['action_index']} 的概览当时未展开 "
                  f"{overview['unexpanded_count']} 个关键记录；不代表最终仍未展开。"
                  "未展开不等于正常或排除。", ""]
    if evidence.formal_detail_omissions:
        lines += ["部分正式细节仅保存在完整核心结果；未显示不等于已查正常。", ""]
    lines += ["## 实际调查过程", "", "公开目的和后续回应均为历史模型陈述，"
              "不是已证实事实，也不是隐藏思维链。故障窗口与调查耗时不可混用。", ""]
    organized = {} if response is None else {s.step_ref: s for s in response.workflow_steps}
    status_names = dict(completed="已完成", rejected="已拒绝", interrupted="中断／完成未知",
                        reserved="已预留／完成未知", not_dispatched="未调度")
    for step in details.steps:
        ref, planned, executed = step["step_ref"], step["planned"], step["executed"]
        lines += [f"### 步骤 {step['decision_index']}（{ref}）", ""]
        questions = planned.get("questions", [])
        lines += ["公开目的（未验证）：" + ("；".join(_excerpt(q) for q in questions)
                                            if questions else "未记录动作目的"), ""]
        lines += ["实际操作：" + _text(executed.get("tool", "无已调度工具")) + "；"
                  + status_names.get(executed["status"], _text(executed["status"])), ""]
        params = executed.get("parameters", {})
        display = {k: aliases.get(v, v) if type(v) is str else v for k, v in params.items()}
        lines += ["参数：" + _text(canonical_json_bytes(display).decode()), ""]
        if executed.get("failure_type"):
            lines += ["失败类型：" + _text(executed["failure_type"]), ""]
        refs = step["observed_fact_refs"]
        lines += [("实际交付事实：" + _refs(refs) if refs else
                   "没有交付可用于诊断断言的事实；不能把目录、拒绝或缺失当作健康结论。"), ""]
        workflow = organized.get(ref)
        if workflow:
            if any(by_id[key].surface_status == "blocked" for key in
                   workflow.observation_claim_ids + workflow.synthesis_claim_ids):
                lines += ["本步部分解释未通过来源或语言检查；隔离原文和原因见证据附件，"
                          "不能把它当作已取得的诊断结论。", ""]
            claims(workflow.observation_claim_ids)
        visible = 0
        statuses: set[str] = set()
        for fact_ref in set(refs):
            observed = facts[fact_ref]
            if observed.origin != "query_scope_metadata":
                visible += 1
                statuses.update(str(v.value) for v in observed.fact.values
                    if v.name in {"truth_value", "support_value"})
        if visible:
            meanings = [label for value, label in (("true", "条件成立"),
                ("false", "条件不成立"), ("unknown", "仍有未知")) if value in statuses]
            lines += ["可展示观测（代码摘要）：交付 " + str(visible)
                      + " 项有来源事实；" + "、".join(meanings or ["范围／目录信息"])
                      + "。各项范围不同，不代表整个服务健康或故障已确认；"
                      "条件的 AND/OR 与反证仍按上面的关键局限及附件解释。", ""]
        if step["responded_to"]:
            for reply in step["responded_to"]:
                lines += [f"后续决策 {reply['decision_index']} 的公开回应（未验证）："
                          + _excerpt(reply["public_summary"]), ""]
        else:
            lines += ["未记录后续回应；不补写当时的转向或收敛理由。", ""]
        if workflow and workflow.synthesis_claim_ids:
            lines += ["报告阶段事后综合（不冒充当时已知）：", ""]
            claims(workflow.synthesis_claim_ids)
    termination = {"decision_budget": "决策预算耗尽（代码记录，不等于主动收敛）",
        "model_stop": "模型选择停止（公开理由仍未验证）", "unavailable": "调查不可用或中断",
        "disabled": "调查未启用"}[details.termination]
    lines += ["停止状态：" + termination, "", "## 未决问题与下一步", "",
              "以下建议均未执行，不构成处置授权或恢复验证。", ""]
    if response:
        claims(response.unresolved_claim_ids)
    if details.steps:
        for unresolved in details.steps[-1]["planned"].get("unresolved", []):
            lines += ["历史公开未决项（未验证）：" + _excerpt(unresolved), ""]
    lines += ["## 附录", "", "### 完整正式排名与服务映射", "",
        "| 排名 | 别名 | 准入服务 ID | 总分 | 局部贡献 | 传播贡献 | 证据状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in evidence.ranking:
        lines.append(f"| {row.rank} | {aliases[row.candidate_service_key]} | "
            f"{_text(row.candidate_service_key)} | {row.quantized_total_score} | "
            f"{row.local_score_component:.12g} | {row.propagation_score_component:.12g} | "
            f"{_text(row.local_evidence_status)} |")
    lines += ["", "正式结果摘要：" + context.core_result_digest, "",
        "核心状态：completed；调查状态：" + evidence.investigation_status
        + "；报告状态：" + ("部分陈述隔离" if any(c.surface_status == "blocked" for c in checks)
                            else "待语义验收" if accepted else "不可用／确定性替代"), "",
        "原始记录：本次运行的核心结果、investigation/state.json、conversation.json "
        "及报告原响应由运行所有者保存；下列摘要不是替代品。", "",
        "成本与模块耗时：" + ("未提供" if accounting is None else
                          _text(canonical_json_bytes(accounting).decode())), "",
        "已记录工具秒数：" + _text(str(details.action_seconds)), "",
        "已记录决策秒数：" + _text(str(details.decision_seconds)), "",
        "### 事实、来源与范围", ""]
    if native:
        lines += ["完整原生事实树（别名按树内路径及共享来源精确解析）：", "",
                  _text(canonical_json_bytes(native.fact_store.to_packet()).decode()), ""]
        for ref in facts:
            lines += [f'<a id="{ref}"></a>', "", "- " + ref + "：见上方完整树的同名别名。", ""]
    else:
        for item in facts.values():
            lines += ["- " + item.fact.fact_ref + "："
                      + _text(canonical_json_bytes(item.model_dump(mode="json")).decode()), ""]
    lines += ["### 通用规则的逐记录范围映射", ""]
    for code, refs in general.items():
        if native:
            lines += ["- " + _text(code) + "：" + _refs(refs)
                      + "；每项范围、来源及路径完整保存在原生树。", ""]
            continue
        lines += ["- " + _text(code) + "：" + _text(canonical_json_bytes([
            dict(fact_ref=f.fact.fact_ref, target_service_keys=f.fact.target_service_keys,
                 intervals=[i.model_dump(mode="json") for i in f.fact.intervals],
                 source_refs=f.source_refs, source_path=f.source_path)
            for f in (facts[ref] for ref in refs)]).decode()), ""]
    lines += ["### 步骤与原记录关联", "",
              _text(canonical_json_bytes(details.record_groups).decode()), "",
              _text(canonical_json_bytes(details.steps).decode()), ""]
    for check in checks:
        if check.surface_status == "blocked":
            lines += ["隔离陈述 " + _text(check.proposal.claim.claim_id) + "："
                      + _text(", ".join(check.reason_codes)), ""]
    return "\n".join(lines)


def _cost_lines(accounting: dict[str, Any] | None) -> list[str]:
    if accounting is None:
        return ["成本与模块耗时：未提供；不能推断为零。", ""]
    lines = []
    selected = accounting.get("usage_scope") == "selected_source_and_previous_reports_only"
    for key, label in (("source_usage", "所列旧调查与已消费报告" if selected else "旧调查"),
                       ("report_usage", "新增报告"),
                       ("usage", "所列历史与本次累计（非全部历史）" if selected else "全案累计")):
        usage = accounting.get(key)
        if not isinstance(usage, dict):
            continue
        known_in = usage.get("reported_input_tokens", usage.get("input_tokens"))
        known_out = usage.get("reported_output_tokens", usage.get("output_tokens"))
        unknown_in, unknown_out = (usage.get("unknown_input_attempts"),
                                   usage.get("unknown_output_attempts"))
        def show(value: Any) -> str:
            return "未提供" if value is None else str(value)
        precise = "精确总量未知" if usage.get("input_tokens") is None or (
            usage.get("output_tokens") is None) else "用量完整"
        lines += [f"{label}：已知输入 {show(known_in)} tokens，已知输出 {show(known_out)} tokens；"
                  f"输入未知尝试 {show(unknown_in)} 次、输出未知尝试 {show(unknown_out)} 次；"
                  f"{precise}。", ""]
    if not lines:
        lines += ["用量：未提供已核验的尝试账本，不能推断为零。", ""]
    if accounting.get("untracked_send_unknown"):
        lines += ["另有一次已开始但传输账本缺失的发送，用量未知，不包含在上述已知小计中。", ""]
    lines += ["准备不计方法时间；历史与新增耗时分列于证据附件，不合并冒称本次壁钟。", ""]
    return lines


def render_sre_report_pair(
    *, context: OpsSreReportContext, response: OpsSreReportResponse | None,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
    references: dict[str, Any] | None = None, accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None, preview: bool = False,
) -> dict[str, str]:
    """Derived two-file presentation. Original context/result/response remain authoritative."""
    return _render_report_pair(context=context, response=response,
        provided_fact_refs=provided_fact_refs, provided_step_refs=provided_step_refs,
        references=references, accounting=accounting, public_history=public_history,
        preview=preview, policy=RENDER_POLICY)


def _render_report_pair(
    *, context: OpsSreReportContext | NativeSreReportContext,
    response: OpsSreReportResponse | None,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
    references: dict[str, Any] | None = None, accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None, preview: bool = False,
    policy: str, raw_response_json: str | None = None,
) -> dict[str, str]:
    response_value = None if response is None else response.model_dump(mode="json")
    identity = dict(policy=policy, context_sha256=context.context_sha256,
        response=response_value, references=references, accounting=accounting,
        public_history=public_history, provided_fact_refs=provided_fact_refs,
        provided_step_refs=provided_step_refs, preview=preview)
    if isinstance(context, NativeSreReportContext):
        identity["raw_response_json"] = raw_response_json
    pin = sha256_hex(canonical_json_bytes(identity))
    full = _render_sre_material(context=context, response=response,
        provided_fact_refs=provided_fact_refs, provided_step_refs=provided_step_refs,
        references=references, accounting=accounting)
    main, appendix = full.split("## 附录", 1)
    stamp = "<!-- " + policy + " binding=" + pin + " -->\n\n"
    if preview:
        main = main.replace("## 当前判断", "本文件是旧响应的离线诊断对照，"
            "不是新模型结果，也不证明新调查已通过。\n\n## 当前判断", 1)
    if accounting and "source_usage" in accounting:
        main = main.replace("## 当前判断",
            "本报告继承原版本已完成的真实调查；本次只作报告阶段综合，"
            "未重新调查、准备数据或计算排名。原来未检查的问题仍未检查。\n\n## 当前判断", 1)
    # Turn alias ranges into links to the first anchor; every alias remains in the text.
    main = re.sub(r"\[(f[0-9]+)([^\]]*)\]",
                  r"[\1\2](report-evidence.md#\1)", main)
    evidence = stamp + "# NURSE 完整证据附件\n\n" + appendix.lstrip()
    if not isinstance(context, NativeSreReportContext):
        evidence = re.sub(r"(?m)^- (f[0-9]+)：", r'<a id="\1"></a>\n\n- \1：', evidence)
    evidence += "\n\n## 原始报告响应（未改写，隔离内容不是事实）\n\n" + _text(
        canonical_json_bytes(response_value).decode())
    evidence += "\n\n## 原生公开调查文字（模型陈述，不是事实）\n\n" + (
        _text(canonical_json_bytes(public_history).decode()) if public_history is not None else
        "未另行提供完整会话；步骤中已标注摘录，完整原文由调查 conversation.json 保存。")
    if isinstance(context, NativeSreReportContext):
        evidence += "\n\n## 原生模型响应（短引用还原前，未改写）\n\n" + (
            _text(raw_response_json) if raw_response_json is not None else
            "未提供原生响应；上方解析结果不能冒充原生文字。")
        evidence += "\n\n## 完整引用映射\n\n" + _text(canonical_json_bytes(references).decode())
    main = stamp + main.rstrip() + "\n\n## 成本与证据入口\n\n" + "\n".join(
        _cost_lines(accounting)) + "\n[完整排名、来源、原文与隔离原因](report-evidence.md)。\n"
    return dict(report=main, evidence=evidence + "\n", binding_sha256=pin)


def render_sre_report(**kwargs: Any) -> str:
    """Compatibility display helper; persistent delivery must save both members."""
    return render_sre_report_pair(**kwargs)["report"]
