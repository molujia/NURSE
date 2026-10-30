"""Deterministic rank/negative-evidence display is never authored by the model."""

# Chinese report prose intentionally uses full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import html
import re

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_report_context import OpsReportContext
from nurse_rcl.v2.live.ops_report_schema import OpsClaimCheck


def _text(value: str) -> str:
    # Show text, not Markdown/HTML or model-authored links/tables.
    value = html.escape(value, quote=False).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", value)


def render_ops_report(*, context: OpsReportContext, checks: tuple[OpsClaimCheck, ...]) -> str:
    lines = [
        "# NURSE 隔离 RCL 报告",
        "",
        "正式结果摘要：" + context.core_result_digest,
        "",
        "调查状态："
        + context.investigation_status
        + "；失败类型："
        + str(context.investigation_failure_type),
        "",
        "## 唯一正式排名（代码渲染）",
        "",
        "| 排名 | 服务 | 总分 | 局部贡献 | 传播贡献 | 证据状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in context.ranking:
        lines.append(
            f"| {row.rank} | {_text(row.candidate_service_key)} | "
            f"{row.quantized_total_score} | {row.local_score_component:.12g} | "
            f"{row.propagation_score_component:.12g} | {_text(row.local_evidence_status)} |"
        )
    lines.extend(
        [
            "",
            "分数不是根因概率；并列破除不代表证据优势。",
            "",
            "## 确定性摘要（不计模型成功）",
            "",
            "formal_record 是核心实际评估记录，不意味着每项均有正贡献；"
            "补充调查未参与本次正式评分，同源复用不算独立支持。",
            "",
            "机械检查通过不代表语义正确，也不确认因果。",
            "",
        ]
    )
    for title, role in (
        ("正式依据及关键缺口", "formal_record"),
        ("补充调查及反证/未知", "supplementary"),
    ):
        lines.extend(["### " + title, ""])
        for record in context.facts:
            if record.role != role or record.fact.fact_ref not in context.mandatory_fact_refs:
                continue
            fact = record.fact
            value = dict(
                targets=fact.target_service_keys,
                values={v.name: v.value for v in fact.values},
                codes=fact.status_codes,
                intervals=[i.model_dump(mode="json") for i in fact.intervals],
                evidence_ids=record.evidence_ids,
                source_refs=record.source_refs,
                origin=record.origin,
                reused_core_evidence_ids=record.reused_core_evidence_ids,
            )
            lines.extend(
                ["- " + fact.fact_ref + ": " + _text(canonical_json_bytes(value).decode()), ""]
            )
    if context.formal_detail_omissions:
        lines.extend(
            [
                "未展示的正式细节仍保存在完整核心结果；不代表已查正常。",
                "",
                _text(
                    canonical_json_bytes(
                        [o.model_dump(mode="json") for o in context.formal_detail_omissions]
                    ).decode()
                ),
                "",
            ]
        )
    if not context.formal_projection_available:
        lines.extend(["正式报告投影超过预算：仅保留代码排名，不能宣称完整解释。", ""])
    for section, title in (
        ("formal_grounds", "模型解释：排名实际依据"),
        ("supplementary_findings", "模型解释：补充发现（未参与评分）"),
        ("limitations", "模型解释：局限"),
        ("conditional_next_steps", "模型解释：有条件下一步（未执行）"),
    ):
        lines.extend(["## " + title, "", "以下原文语义仍未证实。", ""])
        for check in checks:
            if check.proposal.section != section:
                continue
            claim = check.proposal.claim
            if check.surface_status == "blocked":
                lines.extend(
                    [
                        "- "
                        + _text(claim.claim_id)
                        + " 已隔离："
                        + _text(", ".join(check.reason_codes)),
                        "",
                    ]
                )
            else:
                lines.extend(
                    ["- " + _text(claim.surface_text) + " [" + ", ".join(claim.fact_refs) + "]", ""]
                )
                if claim.recommendation is not None:
                    advice = claim.recommendation
                    lines.extend(
                        [
                            "  条件建议（未执行）："
                            + _text(canonical_json_bytes(advice.model_dump(mode="json")).decode()),
                            "",
                        ]
                    )
    return "\n".join(lines)
