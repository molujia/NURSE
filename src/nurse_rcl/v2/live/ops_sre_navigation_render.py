"""D17 presentation only: no inferred outcomes, claim rewrites or permission changes."""

# ruff: noqa: RUF001 -- Chinese human report
from __future__ import annotations

from typing import Any

from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_sre_navigation_failure import OpsNavigationFailureDetail
from nurse_rcl.v2.live.ops_sre_readable_facts import _IDENTITY

SCORE_TYPES = {
    "score_lineage_summary": "评分摘要（排名、分项与聚合配置；下级入口不代表已读明细）",
    "score_family": "家族选择（代表项、所选家族及最终使用层次分别核对）",
    "score_term": "单项输入（原计算、池内选择与角色来源，不外推其他项）",
    "score_propagation": "传播汇总（零值不等于已证无传播）",
    "score_path": "传播路径（仅此路径的实供输入与缺口）",
}
_REASONS = {
    "step_evidence_not_delivered": "引用非本步实供",
    "observation_scope_missing": "缺少合法观测范围",
    "summary_not_observed_precondition": "摘要不能作为观测前提",
    "ranking_basis_missing": "缺少正式排名比较依据",
    "evidence_not_provided": "引用未提供的证据",
}
_ERRORS = {
    "sre_question_not_previously_visible": "问题引用不在当时可用集合",
    "sre_response_not_previously_visible": "历史回应引用不是该动作实际可见回执",
    "sre_latest_result_not_addressed": "未正确关联最新工具反馈",
}


def readable_score_fact(item: OpsReportFact, *, source_kind: str) -> str:
    """Use the retained source kind for every nested field, not guessed predicates."""
    meaning = SCORE_TYPES[source_kind]
    values = {v.name: v.value for v in item.fact.values}
    shown = [
        f"{k}={v}"
        for k, v in values.items()
        if k not in _IDENTITY
        and k not in {"kind", "truth_value"}
        and not k.endswith(("_sha256", "_id", "_ref", "_ids", "_refs"))
        and not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v))
    ]
    return (
        "评分来源："
        + meaning
        + "；"
        + ("；".join(shown) or "此层未提供额外可读标量")
        + ("。来源记录的通用 unknown 不是独立健康判断；完整精确字段、窗口、身份与来源见附件。")
    )


def finding_state(*, total: int, blocked: int, reasons: list[str]) -> str:
    if not 0 <= blocked <= total:
        raise ValueError("finding count differs")
    if total == 0:
        return "模型未提供本步发现；不补造结论。"
    if blocked == 0:
        return f"保留发现 {total} 条（机械未拒绝，仍待语义审阅）。"
    return (
        ("全部隔离" if blocked == total else "部分隔离")
        + f"：{blocked}/{total} 条；"
        + ("、".join(_REASONS.get(r, r) for r in sorted(set(reasons))))
        + "。隔离原文只在附件，不作为本步事实。"
    )


def last_admitted_unresolved(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        planned = step["planned"]
        # Only admitted response projection has unresolved, even when it is empty.
        if "unresolved" in planned:
            return dict(decision_index=step["decision_index"], unresolved=planned["unresolved"])
    return None


def failure_text(detail: OpsNavigationFailureDetail) -> str:
    prefix = f"实际失败阶段：{detail.stage}；决策编号：{detail.decision_index or '未记录'}；"
    reason = _ERRORS.get(
        detail.exchange_code or "", "具体原因未记录；失败类型：" + detail.failure_type
    )
    suffix = (
        "。决策未准入，工具未执行。"
        if detail.stage == "investigation_provider"
        else "。不补造后续执行。"
    )
    return prefix + reason + suffix
