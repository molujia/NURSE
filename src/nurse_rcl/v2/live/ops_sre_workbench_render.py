# ruff: noqa: RUF001 -- Chinese human report
"""Registered human score fields only; original facts and model text stay intact."""

from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_sre_navigation_render import SCORE_TYPES

_LABELS = {
    "rank": "正式位次",
    "primary_family": "实际家族",
    "local_strength": "该项局部支持值",
    "unknown_term_count": "数值不可得项",
    "numeric_term_count": "数值可得项",
    "independent_pool_count": "独立池数",
    "family_strength": "家族支持值",
    "local_score_component": "局部分项",
    "propagation_score_component": "传播分项",
    "used_in_aggregation": "参与家族聚合",
    "selected_in_pool": "池内选用",
    "family_representative": "家族代表",
    "family_used_in_aggregation": "家族参与最终聚合",
    "selected": "本层选用",
    "observed_value": "实供观测值",
    "baseline_value": "实供基线值",
    "deviation": "偏离",
    "coverage": "覆盖率",
    "availability": "可用性",
    "path_strength": "路径支持值",
    "path_score": "路径得分",
    "measured_value": "实供值",
    "value": "实供值",
    "contribution": "记录分项",
}


def readable_score_fact(item: OpsReportFact, *, source_kind: str) -> str:
    """No scalar sweep, semantic inference, score recomputation or prose rewriting."""
    values = {v.name: v.value for v in item.fact.values}
    shown = []
    for name, label in _LABELS.items():
        if name not in values:
            continue
        value = values[name]
        text = (
            "数值不可得"
            if value is None
            else "是"
            if value is True
            else "否"
            if value is False
            else str(value)
        )
        shown.append(label + "：" + text)
    return (
        "评分来源："
        + SCORE_TYPES[source_kind]
        + "；"
        + ("；".join(shown) or "此层没有登记的主文标量")
        + "。完整精确字段及未作中文解释的字段见附件。"
    )
