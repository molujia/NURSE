"""Read-only projection of registered measurement semantics, not a new checker."""

# ruff: noqa: RUF001 -- Chinese report text
from __future__ import annotations

from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.repair_measurements import _NORMAL_PREDICATES, _UP_PREDICATES

_RULES = dict.fromkeys(_UP_PREDICATES, "按注册信号语义检查持续上升；未满足语义时保留未知")
_RULES.update(dict.fromkeys(_NORMAL_PREDICATES, "按完整观测与带内规则检查 verified_normal"))
_RULES.update(
    {
        "local_anomaly_observed": "检查已准入的一般偏离 general_deviation_present",
        "workload_unavailability_observed": "检查已注册生命周期的不可用支持",
        "workload_availability_verified_normal": "检查生命周期不可用反证及 verified_normal",
        "lifecycle_transition_observed": "检查已准入生命周期事件的起止界限",
    }
)
_TRUTH = {"true": "成立", "false": "不成立", "unknown": "未知"}
_IDENTITY = {
    "record_path",
    "source_path",
    "representation",
    "source_role",
    "schema_version",
    "candidate_service_key",
    "support_source_key",
    "canonical_series_key",
}


def readable_fact(item: OpsReportFact) -> str:
    """Keep every scalar except machine identity; preserve independent adverse states."""
    values = {v.name: v.value for v in item.fact.values}
    predicate = values.get("predicate_kind")
    if isinstance(predicate, str) and predicate in _RULES:
        rule = predicate + "：" + str(_RULES[predicate])
    elif {"persistent_up", "persistent_down", "verified_normal"} & values.keys():
        rule = "已测信号状态（episode）；各字段是不同判断，不合并为健康结论"
    else:
        rule = "未注册的可读规则：" + str(predicate or values.get("kind", item.fact.kind))
    shown = []
    for name, value in values.items():
        if name in _IDENTITY or name.endswith(("_sha256", "_id", "_ref")):
            continue
        meaning = (
            _TRUTH.get(str(value))
            if name
            in {
                "truth_value",
                "support_value",
                "persistent_up",
                "persistent_down",
                "persistent_deviation_present",
                "general_deviation_present",
                "verified_normal",
            }
            else None
        )
        shown.append(f"{name}={value}" + (f"（{meaning}）" if meaning else ""))
    window = "、".join(f"[{i.start_unix_ns}, {i.end_unix_ns}] ns" for i in item.fact.intervals)
    return (
        rule
        + "；"
        + "；".join(shown)
        + "；窗口："
        + (window or "未给出")
        + "。单项不成立或未知不表示服务整体健康；完整原值、身份与来源见附件。"
    )
