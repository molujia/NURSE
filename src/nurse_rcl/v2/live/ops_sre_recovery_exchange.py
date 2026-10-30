"""D15 prompt identities; reuse unchanged response parsing and source checks."""

# ruff: noqa: RUF001 -- Chinese user-facing instructions

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_exchange import (
    CONDITION_REPORT_OUTPUT_SCHEMA,
    CONDITION_REPORT_SYSTEM,
    FEEDBACK_DECISION_SYSTEM,
)

RECOVERY_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/9"
RECOVERY_DECISION_SYSTEM = (
    FEEDBACK_DECISION_SYSTEM
    + """
公开 question/public_summary/unresolved、investigation_question 的说明文字和
latest_result_update 的 summary/remaining_question 明确使用中文自然语言。
保留工具名、字段、枚举、ID和精确原值，不翻译标识符或遥测原文。
拒绝原因来自 previous_feedback.parameter_feedback.diagnostic 的真实守卫。
object_contract 说明实际卡种类、全部端点和合法操作，不是观测证据；scoring 不等于 mechanisms。
仅对已分类参数错误指出原样重复不能修复，未知内部异常不能声称瞬时故障。
恢复选项在 investigation_routes.recovery_feedback.route_bindings 关联本轮实际入口，
不是额外行动或新事实，不自动执行。回应最近失败，若重读相同请求需公开说明理由。
仍可自主换目标、读合法内容、续页、暂缓或停止；失败不证明目标健康或机制不存在。
"""
)
RECOVERY_DECISION_PROMPT_SHA256 = sha256_hex(
    encode_json(
        dict(
            protocol=RECOVERY_DECISION_PROTOCOL,
            system=RECOVERY_DECISION_SYSTEM,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
        CONTEXT_LIMITS,
    )
)

RECOVERY_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/11"
RECOVERY_REPORT_INPUT_SCHEMA = "nurse.v2.sre-report-input/10"
RECOVERY_REPORT_SYSTEM = (
    CONDITION_REPORT_SYSTEM
    + """
本报告面向中文读者。surface_text、possible_outcomes.interpretation、safety_limitations
以及其他自然语言说明明确使用中文；工具名、ID、字段、枚举、单位及精确原值保持原样。
executive_summary 是独立且有证据的简短判断，不是将前五名塞进一条 explain_priority。
explain_priority 恰好一目标并引用本次实供的正式 ranking；compare_candidates 恰好同序两目标
并引用实际 competition。两者 interval=null。没有合法依据时保留缺口，不制造摘要。
每条 step_findings 必有真实 step_ref，全部证据当时交付；跨步材料只放 final_synthesis。
每个引用和精确断言仍遵循当前实供指南。不要凭目录或失败诊断声称取得新观测。
next_checks.possible_outcomes 用“若后续只读检查……则……”明确未来条件句，
解释支持增强／降低或保留不确定性；不得改写冻结排名，也不得声称已经执行或恢复。
先总结当前判断与关键保留，再讲证据意义、竞争解释、实际转折和未决项，避免只复述分数。
工具参数错误只表示该步未取得观测，不表示目标机制不存在。原模型公开历史不得补写。
"""
)
RECOVERY_REPORT_PROMPT_SHA256 = sha256_hex(
    encode_json(
        dict(
            protocol=RECOVERY_REPORT_PROTOCOL,
            system=RECOVERY_REPORT_SYSTEM,
            output_schema=CONDITION_REPORT_OUTPUT_SCHEMA,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
        REPORT_LIMITS,
    )
)
