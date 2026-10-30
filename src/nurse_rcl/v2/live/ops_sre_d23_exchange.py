"""D23 prompt identities for balanced triage and truthful process reporting."""

# ruff: noqa: RUF001 -- Chinese punctuation is intentional model-visible guidance.

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import (
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
    SRE_BALANCED_DECISION_PROTOCOL,
    SRE_BALANCED_REPORT_PROTOCOL,
)
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_exchange import COMPARABLE_REPORT_OUTPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_d22_exchange import (
    RECEIPT_REUSE_DECISION_SYSTEM,
    RECEIPT_REUSE_REPORT_SYSTEM,
)
from nurse_rcl.v2.live.ops_sre_skills import balanced_common_guidance

BALANCED_DECISION_SYSTEM = (
    RECEIPT_REUSE_DECISION_SYSTEM.replace(
        "nurse.v2.sre-model-view/7", "nurse.v2.sre-model-view/8"
    ).replace(
        "latest_result_update 首次为null；以后恰好含四项：",
        "latest_result_update 首次为null；以后恰好含五项：",
    )
    + """
D23：比较首页的role_contrast是双侧正式已用评分、症状和机制的冻结来源摘要，
不是新观测或未读完整卡。先查看investigation_workbench的最近实际回执、
逐侧逐角色的complete/brief/partial/未读覆盖及合法cursor绑定，再决定读卡、
转向或诚实停止。重复完整表示的no_new_content不得写成新发现；
latest_result_update新增content_relation，必须原样回显最近实际状态枚举
new_content/no_new_content/none_delivered，首轮无上一步仍为null。
actions_available_now大于0时不可用budget_exhausted停止；decisions_after_this为0
只表示本轮后不再购买决策，不表示当前动作已耗尽。可因no_useful_check自主停止。
角色计数、来源集合digest和导航入口均不是因果票数；旧执行前意图不是最新反馈。
"""
)


def balanced_decision_policy() -> dict[str, Any]:
    guidance = balanced_common_guidance()
    return dict(
        protocol=SRE_BALANCED_DECISION_PROTOCOL,
        system=BALANCED_DECISION_SYSTEM,
        schema="nurse.v2.sre-investigation-input/15",
        profile="hermes-sre-tools/14",
        guidance=guidance,
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=SRE_BALANCED_DECISION_PROTOCOL,
            system=BALANCED_DECISION_SYSTEM,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ), CONTEXT_LIMITS)),
    )


BALANCED_REPORT_SYSTEM = (
    RECEIPT_REUSE_REPORT_SYSTEM.replace("model-view/8", "model-view/9")
    + """
D23报告先总结固定排名、实际已核验来源和因果局限，再按真实调查动作说明
为何检查、回执是否新增、何时转向或诚实停止。必须严格区分：
单张complete_record与整个domain_unread；unknown_term与not_measured；
core_published与formal_used；no_new_content与新观测。
角色摘要只能作导航，未读细节不能断言；具体数值仍仅来自本次实供事实。
保留竞争解释和反证/unknown，不把条件计数写成独立支持票。
原response/4、逐项事实、条件角色与建议前提检查完整保留。
"""
)


def balanced_report_policy() -> dict[str, Any]:
    return dict(
        protocol=SRE_BALANCED_REPORT_PROTOCOL,
        system=BALANCED_REPORT_SYSTEM,
        schema="nurse.v2.sre-report-input/16",
        profile="hermes-sre-tools/14",
        output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=SRE_BALANCED_REPORT_PROTOCOL,
            system=BALANCED_REPORT_SYSTEM,
            output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ), REPORT_LIMITS)),
    )
