"""D21 decision prompt identity; prior workbench wire remains immutable."""

# ruff: noqa: RUF001 -- Chinese punctuation is part of the model-visible contract.

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import (
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
    SRE_USED_SOURCE_DECISION_PROTOCOL,
    SRE_USED_SOURCE_REPORT_PROTOCOL,
)
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_exchange import COMPARABLE_REPORT_OUTPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_skills import used_source_common_guidance
from nurse_rcl.v2.live.ops_sre_workbench_exchange import (
    WORKBENCH_DECISION_SYSTEM,
    WORKBENCH_REPORT_SYSTEM,
)

USED_SOURCE_DECISION_PROTOCOL = SRE_USED_SOURCE_DECISION_PROTOCOL
USED_SOURCE_DECISION_SYSTEM = (
    WORKBENCH_DECISION_SYSTEM.replace(
        "nurse.v2.sre-model-view/5", "nurse.v2.sre-model-view/6"
    ).replace(
        "重复与失败均计动作。",
        "实际派发的重复与失败均计动作；只有可验证的同参完整已读拟议在派发前进入一次独立纠正。",
    )
    + """
D21工作台的post_action_state来自实际ledger和回执，public_working_memory是上轮执行前公开意图，
不是尚待执行命令。先对照最后真实动作、实际交付字段和无新增状态，再自主提出新的区分性问题或停止。
若系统反馈already_delivered_no_new_content，可复用原完整回执；纠正请求单独计费且全案最多一次。
不要将评分概要当未读子卡或新观测；仅复制字段可作有限评分解释，机制判断仍需实际核验。
"""
)


def used_source_decision_policy() -> dict[str, Any]:
    guidance = used_source_common_guidance()
    return dict(
        protocol=USED_SOURCE_DECISION_PROTOCOL,
        system=USED_SOURCE_DECISION_SYSTEM,
        schema="nurse.v2.sre-investigation-input/13",
        profile="hermes-sre-tools/12",
        guidance=guidance,
        prompt_sha256=sha256_hex(
            encode_json(
                dict(
                    protocol=USED_SOURCE_DECISION_PROTOCOL,
                    system=USED_SOURCE_DECISION_SYSTEM,
                    output_policy=OUTPUT_POLICY,
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                CONTEXT_LIMITS,
            )
        ),
    )


USED_SOURCE_REPORT_SYSTEM = (
    WORKBENCH_REPORT_SYSTEM.replace("model-view/5", "model-view/7")
    + """
D21首屏的used_source_contrast只复制固定排序实际已用的评分来源字段，
只能作排名输入的部分解释，不是新症状观测或机制、方向、时序的独立核验。
先汇报有限判断、真实支撑、反证/未知及未做检查；仅做评分来源对照时，
必须说明故障机制调查仍未完成。未派发的重复读卡拟议只是付费决策，
不作工具步骤或新证据；纠正请求另计费，仅实际派发动作进入步骤。
原response/4、当步实供、窗口、条件角色和建议前提检查全部保持。
"""
)


def used_source_report_policy() -> dict[str, Any]:
    return dict(
        protocol=SRE_USED_SOURCE_REPORT_PROTOCOL,
        system=USED_SOURCE_REPORT_SYSTEM,
        schema="nurse.v2.sre-report-input/14",
        profile="hermes-sre-tools/12",
        output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
        prompt_sha256=sha256_hex(
            encode_json(
                dict(
                    protocol=SRE_USED_SOURCE_REPORT_PROTOCOL,
                    system=USED_SOURCE_REPORT_SYSTEM,
                    output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
                    output_policy=OUTPUT_POLICY,
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                REPORT_LIMITS,
            )
        ),
    )
