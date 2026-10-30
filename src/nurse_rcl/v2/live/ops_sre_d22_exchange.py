"""D22 prompt identity: receipt replay and strictly supplied report numerals."""

# ruff: noqa: RUF001 -- Chinese punctuation is intentional model-visible guidance.

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import (
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
    SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
    SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
)
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_exchange import COMPARABLE_REPORT_OUTPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_d21_exchange import (
    USED_SOURCE_DECISION_SYSTEM,
    USED_SOURCE_REPORT_SYSTEM,
)
from nurse_rcl.v2.live.ops_sre_skills import used_source_common_guidance

RECEIPT_REUSE_DECISION_SYSTEM = (
    USED_SOURCE_DECISION_SYSTEM.replace("nurse.v2.sre-model-view/6", "nurse.v2.sre-model-view/7")
    + """
D22：若精确完整已读回执被复用，最新动作会显示receipt_reuse和no_new_content。
这是一次已执行但没有新证据的只读动作，不是新观测、独立支持或待重复的旧计划。
在剩余额度内，自主比较尚未解决的竞争问题；可选症状、机制条件、依赖方向、
时序等现有工具，也可诚实停止。不要把评分来源对照当作机制核验。
"""
)


def receipt_reuse_decision_policy() -> dict[str, Any]:
    guidance = used_source_common_guidance()
    return dict(
        protocol=SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
        system=RECEIPT_REUSE_DECISION_SYSTEM,
        schema="nurse.v2.sre-investigation-input/14",
        profile="hermes-sre-tools/13",
        guidance=guidance,
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
            system=RECEIPT_REUSE_DECISION_SYSTEM,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ), CONTEXT_LIMITS)),
    )


RECEIPT_REUSE_REPORT_SYSTEM = (
    USED_SOURCE_REPORT_SYSTEM.replace("model-view/7", "model-view/8")
    + """
D22：重复已读动作按真实步骤汇报，但标为无新增，不能将复用回执当第二份独立证据。
正文中的每个具体小数必须准确来自该claim实际引用且本次已提供的事实字段；
隐藏完整卡的值、记忆中的值、其他目标或其他步骤的数值都不能补写。
若没有本轮实供，改用“尚未核验”而非猜数；不要靠尾部免责声明抵消前文断言。
"""
)


def receipt_reuse_report_policy() -> dict[str, Any]:
    return dict(
        protocol=SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
        system=RECEIPT_REUSE_REPORT_SYSTEM,
        schema="nurse.v2.sre-report-input/15",
        profile="hermes-sre-tools/13",
        output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
            system=RECEIPT_REUSE_REPORT_SYSTEM,
            output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ), REPORT_LIMITS)),
    )
