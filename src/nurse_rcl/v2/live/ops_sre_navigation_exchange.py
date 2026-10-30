"""D17 versioned navigation contract; old prompts remain unchanged."""

# ruff: noqa: RUF001 -- Chinese model-facing contract
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_exchange import (
    COMPARABLE_DECISION_SYSTEM,
    COMPARABLE_REPORT_OUTPUT_SCHEMA,
    COMPARABLE_REPORT_SYSTEM,
)
from nurse_rcl.v2.live.ops_sre_delivered_navigation import NAVIGATION_INPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_section_guidance import REPORT_CONTRACT_EXAMPLES

NAVIGATION_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/11"
NAVIGATION_DECISION_SYSTEM = (
    COMPARABLE_DECISION_SYSTEM
    + """
delivered_navigation 仅登记成功回执实际给出的详情链接，source 标明源卡与记录，
links 标明实际 kind、primary_family/term_id/path_id 和精确 source_path。
这些目标可用于 investigation_question.card_ids 和合法 read_node 参数，尚不代表读到目标正文。
responds_to 仍只能引用 actions 的 visible_card_ids；看到子入口时回应实际给出入口的源卡。
评分摘要不是家族或 term；使用真实 kind/primary_family，不按短ID编号或 selected_family_keys
顺序猜 family_details 类型。重复读摘要不会展开子项。可自主续页、读详情、转向或停止。
只有未读明细问题而无观测前提时，将其保留在 unresolved，不编造条件预测。
"""
)
NAVIGATION_DECISION_PROMPT_SHA256 = sha256_hex(
    encode_json(
        dict(
            protocol=NAVIGATION_DECISION_PROTOCOL,
            system=NAVIGATION_DECISION_SYSTEM,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
        CONTEXT_LIMITS,
    )
)


def navigation_decision_policy() -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_skills import navigation_common_guidance

    return dict(
        protocol=NAVIGATION_DECISION_PROTOCOL,
        system=NAVIGATION_DECISION_SYSTEM,
        schema=NAVIGATION_INPUT_SCHEMA,
        profile="hermes-sre-tools/10",
        guidance=navigation_common_guidance(),
        prompt_sha256=NAVIGATION_DECISION_PROMPT_SHA256,
    )


NAVIGATION_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/13"
NAVIGATION_REPORT_SYSTEM = (
    COMPARABLE_REPORT_SYSTEM
    + """
section_reference_guide 指向原 supply，不是新增证据。正式比较用 formal_interpretations 的
ranking_refs/competition_refs；step_findings 只引用各步原 delivered 集合，不能借全局正式事实。
只有评分材料也可按实供字段作有限综合；摘要、家族、term、传播汇总和路径层次不得混淆。
缺少观测前提时不构造 next_checks；保留实际公开未决问题，不把未读入口当成条件预测。
investigation_failure 是实际捕获的有界过程元信息，null 表示没有可用的新版记录，不能猜原因。
决策未准入不等于执行了模型拟议工具。只解释最后合法公开问题，不将未准入回复补进调查历史。
以下为独立假设 fixture 的格式示例，不是本案事实，禁止照搬示例引用、目标或值。
示例假设 f2 是双目标正式 competition；f3/f4 是 s1 实供评分字段。
"""
    + canonical_json_bytes(REPORT_CONTRACT_EXAMPLES).decode()
)
NAVIGATION_REPORT_PROMPT_SHA256 = sha256_hex(
    encode_json(
        dict(
            protocol=NAVIGATION_REPORT_PROTOCOL,
            system=NAVIGATION_REPORT_SYSTEM,
            output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
            output_policy=OUTPUT_POLICY,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
        REPORT_LIMITS,
    )
)


def navigation_report_policy() -> dict[str, str]:
    return dict(
        protocol=NAVIGATION_REPORT_PROTOCOL,
        system=NAVIGATION_REPORT_SYSTEM,
        schema="nurse.v2.sre-report-input/12",
        profile="hermes-sre-tools/10",
        output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
        prompt_sha256=NAVIGATION_REPORT_PROMPT_SHA256,
    )
