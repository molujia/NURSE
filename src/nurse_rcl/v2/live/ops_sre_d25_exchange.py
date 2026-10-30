"""D25 model contract: source-bound progress outranks old public intent."""

# ruff: noqa: RUF001 -- intentional model-facing Chinese punctuation.

from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_comparable_exchange import COMPARABLE_REPORT_OUTPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_d23_exchange import BALANCED_DECISION_SYSTEM, BALANCED_REPORT_SYSTEM
from nurse_rcl.v2.live.ops_sre_skills import replanning_common_guidance

REPLANNING_DECISION_SYSTEM = BALANCED_DECISION_SYSTEM.replace(
    "nurse.v2.sre-model-view/8", "nurse.v2.sre-model-view/9"
).replace(
    "investigation_workbench", "investigation_progress"
).replace(
    "D22：若精确完整已读回执被复用，最新动作会显示receipt_reuse和no_new_content。\n"
    "这是一次已执行但没有新证据的只读动作，不是新观测、独立支持或待重复的旧计划。\n",
    "D25：精确完整已读拟议不会重放或再次派发；解释可引用已保存来源。\n",
).replace(
    "可复用原完整回执；纠正请求", "可在解释中引用原完整回执，不再次派发；纠正请求"
).replace(
    "允许有理由的重读/转向/停止", "允许读取尚未完整交付或来源变化的记录、转向或停止"
) + """
D25：首先查看investigation_progress中的最近实际回执、双方已读/未读角色及有限字段对照。
对照只是同epoch完整实供、正式已用卡的原值；same不代表机制相同或因果已确认，
different不代表分差的因果贡献，not_supplied或容量省略不能补猜。
public_working_memory仅保留历史问题和回执关联，不把旧public_summary/unresolved意图
当当前待执行命令；原文在原conversation，不由程序判问题已解决。
在解释中引用已有回执无需再次read_node。精确同源同参完整重复不会提供新信息，
首次可获全案唯一纠正：根据已读来源、尚未读角色和合法入口自主重新选择或停止。
纠正仍精确重复或纠正已用后再重复，将partial/no_progress_after_correction停止，
不派发、不重放、不占工具动作；不得把剩余动作当已执行或把未查范围写成排除。
角色入口不规定顺序、不替你选择下一动作，固定排名不由调查结果或你的判断改变。
"""
REPLANNING_REPORT_SYSTEM = BALANCED_REPORT_SYSTEM.replace(
    "model-view/9", "model-view/10"
) + """
D25报告必须如实保留no_progress_after_correction部分终态和未查范围。
进度板的字段对照不增加独立证据；技术陈述仍仅引用原始实供卡和实际步骤。
未派发的重复拟议或纠正不是成功调查动作，不编造转向、收敛或检查结果。
"""


def replanning_decision_policy() -> dict[str, Any]:
    protocol = "nurse.v2.sre-decision-exchange/16"
    return dict(
        protocol=protocol, system=REPLANNING_DECISION_SYSTEM,
        schema="nurse.v2.sre-investigation-input/16", profile="hermes-sre-tools/15",
        guidance=replanning_common_guidance(),
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=protocol, system=REPLANNING_DECISION_SYSTEM,
            output_policy=OUTPUT_POLICY, max_tokens=MAX_OUTPUT_TOKENS,
        ), CONTEXT_LIMITS)),
    )


def replanning_report_policy() -> dict[str, Any]:
    protocol = "nurse.v2.sre-report-exchange/18"
    return dict(
        protocol=protocol, system=REPLANNING_REPORT_SYSTEM,
        schema="nurse.v2.sre-report-input/17", profile="hermes-sre-tools/15",
        output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
        prompt_sha256=sha256_hex(encode_json(dict(
            protocol=protocol, system=REPLANNING_REPORT_SYSTEM,
            output_schema=COMPARABLE_REPORT_OUTPUT_SCHEMA,
            output_policy=OUTPUT_POLICY, max_tokens=MAX_OUTPUT_TOKENS,
        ), REPORT_LIMITS)),
    )
