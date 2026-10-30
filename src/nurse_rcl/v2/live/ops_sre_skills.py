"""Versioned SRE skills shipped with NURSE, independent of a Hermes installation."""

# ruff: noqa: RUF001 -- Chinese punctuation is intentional in model-facing skill guidance.

from __future__ import annotations

from importlib.resources import files
from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live.ops_sre_tools import SRE_SKILL_TOOLS

_EXAMPLES: dict[str, list[dict[str, Any]]] = {
    "boundary": [dict(tool="inspect_case", parameters=dict(cursor=None))],
    "symptoms": [
        dict(
            tool="inspect_candidate",
            parameters=dict(candidate_service_key="<admitted_candidate>", cursor=None),
        ),
        dict(tool="measure_signal", parameters=dict(request_key="<registered_request>")),
    ],
    "dependency": [
        dict(
            tool="inspect_dependency",
            parameters=dict(
                caller_service_key="<registered_caller>",
                callee_service_key="<registered_callee>",
                cursor=None,
            ),
        )
    ],
    "timing": [
        dict(
            tool="compare_timing",
            parameters=dict(
                source_onset_id="<measured_source_onset>", target_onset_id="<measured_target_onset>"
            ),
        )
    ],
    "mechanisms": [dict(tool="check_mechanism", parameters=dict(card_id="<published_mechanism>"))],
    "scoring": [
        dict(
            tool="compare_candidates",
            parameters=dict(
                left_candidate_service_key="<admitted_left>",
                right_candidate_service_key="<different_admitted_right>",
                cursor=None,
            ),
        )
    ],
}


def load_sre_skill(skill_id: str) -> str:
    if skill_id.endswith("/20"):
        from nurse_rcl.v2.live.ops_sre_focused_tools import FOCUSED_SKILLS

        if skill_id in FOCUSED_SKILLS:
            return load_sre_skill(skill_id.removesuffix("/20") + "/19")
    from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import ATOMIC_SKILLS

    if skill_id in ATOMIC_SKILLS:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import ATOMIC_SKILL_GUIDANCE

        return ATOMIC_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_typed_purpose_tools import TYPED_SKILLS

    if skill_id in TYPED_SKILLS:
        from nurse_rcl.v2.live.ops_sre_typed_purpose_exchange import TYPED_SKILL_GUIDANCE

        return TYPED_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_observable_tools import OBSERVABLE_SKILLS

    if skill_id in OBSERVABLE_SKILLS:
        from nurse_rcl.v2.live.ops_sre_observable_exchange import OBSERVABLE_SKILL_GUIDANCE

        return OBSERVABLE_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS

    if skill_id in ENTRY_SKILLS:
        from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_SKILL_GUIDANCE

        return ENTRY_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_directed_tools import DIRECTED_SKILL_TOOLS

    if skill_id in DIRECTED_SKILL_TOOLS:
        from nurse_rcl.v2.live.ops_sre_directed_exchange import DIRECTED_SKILL_GUIDANCE

        return DIRECTED_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_d26_tools import QUESTION_SKILL_TOOLS

    if skill_id in QUESTION_SKILL_TOOLS:
        from nurse_rcl.v2.live.ops_sre_d26_exchange import QUESTION_SKILL_GUIDANCE

        return QUESTION_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
    from nurse_rcl.v2.live.ops_sre_d25_tools import REPLANNING_SKILL_TOOLS

    if skill_id in REPLANNING_SKILL_TOOLS:
        return load_sre_skill(skill_id.removesuffix("/12") + "/11")
    from nurse_rcl.v2.live.ops_sre_d23_tools import BALANCED_SKILL_TOOLS

    if skill_id in BALANCED_SKILL_TOOLS:
        return load_sre_skill(skill_id.removesuffix("/11") + "/10")
    from nurse_rcl.v2.live.ops_sre_d22_tools import RECEIPT_REUSE_SKILL_TOOLS

    if skill_id in RECEIPT_REUSE_SKILL_TOOLS:
        return load_sre_skill(skill_id.removesuffix("/10") + "/9")
    from nurse_rcl.v2.live.ops_sre_d21_tools import USED_SOURCE_SKILL_TOOLS

    if skill_id in USED_SOURCE_SKILL_TOOLS:
        return load_sre_skill(skill_id.removesuffix("/9") + "/8")
    from nurse_rcl.v2.live.ops_sre_workbench_tools import WORKBENCH_SKILL_TOOLS

    if skill_id in WORKBENCH_SKILL_TOOLS:
        if skill_id == "sre-scoring/8":
            return (
                files("nurse_rcl.v2.live")
                .joinpath("ops_skills")
                .joinpath("sre-workbench-scoring.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return load_sre_skill(skill_id.removesuffix("/8") + "/6")
    from nurse_rcl.v2.live.ops_sre_navigation_tools import NAVIGATION_SKILL_TOOLS

    if skill_id in NAVIGATION_SKILL_TOOLS:
        original = load_sre_skill(skill_id.removesuffix("/7") + "/6")
        if skill_id == "sre-scoring/7":
            original += (
                "\n\n"
                + files("nurse_rcl.v2.live")
                .joinpath("ops_skills")
                .joinpath("sre-navigation-scoring.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return original
    from nurse_rcl.v2.live.ops_sre_brief_tools import BRIEF_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_comparable_tools import COMPARABLE_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_condition_tools import CONDITION_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_recovery_tools import RECOVERY_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_working_tools import WORKING_SKILL_TOOLS

    if (
        skill_id not in SRE_SKILL_TOOLS
        and skill_id not in WORKING_SKILL_TOOLS
        and (skill_id not in BRIEF_SKILL_TOOLS)
        and skill_id not in CONDITION_SKILL_TOOLS
        and skill_id not in RECOVERY_SKILL_TOOLS
        and skill_id not in COMPARABLE_SKILL_TOOLS
    ):
        raise ValueError("SRE skill ID is not registered")
    name = skill_id.rsplit("/", 1)[0]
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    if skill_id in COMPARABLE_SKILL_TOOLS:
        if name in {"sre-mechanisms", "sre-scoring"}:
            return (
                folder.joinpath(name.replace("sre-", "sre-comparable-", 1) + ".md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return load_sre_skill(name + "/5")
    if (
        skill_id in BRIEF_SKILL_TOOLS
        or skill_id in CONDITION_SKILL_TOOLS
        or skill_id in RECOVERY_SKILL_TOOLS
    ):
        name = name.replace("sre-", "sre-brief-", 1)
    if (skill_id in CONDITION_SKILL_TOOLS or skill_id in RECOVERY_SKILL_TOOLS) and name in {
        "sre-brief-mechanisms",
        "sre-brief-symptoms",
        "sre-brief-scoring",
    }:
        name = name.replace("sre-brief-", "sre-condition-", 1)
    text = folder.joinpath(name + ".md").read_text(encoding="utf-8").strip()
    if skill_id in WORKING_SKILL_TOOLS:
        text += (
            "\n\n" + folder.joinpath("sre-working-feedback.md").read_text(encoding="utf-8").strip()
        )
    return text


def brief_common_guidance() -> dict[str, str]:
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    text = folder.joinpath("sre-brief-common.md").read_text(encoding="utf-8").strip()
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def condition_common_guidance() -> dict[str, str]:
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    text = folder.joinpath("sre-condition-common.md").read_text(encoding="utf-8").strip()
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def recovery_common_guidance() -> dict[str, str]:
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    text = (
        condition_common_guidance()["guidance"]
        + "\n\n"
        + folder.joinpath("sre-recovery-common.md").read_text(encoding="utf-8").strip()
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def comparable_common_guidance() -> dict[str, str]:
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    text = (
        recovery_common_guidance()["guidance"]
        + "\n\n"
        + folder.joinpath("sre-comparable-common.md").read_text(encoding="utf-8").strip()
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def sre_skill_index(*, profile: str = "hermes-sre-tools/4") -> tuple[dict[str, Any], ...]:
    if profile == "hermes-sre-tools/25":
        return tuple(
            dict(row, skill_id=row["skill_id"].removesuffix("/19") + "/20")
            for row in sre_skill_index(profile="hermes-sre-tools/23")
        )
    if profile in {"hermes-sre-tools/22", "hermes-sre-tools/23"}:
        from nurse_rcl.v2.live.ops_sre_typed_purpose_exchange import TYPED_SKILL_GUIDANCE
        from nurse_rcl.v2.live.ops_sre_typed_purpose_tools import TYPED_SKILLS

        if profile == "hermes-sre-tools/23":
            from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import ATOMIC_SKILL_GUIDANCE
            from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import ATOMIC_SKILLS

            active_skills, active_guidance = ATOMIC_SKILLS, ATOMIC_SKILL_GUIDANCE
        else:
            active_skills, active_guidance = TYPED_SKILLS, TYPED_SKILL_GUIDANCE
        rows = []
        for skill_id, tools in active_skills.items():
            node = skill_id.removeprefix("sre-").rsplit("/", 1)[0]
            guidance = active_guidance[node]
            rows.append(
                dict(
                    skill_id=skill_id,
                    node=node,
                    tools=list(tools),
                    guidance=guidance,
                    examples=[
                        "<current_issued_action_entry>"
                        if profile == "hermes-sre-tools/23"
                        else dict(entry_ref="<current_issued_action_entry>"),
                        dict(
                            navigation=dict(
                                kind="candidate",
                                candidate_service_key="<issued_service>",
                                focus=node
                                if node in {"symptoms", "mechanisms", "scoring"}
                                else "all",
                                cursor=None,
                            ),
                            question="Find published sources for this scope.",
                        ),
                    ],
                    example_policy="placeholders are not IDs; copy exact values from receipts",
                    parameter_contracts={t: "capabilities.parameter_contracts." + t for t in tools},
                    content_sha256=sha256_hex(guidance.encode()),
                )
            )
        return tuple(rows)
    if profile == "hermes-sre-tools/21":
        from nurse_rcl.v2.live.ops_sre_single_owner_exchange import SINGLE_OWNER_SKILL_GUIDANCE

        rows = []
        for original in sre_skill_index(profile="hermes-sre-tools/19"):
            skill_id = original["skill_id"].removesuffix("/15") + "/16"
            guidance = SINGLE_OWNER_SKILL_GUIDANCE[skill_id.removeprefix("sre-").rsplit("/", 1)[0]]
            rows.append(
                dict(
                    original,
                    skill_id=skill_id,
                    guidance=guidance,
                    content_sha256=sha256_hex(guidance.encode()),
                )
            )
        return tuple(rows)
    if profile == "hermes-sre-tools/20":
        rows = []
        for original in sre_skill_index(profile="hermes-sre-tools/19"):
            skill_id = original["skill_id"].removesuffix("/15") + "/16"
            guidance = load_sre_skill(skill_id)
            rows.append(
                dict(
                    original,
                    skill_id=skill_id,
                    guidance=guidance,
                    content_sha256=sha256_hex(guidance.encode()),
                )
            )
        return tuple(rows)
    from nurse_rcl.v2.live.ops_sre_brief_tools import BRIEF_PROFILE, BRIEF_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_comparable_tools import (
        COMPARABLE_PROFILE,
        COMPARABLE_SKILL_TOOLS,
    )
    from nurse_rcl.v2.live.ops_sre_condition_tools import CONDITION_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_d21_tools import USED_SOURCE_PROFILE, USED_SOURCE_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_d22_tools import (
        RECEIPT_REUSE_PROFILE,
        RECEIPT_REUSE_SKILL_TOOLS,
    )
    from nurse_rcl.v2.live.ops_sre_d23_tools import BALANCED_PROFILE, BALANCED_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_d25_tools import REPLANNING_PROFILE, REPLANNING_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_d26_tools import QUESTION_PROFILE, QUESTION_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_directed_tools import DIRECTED_PROFILE, DIRECTED_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_feedback_context import CONDITION_PROFILE
    from nurse_rcl.v2.live.ops_sre_navigation_tools import (
        NAVIGATION_PROFILE,
        NAVIGATION_SKILL_TOOLS,
    )
    from nurse_rcl.v2.live.ops_sre_recovery_tools import RECOVERY_PROFILE, RECOVERY_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_workbench_tools import WORKBENCH_PROFILE, WORKBENCH_SKILL_TOOLS
    from nurse_rcl.v2.live.ops_sre_working_tools import WORKING_PROFILE, WORKING_SKILL_TOOLS

    if profile not in {
        "hermes-sre-tools/4",
        WORKING_PROFILE,
        BRIEF_PROFILE,
        CONDITION_PROFILE,
        RECOVERY_PROFILE,
        COMPARABLE_PROFILE,
        NAVIGATION_PROFILE,
        WORKBENCH_PROFILE,
        USED_SOURCE_PROFILE,
        RECEIPT_REUSE_PROFILE,
        BALANCED_PROFILE,
        REPLANNING_PROFILE,
        QUESTION_PROFILE,
        "hermes-sre-tools/17",
        DIRECTED_PROFILE,
        "hermes-sre-tools/19",
    }:
        raise ValueError("SRE skill profile differs")
    working = profile == WORKING_PROFILE
    brief = profile in {
        BRIEF_PROFILE,
        CONDITION_PROFILE,
        RECOVERY_PROFILE,
        COMPARABLE_PROFILE,
        NAVIGATION_PROFILE,
        WORKBENCH_PROFILE,
        USED_SOURCE_PROFILE,
        RECEIPT_REUSE_PROFILE,
        BALANCED_PROFILE,
        REPLANNING_PROFILE,
        QUESTION_PROFILE,
        "hermes-sre-tools/17",
        DIRECTED_PROFILE,
        "hermes-sre-tools/19",
    }
    from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS

    registry = (
        ENTRY_SKILLS
        if profile == "hermes-sre-tools/19"
        else DIRECTED_SKILL_TOOLS
        if profile == DIRECTED_PROFILE
        else QUESTION_SKILL_TOOLS
        if profile in {QUESTION_PROFILE, "hermes-sre-tools/17"}
        else REPLANNING_SKILL_TOOLS
        if profile == REPLANNING_PROFILE
        else BALANCED_SKILL_TOOLS
        if profile == BALANCED_PROFILE
        else RECEIPT_REUSE_SKILL_TOOLS
        if profile == RECEIPT_REUSE_PROFILE
        else USED_SOURCE_SKILL_TOOLS
        if profile == USED_SOURCE_PROFILE
        else WORKBENCH_SKILL_TOOLS
        if profile == WORKBENCH_PROFILE
        else NAVIGATION_SKILL_TOOLS
        if profile == NAVIGATION_PROFILE
        else COMPARABLE_SKILL_TOOLS
        if profile == COMPARABLE_PROFILE
        else RECOVERY_SKILL_TOOLS
        if profile == RECOVERY_PROFILE
        else CONDITION_SKILL_TOOLS
        if profile == CONDITION_PROFILE
        else BRIEF_SKILL_TOOLS
        if brief
        else WORKING_SKILL_TOOLS
        if working
        else SRE_SKILL_TOOLS
    )
    rows = []
    for skill_id, tools in registry.items():
        node = skill_id.removeprefix("sre-").rsplit("/", 1)[0]
        guidance = load_sre_skill(skill_id)
        if profile == "hermes-sre-tools/17":
            from nurse_rcl.v2.live.ops_sre_action_exchange import ACTION_SKILL_GUIDANCE

            guidance = ACTION_SKILL_GUIDANCE[node]
        examples = [dict(item, parameters=dict(item["parameters"])) for item in _EXAMPLES[node]]
        if working or brief:
            for item in examples:
                if item["tool"] in {"inspect_candidate", "compare_candidates"}:
                    item["parameters"]["focus"] = (
                        node if node in {"symptoms", "mechanisms", "scoring"} else "all"
                    )
        if brief and node == "mechanisms":
            examples.append(
                dict(
                    tool="read_node",
                    parameters=dict(
                        card_id="<visible_exact_mechanism>", section="payload", cursor=None
                    ),
                )
            )
        if profile == DIRECTED_PROFILE and node == "symptoms":
            examples.append(
                dict(
                    tool="inspect_sources",
                    parameters=dict(
                        card_ids=["<issued_symptom_card>"],
                        candidate_service_keys=["<issued_candidate>"],
                    ),
                )
            )
        if profile == "hermes-sre-tools/19":
            examples = [
                dict(
                    entry_ref="<current_issued_action_entry>",
                    question="Check the actual supplied source.",
                ),
                dict(
                    navigation=dict(
                        kind="candidate",
                        candidate_service_key="<issued_service>",
                        focus=node if node in {"symptoms", "mechanisms", "scoring"} else "all",
                        cursor=None,
                    ),
                    question="Find admitted sources for this scope.",
                ),
            ]
        rows.append(
            dict(
                skill_id=skill_id,
                node=node,
                tools=list(tools),
                guidance=guidance,
                examples=examples,
                example_policy="placeholders are not IDs; copy exact values from receipts",
                # The single capabilities table contains exact parameter contracts;
                # six skills should not repeat the same navigation/tool schemas.
                parameter_contracts={t: "capabilities.parameter_contracts." + t for t in tools},
                content_sha256=sha256_hex(guidance.encode("utf-8")),
            )
        )
    return tuple(rows)


def workbench_common_guidance() -> dict[str, str]:
    text = (
        files("nurse_rcl.v2.live")
        .joinpath("ops_skills")
        .joinpath("sre-workbench-common.md")
        .read_text(encoding="utf-8")
        .strip()
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def used_source_common_guidance() -> dict[str, str]:
    text = workbench_common_guidance()["guidance"] + (
        "\n\nD21：public_working_memory是执行前意图；先核对post_action_state里的实际最后动作、"
        "精确交付和剩余动作。双侧used_source_contrast仅为实际已用来源的有损字段副本，"
        "不能推断未读子卡、健康或机制相同。精确完整重复可被派发前阻止，纠正时自主改选或停止。"
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def balanced_common_guidance() -> dict[str, str]:
    text = used_source_common_guidance()["guidance"] + (
        "\n\nD23：先读最新真实回执及content_relation，再读双侧三角色来源摘要与已读覆盖。"
        "complete只表示该完整表示已交付，不等于候选域已读；brief/partial与未读另列。"
        "切换focus时cursor=null；不得复用不同tool、scope、focus或prefix的cursor。"
        "余额大于0不得称budget_exhausted，可因无有益检查诚实停止；不要把目录计数当票数。"
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def navigation_common_guidance() -> dict[str, str]:
    folder = files("nurse_rcl.v2.live").joinpath("ops_skills")
    text = (
        comparable_common_guidance()["guidance"]
        + "\n\n"
        + folder.joinpath("sre-navigation-common.md").read_text(encoding="utf-8").strip()
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))


def replanning_common_guidance() -> dict[str, str]:
    text = balanced_common_guidance()["guidance"].replace(
        "精确重复不会增加独立支持，仍可为核对而选择且照常计预算；",
        "精确同源同参完整重复不会再次派发；解释可引用已保存回执；",
    ).replace(
        "可以续页、直接定向、重读、转向或停止",
        "可以续页、直接定向、读取未完整交付或来源变化的记录、转向或停止",
    ) + (
        "\n\nD25：先用investigation_progress的实际已读来源、最新反馈和未读角色决定下一检查。"
        "同家族对照只是完整实供原值，不是新证据或因果结论；缺侧/unknown/省略不推断相等。"
        "旧public_summary/unresolved意图不再作为当前待执行计划；问题是否已解由你判断。"
        "解释中复用旧回执无需read_node；唯一纠正后仍精确重复则部分停止，未派发不算动作。"
    )
    return dict(guidance=text, content_sha256=sha256_hex(text.encode("utf-8")))
