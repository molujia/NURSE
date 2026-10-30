# ruff: noqa: RUF001 -- Chinese reporting punctuation
"""D13 human report: condition-level caveats, complete bound local appendix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_report_render import _text
from nurse_rcl.v2.live.ops_sre_brief_claims import (
    BriefReportResponse,
    ConditionReportResponse,
)
from nurse_rcl.v2.live.ops_sre_brief_report import (
    AtomicReportMetadata,
    BriefSreReportContext,
    ComparableReportMetadata,
    ConditionReportMetadata,
    FocusedReportMetadata,
    NavigationReportMetadata,
    QuestionReportMetadata,
    RecoveryReportMetadata,
    WorkbenchReportMetadata,
    brief_report_supply,
    validate_brief_report,
)
from nurse_rcl.v2.live.ops_sre_comparable_claims import ComparableReportResponse
from nurse_rcl.v2.live.ops_sre_report_check_policy import (
    _AdmittedBriefChecks,
    _require_admitted_checks,
    admit_brief_checks,
    check_identity,
)
from nurse_rcl.v2.live.ops_sre_report_context import critical_fact
from nurse_rcl.v2.live.ops_sre_report_delivery import _inspect_report_pair, _save_report_pair
from nurse_rcl.v2.live.ops_sre_report_render import (
    _GENERAL_RULES,
    _LIMIT_TEXT,
    _cost_lines,
    _excerpt,
    service_aliases,
)

BRIEF_RENDER_POLICY = "sre-readable-pair/4"
CONDITION_RENDER_POLICY = "sre-readable-pair/5"
RECOVERY_RENDER_POLICY = "sre-readable-pair/6"


def brief_render_policy(context: BriefSreReportContext) -> str:
    if type(context.metadata) is FocusedReportMetadata:
        return "sre-readable-pair/15"
    if type(context.metadata) is AtomicReportMetadata:
        return "sre-readable-pair/14"
    if isinstance(context.metadata, QuestionReportMetadata):
        return "sre-readable-pair/11"
    if isinstance(context.metadata, WorkbenchReportMetadata):
        return "sre-readable-pair/9"
    if isinstance(context.metadata, NavigationReportMetadata):
        return "sre-readable-pair/8"
    if isinstance(context.metadata, ComparableReportMetadata):
        return "sre-readable-pair/7"
    if isinstance(context.metadata, RecoveryReportMetadata):
        return RECOVERY_RENDER_POLICY
    return (
        CONDITION_RENDER_POLICY
        if isinstance(context.metadata, ConditionReportMetadata)
        else BRIEF_RENDER_POLICY
    )


def only_directory_or_scoring(
    completed: list[dict[str, Any]],
    meanings: dict[str, Any],
) -> bool:
    """Describe delivered scope only; never judge a cause or choose a next tool."""
    score_kinds = {
        "score_lineage_summary",
        "score_family",
        "score_term",
        "score_propagation",
        "score_path",
    }
    if not completed:
        return False
    for row in completed:
        tool, params = row.get("tool"), row.get("parameters", {})
        if tool == "inspect_case" or (
            tool in {"inspect_candidate", "compare_candidates"} and params.get("focus") == "scoring"
        ):
            continue
        bindings = [b for b in meanings["bindings"] if row["action_index"] in b["action_indices"]]
        if (
            tool != "read_node"
            or not bindings
            or any(
                not b["entry_keys"] or not set(b["entry_keys"]) <= score_kinds | {"boundary"}
                for b in bindings
            )
        ):
            return False
    return True


_ROLES = {"necessary": "必要条件", "refuting": "反证条件", "optional": "可选条件"}
_TRUTH = {"true": "成立", "false": "不成立", "unknown": "未知"}
_RESOLUTION = {
    "decisive_known_leaf": "仅补齐未知不会翻转当前真值。",
    "fully_observed": "当前没有待补齐的适用未知；不表示因果确认。",
    "unresolved_applicable_leaves": "未知项可能改变当前条件，是否有可用检查另看真实工具入口。",
    "no_applicable_evidence": "没有适用叶子，无法判断，不能当作成立或正常。",
}


def _window_text(window: list[int] | None) -> str:
    if window is None:
        return "未给出时间范围"
    return (
        " — ".join(datetime.fromtimestamp(v / 1e9, timezone.utc).isoformat() for v in window)
        + "（UTC，精确 ns 见附件）"
    )


def condition_lines(
    briefs: list[dict[str, Any]], aliases: dict[str, str], *, readable: bool | None = None
) -> list[str]:
    """Show every condition, not only model-selected or favorable conditions."""
    if readable is None:
        readable = any("resolution_state" in b["fields"] for b in briefs)
    lines = [
        "计数描述同一条件内部的叶子，不是独立支持票数；OR 成立不表示所有叶子成立，未知不等于正常。",
        "",
    ]
    previous = None
    for brief in briefs:
        identity = brief["source_record_sha256"]
        fields = brief["fields"]
        if identity != previous:
            labels = "、".join(aliases.get(k, k) for k in brief["target_service_keys"])
            lines += [
                "### " + _text(labels + " / " + str(fields.get("mechanism_key", "机制"))),
                "",
                "来源角色："
                + _text(brief["source_role"])
                + ("；窗口：" if readable else "；窗口 ns：")
                + _text(_window_text(brief["window_ns"]) if readable else str(brief["window_ns"]))
                + (
                    "。原卡片身份、完整来源及精确叶子见本组证据链接的附件。"
                    if readable
                    else "；原卡片："
                    + _text(str(brief["card_id"]))
                    + "。完整来源及精确叶子见附件。"
                ),
                "",
            ]
            if brief["limitations"]:
                lines += [
                    "来源限制："
                    + _text(
                        "；".join(_LIMIT_TEXT.get(code, code) for code in brief["limitations"])
                    ),
                    "",
                ]
            previous = identity
        ref = brief["evidence_ref"]
        link = f"[{ref}](report-evidence.md#{ref})"
        if brief["granularity"] in {"mechanism_decision", "formal_mechanism_reference"}:
            lines += [
                (
                    "补充复核引用的原机制判定（不是新的正式判定）："
                    if brief["granularity"] == "formal_mechanism_reference"
                    else "原机制判定："
                )
                + _text(str(fields.get("disposition", "未记录")))
                + "；因果状态："
                + _text(str(fields.get("causal_status", "未记录")))
                + "。"
                + link,
                "",
            ]
            continue
        if brief["granularity"] == "supplementary_support":
            lines += [
                "补充复核必要条件汇总："
                + _text(
                    _TRUTH.get(
                        fields["supplementary_necessary_support"],
                        fields["supplementary_necessary_support"],
                    )
                )
                + "；没有正式评分权，不是新的正式判定。"
                + link,
                "",
            ]
            continue
        role = _ROLES.get(fields.get("requirement"), str(fields.get("requirement")))
        condition_label = (
            ref if readable else fields.get("condition_id", fields.get("formal_condition_id"))
        )
        truth = _TRUTH.get(fields.get("truth_value"), str(fields.get("truth_value")))
        counts = "、".join(
            f"{key}={fields.get(key + '_count', 0)}" for key in ("true", "false", "unknown")
        )
        reasons = brief["lists"].get("reason_codes", [])
        impact = (
            "必要项尚未成立，不能视为机制已证实。"
            if role == "必要条件" and truth != "成立"
            else "反证成立，必须与支持条件共同解释。"
            if role == "反证条件" and truth == "成立"
            else "保留条件本身的范围，不外推全服务健康或因果。"
        )
        lines += [
            "- "
            + _text(
                f"{role} {condition_label}："
                f"{str(fields.get('operator')).upper()}，{truth}；{counts}；"
                f"非适用={fields.get('non_applicable_count', 0)}；"
                f"原因={','.join(reasons) or '未列出'}。" + impact
            )
            + " "
            + link,
            "",
        ]
        if "resolution_state" in fields:
            lines += [
                "  "
                + _RESOLUTION[fields["resolution_state"]]
                + "前提是同一条件结构与窗口、既有已知叶子不变；"
                "新的测量若替换已知叶子，结论可能改变。",
                "",
            ]
    return lines


def grouped_limit_lines(
    context: BriefSreReportContext,
    supply: dict[str, Any],
    excluded: set[str],
    aliases: dict[str, str],
    *,
    readable_facts: bool = False,
    navigation: bool = False,
    workbench: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Share inherited coverage wording only; keep individual semantic states and refs."""
    source_ids = {
        r: b["record_sha256"] for b in context.metadata.source_bindings for r in b["fact_refs"]
    }
    score_kinds: dict[str, str] = {}
    if navigation:
        from nurse_rcl.v2.live.ops_sre_navigation_render import SCORE_TYPES

        for binding in context.metadata.source_bindings:
            kinds = {
                v.value
                for ref in binding["fact_refs"]
                for v in context.fact_store.resolve(ref).fact.values
                if v.name == "kind" and isinstance(v.value, str) and v.value in SCORE_TYPES
            }
            if len(kinds) == 1:
                score_kinds.update(dict.fromkeys(binding["fact_refs"], next(iter(kinds))))
    groups: dict[bytes, dict[str, Any]] = {}
    for ref in supply["provided_fact_refs"]:
        if ref in excluded:
            continue
        item = context.fact_store.resolve(ref)
        if not critical_fact(item):
            continue
        values = {v.name: v.value for v in item.fact.values}
        codes = sorted(set(item.fact.status_codes) - _GENERAL_RULES)
        # Predicate/episode results are independent even when their coverage is inherited.
        distinct = bool(
            set(values)
            & {
                "truth_value",
                "support_value",
                "persistent_up",
                "verified_normal",
                "predicate_id",
                "episode_id",
                "condition_id",
                "status",
            }
        )
        if not codes and not distinct and item.fact.kind not in {"limit", "uncertainty"}:
            continue
        identity = dict(
            source=source_ids.get(ref, ref),
            targets=list(item.fact.target_service_keys),
            intervals=[i.model_dump(mode="json") for i in item.fact.intervals],
            role=item.role,
            origin=item.origin,
            codes=codes,
        )
        if navigation:
            # Formal mechanism summaries also carry condition roles, even when
            # the agent has not obtained a supplementary condition brief.
            # Keep their supplied semantics separate; never infer a missing window.
            identity["scope_fields"] = {
                key: values[key]
                for key in (
                    "mechanism_key",
                    "requirement",
                    "predicate_kind",
                    "necessary_conditions",
                    "refuting_conditions",
                    "optional_conditions",
                )
                if key in values
            }
        key = canonical_json_bytes(identity)
        group = groups.setdefault(key, dict(identity, refs=[], independent=[]))
        group["refs"].append(ref)
        if distinct:
            group["independent"].append(dict(ref=ref, path=item.source_path, values=values))
    lines = []
    mapping = {}
    for index, group in enumerate(groups.values(), 1):
        anchor = f"limit-group-{index}"
        mapping[anchor] = group
        scope = "、".join(aliases.get(k, k) for k in group["targets"]) or "案例范围"
        scope_fields = group.get("scope_fields", {})
        if scope_fields:
            labels = {
                "mechanism_key": "机制",
                "requirement": "条件角色",
                "predicate_kind": "谓词",
                "necessary_conditions": "必要条件",
                "refuting_conditions": "反证条件",
                "optional_conditions": "可选条件",
            }
            for key, value in scope_fields.items():
                shown = _ROLES.get(str(value), str(value)) if key == "requirement" else str(value)
                scope += "；" + labels[key] + "：" + (shown or "本字段为空，不推断正常")
            scope += "；窗口：" + (
                "、".join(
                    _window_text([i["start_unix_ns"], i["end_unix_ns"]]) for i in group["intervals"]
                )
                or _window_text(None)
            )
        meaning = (
            "查询范围可见性，不是服务健康判断"
            if group["origin"] == "query_scope_metadata"
            else "同源记录的范围限制"
        )
        lines += [
            "- "
            + _text(
                scope
                + "；"
                + meaning
                + "："
                + "；".join(_LIMIT_TEXT.get(c, c) for c in group["codes"])
            )
            + f"（对应 {len(group['refs'])} 项子字段）[完整映射](report-evidence.md#{anchor})",
            "",
        ]
        for state in group["independent"]:
            from nurse_rcl.v2.live.ops_sre_navigation_render import readable_score_fact
            from nurse_rcl.v2.live.ops_sre_readable_facts import readable_fact
            from nurse_rcl.v2.live.ops_sre_workbench_render import (
                readable_score_fact as workbench_score_fact,
            )

            score_text = workbench_score_fact if workbench else readable_score_fact

            displayed_values = {
                key: value
                for key, value in state["values"].items()
                if key not in {"record_path", "representation", "source_role"}
            }
            lines += [
                ("  - 评分材料：" if state["ref"] in score_kinds else "  - 独立判断：")
                + _text(
                    score_text(
                        context.fact_store.resolve(state["ref"]),
                        source_kind=score_kinds[state["ref"]],
                    )
                    if state["ref"] in score_kinds
                    else readable_fact(context.fact_store.resolve(state["ref"]))
                    if readable_facts
                    else canonical_json_bytes(displayed_values).decode()
                )
                + f" [{state['ref']}](report-evidence.md#{state['ref']})",
                "",
            ]
    return lines, mapping


def compact_workbench_limits(groups: dict[str, Any]) -> list[str]:
    """Main-text index only; every complete group remains in the evidence appendix."""
    counts: dict[str, int] = {}
    facts = 0
    labels = {
        "formal_record": "正式记录",
        "supplementary": "补充材料",
        "query_scope_metadata": "查询范围元信息",
        "investigation_core_reuse": "复用的核心材料",
    }
    for group in groups.values():
        role = str(group["role"])
        label = labels.get(role, "其他来源")
        counts[label] = counts.get(label, 0) + 1
        facts += len(group["refs"])
    if not groups:
        return ["逐项来源与范围限制：0 组；这不表示已证实正常。", ""]
    by_role = "、".join(f"{label} {count} 组" for label, count in sorted(counts.items()))
    return [
        f"其他逐项来源与范围限制共 {len(groups)} 组，关联 {facts} 项已实供事实；"
        f"按角色为{by_role}。数量不代表支持强度或排除故障。",
        "",
        "每组的窗口、条件角色、反证、数值不可得与原字段均保留在"
        "[完整证据附件](report-evidence.md)；这里不把未列细项视为正常或已排除。",
        "",
    ]


def _latest_reply_lines(
    reply: dict[str, Any], *, profile: str, question_state: dict[str, Any] | None = None
) -> list[str]:
    """Pure receipt display; a new profile never opens a second lifecycle writer.

    A caller may supply the already saved projection for this acknowledged action.
    No binding is inferred from receipt completeness or the agent's summary.
    This helper grants no report/profile permission.
    """
    update = reply.get("latest_result_update")
    if update is None:
        return []
    typed_purpose = profile in {"hermes-sre-tools/22", "hermes-sre-tools/23", "hermes-sre-tools/25"}
    single_owner = profile == "hermes-sre-tools/21" or typed_purpose
    if single_owner:
        from nurse_rcl.v2.live.ops_sre_single_owner_decision import LATEST_FIELDS

        if type(update) is not dict or set(update) != LATEST_FIELDS:
            raise ValueError("single-owner display receipt fields differ")
    lines = [
        f"后续决策 {reply['decision_index']} 对最新结果的公开回应（未验证）："
        + _text(
            update["summary"]
            if single_owner
            else update["answer_status"] + "；" + update["summary"]
        ),
        "",
    ]
    if single_owner:
        question = None
        if question_state is not None:
            expected = (
                "nurse.v2.sre-question-state/3"
                if typed_purpose
                else "nurse.v2.sre-question-state/2"
            )
            if question_state.get("schema_version") != expected:
                raise ValueError("single-owner display question identity differs")
            identity = question_state["latest_action_question_id"]
            if identity is not None:
                rows = [r for r in question_state["questions"] if r["question_id"] == identity]
                if len(rows) != 1:
                    raise ValueError("single-owner display question binding differs")
                question = rows[0]
        lines += [
            "保存的问题状态（非诊断事实）："
            + (
                "未绑定"
                if question is None
                else _text(question["question_id"] + " / " + question["state"])
            ),
            "",
            "题状态仅以保存的问题投影为准；本摘要不更新生命周期。",
            "",
        ]
    elif update["remaining_question"] is not None:
        lines += ["尚缺：" + _text(update["remaining_question"]), ""]
    return lines


def _typed_purpose_lines(plan: dict[str, Any]) -> list[str]:
    """Pure display of an already admitted plan, without granting report authority."""
    if type(plan) is not dict or set(plan) != {"updates", "current_question_id", "action_purpose"}:
        raise ValueError("typed display plan fields differ")
    key, purpose = plan["current_question_id"], plan["action_purpose"]
    if key is None:
        if purpose is not None:
            raise ValueError("typed display purpose binding differs")
        return []
    if (
        type(purpose) is not dict
        or set(purpose) != {"question_id", "why_now"}
        or purpose["question_id"] != key
        or type(purpose["why_now"]) is not str
    ):
        raise ValueError("typed display purpose fields differ")
    rows = [r for r in plan["updates"] if r["question_id"] == key]
    if len(rows) != 1:
        raise ValueError("typed display question binding differs")
    row = rows[0]
    labels = {
        "program_from_admitted_observable_scope": "程序从已准入 scope 生成的窄题标题：",
        "agent_navigation_question": "模型公开的导航问题：",
    }
    if row["question_origin"] not in labels:
        raise ValueError("typed display question origin differs")
    return [
        "关联公开问题：" + _text(key),
        "",
        labels[row["question_origin"]] + _text(row["question"]),
        "",
        "模型公开动机／未经语义认证：" + _text(purpose["why_now"]),
        "",
    ]


def render_brief_report_pair(
    *,
    context: BriefSreReportContext,
    response: BriefReportResponse | ConditionReportResponse | ComparableReportResponse | None,
    references: dict[str, Any] | None = None,
    accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None,
    raw_response_json: str | None = None,
    preview: bool = False,
) -> dict[str, str]:
    retained = validate_brief_report(context)
    return _render_admitted_brief_report_pair(
        context=context,
        retained=retained,
        response=response,
        admitted_checks=admit_brief_checks(context, response, retained=retained),
        references=references,
        accounting=accounting,
        public_history=public_history,
        raw_response_json=raw_response_json,
        preview=preview,
    )


def _render_admitted_brief_report_pair(
    *,
    context: BriefSreReportContext,
    retained: dict[str, Any],
    response: BriefReportResponse | ConditionReportResponse | ComparableReportResponse | None,
    admitted_checks: _AdmittedBriefChecks,
    references: dict[str, Any] | None = None,
    accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None,
    raw_response_json: str | None = None,
    preview: bool = False,
) -> dict[str, str]:
    """Private owner reuse; exact current content still must match admitted content."""
    from nurse_rcl.v2.live.ops_sre_brief_report import _require_brief_unchanged

    _require_brief_unchanged(context, canonical_json_bytes(retained))
    workbench = isinstance(context.metadata, WorkbenchReportMetadata)
    navigation = isinstance(context.metadata, NavigationReportMetadata) or workbench
    comparable = isinstance(context.metadata, ComparableReportMetadata) or navigation
    condition = (
        isinstance(context.metadata, ConditionReportMetadata | RecoveryReportMetadata) or comparable
    )
    expected_response = (
        ComparableReportResponse
        if comparable
        else (ConditionReportResponse if condition else BriefReportResponse)
    )
    if response is not None and type(response) is not expected_response:
        raise ValueError("report response version differs from context")
    policy = brief_render_policy(context)
    evidence = context.metadata.evidence
    supply = brief_report_supply(context)
    candidates = tuple(r.candidate_service_key for r in evidence.ranking)
    aliases = service_aliases(candidates, references, snapshot=evidence.case_snapshot_id)
    _require_admitted_checks(context, response, retained=retained, admitted=admitted_checks)
    checks = admitted_checks.checks
    by_id = {c.proposal.claim_id: c for c in checks}
    grounding = None
    if (
        type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}
        and response is not None
        and raw_response_json is not None
    ):
        from nurse_rcl.v2.live.ops_sre_report_grounding import grounding_diagnostics

        grounding = grounding_diagnostics(
            response, raw_response_json, context.supply, context.fact_store, checks
        )
    identity = dict(
        policy=policy,
        context_sha256=sha256_hex(canonical_json_bytes(retained)),
        response=None if response is None else response.model_dump(mode="json"),
        references=references,
        accounting=accounting,
        public_history=public_history,
        raw_response_json=raw_response_json,
        preview=preview,
        **check_identity(admitted_checks),
    )
    if type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
        identity["grounding"] = grounding
    pin = sha256_hex(canonical_json_bytes(identity))
    stamp = "<!-- " + policy + " binding=" + pin + " -->\n\n"
    lines = ["# NURSE 运维调查报告", "", "## 当前判断", ""]
    if preview:
        lines += ["这是保存材料的离线投影，不是新模型调查或解释验收。", ""]
    if type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
        lines += [
            "程序限定：本报告使用独立绑定的原生调查材料；公开 note／问题状态不是诊断事实。",
            "",
        ]
        if type(context.metadata) is FocusedReportMetadata:
            lines += [
                "程序限定：指标、单位与请求／观测／来源链是同一实际交付记录的归属信息，"
                "不是新增观测或独立支持票。",
                "",
            ]
        if context.metadata.termination == "stage_pause":
            lines += [
                "程序限定：stage_pause 仅为获批片段的程序段限，不是模型主动收敛或整案调查完成。",
                "",
            ]
        last_action = next(
            (s for s in reversed(context.metadata.steps) if s["action_index"] is not None), None
        )
        if last_action is not None and last_action["response_state"] == "not_recorded":
            lines += [
                "程序限定：最后行动未记录后续回应；报告可事后综合该回执，"
                "不代表调查模型已读取或利用，不补写转向或收敛。",
                "",
            ]
    if evidence.ranking:
        lines += [
            "确定性小模型将"
            + _text(aliases[evidence.ranking[0].candidate_service_key])
            + "列为正式首位；精确排名与分数在附件中仅展示一次。",
            "",
        ]
    lines += [
        "核心自主查询与评分；智能体编排补充调查及解释，不改变正式定位。"
        "分数不是根因概率，已发布也不等于实际进入评分。",
        "",
        "解释状态：待助手语义审阅和用户认可；机械未拒绝的文字仍未验证。",
        "",
    ]
    accepted = [c for c in checks if c.surface_status != "blocked"]
    if not accepted:
        lines += ["当前只有确定性材料可展示，不能算作模型解释成功。", ""]
    blocked = sum(c.surface_status == "blocked" for c in checks)
    if blocked:
        lines += [f"有 {blocked} 条陈述被隔离；原文与原因留在附件，不作为事实展示。", ""]
    shown: set[str] = set()

    def show(ids: list[str] | tuple[str, ...]) -> None:
        for claim_id in ids:
            checked = by_id[claim_id]
            if checked.surface_status == "blocked" or claim_id in shown:
                continue
            shown.add(claim_id)
            claim = checked.proposal
            lines.extend(
                [
                    _text(claim.surface_text) + "（模型解释，未作语义确认）",
                    "",
                    "依据："
                    + "、".join(f"[{r}](report-evidence.md#{r})" for r in claim.evidence_refs),
                    "",
                ]
            )
            if claim.recommendation is not None:
                lines.extend(["建议尚未执行：" + _text(claim.recommendation.action_class), ""])
                question = getattr(claim.recommendation, "verification_question", None)
                if question is not None:
                    lines.extend(["待核问题：" + _text(question), ""])
                    subject = getattr(claim.recommendation, "outcome_subject_ref", None)
                    if subject is not None:
                        lines.extend(
                            [f"条件结果主体：[{subject}](report-evidence.md#{subject})", ""]
                        )
                    else:
                        lines.extend(["尚无可靠结果分支；不据此推演机制或评分变化。", ""])
                for result in claim.recommendation.possible_outcomes:
                    lines.extend(["- " + _text(result.outcome + "：" + result.interpretation), ""])
                if grounding is not None:
                    for branch in grounding["branches"]:
                        if branch["claim_id"] != claim.claim_id:
                            continue
                        boundary = branch["boundary"]
                        lines.extend(
                            ["程序条件边界（非模型诊断）：" + _text(str(boundary["role"])), ""]
                        )
                        for effect in branch["effects"]:
                            lines.extend(
                                [
                                    "程序条件作用："
                                    + _text(effect["outcome"] + " → " + effect["condition_effect"]),
                                    "",
                                ]
                            )
                        for mechanism in boundary["mechanisms"]:
                            for condition_item in mechanism["necessary_conditions"]:
                                sources = condition_item["sources"]
                                state_text = (
                                    "；".join(
                                        r["evidence_ref"] + "=" + r["truth_value"] for r in sources
                                    )
                                    if sources
                                    else "not_supplied"
                                )
                                lines.extend(
                                    [
                                        "程序必要条件位置："
                                        + _text(
                                            mechanism["mechanism_key"]
                                            + " / "
                                            + condition_item["predicate_kind"]
                                            + " / "
                                            + state_text
                                        ),
                                        "",
                                    ]
                                )
                lines.extend(
                    ["建议边界：" + _text("；".join(claim.recommendation.safety_limitations)), ""]
                )

    if response is not None:
        show(
            [c.claim_id for c in response.executive_summary]
            if isinstance(response, ConditionReportResponse | ComparableReportResponse)
            else response.summary_claim_ids
        )
    if grounding is not None:
        lines += ["## 最终条件竞争交接", "", "程序覆盖诊断仅核来源结构，未作语义确认。", ""]
        if grounding["competition"]["status"] == "missing":
            lines += [
                "condition_competition_handoff_missing：存在合法条件组合，但最终交接尚未覆盖；"
                "不补写结论或自动重抽。",
                "",
            ]
        elif grounding["competition"]["status"] == "not_available":
            lines += ["没有可用的同机制／同定义／同窗跨候选条件组合；保留未覆盖，不强制比较。", ""]
        if response is not None:
            show([c.claim_id for c in response.final_synthesis])
    lines += ["## 关键依据与竞争解释", ""]
    show(
        [
            c.proposal.claim_id
            for c in checks
            if c.section in {"formal_interpretations", "final_synthesis"}
        ]
    )
    if (
        type(context.metadata) not in {AtomicReportMetadata, FocusedReportMetadata}
        and condition
        and not any(
            c.proposal.operation in {"compare_candidates", "compare_mechanisms"}
            and c.surface_status != "blocked"
            for c in checks
        )
    ):
        lines += ["尚未形成独立的竞争解释；不把分数差或单候选陈述冒充已完成的竞争核验。", ""]
    limits_start = len(lines)
    lines += ["## 条件级证据与局限", ""]
    if workbench:
        lines += [
            "评分数值 unknown 表示数值不可得，不是未读项，也不是独立健康判断；"
            "完整来源和精确字段见附件。",
            "",
        ]
    lines += condition_lines(supply["briefs"], aliases, readable=condition)
    summarized = {b["source_record_sha256"] for b in supply["briefs"]}
    detailed = {
        ref
        for b in context.metadata.source_bindings
        if b["record_sha256"] in summarized
        for ref in b["fact_refs"]
    }
    general: dict[str, list[str]] = {}
    grouped_limits: dict[str, Any] = {}
    if condition:
        shared_lines, grouped_limits = grouped_limit_lines(
            context,
            supply,
            detailed,
            aliases,
            readable_facts=isinstance(
                context.metadata,
                RecoveryReportMetadata
                | ComparableReportMetadata
                | NavigationReportMetadata
                | WorkbenchReportMetadata,
            ),
            navigation=navigation,
            workbench=workbench,
        )
        lines += compact_workbench_limits(grouped_limits) if workbench else shared_lines
    for ref in supply["provided_fact_refs"]:
        if ref in detailed:
            continue  # Its full condition, including contrary leaves, is above.
        item = context.fact_store.resolve(ref)
        if not critical_fact(item):
            continue
        codes = set(item.fact.status_codes)
        for code in sorted(codes & _GENERAL_RULES):
            general.setdefault(code, []).append(ref)
        if condition:
            continue
        values = {v.name: v.value for v in item.fact.values}
        adverse = any(
            values.get(k) in {"false", "unknown"} for k in ("truth_value", "support_value")
        )
        specific = sorted(codes - _GENERAL_RULES)
        if not specific and not adverse and item.fact.kind not in {"limit", "uncertainty"}:
            continue
        scope = "、".join(aliases.get(k, k) for k in item.fact.target_service_keys) or "案例范围"
        meaning = (
            "查询范围元信息，不是健康判断"
            if item.origin == "query_scope_metadata"
            else ("正式限制" if item.role == "formal_record" else "实际交付的非机制限制")
        )
        lines += [
            "- "
            + _text(
                scope
                + "；"
                + meaning
                + "；"
                + item.source_path
                + "："
                + ", ".join(_LIMIT_TEXT.get(code, code) for code in specific)
                + (
                    "；原判断=" + str(values.get("truth_value", values.get("support_value")))
                    if adverse
                    else ""
                )
            )
            + f" [{ref}](report-evidence.md#{ref})",
            "",
        ]
    if general:
        lines += ["### 通用解释边界", ""]
        for code, refs in general.items():
            lines += [
                "- "
                + _text(_LIMIT_TEXT.get(code, code))
                + f"；适用 {len(refs)} 项原记录，逐项映射见附件。",
                "",
            ]
    deferred_limits = lines[limits_start:] if workbench else []
    if workbench:
        del lines[limits_start:]
    lines += ["## 实际调查过程", ""]
    for step in supply["steps"]:
        planned, executed = step["planned"], step["executed"]
        lines += [f"### 决策 {step['decision_index']}（{step['step_ref']}）", ""]
        for question in planned.get("questions", []):
            lines += ["当时的问题（模型公开说明）：" + _excerpt(question), ""]
        question = planned.get("investigation_question")
        if type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
            if "investigation_plan" in planned:
                lines += _typed_purpose_lines(planned["investigation_plan"])
            question = None  # atomic titles/motives do not contain legacy discriminating_value
        elif isinstance(context.metadata, QuestionReportMetadata):
            plan = planned.get("investigation_plan", {})
            question = plan.get("action_purpose")
            if plan.get("current_question_id") is not None:
                lines += ["关联公开问题：" + _text(plan["current_question_id"]), ""]
        if question is not None:
            lines += [
                "区分目的：" + _text(question["discriminating_value"]),
                "",
                "此时检查的理由：" + _text(question["why_now"]),
                "",
            ]
        lines += [
            "执行："
            + _text(executed.get("tool", "未调工具") + " / " + executed["status"])
            + (
                "；来源：" + _text(str(executed.get("origin", "未记录"))) + "；完整参数见附件。"
                if condition
                else "；参数："
                + _text(canonical_json_bytes(executed.get("parameters", {})).decode())
            ),
            "",
            f"实际交付：{len(step['delivered_brief_refs'])} 个简报字段组、"
            f"{len(step['delivered_fact_refs'])} 项精确事实（数量不是支持强度）。",
            "",
        ]
        findings = [
            c.proposal.claim_id
            for c in checks
            if c.section == "step_findings"
            and getattr(c.proposal, "step_ref", None) == step["step_ref"]
        ]
        if type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import program_fact_rows

            rows = program_fact_rows(supply, context.fact_store, step_ref=step["step_ref"])
            lines += ["程序事实（原来源字段，非模型诊断）：", ""]
            for fact_row in rows:
                lines += ["- " + _text(canonical_json_bytes(fact_row).decode()), ""]
            if not rows:
                lines += ["本步无已完成交付的精确事实；不补造测量。", ""]
            lines += ["模型解释（原生选择展开后仍须检查与语义审阅）：", ""]
        show(findings)
        if navigation:
            from nurse_rcl.v2.live.ops_sre_navigation_render import finding_state

            isolated = [by_id[key] for key in findings if by_id[key].surface_status == "blocked"]
            lines += [
                finding_state(
                    total=len(findings),
                    blocked=len(isolated),
                    reasons=[r for check in isolated for r in check.reason_codes],
                ),
                "",
            ]
        if context.metadata.profile == "hermes-sre-tools/9" and not findings:
            lines += ["模型未提供本步发现；保留实际执行及回应，不补造结论。", ""]
        for reply in step["responded_to"]:
            if condition:
                update = reply.get("latest_result_update")
                if update is not None:
                    lines += _latest_reply_lines(reply, profile=context.metadata.profile)
                continue
            lines += [
                f"后续决策 {reply['decision_index']} 的公开回应（未验证）："
                + _excerpt(reply["public_summary"]),
                "",
            ]
        if not step["responded_to"] or (
            condition and not any("latest_result_update" in r for r in step["responded_to"])
        ):
            lines += ["未记录后续回应；不补写当时的转向或收敛理由。", ""]
        if planned.get("stop_reason") is not None:
            lines += ["模型停止原话：" + _excerpt(planned["stop_reason"]), ""]
            lines += [
                "额度／停止核对："
                + _text(
                    ", ".join(planned.get("stop_diagnostics", []))
                    or "未发现已注册的额度矛盾；不表示未查问题均已排除。"
                ),
                "",
            ]
    if workbench:
        lines += ["真实停止状态：" + _text(context.metadata.termination), "", *deferred_limits]
    lines += [
        *([] if workbench else ["真实停止状态：" + _text(context.metadata.termination)]),
        "",
        "## 未决问题与未执行建议",
        "",
    ]
    if isinstance(context.metadata, QuestionReportMetadata):
        state = context.metadata.question_state
        lines += [
            "### 全局问题交接",
            "",
            "以下为模型公开任务状态，不是诊断事实；局部 answered 不关闭全案问题。",
            "",
        ]
        completed = [
            dict(s["executed"], action_index=s["action_index"])
            for s in supply["steps"]
            if s["executed"]["status"] == "completed"
        ]
        if only_directory_or_scoring(completed, context.metadata.diagnostic_semantics):
            lines += ["执行范围提示：仅核对目录／评分资料，故障证据调查尚未展开。", ""]
        for row in state["questions"]:
            lines += [
                "- " + _text(row["question_id"] + "（" + row["state"] + "）：" + row["question"]),
                "",
            ]
            if row["remaining_gap"]:
                lines += ["  尚缺：" + _text(row["remaining_gap"]), ""]
            if row["superseded_by"] is not None:
                lines += ["  显式替代为：" + _text(row["superseded_by"]), ""]
        assessment = state["stop_assessment"]
        if assessment is None:
            lines += ["没有被接受的模型全局停止交接；上列未决问题仍保留。", ""]
        else:
            lines += ["模型说明的已完成范围：" + _text(assessment["completed_scope"]), ""]
            for row in assessment["remaining_questions"]:
                lines += ["- " + _text(row["question_id"] + " 的停止理由：" + row["why_stop"]), ""]
            for scope in assessment["unchecked_scope"]:
                lines += ["- 未查范围（模型说明）：" + _text(scope), ""]
    if (
        isinstance(context.metadata, NavigationReportMetadata | WorkbenchReportMetadata)
        and context.metadata.investigation_failure is not None
    ):
        from nurse_rcl.v2.live.ops_sre_navigation_render import failure_text

        lines += [_text(failure_text(context.metadata.investigation_failure)), ""]
    show([c.proposal.claim_id for c in checks if c.section in {"limitations", "next_checks"}])
    if navigation:
        from nurse_rcl.v2.live.ops_sre_navigation_render import last_admitted_unresolved

        last = last_admitted_unresolved(supply["steps"])
        if last is not None:
            for item in last["unresolved"]:
                lines += [
                    f"决策 {last['decision_index']} 的最后合法公开未决项（未验证）："
                    + _excerpt(item),
                    "",
                ]
    elif supply["steps"]:
        for item in supply["steps"][-1]["planned"].get("unresolved", []):
            lines += ["历史公开未决项（未验证）：" + _excerpt(item), ""]
    lines += [
        "## 成本与证据入口",
        "",
        *_cost_lines(accounting),
        "[完整排名、证据树、来源、原文与隔离原因](report-evidence.md)。",
        "",
    ]
    appendix = [
        "# NURSE 完整证据附件",
        "",
        "## 完整正式排名与服务映射",
        "",
        "| 排名 | 服务 | 准入 ID | 总分 | 局部贡献 | 传播贡献 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in evidence.ranking:
        appendix += [
            f"| {row.rank} | {_text(aliases[row.candidate_service_key])} | "
            f"{_text(row.candidate_service_key)} | {row.quantized_total_score} | "
            f"{row.local_score_component:.12g} | {row.propagation_score_component:.12g} |"
        ]
    appendix += [
        "",
        "## 完整来源与唯一事实树",
        "",
        _text(canonical_json_bytes(retained).decode()),
        "",
    ]
    for ref in (*context.fact_store.refs(), *supply["provided_brief_refs"]):
        appendix += [f'<a id="{ref}"></a>', "", f"{ref}：见对应原生树或简报条目。", ""]
    for anchor, group in grouped_limits.items():
        appendix += [f'<a id="{anchor}"></a>', "", _text(canonical_json_bytes(group).decode()), ""]
    for title, value in (
        ("原始未验收响应（含隔离内容，非结论）：解析结果", identity["response"]),
        ("原始未验收响应（含隔离内容，非结论）：原生模型响应（还原前，未改写）", raw_response_json),
        ("完整公开调查原文（非事实）", public_history),
        ("检查与隔离原因", [c.model_dump(mode="json") for c in checks]),
        ("通用规则逐记录映射", general),
        ("完整引用映射", references),
        (
            "成本及模块耗时",
            dict(
                accounting=accounting,
                action_seconds=context.metadata.action_seconds,
                decision_seconds=context.metadata.decision_seconds,
            ),
        ),
    ):
        appendix += [
            "## " + title,
            "",
            _text(value if type(value) is str else canonical_json_bytes(value).decode()),
            "",
        ]
    return dict(
        report=stamp + "\n".join(lines), evidence=stamp + "\n".join(appendix), binding_sha256=pin
    )


def save_brief_report_pair(directory: Path, **kwargs: Any) -> dict[str, Any]:
    return _save_report_pair(
        directory,
        pair=render_brief_report_pair(**kwargs),
        policy=brief_render_policy(kwargs["context"]),
    )


def _save_admitted_brief_report_pair(directory: Path, **kwargs: Any) -> dict[str, Any]:
    return _save_report_pair(
        directory,
        pair=_render_admitted_brief_report_pair(**kwargs),
        policy=brief_render_policy(kwargs["context"]),
    )


def inspect_brief_report_pair(
    directory: Path,
    *,
    condition_explanations: bool = False,
    recovery: bool = False,
    comparable: bool = False,
    navigation: bool = False,
    workbench: bool = False,
    question_led: bool = False,
    legacy_question: bool = False,
    atomic_report: bool = False,
) -> dict[str, Any]:
    return _inspect_report_pair(
        directory,
        policy="sre-readable-pair/14"
        if atomic_report
        else "sre-readable-pair/10"
        if legacy_question
        else "sre-readable-pair/11"
        if question_led
        else "sre-readable-pair/9"
        if workbench
        else "sre-readable-pair/8"
        if navigation
        else "sre-readable-pair/7"
        if comparable
        else RECOVERY_RENDER_POLICY
        if recovery
        else CONDITION_RENDER_POLICY
        if condition_explanations
        else BRIEF_RENDER_POLICY,
    )
