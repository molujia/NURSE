"""SRE narratives bind actual steps and facts separately; no network required."""

# ruff: noqa: F811, RUF001 -- imported fixtures and Chinese report text

import importlib
import importlib.util
import json
import re

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreInvestigationLedger
from tests.test_v2_live_ops_report import (
    complete,  # noqa: F401
    prepared_cases,  # noqa: F401
    proposal,
    report_inputs,  # noqa: F401
)
from tests.test_v2_live_sre_context import decision, project, record
from tests.test_v2_live_sre_ledger import action, setup


def api(suffix=""):
    name = "nurse_rcl.v2.live.ops_sre_report" + suffix
    assert importlib.util.find_spec(name), "SRE report API not implemented: " + suffix
    return importlib.import_module(name)


def test_report_api_exists():
    for suffix in ("", "_context", "_exchange", "_render"):
        api(suffix)


@pytest.fixture(scope="module")
def history(complete, tmp_path_factory):
    options, tool = setup(complete, tmp_path_factory.mktemp("sre-report"), core=True)
    key = next(k for k, c in tool.cards.items()
               if c.node == "mechanisms" and c.truth_value == "unknown")
    decisions = []
    with OpsSreInvestigationLedger(**options) as ledger:
        for index, request in enumerate((action(cursor=None), action(cursor=None),
                action("read_node", "mechanisms", card_id=key, section="payload", cursor=None)), 1):
            before = project(ledger, decisions=tuple(decisions), index=index)
            decisions.append(record(before, decision(request, summary="核对未知条件"),
                                    index=index, action_index=index))
            ledger.dispatch(request)
        return tuple(decisions), ledger.records


@pytest.fixture(scope="module")
def context(report_inputs, history):
    data, core, _ = report_inputs
    decisions, records = history
    return api("_context").build_sre_report_context(
        core=core, sources=data.sources, decisions=decisions, records=records,
        investigation_status="completed", termination="decision_budget")


def draft(context, **changes):
    # Existing legal claims remain legal; organization is a separate envelope.
    value = proposal(context.evidence).model_dump(mode="json")
    fact = next(f.fact for f in context.evidence.facts if f.role == "supplementary"
                and any(v.name == "truth_value" and v.value == "unknown" for v in f.fact.values))
    value["claims"][0]["claim"].update(fact_refs=[fact.fact_ref],
        target_service_keys=fact.target_service_keys,
        assertions=[dict(fact_ref=fact.fact_ref, field="truth_value", value="unknown")],
        surface_text=f"{fact.fact_ref}: 判断仍未知，不能确认因果。")
    value.update(schema_version="nurse.v2.sre-report-response/1",
                 summary_claim_ids=["claim1"], finding_claim_ids=[],
                 workflow_steps=[dict(step_ref="s1", observation_claim_ids=[],
                                      synthesis_claim_ids=["claim1"])],
                 unresolved_claim_ids=[])
    value.update(changes)
    return api().parse_sre_report(canonical_json_bytes(value))


def check(context, response):
    return api().check_sre_report(response, context=context,
        provided_fact_refs=tuple(f.fact.fact_ref for f in context.evidence.facts),
        provided_step_refs=tuple(s["step_ref"] for s in context.steps))


def render(context, response=None, *, attachment=False):
    pair = api("_render").render_sre_report_pair(context=context, response=response,
        provided_fact_refs=tuple(f.fact.fact_ref for f in context.evidence.facts),
        provided_step_refs=tuple(s["step_ref"] for s in context.steps))
    value = pair["evidence" if attachment else "report"]
    return re.sub(r"\\(.)", r"\1", value)


def test_distinct_schema_closes_organization_and_disallows_free_prose(context):
    response = draft(context)
    assert check(context, response)[0].reason_codes == ()
    with pytest.raises(ValueError):
        draft(context, summary_claim_ids=["absent"])
    with pytest.raises(ValueError):
        draft(context, workflow_steps=[dict(step_ref="s1", text="已确认故障")])
    with pytest.raises(ValueError):
        draft(context, ranking=["alternative"])
    with pytest.raises(ValueError):
        from nurse_rcl.v2.live.ops_report_schema import parse_ops_report

        parse_ops_report(canonical_json_bytes(response.model_dump(mode="json")), empirical=True)


def test_step_reference_cannot_replace_a_fact_or_invent_a_step(context):
    value = draft(context).model_dump(mode="json")
    value["claims"][0]["claim"]["fact_refs"] = ["s1"]
    with pytest.raises(ValueError):
        api().parse_sre_report(canonical_json_bytes(value))
    response = draft(context, workflow_steps=[dict(step_ref="s8",
        observation_claim_ids=[], synthesis_claim_ids=["claim1"])])
    with pytest.raises(ValueError, match="step"):
        check(context, response)


def test_late_facts_cannot_become_earlier_observations(context):
    late = next(f for f in context.evidence.facts if f.action_index == 3
                and any(v.name == "truth_value" and v.value == "unknown" for v in f.fact.values))
    value = draft(context).model_dump(mode="json")
    claim = value["claims"][0]["claim"]
    claim.update(fact_refs=[late.fact.fact_ref], target_service_keys=late.fact.target_service_keys,
        assertions=[dict(fact_ref=late.fact.fact_ref, field="truth_value", value="unknown")],
        surface_text=f"{late.fact.fact_ref}: 条件仍未知，不能确认因果。")
    value["workflow_steps"] = [dict(step_ref="s1", observation_claim_ids=["claim1"],
                                    synthesis_claim_ids=[])]
    response = api().parse_sre_report(canonical_json_bytes(value))
    assert "step_fact_not_delivered" in check(context, response)[0].reason_codes
    value["workflow_steps"][0].update(observation_claim_ids=[], synthesis_claim_ids=["claim1"])
    assert check(context, api().parse_sre_report(canonical_json_bytes(value)))[0].reason_codes == ()


@pytest.mark.parametrize("patch,reason", [
    ({"surface_text": "已经执行回滚并确认恢复。"}, "recommendation_executed"),
    ({"surface_text": "修正排名，另一服务第一。"}, "model_ranking_override"),
    ({"target_service_keys": ["foreign-service"]}, "target_scope_mismatch"),
    ({"requested_claim_strength": "confirmed_cause"}, "operation_strength_mismatch"),
])
def test_workflow_claims_use_the_same_checks(context, patch, reason):
    value = draft(context).model_dump(mode="json")
    value["claims"][0]["claim"].update(patch)
    response = api().parse_sre_report(canonical_json_bytes(value))
    assert reason in check(context, response)[0].reason_codes
    assert patch.get("surface_text", "NEVER") not in render(context, response)


def test_duplicate_receipts_share_facts_and_keep_at_time_knowledge(context, history):
    assert context.schema_version == "nurse.v2.sre-report-context/4"
    assert context.profile == "hermes-sre-tools/4"
    assert context.steps[0]["observed_fact_refs"] == context.steps[1]["observed_fact_refs"]
    assert context.steps[-1]["response_state"] == "not_recorded"
    assert context.steps[-1]["responded_to"] == []
    assert "核对未知条件" not in canonical_json_bytes([
        f.model_dump(mode="json") for f in context.evidence.facts]).decode()
    assert all("record_ids" not in {v.name for v in f.fact.values}
               for f in context.evidence.facts)
    assert any(g["representation"] == "candidate_overview_not_full_judgment"
               for g in context.record_groups)
    assert any(g["representation"] == "full_semantic_record" for g in context.record_groups)
    assert all(f.origin != "core_evaluation" for f in context.evidence.facts
               if f.role == "supplementary")


def test_report_wire_keeps_questions_caveats_and_exact_table_values(context):
    from nurse_rcl.v2.live.ops_context_view import expand_report_facts

    projected = api("_exchange").sre_report_projection(context)
    assert projected.status == "ready"
    wire = json.loads(projected.wire_bytes)
    assert wire["max_tokens"] == 65536
    packet = json.loads(wire["messages"][1]["content"])["context"]
    assert packet["schema_version"] == "nurse.v2.sre-model-view/2"
    view = json.loads(projected.view_json)
    assert view["schema_version"] == "nurse.v2.sre-report-input/6"
    assert "核对未知条件" in canonical_json_bytes(view["steps"]).decode()
    expanded = expand_report_facts(view)
    originals = {f.fact.fact_ref: f for f in context.evidence.facts}
    def normalized(item):
        row = item.model_dump(mode="json")
        row["fact"]["values"] = sorted(row["fact"]["values"], key=lambda v: v["name"])
        return canonical_json_bytes(row)  # Exact JSON types, not Python false == 0.

    assert all(normalized(f) == normalized(originals[f.fact.fact_ref]) for f in expanded)
    assert set(context.critical_fact_refs) <= set(projected.provided_fact_refs)
    assert "payload_json" not in canonical_json_bytes(view).decode()


def test_report_public_history_is_not_exempt_from_outbound_checks(context):
    steps = json.loads(canonical_json_bytes(context.steps))
    steps[0]["planned"]["public_summary"]["text"] = "ts0-mysql-container-kill-9t6n24"
    changed = context.model_copy(update={"steps": tuple(steps)})
    with pytest.raises(ValueError, match="private"):
        api("_exchange").sre_report_projection(changed)


def test_chinese_summary_precedes_workflow_and_appendix_without_hiding_unknowns(context):
    text = render(context, draft(context))
    evidence = render(context, draft(context), attachment=True)
    assert text.index("## 当前判断") < text.index("## 实际调查过程")
    assert "## 附录" not in text and "report-evidence.md" in text
    assert "## 关键局限" in text.split("## 附录")[0]
    assert "未知" in text.split("## 附录")[0]
    assert "未记录后续回应" in text
    assert "事后综合" in text and "核心自主执行" in text
    assert "未映射服务" in text and context.core_result_digest in evidence
    assert "尚待人工语义核对与用户认可" in text
    assert "预算" in text
    for row in context.evidence.ranking:
        assert row.candidate_service_key in evidence
    # Rendering cannot manufacture absent cost accounting.
    assert "成本与模块耗时：未提供" in text


def test_missing_model_report_is_explicitly_not_explanation_success(context):
    text = render(context)
    assert "确定性替代摘要" in text and "不计模型解释成功" in text
    assert "未执行" in text


def claim_for(context, item, operation, field):
    value = draft(context).model_dump(mode="json")
    claim = value["claims"][0]["claim"]
    claim.update(operation=operation, fact_refs=[item.fact.fact_ref],
        target_service_keys=item.fact.target_service_keys,
        assertions=[dict(fact_ref=item.fact.fact_ref, field=field,
                         value=next(v.value for v in item.fact.values if v.name == field))],
        surface_text="保留实际比较范围与证据局限，不确认因果。")
    value["claims"][0]["section"] = "formal_grounds"
    return value


def test_executable_single_and_double_target_examples_keep_numeric_type(context):
    ranked = next(f for f in context.evidence.facts if f.fact.kind == "ranking"
                  and any(v.name == "quantized_total_score" for v in f.fact.values))
    value = claim_for(context, ranked, "explain_priority", "quantized_total_score")
    assert check(context, api().parse_sre_report(canonical_json_bytes(value)))[0].reason_codes == ()
    assertion = value["claims"][0]["claim"]["assertions"][0]
    assert type(assertion["value"]) is str
    assertion["value"] = float(assertion["value"])
    assert "assertion_value_mismatch" in check(context, api().parse_sre_report(
        canonical_json_bytes(value)))[0].reason_codes
    pair = next(f for f in context.evidence.facts if f.fact.kind == "competition")
    scalar = next(v.name for v in pair.fact.values if type(v.value) in (str, int, float, bool))
    value = claim_for(context, pair, "compare_candidates", scalar)
    assert check(context, api().parse_sre_report(canonical_json_bytes(value)))[0].reason_codes == ()
    value["claims"][0]["claim"]["target_service_keys"] = pair.fact.target_service_keys[:1]
    assert "target_scope_mismatch" in check(context, api().parse_sre_report(
        canonical_json_bytes(value)))[0].reason_codes


def test_executable_recommendation_retains_per_target_preconditions(context):
    item = next(f for f in context.evidence.facts if f.role == "supplementary"
                and len(f.fact.target_service_keys) == 1 and f.fact.kind == "observation"
                and any(v.name == "truth_value" for v in f.fact.values))
    value = claim_for(context, item, "propose_verification", "truth_value")
    value["claims"][0]["section"] = "conditional_next_steps"
    claim = value["claims"][0]["claim"]
    claim.update(statement_type="recommendation",
        requested_claim_strength="conditional_recommendation",
        surface_text="建议只读核验该候选的未决条件，尚未执行。",
        recommendation=dict(status="proposed", action_class="read_only_diagnostic_check",
            precondition_refs=[item.fact.fact_ref], possible_outcomes=[
                dict(outcome=v, interpretation="按实际检查结果保留支持、反证或未知。")
                for v in ("true", "false", "unknown")],
            safety_limitations=["仅限冻结窗口与该候选，不执行干预，不确认因果。"]))
    response = api().parse_sre_report(canonical_json_bytes(value))
    assert check(context, response)[0].reason_codes == ()
    assert "条件建议（未执行）" in render(context, response)
    other = next(r.candidate_service_key for r in context.evidence.ranking
                 if r.candidate_service_key not in item.fact.target_service_keys)
    claim["target_service_keys"] = [*item.fact.target_service_keys, other]
    assert "recommendation_precondition_invalid" in check(context, api().parse_sre_report(
        canonical_json_bytes(value)))[0].reason_codes


def test_rejected_action_keeps_intent_but_cannot_claim_diagnostic_results(
    complete, report_inputs, tmp_path,
):
    options, _ = setup(complete, tmp_path)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        request = action("inspect_candidate", candidate_service_key="not-admitted", cursor=None)
        planned = record(before, decision(request))
        ledger.dispatch(request)
        data, core, _ = report_inputs
        ctx = api("_context").build_sre_report_context(core=core, sources=data.sources,
            decisions=(planned,), records=ledger.records, investigation_status="partial",
            termination="unavailable")
    assert not ctx.steps[0]["observed_fact_refs"]
    assert all(f.role == "formal_record" for f in ctx.evidence.facts)
    assert "已拒绝" in render(ctx)


def test_aliases_do_not_depend_on_formal_rank(context):
    module = api("_render")
    keys = tuple(r.candidate_service_key for r in context.evidence.ranking)
    assert module.service_aliases(keys) == module.service_aliases(tuple(reversed(keys)))


def test_offline_report_preview_is_saved(context, tmp_path):
    rendered = render(context, draft(context))
    path = tmp_path / "sre-report-fixture-preview.md"
    path.write_text(rendered, encoding="utf-8")
    print("SRE_FIXTURE_PREVIEW=" + str(path))
    assert "# NURSE 运维调查报告" in path.read_text(encoding="utf-8")


def test_main_caveats_are_chinese_and_group_shared_limits_without_losing_refs(context):
    text = render(context)
    main = text.split("## 附录")[0]
    assert "必要条件仍有未知项" in main
    assert "尚未测量的条件仍为未知" in main
    assert "necessary_predicate_unknown" not in main
    assert "unmeasured_conditions_remain_unknown" not in main
    assert "反证／未知" in main
    assert "同类局限合并展示" in main


def test_delivered_onset_retains_the_tool_bound_candidate(complete, report_inputs, tmp_path):
    options, tool = setup(complete, tmp_path, core=True)
    key = next(k for k, c in tool.cards.items() if c.node == "timing"
               and "onset" in json.loads(c.payload_json))
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        request = action("read_node", "timing", card_id=key, section="payload", cursor=None)
        planned = record(before, decision(request))
        ledger.dispatch(request)
        delivered = json.loads(ledger.records[0].model_view_json)["semantic_record"]
        assert delivered["targets"], "fixture must contain a genuinely bound onset"
        data, core, _ = report_inputs
        ctx = api("_context").build_sre_report_context(core=core, sources=data.sources,
            decisions=(planned,), records=ledger.records, investigation_status="completed",
            termination="decision_budget")
    onset = next(f for f in ctx.evidence.facts if f.source_path == "payload.onset")
    assert onset.fact.target_service_keys == tuple(delivered["targets"])


def test_empty_aggregate_page_does_not_become_a_partial_health_card():
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    view = dict(schema_version="nurse.v2.sre-tool-view/2", tool="inspect_candidate",
        card_id="aggregate", truth_value="unknown",
        source_role="aggregation_not_independent_evidence",
        page=dict(records=[], count=0), critical_record_ids=[], limitations=["not_published"])
    assert _delivered_records(view) == []


def test_nested_unknown_caveat_keeps_true_or_parent_and_role(context):
    from nurse_rcl.v2.live.ops_report_context import OpsReportFact
    from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact, ReportEvidenceValue

    rows = []
    for ref, path, values in (("f900001", "payload.conditions[0]",
            dict(condition_id="availability", requirement="necessary", operator="or",
                 truth_value="true")),
            ("f900002", "payload.conditions[0].predicates[0]", dict(truth_value="unknown"))):
        rows.append(OpsReportFact(fact=ReportEvidenceFact(fact_ref=ref, kind="observation",
            target_service_keys=(), values=tuple(ReportEvidenceValue(name=k, value=v)
            for k, v in values.items()), allowed_claim_strength="frozen_observation",
            scope_basis="bound_observation"), role="supplementary", evidence_ids=(),
            source_refs=(), source_path=path, origin="fixture", action_index=1))
    evidence = context.evidence.model_copy(update={"facts": (*context.evidence.facts, *rows),
        "mandatory_fact_refs": (*context.evidence.mandatory_fact_refs, "f900002")})
    steps = json.loads(canonical_json_bytes(context.steps))
    steps[0]["observed_fact_refs"].extend(["f900001", "f900002"])
    ctx = context.model_copy(update=dict(evidence=evidence,
        steps=tuple(steps),
        critical_fact_refs=(*context.critical_fact_refs, "f900002"),
        record_groups=(*context.record_groups, dict(card_id="conditional-record",
            record_sha256="e" * 64, source_role="fixture",
            fact_refs=["f900001", "f900002"], action_indices=[1]))))
    main = render(ctx).split("## 附录")[0]
    assert "必要条件 availability" in main
    assert "OR" in main and "上层判断成立" in main and "子项未知" in main


def test_unexpanded_counts_are_historical_not_claimed_as_final_state(context):
    main = render(context).split("## 附录")[0]
    assert "当时未展开" in main
    assert "不代表最终仍未展开" in main


def test_dependency_absence_and_self_time_gap_remain_scoped_query_limits(
    complete, report_inputs, tmp_path,
):
    options, tool = setup(complete, tmp_path, core=True)
    # This fixture has admitted candidates but no observed dependency edge.
    caller, callee = report_inputs[1].evidence.ranking.ordered_service_keys[:2]
    edge = dict(caller_service_key=caller, callee_service_key=callee)
    request = action("inspect_dependency", "dependency",
        caller_service_key=edge["callee_service_key"],
        callee_service_key=edge["caller_service_key"], cursor=None)
    with OpsSreInvestigationLedger(**options) as ledger:
        before = project(ledger)
        planned = record(before, decision(request))
        ledger.dispatch(request)
        actual = json.loads(ledger.records[0].model_view_json)
        assert actual["dependency"]["edge_state"] == "not_observed"
        data, core, _ = report_inputs
        ctx = api("_context").build_sre_report_context(core=core, sources=data.sources,
            decisions=(planned,), records=ledger.records, investigation_status="completed",
            termination="decision_budget")
    found = next((f for f in ctx.evidence.facts if any(
        v.name == "callee_self_time" and v.value == "not_available" for v in f.fact.values)), None)
    assert found is not None, "actual query coverage must not disappear from report"
    assert found.fact.kind == "uncertainty" and found.role == "supplementary"
    assert found.fact.scope_basis == "frozen_assessment"
    assert set(found.fact.target_service_keys) == {
        edge["caller_service_key"], edge["callee_service_key"]}
    assert found.fact.fact_ref in ctx.steps[0]["observed_fact_refs"]
