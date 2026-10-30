"""Readable D9 delivery is a bound pair, not an edited historical model report."""

# ruff: noqa: F811 -- shared fixtures

import importlib
import re

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report_exchange import sre_report_projection
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from tests.test_v2_live_sre_report import (  # noqa: F401
    complete,
    context,
    draft,
    history,
    prepared_cases,
    report_inputs,
)


def options(context):
    book = SreReferenceBook(snapshot=context.evidence.case_snapshot_id, run_id="a" * 64)
    # Deliberately choose a different order from sorted IDs and formal rank.
    for row in reversed(context.evidence.ranking):
        book.issue("service", row.candidate_service_key, visible_at=1)
    refs = book.to_state()
    projection = sre_report_projection(context, references=refs)
    return dict(context=context, response=draft(context), references=refs,
        provided_fact_refs=projection.provided_fact_refs,
        provided_step_refs=tuple(s["step_ref"] for s in context.steps),
        public_history=[dict(public_summary="完整原生公开文字" * 500)])


def pair_api():
    module = importlib.import_module("nurse_rcl.v2.live.ops_sre_report_render")
    assert hasattr(module, "render_sre_report_pair"), "bound readable report pair is missing"
    return module.render_sre_report_pair


def test_report_prompt_requests_summary_and_delta_not_rank_repetition():
    from nurse_rcl.v2.live.ops_sre_report_exchange import SRE_REPORT_SYSTEM

    assert "不要逐项重抄 Top-5" in SRE_REPORT_SYSTEM
    assert "新增或改变判断" in SRE_REPORT_SYSTEM
    assert "合法工具" in SRE_REPORT_SYSTEM


def test_same_book_names_and_unmodified_text_in_separate_evidence(context):
    args = options(context)
    pair = pair_api()(**args)
    main, evidence = pair["report"], pair["evidence"]
    mapping = {r[2]: r[1] for r in args["references"]["entries"] if r[0] == "service"}
    top = context.evidence.ranking[0].candidate_service_key
    assert "服务" + mapping[top] in main and "服务 A" not in main
    assert "report-evidence.md" in main
    assert "### 事实、来源与范围" not in main
    assert "完整原生公开文字" * 500 in evidence
    assert "完整原生公开文字" not in main
    assert all(item.fact.fact_ref in evidence for item in context.evidence.facts)
    unescaped = re.sub(r"\\(.)", r"\1", evidence)
    assert all(row.candidate_service_key in unescaped for row in context.evidence.ranking)
    assert pair["binding_sha256"] in main and pair["binding_sha256"] in evidence
    assert args["response"].claims[0].claim.surface_text in unescaped


def test_main_includes_parameters_observation_and_isolation_not_just_aliases(context):
    args = options(context)
    value = args["response"].model_dump(mode="json")
    value["workflow_steps"][0].update(observation_claim_ids=["claim1"], synthesis_claim_ids=[])
    value["claims"][0]["claim"]["surface_text"] = "已经确认根因。"
    from nurse_rcl.v2.live.ops_sre_report import parse_sre_report
    args["response"] = parse_sre_report(canonical_json_bytes(value))
    pair = pair_api()(**args)
    main = pair["report"]
    assert "参数" in main and "cursor" in main
    assert "解释未通过" in main and "已经确认根因。" not in main
    assert "可展示观测" in main and "未知" in main
    assert "causal_overclaim" in re.sub(r"\\(.)", r"\1", pair["evidence"])
    assert main.index("## 当前判断") < main.index("## 实际调查过程")


def test_known_and_unknown_attempt_costs_are_both_visible(context):
    args = options(context)
    usage = summarize_attempts([
        dict(phase="report", slot=1, attempt=1, usage=None),
        dict(phase="report", slot=1, attempt=2,
             usage=dict(prompt_tokens=101849, completion_tokens=9825, total_tokens=111674)),
    ])
    pair = pair_api()(**args, accounting=dict(usage=usage))
    assert "已知输入 101849" in pair["report"]
    assert "已知输出 9825" in pair["report"]
    assert "未知尝试 1" in pair["report"] and "精确总量未知" in pair["report"]
    assert "None" not in pair["report"]
    changed = pair_api()(**args, accounting=dict(usage=dict(usage, reported_input_tokens=101850)))
    assert pair["binding_sha256"] != changed["binding_sha256"]


def test_delivery_recovers_missing_attachment_without_redrawing(context, tmp_path):
    pair_api()  # Expected red assertion, not an import error during collection.
    delivery = importlib.import_module("nurse_rcl.v2.live.ops_sre_report_delivery")
    args = options(context)
    first = delivery.save_sre_report_pair(tmp_path, **args)
    assert delivery.inspect_sre_report_pair(tmp_path) == first
    (tmp_path / "report-evidence.md").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        delivery.inspect_sre_report_pair(tmp_path)
    assert delivery.save_sre_report_pair(tmp_path, **args) == first
    assert delivery.inspect_sre_report_pair(tmp_path) == first
    text = (tmp_path / "report.md").read_text()
    (tmp_path / "report.md").write_text(text + "changed")
    with pytest.raises(ValueError, match="binding"):
        delivery.inspect_sre_report_pair(tmp_path)
