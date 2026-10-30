"""Layered explanations cannot rewrite RCL or promote supplementary evidence."""

import importlib
import importlib.util
import json
import re

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence, OpsNodeEvidence
from nurse_rcl.v2.live.ops_result import seal_core_result
from tests.test_v2_live_normal_queries import prepared_cases  # noqa: F401
from tests.test_v2_live_ops_contracts import identity
from tests.test_v2_live_ops_tools import complete  # noqa: F401


def api(name):
    qualified = "nurse_rcl.v2.live.ops_report_" + name
    assert importlib.util.find_spec(qualified), "missing isolated report " + name
    return importlib.import_module(qualified)


@pytest.fixture(scope="module")
def report_inputs(complete):  # noqa: F811
    from nurse_rcl.v2.live import ops_contracts

    data, _, evidence, _ = complete
    core = seal_core_result(identity=identity(ops_contracts, evidence), evidence=evidence)
    non_top = next(
        k
        for k in evidence.ranking.ordered_service_keys
        if k not in evidence.ranking.top_k_service_keys
    )
    node = OpsNodeEvidence(
        case_snapshot_id=core.identity.case_snapshot_id,
        node="timing",
        tool_version="fixture/1",
        parameters_json="{}",
        evidence_ids=("late-onset",),
        source_refs=("c" * 64,),
        unit="ms",
        window_ns=(0, 10),
        coverage=1.0,
        availability="partial",
        truth_value="unknown",
        limitations=("onset_overlap", "not_formal_scoring_input"),
        payload_json=canonical_json_bytes(
            dict(
                candidate_service_key=non_top,
                value=42,
                verified_normal="unknown",
                source_bounds_ns=[5, 7],
                target_bounds_ns=[4, 6],
            )
        ).decode(),
    )
    finding = OpsInvestigationEvidence(
        action_index=1, evidence=node, origin="measured", reused_core_evidence_ids=()
    )
    return data, core, finding


def context(report_inputs):
    data, core, finding = report_inputs
    return api("context").build_report_context(
        core=core, sources=data.sources, investigation=(finding,)
    )


def proposal(ctx, *, section="supplementary_findings", **changes):
    fact = next(f.fact for f in ctx.facts if f.role == "supplementary")
    claim = dict(
        claim_id="claim1",
        statement_type="inference",
        operation="explain_uncertainty",
        fact_refs=[fact.fact_ref],
        target_service_keys=list(fact.target_service_keys),
        interval=None,
        requested_claim_strength="ranking_inference",
        assertions=[dict(fact_ref=fact.fact_ref, field="truth_value", value="unknown")],
        recommendation=None,
        surface_text=f"{fact.fact_ref}: onset remains unknown.",
    )
    claim.update(changes)
    return api("schema").parse_ops_report(
        canonical_json_bytes(
            dict(
                schema_version="nurse.v2.ops-layered-report-response/1",
                core_result_digest=ctx.core_result_digest,
                claims=[dict(section=section, claim=claim)],
            )
        )
    )


def test_projection_binds_full_ranking_and_non_top_supplement(report_inputs):
    ctx = context(report_inputs)
    _, core, finding = report_inputs
    assert ctx.core_result_digest == core.core_result_digest
    assert tuple(r.candidate_service_key for r in ctx.ranking) == (
        core.evidence.ranking.ordered_service_keys
    )
    fact = next(f for f in ctx.facts if f.role == "supplementary")
    assert fact.fact.target_service_keys[0] not in ctx.formal_top_k
    assert fact.evidence_ids == finding.evidence.evidence_ids
    assert fact.source_refs == finding.evidence.source_refs
    assert fact.fact.fact_ref in ctx.mandatory_fact_refs
    assert "v2-live-evidence-repair" not in ctx.model_dump_json()
    checks = api("checks").check_ops_report(proposal(ctx), context=ctx)
    assert checks[0].reason_codes == ()
    assert checks[0].surface_status == "model_unverified"


@pytest.mark.parametrize(
    "patch,reason",
    [
        ({"section": "formal_grounds"}, "supplementary_as_formal"),
        ({"surface_text": "This service is the root cause."}, "causal_overclaim"),
        ({"surface_text": "The observation is normal."}, "missing_evidence_upgraded"),
        ({"surface_text": "The corrected Top-1 is another service."}, "model_ranking_override"),
        ({"surface_text": "已经执行回滚并确认恢复。"}, "recommendation_executed"),
        ({"requested_claim_strength": "confirmed_cause"}, "operation_strength_mismatch"),
        ({"target_service_keys": ["foreign-service"]}, "target_scope_mismatch"),
        ({"interval": {"start_unix_ns": 20, "end_unix_ns": 30}}, "interval_source_mismatch"),
    ],
)
def test_domain_or_surface_violation_is_retained_not_accepted(report_inputs, patch, reason):
    ctx = context(report_inputs)
    raw = proposal(ctx, **patch)
    check = api("checks").check_ops_report(raw, context=ctx)[0]
    assert reason in check.reason_codes
    assert check.surface_status == "blocked"
    assert check.proposal == raw.claims[0]


@pytest.mark.parametrize(
    "field,value", [("value", 43), ("unit", "s"), ("truth_value", "true"), ("source_ref", "forged")]
)
def test_existing_reference_cannot_lend_false_scalar_or_source(report_inputs, field, value):
    ctx = context(report_inputs)
    ref = next(f.fact.fact_ref for f in ctx.facts if f.role == "supplementary")
    draft = proposal(ctx, assertions=[dict(fact_ref=ref, field=field, value=value)])
    check = api("checks").check_ops_report(draft, context=ctx)[0]
    assert any(r.startswith("assertion_") for r in check.reason_codes)


def test_code_summary_keeps_omitted_counterevidence_and_original_rank(report_inputs):
    ctx = context(report_inputs)
    text = re.sub(r"\\(.)", r"\1", api("render").render_ops_report(context=ctx, checks=()))
    assert "onset_overlap" in text and "unknown" in text
    assert ctx.ranking[0].candidate_service_key in text
    assert "补充调查" in text and "未参与" in text
    assert "确定性摘要" in text and "不代表语义正确" in text
    bad = proposal(ctx, surface_text="The corrected Top-1 is another service.")
    checks = api("checks").check_ops_report(bad, context=ctx)
    rendered = re.sub(r"\\(.)", r"\1", api("render").render_ops_report(context=ctx, checks=checks))
    assert "The corrected Top-1" not in rendered
    assert "model_ranking_override" in rendered


def test_wrong_snapshot_and_wrong_core_response_fail_closed(report_inputs):
    data, core, finding = report_inputs
    changed = finding.model_copy(
        update={
            "evidence": finding.evidence.model_copy(
                update={
                    "case_snapshot_id": "0" * 64,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="snapshot"):
        api("context").build_report_context(
            core=core, sources=data.sources, investigation=(changed,)
        )
    ctx = context(report_inputs)
    wrong = proposal(ctx).model_copy(update={"core_result_digest": "0" * 64})
    with pytest.raises(ValueError, match="core"):
        api("checks").check_ops_report(wrong, context=ctx)


def test_schema_never_accepts_model_rank_table(report_inputs):
    draft = proposal(context(report_inputs)).model_dump(mode="json")
    draft["ranking"] = ["replacement"]
    with pytest.raises(ValueError):
        api("schema").parse_ops_report(canonical_json_bytes(draft))


def test_free_text_value_unit_error_rejected_with_valid_structured_assertion(report_inputs):
    ctx = context(report_inputs)
    ref = next(f.fact.fact_ref for f in ctx.facts if f.role == "supplementary")
    draft = proposal(ctx, surface_text=f"{ref}.value = 42 s")
    checks = api("checks").check_ops_report(draft, context=ctx)
    assert "source_unit_mismatch" in checks[0].reason_codes


def test_report_context_is_data_only_and_bounded(report_inputs):
    ctx = context(report_inputs)
    body = json.loads(ctx.model_dump_json())
    assert "points" not in body and "api_key" not in body
    with pytest.raises(ValueError):
        api("context").build_report_context(core=None, sources=(), investigation=())
