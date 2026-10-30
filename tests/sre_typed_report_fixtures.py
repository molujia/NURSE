"""New independent apparatus facts, never old model answers or private gold."""

from importlib import import_module, util

from nurse_rcl.artifacts import canonical_json_bytes

SECTIONS = (
    "executive_summary",
    "formal_interpretations",
    "step_findings",
    "final_synthesis",
    "limitations",
    "next_checks",
)
NATIVE_SCHEMA = "nurse.v2.atomic-typed-report-expression-response/1"


def api(suffix):
    name = "nurse_rcl.v2.live.ops_sre_typed_report_" + suffix
    assert util.find_spec(name), f"missing explicit normal typed report {suffix}"
    return import_module(name)


def fact(
    ref,
    values,
    *,
    start=10,
    end=20,
    path="payload.window",
    formal=False,
    target="other-service",
    kind="observation",
):
    from nurse_rcl.v2.live.ops_report_context import OpsReportFact
    from nurse_rcl.v2.observation_schemas import TimeInterval
    from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact, ReportEvidenceValue

    return OpsReportFact(
        fact=ReportEvidenceFact(
            fact_ref=ref,
            kind=kind,
            target_service_keys=(target,),
            intervals=(TimeInterval(start_unix_ns=start, end_unix_ns=end),),
            values=tuple(ReportEvidenceValue(name=k, value=v) for k, v in values.items()),
            allowed_claim_strength="frozen_observation",
            scope_basis="bound_observation",
        ),
        role="formal_record" if formal else "supplementary",
        evidence_ids=(),
        source_refs=(),
        source_path=path,
        origin="core_published",
        action_index=1,
    )


def material(step="s7", target="other-service"):
    api("binding")  # Missing implementation is testable without a platform-codec substitute.
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store

    rows = [
        fact(
            "f1",
            dict(
                baseline_denominator=0,
                baseline_denominator_unit=None,
                fraction=None,
                fraction_unit=None,
                unit="cores",
                unknown="unknown",
            ),
            target=target,
            formal=True,
        ),
        fact(
            "f2",
            dict(
                fault_denominator=0,
                fault_denominator_unit=None,
                unit="requests",
                coverage="unknown",
            ),
            start=20,
            end=30,
            target=target,
        ),
        fact(
            "f3",
            dict(predicate_kind="local_pressure_observed", truth_value="unknown"),
            path="payload.window.child",
            target=target,
            kind="predicate",
        ),
    ]
    supply = dict(
        schema_version="nurse.v2.sre-brief-supply/3",
        core_result_digest="d" * 64,
        provided_fact_refs=[r.fact.fact_ref for r in rows],
        formal_fact_refs=["f1"],
        briefs=[],
        provided_brief_refs=[],
        steps=[
            dict(
                step_ref=step,
                executed=dict(status="completed"),
                delivered_fact_refs=[r.fact.fact_ref for r in rows],
                delivered_brief_refs=[],
            )
        ],
    )
    return supply, build_fact_store(rows)


def empty_native():
    return dict(
        schema_version=NATIVE_SCHEMA, **{s: {} if s == "step_findings" else [] for s in SECTIONS}
    )


def native(
    expression="e1",
    *,
    section="executive_summary",
    step="s7",
    primary=None,
    text="原样  0 unit wording; unknown remains unknown.\n第二行。",
    outcomes=None,
):
    body = empty_native()
    selection = dict(
        claim_id="claim_1",
        expression_ref=expression,
        primary_fields=[] if primary is None else primary,
        surface_text=text,
        outcomes=outcomes,
    )
    body[section] = {step: [selection]} if section == "step_findings" else [selection]
    return body


def raw(value):
    return canonical_json_bytes(value)


def original_catalog(supply, store):
    from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import expression_catalog

    return expression_catalog(supply, store)
