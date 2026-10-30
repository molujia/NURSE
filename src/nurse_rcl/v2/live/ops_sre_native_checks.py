"""Local fact resolution with complete vocabulary and unchanged claim rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from nurse_rcl.v2.live.ops_report_checks import check_ops_report_records
from nurse_rcl.v2.live.ops_report_schema import OpsClaimCheck, OpsEmpiricalReportResponse
from nurse_rcl.v2.live.ops_sre_native_report import NativeFactLookup, NativeSreReportContext
from nurse_rcl.v2.live.ops_sre_report import OpsSreReportResponse, check_sre_organization
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact
from nurse_rcl.v2.report_language_rules import (
    SRE_LANGUAGE_POLICY,
    LanguageFactIndex,
    check_indexed_fact_claim_language,
)
from nurse_rcl.v2.report_presentation_schemas import ReportLanguageFieldCheck


class _EvidenceLookup(Mapping[str, ReportEvidenceFact]):
    def __init__(self, records: NativeFactLookup) -> None:
        self.records = records

    def __getitem__(self, key: str) -> ReportEvidenceFact:
        return self.records[key].fact

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, key: object) -> bool:
        return key in self.records


def check_native_sre_report(
    response: OpsSreReportResponse, *, context: NativeSreReportContext,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
) -> tuple[OpsClaimCheck, ...]:
    if response.core_result_digest != context.core_result_digest:
        raise ValueError("report response core digest differs")
    metadata = context.metadata
    steps = {s["step_ref"]: s for s in metadata.steps}
    if len(set(provided_step_refs)) != len(provided_step_refs) or (
        not set(provided_step_refs) <= steps.keys()
        or any(s.step_ref not in provided_step_refs for s in response.workflow_steps)
    ):
        raise ValueError("SRE provided step references differ")
    records = NativeFactLookup(context.fact_store)
    facts = _EvidenceLookup(records)
    index = LanguageFactIndex.build(facts.values())
    candidates = tuple(r.candidate_service_key for r in metadata.evidence.ranking)

    def language(claim: ReportCompositionClaim) -> tuple[ReportLanguageFieldCheck, ...]:
        return check_indexed_fact_claim_language(claim, facts=facts, index=index,
            candidate_service_keys=candidates, language_policy=SRE_LANGUAGE_POLICY)

    checked = check_ops_report_records(OpsEmpiricalReportResponse(
        schema_version="nurse.v2.ops-layered-report-response/2",
        core_result_digest=response.core_result_digest, claims=response.claims),
        records=records, facts=facts, candidates=candidates,
        provided_fact_refs=provided_fact_refs, language_checker=language)
    return check_sre_organization(response, checked=checked, steps=steps, profile=metadata.profile)
