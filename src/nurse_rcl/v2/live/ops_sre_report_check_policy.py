"""One finite parser/check policy and an internal, content-bound admission result."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_brief_claims import (
    BriefClaimCheck,
    BriefReportResponse,
    ConditionReportResponse,
    check_brief_report,
    parse_brief_report,
)
from nurse_rcl.v2.live.ops_sre_brief_report import (
    ActionReportMetadata,
    AtomicReportMetadata,
    BalancedReportMetadata,
    BriefReportMetadata,
    BriefSreReportContext,
    ComparableReportMetadata,
    ConditionReportMetadata,
    FocusedReportMetadata,
    NavigationReportMetadata,
    QuestionReportMetadata,
    ReceiptReuseReportMetadata,
    RecoveryReportMetadata,
    ReplanningReportMetadata,
    UsedSourceReportMetadata,
    WorkbenchReportMetadata,
    _require_brief_unchanged,
    validate_brief_report,
)
from nurse_rcl.v2.live.ops_sre_comparable_claims import (
    ComparableClaimCheck,
    ComparableReportResponse,
)
from nurse_rcl.v2.live.ops_sre_d22_numbers import EXACT_DECIMAL_STRING, FLOAT_ONLY
from nurse_rcl.v2.live.ops_sre_staged_report_metadata import StagedReportMetadata

BriefResponse = BriefReportResponse | ConditionReportResponse | ComparableReportResponse
BriefChecks = tuple[BriefClaimCheck, ...] | tuple[ComparableClaimCheck, ...]
_ADMISSION = object()


def report_check_policy(context: BriefSreReportContext) -> dict[str, Any]:
    metadata = context.metadata
    known = (
        BriefReportMetadata,
        ConditionReportMetadata,
        RecoveryReportMetadata,
        ComparableReportMetadata,
        NavigationReportMetadata,
        WorkbenchReportMetadata,
        UsedSourceReportMetadata,
        ReceiptReuseReportMetadata,
        BalancedReportMetadata,
        ReplanningReportMetadata,
        QuestionReportMetadata,
        ActionReportMetadata,
        StagedReportMetadata,
        AtomicReportMetadata,
        FocusedReportMetadata,
    )
    if type(metadata) not in known:
        raise ValueError("report check metadata version differs")
    policy = dict(
        schema_version="nurse.v2.sre-report-check-policy/1",
        metadata_schema=metadata.schema_version,
        comparable_evidence=isinstance(
            metadata, ComparableReportMetadata | NavigationReportMetadata | WorkbenchReportMetadata
        ),
        condition_explanations=isinstance(
            metadata, ConditionReportMetadata | RecoveryReportMetadata
        ),
        require_supplied_decimals=isinstance(metadata, ReceiptReuseReportMetadata),
    )
    if type(metadata) in {StagedReportMetadata, AtomicReportMetadata, FocusedReportMetadata}:
        policy.update(
            schema_version="nurse.v2.sre-report-check-policy/2",
            decimal_value_policy=EXACT_DECIMAL_STRING,
        )
    if type(metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
        policy.update(
            schema_version="nurse.v2.sre-report-check-policy/4",
            named_field_scope="exact_supplied_identifiers/1",
            count_phrase_scope="finite_english_integer_field_aliases/1",
            candidate_denial_scope="does_not_establish_which_candidate_is_the_root_cause/1",
        )
    if type(metadata) is FocusedReportMetadata:
        policy.update(
            schema_version="nurse.v2.sre-report-check-policy/5",
            source_attribution_policy="same_record_not_independent_evidence/1",
        )
    return policy


def parse_checked_brief(context: BriefSreReportContext, raw: bytes) -> BriefResponse:
    policy = report_check_policy(context)
    return parse_brief_report(
        raw,
        comparable_evidence=policy["comparable_evidence"],
        condition_explanations=policy["condition_explanations"],
    )


@dataclass(frozen=True)
class _AdmittedBriefChecks:
    policy_json: str
    context_sha256: str
    response_sha256: str
    checks: BriefChecks
    binding_sha256: str
    _marker: object = field(repr=False, compare=False)


def _identity(
    policy: dict[str, Any], context_pin: str, response_pin: str, checks: BriefChecks
) -> dict[str, Any]:
    return dict(
        policy=policy,
        context_sha256=context_pin,
        response_sha256=response_pin,
        checks=[check.model_dump(mode="json") for check in checks],
    )


def admit_brief_checks(
    context: BriefSreReportContext, response: BriefResponse | None, *, retained: dict[str, Any]
) -> _AdmittedBriefChecks:
    _require_brief_unchanged(context, canonical_json_bytes(retained))
    policy = report_check_policy(context)
    context_pin = sha256_hex(canonical_json_bytes(retained))
    response_pin = sha256_hex(
        canonical_json_bytes(None if response is None else response.model_dump(mode="json"))
    )
    checks = (
        ()
        if response is None
        else check_brief_report(
            response,
            supply=context.supply,
            fact_store=context.fact_store,
            candidates=tuple(r.candidate_service_key for r in context.metadata.evidence.ranking),
            require_supplied_decimals=policy["require_supplied_decimals"],
            decimal_value_policy=policy.get("decimal_value_policy", FLOAT_ONLY),
        )
    )
    if type(context.metadata) in {AtomicReportMetadata, FocusedReportMetadata}:
        from nurse_rcl.v2.live.ops_sre_report_grounding import (
            named_field_reasons,
            scalar_field_scope,
        )

        scope = scalar_field_scope(context.supply, context.fact_store)
        grounded = []
        for c in checks:
            reasons = tuple(
                sorted(
                    set(c.reason_codes)
                    | set(
                        named_field_reasons(
                            c.proposal, context.supply, context.fact_store, field_scope=scope
                        )
                    )
                )
            )
            grounded.append(
                c.model_copy(
                    update=dict(
                        reason_codes=reasons,
                        surface_status="blocked" if reasons else "model_unverified",
                    )
                )
            )
        checks = cast(BriefChecks, tuple(grounded))
    return _AdmittedBriefChecks(
        canonical_json_bytes(policy).decode(),
        context_pin,
        response_pin,
        checks,
        sha256_hex(canonical_json_bytes(_identity(policy, context_pin, response_pin, checks))),
        _ADMISSION,
    )


def _require_admitted_checks(
    context: BriefSreReportContext,
    response: BriefResponse | None,
    *,
    retained: dict[str, Any],
    admitted: _AdmittedBriefChecks,
) -> None:
    _require_brief_unchanged(context, canonical_json_bytes(retained))
    policy = report_check_policy(context)
    context_pin = sha256_hex(canonical_json_bytes(retained))
    response_pin = sha256_hex(
        canonical_json_bytes(None if response is None else response.model_dump(mode="json"))
    )
    if (
        type(admitted) is not _AdmittedBriefChecks
        or admitted._marker is not _ADMISSION
        or (
            (admitted.policy_json, admitted.context_sha256, admitted.response_sha256)
            != (canonical_json_bytes(policy).decode(), context_pin, response_pin)
            or admitted.binding_sha256
            != sha256_hex(
                canonical_json_bytes(_identity(policy, context_pin, response_pin, admitted.checks))
            )
        )
    ):
        raise ValueError("report admitted checks binding differs")


def response_admission(
    context: BriefSreReportContext, raw: str, *, retained: dict[str, Any] | None = None
) -> _AdmittedBriefChecks:
    return admit_brief_checks(
        context,
        parse_checked_brief(context, raw.encode()),
        retained=validate_brief_report(context) if retained is None else retained,
    )


def check_identity(admitted: _AdmittedBriefChecks) -> dict[str, Any]:
    return dict(
        check_policy=json.loads(admitted.policy_json), checks_binding_sha256=admitted.binding_sha256
    )
