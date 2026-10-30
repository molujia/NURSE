"""Empirical identities, approved request capacity and independent resource bounds."""

from typing import Literal

from nurse_rcl.v2.exchange_json import ExchangeLimits

DECISION_PROTOCOL = "nurse.v2.ops-decision-exchange/3"
REPORT_PROTOCOL = "nurse.v2.ops-report-exchange/3"
SRE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/5"
SRE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/7"
SRE_WORKING_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/6"
SRE_NATIVE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/8"
SRE_BRIEF_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/7"
SRE_BRIEF_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/9"
SRE_FEEDBACK_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/8"
SRE_CONDITION_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/10"
SRE_RECOVERY_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/9"
SRE_RECOVERY_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/11"
SRE_COMPARABLE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/10"
SRE_COMPARABLE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/12"
SRE_NAVIGATION_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/11"
SRE_NAVIGATION_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/13"
SRE_WORKBENCH_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/12"
SRE_WORKBENCH_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/14"
SRE_USED_SOURCE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/13"
SRE_USED_SOURCE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/15"
SRE_RECEIPT_REUSE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/14"
SRE_RECEIPT_REUSE_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/16"
SRE_BALANCED_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/15"
SRE_BALANCED_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/17"
SRE_REPLANNING_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/16"
SRE_REPLANNING_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/18"
SRE_QUESTION_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/17"
SRE_QUESTION_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/19"
SRE_ACTION_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/18"
SRE_ACTION_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/20"
SRE_LEGACY_ATOMIC_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/21"
SRE_PRIOR_ATOMIC_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/22"
SRE_ATOMIC_REPORT_PROTOCOL = "nurse.v2.sre-report-exchange/23"
SRE_TYPED_REPORT_PROTOCOL = "nurse.v2.sre-typed-report-exchange/1"
SRE_GPLUS_REPORT_PROTOCOL = "nurse.v2.sre-gplus-report-exchange/1"
SRE_DIRECTED_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/19"
SRE_ENTRY_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/20"
SRE_OBSERVABLE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/21"
SRE_SINGLE_OWNER_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/22"
SRE_TYPED_PURPOSE_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/23"
SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/24"
SRE_FEEDBACK_FACT_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/25"
SRE_FOCUSED_DECISION_PROTOCOL = "nurse.v2.sre-decision-exchange/26"
SRE_DECISION_PROTOCOLS = (
    SRE_FOCUSED_DECISION_PROTOCOL,
    SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
    SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
    SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
    SRE_SINGLE_OWNER_DECISION_PROTOCOL,
    SRE_OBSERVABLE_DECISION_PROTOCOL,
    SRE_ENTRY_DECISION_PROTOCOL,
    SRE_DIRECTED_DECISION_PROTOCOL,
    SRE_ACTION_DECISION_PROTOCOL,
    SRE_QUESTION_DECISION_PROTOCOL,
    SRE_REPLANNING_DECISION_PROTOCOL,
    SRE_BALANCED_DECISION_PROTOCOL,
    SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
    SRE_USED_SOURCE_DECISION_PROTOCOL,
    SRE_WORKBENCH_DECISION_PROTOCOL,
    SRE_NAVIGATION_DECISION_PROTOCOL,
    SRE_RECOVERY_DECISION_PROTOCOL,
    SRE_COMPARABLE_DECISION_PROTOCOL,
    SRE_DECISION_PROTOCOL,
    SRE_WORKING_DECISION_PROTOCOL,
    SRE_BRIEF_DECISION_PROTOCOL,
    SRE_FEEDBACK_DECISION_PROTOCOL,
)
SRE_REPORT_PROTOCOLS = (
    SRE_GPLUS_REPORT_PROTOCOL,
    SRE_ATOMIC_REPORT_PROTOCOL,
    SRE_PRIOR_ATOMIC_REPORT_PROTOCOL,
    SRE_LEGACY_ATOMIC_REPORT_PROTOCOL,
    SRE_ACTION_REPORT_PROTOCOL,
    SRE_QUESTION_REPORT_PROTOCOL,
    SRE_REPLANNING_REPORT_PROTOCOL,
    SRE_BALANCED_REPORT_PROTOCOL,
    SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
    SRE_USED_SOURCE_REPORT_PROTOCOL,
    SRE_WORKBENCH_REPORT_PROTOCOL,
    SRE_NAVIGATION_REPORT_PROTOCOL,
    SRE_RECOVERY_REPORT_PROTOCOL,
    SRE_COMPARABLE_REPORT_PROTOCOL,
    SRE_REPORT_PROTOCOL,
    SRE_NATIVE_REPORT_PROTOCOL,
    SRE_BRIEF_REPORT_PROTOCOL,
    SRE_CONDITION_REPORT_PROTOCOL,
    SRE_TYPED_REPORT_PROTOCOL,
)
SRE_PROTOCOLS = (*SRE_DECISION_PROTOCOLS, *SRE_REPORT_PROTOCOLS)
SRE_VARIANT: Literal["v2-ops-sre/1"] = "v2-ops-sre/1"
PROTOCOLS = (DECISION_PROTOCOL, REPORT_PROTOCOL, *SRE_PROTOCOLS)
USAGE_POLICY = "ops-empirical-input-cost/1"
OUTPUT_POLICY = "explicit_max_tokens_65536/1"
# User-approved request capacity, not a target length or a whole-case quota.
# Deployment acceptance is unverified until an independently approved live run.
MAX_OUTPUT_TOKENS = 65536
# The existing large report/context envelope now covers both output channels.
# This is a serialization/process safety boundary, unrelated to input targets.
RESPONSE_LIMITS = ExchangeLimits(4_194_304, 64, 100_000)
# Five retained responses plus JSON escaping and receipts. Full responses remain
# in the one existing attempt ledger, not a second accounting store.
LEDGER_BYTES = 64 * 1024 * 1024
