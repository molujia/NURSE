"""Separately granted entry decision stage; original lifetime transport is retained."""

from nurse_rcl.v2.live.ops_empirical_policy import SRE_ENTRY_DECISION_PROTOCOL
from nurse_rcl.v2.live.ops_sre_directed_provider import OpsSreDirectedDecisionProvider


class OpsSreEntryDecisionProvider(OpsSreDirectedDecisionProvider):
    stage_profile = "hermes-sre-tools/19"
    stage_variant = "v2-ops-sre/9"
    stage_protocol = SRE_ENTRY_DECISION_PROTOCOL

    def _allows_bounded_entry(self) -> bool:
        # The inherited constructor verifies the independent ceiling grant first.
        return True
