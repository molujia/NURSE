"""Separately granted bounded presentation; unchanged atomic execution and attempts."""

from nurse_rcl.v2.live.ops_sre_atomic_selector_provider import OpsSreAtomicSelectorDecisionProvider
from nurse_rcl.v2.live.ops_sre_feedback_fact_exchange import FACT_PROTOCOL


class OpsSreFeedbackFactDecisionProvider(OpsSreAtomicSelectorDecisionProvider):
    stage_variant = "v2-ops-sre/14"
    stage_protocol = FACT_PROTOCOL

    def _allows_bounded_feedback_fact(self) -> bool:
        return True
