"""New scope protocol requires its own approved bounded-stage execution."""

from nurse_rcl.v2.live.ops_empirical_policy import SRE_OBSERVABLE_DECISION_PROTOCOL
from nurse_rcl.v2.live.ops_sre_directed_provider import OpsSreDirectedDecisionProvider


class OpsSreObservableDecisionProvider(OpsSreDirectedDecisionProvider):
    stage_profile = "hermes-sre-tools/20"
    stage_variant = "v2-ops-sre/10"
    stage_protocol = SRE_OBSERVABLE_DECISION_PROTOCOL

    def _allows_bounded_observable(self) -> bool:
        # The inherited constructor first verifies both grant and execution approval.
        return True
