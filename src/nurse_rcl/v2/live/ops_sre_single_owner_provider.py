"""No general provider bypass: this protocol requires its own approved grant."""

from nurse_rcl.v2.live.ops_sre_directed_provider import OpsSreDirectedDecisionProvider
from nurse_rcl.v2.live.ops_sre_single_owner_exchange import SINGLE_OWNER_PROTOCOL


class OpsSreSingleOwnerDecisionProvider(OpsSreDirectedDecisionProvider):
    stage_profile = "hermes-sre-tools/21"
    stage_variant = "v2-ops-sre/11"
    stage_protocol = SINGLE_OWNER_PROTOCOL

    def _allows_bounded_single_owner(self) -> bool:
        # The inherited constructor verifies execution approval and stage grant first.
        return True
