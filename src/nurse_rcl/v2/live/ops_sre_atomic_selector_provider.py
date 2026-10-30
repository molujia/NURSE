"""Only a separately approved bounded atomic-selector stage can send decisions."""

from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import ATOMIC_PROTOCOL
from nurse_rcl.v2.live.ops_sre_directed_provider import OpsSreDirectedDecisionProvider


class OpsSreAtomicSelectorDecisionProvider(OpsSreDirectedDecisionProvider):
    stage_profile = "hermes-sre-tools/23"
    stage_variant = "v2-ops-sre/13"
    stage_protocol = ATOMIC_PROTOCOL

    def _allows_bounded_atomic_selector(self) -> bool:
        # The inherited constructor checks exact plan/grant/ceiling before this capability.
        return True
