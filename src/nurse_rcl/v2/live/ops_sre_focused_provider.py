"""Separately granted focused decisions retain all bounded transport guards."""

from nurse_rcl.v2.live.ops_sre_atomic_selector_provider import OpsSreAtomicSelectorDecisionProvider


class OpsSreFocusedDecisionProvider(OpsSreAtomicSelectorDecisionProvider):
    stage_profile = "hermes-sre-tools/25"  # type: ignore[assignment]
    stage_variant = "v2-ops-sre/15"
    stage_protocol = "nurse.v2.sre-decision-exchange/26"

    def _allows_bounded_focused(self) -> bool:
        return True
