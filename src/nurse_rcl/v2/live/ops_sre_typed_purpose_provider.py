"""Typed purpose can only use a separately approved bounded decision grant."""

from nurse_rcl.v2.live.ops_sre_directed_provider import OpsSreDirectedDecisionProvider
from nurse_rcl.v2.live.ops_sre_typed_purpose_exchange import TYPED_PURPOSE_PROTOCOL


class OpsSreTypedPurposeDecisionProvider(OpsSreDirectedDecisionProvider):
    stage_profile = "hermes-sre-tools/22"
    stage_variant = "v2-ops-sre/12"
    stage_protocol = TYPED_PURPOSE_PROTOCOL

    def _allows_bounded_typed_purpose(self) -> bool:
        # The inherited constructor checks execution approval and exact stage grant first.
        return True
