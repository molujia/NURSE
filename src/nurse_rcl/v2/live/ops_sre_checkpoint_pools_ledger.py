"""Version seventeen storage hooks; durable owner semantics remain inherited."""

from __future__ import annotations

from typing import Any, Literal

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_checkpoint_ledger import (
    OpsSreCheckpointInvestigationLedger,
    OpsSreCheckpointLedgerState,
)
from nurse_rcl.v2.live.ops_sre_checkpoint_pools import (
    pack_pooled_source_navigation,
    unpack_pooled_source_navigation,
    validate_pooled_source_envelope,
)


class OpsSrePooledLedgerState(OpsSreCheckpointLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/17"] = (
        "nurse.v2.sre-investigation-ledger/17"  # type: ignore[assignment]
    )


class OpsSrePooledInvestigationLedger(OpsSreCheckpointInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSrePooledLedgerState

    def _state_values(self) -> dict[str, Any]:
        self._indexed._publish()
        source = self._indexed.checkpoint_source
        return dict(
            navigation=pack_pooled_source_navigation(
                self._indexed.export_navigation(
                    exclude_card_ids=frozenset(source.cards) if source else frozenset(),
                ),
                source=source,
            )
        )

    def _decode_navigation(self, navigation: dict[str, Any]) -> dict[str, Any]:
        validate_pooled_source_envelope(navigation)
        if navigation["source"] is not None and self._indexed.checkpoint_source is None:
            restore = self._indexed._restore_score_source
            if restore is not None:
                restore()
        return unpack_pooled_source_navigation(navigation, source=self._indexed.checkpoint_source)
