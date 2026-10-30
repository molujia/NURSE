"""D16 opt-in, source-preserving comparable views over published evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_comparable_briefs import project_comparable_brief
from nurse_rcl.v2.live.ops_sre_comparable_pairs import mechanism_pairs
from nurse_rcl.v2.live.ops_sre_encoding import compact_record, expand
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_recovery_tools import (
    RECOVERY_SKILL_TOOLS,
    OpsRecoveryActionRecord,
    OpsSreRecoveryInvestigationLedger,
    OpsSreRecoveryToolbox,
)
from nurse_rcl.v2.live.ops_sre_score_cards import (
    SCORE_CARD_VERSION,
    SCORE_PAGE_KINDS,
    score_pairs,
)
from nurse_rcl.v2.live.ops_sre_score_lineage import SreScoreLineage
from nurse_rcl.v2.live.ops_sre_source_checkpoint import (
    ScoreSource,
    admit_source,
    check_state_capacity,
    pack_source_navigation,
    require_capacity,
    unpack_source_navigation,
    validate_source_envelope,
)
from nurse_rcl.v2.live.ops_sre_views import semantic_record
from nurse_rcl.v2.live.repair_module_timing import _module_call

COMPARABLE_PROFILE = "hermes-sre-tools/9"
COMPARABLE_SKILL_TOOLS = {k.removesuffix("/5") + "/6": v for k, v in RECOVERY_SKILL_TOOLS.items()}
_RECEIPTS = {f"sre-inspect/{i}" for i in range(1, 7)}


class OpsSreComparableToolbox(OpsSreRecoveryToolbox):
    """New profile only. Existing paid profiles keep their old page semantics."""

    profile_id = COMPARABLE_PROFILE
    skill_tools = COMPARABLE_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/8"
    inspection_version = "sre-inspect/6"
    _restore_score_source: Callable[[], None] | None = None

    @staticmethod
    def _score_pair_units(rows: list[dict[str, Any]], scope: list[str]) -> list[dict[str, Any]]:
        return score_pairs(rows, scope)

    def admit_score_lineage(self, projection: SreScoreLineage) -> None:
        """Trusted publication adapter only, after terminal score publication."""
        if type(projection) is not SreScoreLineage or (
            projection.core_identity.case_snapshot_id != self.case_snapshot_id
            or set(r.score.candidate_service_key for r in projection.candidates)
            != set(self._by_candidate)
        ):
            raise ValueError("score lineage publication identity differs")
        previous = getattr(self, "_score_lineage_sha256", None)
        if previous is not None:
            if previous != projection.projection_sha256:
                raise ValueError("score lineage already binds another published source")
            return
        published = {}
        for key in self._core_cards:
            card = self._cards[key]
            if card.tool_version == "ops-scoring/1":
                body = json.loads(card.payload_json)
                candidate = body["score"]["candidate_service_key"]
                if candidate in published:
                    raise ValueError("conflicting published scores")
                published[candidate] = body
        for row in projection.candidates:
            score = published.get(row.score.candidate_service_key)
            if (
                score is None
                or score["rank"] != row.rank
                or score["score"] != row.score.model_dump(mode="json")
            ):
                raise ValueError("score lineage requires identical published terminal scores")
        source = admit_source(
            projection,
            interval=(self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window),
        )
        self.admit_core(nodes=tuple(source.cards.values()), steps=())
        # Core admission deliberately copies values. Bind the owner's immutable
        # registry to those validated exact values once, not on every save.
        for key, card in source.cards.items():
            if self._cards[key] != card:
                raise ValueError("admitted source card value differs")
            self._cards[key] = card
        self._checkpoint_source = source
        self._score_lineage_sha256 = projection.projection_sha256

    @property
    def checkpoint_source(self) -> ScoreSource | None:
        return getattr(self, "_checkpoint_source", None)

    def _selected(self, tool: str, scope: list[str]) -> list[str]:
        # New source detail is opt-in via scoring focus or explicit read_node;
        # symptoms/all do not silently become a different investigation surface.
        return [
            k
            for k in super()._selected(tool, scope)
            if self._cards[k].tool_version != SCORE_CARD_VERSION
        ]

    def _current_prefix(
        self,
        tool: str,
        scope: list[str],
        *,
        focus: str = "all",
    ) -> dict[str, Any]:
        prefix = super()._current_prefix(tool, scope, focus=focus)
        if focus != "scoring" or tool not in {"inspect_candidate", "compare_candidates"}:
            return prefix
        extra = sorted(
            k
            for k, r in self._semantic.items()
            if (
                self._cards[k].tool_version == SCORE_CARD_VERSION
                and r["payload"].get("kind") in SCORE_PAGE_KINDS
                and any(c in scope for c in r["targets"])
            )
        )
        if extra:
            ids = [
                k for k in prefix["overview_ids"] if self._semantic[k]["node"] != "scoring"
            ] + extra
            prefix.update(
                overview_ids=ids,
                ids=extra,
                record_roles={k: self._semantic[k]["source_role"] for k in ids},
                record_targets={k: self._semantic[k]["targets"] for k in ids},
            )
        return prefix

    def _inspection_record(self, row: dict[str, Any]) -> dict[str, Any]:
        return project_comparable_brief(row)

    def _candidate_overview(self, candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return super()._candidate_overview(
            candidate,
            [
                r
                for r in rows
                if (
                    r.get("tool_version") != SCORE_CARD_VERSION
                    or r["payload"].get("kind") == "score_lineage_summary"
                )
            ],
        )

    def _refresh_semantic(self) -> None:
        super()._refresh_semantic()
        # Restored inspection receipts are navigation, never recursive evidence.
        excluded = {k for k in self._semantic if self._cards[k].tool_version in _RECEIPTS}
        for key in excluded:
            self._semantic.pop(key)
        detail = {k for k in self._semantic if self._cards[k].tool_version == SCORE_CARD_VERSION}
        for ids in self._by_candidate.values():
            # Detailed lineage remains in the explicit scoring surface and node
            # directory. Do not duplicate it into every decision's case situation.
            ids.difference_update(excluded | detail)

    def restore_evidence(self, saved: OpsInvestigationEvidence) -> None:
        if saved.evidence.case_snapshot_id != self.case_snapshot_id:
            raise ValueError("comparable receipt snapshot differs")
        if saved.evidence.tool_version in _RECEIPTS:
            return
        super().restore_evidence(saved)

    def _paired(self, prefix: dict[str, Any]) -> bool:
        return (prefix["tool"] == "compare_candidates" and prefix.get("focus") == "mechanisms") or (
            prefix["tool"] in {"inspect_candidate", "compare_candidates"}
            and prefix.get("focus") == "scoring"
        )

    @_module_call("sre_comparison_view")
    def _pair_units(self, prefix: dict[str, Any]) -> list[dict[str, Any]]:
        # A prefix freezes scope, role, source identities and publication. Cache
        # only this metadata within the toolbox, not observations across runs.
        cache = getattr(self, "_comparison_units", None)
        if cache is None:
            self._comparison_units: dict[str, list[dict[str, Any]]] = {}
            cache = self._comparison_units
        pin = self._pin(prefix)
        if pin not in cache:
            rows = [
                dict(
                    self._semantic[k],
                    source_role=prefix["record_roles"][k],
                    published_by_core=k in prefix["core_cards"],
                    targets=prefix["record_targets"][k],
                )
                for k in prefix["ids"]
            ]
            pair = self._score_pair_units if prefix.get("focus") == "scoring" else mechanism_pairs
            cache[pin] = pair(rows, prefix["scope"])
        return cache[pin]

    def _page_total(self, prefix: dict[str, Any]) -> int:
        return (
            len(self._pair_units(prefix)) if self._paired(prefix) else super()._page_total(prefix)
        )

    def _inspection_page(
        self,
        prefix: dict[str, Any],
        pin: str,
        offset: int,
        count: int,
        parameters: dict[str, Any],
        *,
        capacity: bool = False,
    ) -> dict[str, Any]:
        if not self._paired(prefix):
            return super()._inspection_page(
                prefix, pin, offset, count, parameters, capacity=capacity
            )
        units = self._pair_units(prefix)
        shown = units[offset : offset + count]
        ids = [
            k
            for pair in shown
            for side in ("left", "right")
            for k in pair.get(side, {}).get("card_ids", [])
        ]
        if count > 8 or len(ids) > 16:
            raise ExchangeBoundaryError(
                "json_limit",
                boundary="comparison_page_units_or_sources",
                observed=max(count * 2, len(ids)),
                limit=16,
            )
        # Keep the original prefix in the owner's registry. Only the body selection
        # passed to the existing view builder differs; retained-source checks still
        # require exact frozen source identity, role and one-sided targets.
        value = expand(
            super()._inspection_page(
                dict(prefix, ids=ids), pin, 0, len(ids), parameters, capacity=capacity
            )
        )
        following, total = offset + len(shown), len(units)
        more = following < total
        cursor = (
            "sre_cursor_" + self._pin(dict(prefix=pin, offset=following, previous_offset=offset))
            if more
            else None
        )
        value["page"].update(
            pairs=shown,
            count=len(shown),
            total=total,
            source_record_count=len(ids),
            unit="complete_comparison_unit",
            has_more=more,
            next_cursor=cursor,
            omitted_count=total - following,
            selection="complete_same_kind_pairs/1",
            omitted_reason="unread_complete_pairs_not_evidence_absence",
        )
        value["critical_overview"]["next_cursor"] = cursor
        value["focus_coverage"]["not_selected_count"] = len(prefix["overview_ids"]) - len(
            prefix["ids"]
        )
        value["limitations"].append("pair_container_does_not_expand_source_scope_or_authority")
        return compact_record(value)

    def _execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        result = super()._execute(tool, parameters)
        if tool == "check_mechanism":
            view = expand(json.loads(result.model_view_json))
            if "semantic_record" in view:
                source = semantic_record(result.evidence, published=False, origin=result.origin)
                view["semantic_record"] = project_comparable_brief(source)
                result = result.model_copy(update={"model_view_json": self._serialize_view(view)})
        return result

    def retained_brief_source(self, view: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        if view.get("publication_prefix") is not None:
            return super().retained_brief_source(view, row)
        self._refresh_semantic()
        source = self._semantic.get(row["card_id"])
        if source is None or row.get("mechanism_source_kind") != "supplementary_recheck":
            raise ValueError("direct comparable brief source differs")
        if canonical_json_bytes(project_comparable_brief(source)) != canonical_json_bytes(row):
            raise ValueError("comparable brief differs from retained source")
        return source

    def export_navigation(
        self, *, exclude_card_ids: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        self._publish()
        if exclude_card_ids:
            source = self.checkpoint_source
            if (
                source is None
                or exclude_card_ids != source.cards.keys()
                or any(self._cards.get(k) is not card for k, card in source.cards.items())
            ):
                raise ValueError("checkpoint exclusions differ from admitted source")
        return dict(
            super().export_navigation(exclude_card_ids=exclude_card_ids),
            score_lineage_sha256=getattr(self, "_score_lineage_sha256", None),
        )

    def restore_navigation(self, state: dict[str, Any]) -> None:
        if "score_lineage_sha256" not in state or state["score_lineage_sha256"] != (
            getattr(self, "_score_lineage_sha256", None)
        ):
            raise ValueError("saved score lineage was not independently admitted")
        self._comparison_units = {}
        super().restore_navigation({k: v for k, v in state.items() if k != "score_lineage_sha256"})
        if self.checkpoint_source:
            for key, card in self.checkpoint_source.cards.items():
                if self._cards[key] != card:
                    raise ValueError("restored source card value differs")
                self._cards[key] = card


class OpsSreComparableLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/8"] = (
        "nurse.v2.sre-investigation-ledger/8"
    )
    investigation_profile_id: Literal["hermes-sre-tools/9"] = "hermes-sre-tools/9"
    records: tuple[OpsRecoveryActionRecord, ...] = Field(max_length=8)


class OpsSreComparableInvestigationLedger(OpsSreRecoveryInvestigationLedger):
    _indexed: OpsSreComparableToolbox
    state_type: type[OpsLedgerStateBase] = OpsSreComparableLedgerState
    skill_tools = COMPARABLE_SKILL_TOOLS
    toolbox_type = OpsSreComparableToolbox

    def _state_values(self) -> dict[str, Any]:
        self._indexed._publish()
        source = self._indexed.checkpoint_source
        return dict(
            navigation=pack_source_navigation(
                self._indexed.export_navigation(
                    exclude_card_ids=frozenset(source.cards) if source else frozenset()
                ),
                source=source,
            )
        )

    def _decode_navigation(self, navigation: dict[str, Any]) -> dict[str, Any]:
        validate_source_envelope(navigation)
        if navigation.get("source") is not None and self._indexed.checkpoint_source is None:
            restore = self._indexed._restore_score_source
            if restore is not None:
                restore()
        return unpack_source_navigation(navigation, source=self._indexed.checkpoint_source)

    def _read_state_bytes(self, path: Path) -> bytes:
        from nurse_rcl.v2.live.staged_case import require_plain_path

        require_capacity(
            require_plain_path(path).stat().st_size, self.state_bytes, "checkpoint_file_bytes"
        )
        return super()._read_state_bytes(path)

    def _validate_state_capacity(self, state: OpsLedgerStateBase, raw: bytes) -> None:
        check_state_capacity(
            state.model_dump(mode="json"), file_limit=self.state_bytes, raw_size=len(raw)
        )

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        # The D15 extra feedback validation is retained with the new exact state
        # type. Its final assertion names /6, so invoke the common parent then
        # validate the identical rejection bindings here, without a fake /6 cast.
        from nurse_rcl.v2.live.ops_sre_condition_tools import OpsSreConditionInvestigationLedger

        OpsSreConditionInvestigationLedger._restore_tool_state(self, state)
        assert isinstance(state, OpsSreComparableLedgerState)
        for row in state.records:
            if row.rejection is None or row.model_view_json is None:
                continue
            view = expand(json.loads(row.model_view_json))
            if view.get("status") != "resource_limit" and (
                view.get("parameter_feedback", {}).get("diagnostic")
                != row.rejection.model_dump(mode="json")
                or view.get("route_binding_sha256") != row.request_binding_sha256
            ):
                raise ValueError("saved comparable rejection diagnostic or binding differs")
