"""D23 opt-in balanced first-page triage over the unchanged D22 source tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_d22_tools import (
    RECEIPT_REUSE_SKILL_TOOLS,
    OpsSreReceiptReuseActionRecord,
    OpsSreReceiptReuseInvestigationLedger,
    OpsSreReceiptReuseLedgerState,
    OpsSreReceiptReusePublicationReader,
    OpsSreReceiptReuseToolbox,
)
from nurse_rcl.v2.live.ops_sre_d23_triage import build_role_contrast, build_role_manifest
from nurse_rcl.v2.live.ops_sre_encoding import compact_record, expand
from nurse_rcl.v2.live.ops_sre_score_cards import SCORE_CARD_VERSION, SCORE_PAGE_KINDS
from nurse_rcl.v2.live.ops_sre_views import record_order

BALANCED_PROFILE: Literal["hermes-sre-tools/14"] = "hermes-sre-tools/14"
BALANCED_SKILL_TOOLS = {
    key.removesuffix("/10") + "/11": value
    for key, value in RECEIPT_REUSE_SKILL_TOOLS.items()
}


class OpsSreBalancedToolbox(OpsSreReceiptReuseToolbox):
    profile_id = BALANCED_PROFILE  # type: ignore[assignment]
    skill_tools = BALANCED_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/10"

    def role_source_members(self, pin: str) -> dict[str, Any]:
        """Internal frozen membership for coverage; never a model-visible page."""
        prefix = self._inspection_prefixes.get(pin)
        if (
            type(pin) is not str or prefix is None
            or prefix.get("tool") != "compare_candidates"
            or self._pin(prefix) != pin
        ):
            raise ValueError("D23 role source prefix differs")
        self._restored_prefix_extra_fields(prefix)
        score = prefix["triage_score_prefix"]
        rows = [
            dict(self._semantic[key], source_role=prefix["record_roles"][key],
                 targets=prefix["record_targets"][key])
            for key in prefix["overview_ids"]
            if self._semantic[key]["node"] in {"symptoms", "mechanisms"}
        ] + [
            dict(self._semantic[key], source_role=score["record_roles"][key],
                 targets=score["record_targets"][key])
            for key in score["ids"]
        ]
        manifest = prefix["role_source_manifest"]
        members: dict[str, dict[str, list[list[str]]]] = {}
        for label, candidate in zip(("left", "right"), prefix["scope"], strict=True):
            members[label] = {}
            for node in ("symptoms", "mechanisms", "scoring"):
                identities = sorted(
                    [row["card_id"], sha256_hex(canonical_json_bytes(row))]
                    for row in rows if row["node"] == node and row["targets"] == [candidate]
                )
                source = manifest[label][node]
                if (
                    len(identities) != source["total_records"]
                    or sha256_hex(canonical_json_bytes(identities))
                    != source["source_set_sha256"]
                ):
                    raise ValueError("D23 role source membership differs")
                members[label][node] = identities
        return dict(manifest=manifest, members=members)

    def _restored_prefix_extra_fields(self, prefix: dict[str, Any]) -> set[str]:
        if prefix.get("tool") != "compare_candidates":
            return set()
        extra_fields = {"triage_score_prefix", "role_source_manifest"}
        base_fields = {
            "tool", "scope", "ids", "core_cards", "completed_requests",
            "core_requests", "record_roles", "record_targets", "focus", "overview_ids",
        }
        if (
            not base_fields | extra_fields <= set(prefix)
            or type(prefix["scope"]) is not list
            or len(prefix["scope"]) != 2
            or type(prefix["overview_ids"]) is not list
            or type(prefix["core_cards"]) is not list
            or type(prefix["record_roles"]) is not dict
            or type(prefix["record_targets"]) is not dict
        ):
            raise ValueError("D23 inspection prefix shape differs")
        base = {key: prefix[key] for key in base_fields}
        scope = base["scope"]
        try:
            score_ids = sorted(
                key for key in base["core_cards"]
                if (
                    key in self._semantic
                    and self._cards[key].tool_version == SCORE_CARD_VERSION
                    and self._semantic[key]["payload"].get("kind") in SCORE_PAGE_KINDS
                    and any(candidate in scope for candidate in self._semantic[key]["targets"])
                )
            )
            score = dict(
                base, focus="scoring",
                ids=[key for key in base["overview_ids"]
                     if self._semantic[key]["node"] == "scoring"],
            )
            if score_ids:
                selected = [key for key in base["overview_ids"]
                            if self._semantic[key]["node"] != "scoring"] + score_ids
                score.update(
                    overview_ids=selected,
                    ids=score_ids,
                    record_roles={key: self._semantic[key]["source_role"] for key in selected},
                    record_targets={key: self._semantic[key]["targets"] for key in selected},
                )
            neutral = dict(base, focus="all", ids=base["overview_ids"])
            if base["focus"] == "scoring" and score_ids:
                # The paired scoring focus replaces ordinary scoring records
                # in its overview. Recover the frozen all-focus publication
                # from the saved non-scoring IDs and frozen core scoring cards,
                # never from subsequently acquired observations.
                neutral_ids = sorted(
                    {
                        key for key in base["overview_ids"] if key not in score_ids
                    } | {
                        key for key in base["core_cards"]
                        if key in self._semantic
                        and self._cards[key].tool_version != SCORE_CARD_VERSION
                        and self._semantic[key]["node"] == "scoring"
                        and any(candidate in scope for candidate in self._semantic[key]["targets"])
                    },
                    key=lambda key: record_order(self._semantic[key]),
                )
                neutral.update(
                    overview_ids=neutral_ids,
                    ids=neutral_ids,
                    record_roles={
                        key: self._semantic[key]["source_role"] for key in neutral_ids
                    },
                    record_targets={
                        key: self._semantic[key]["targets"] for key in neutral_ids
                    },
                )
            records = [
                dict(self._semantic[key], source_role=base["record_roles"][key],
                     targets=base["record_targets"][key])
                for key in base["overview_ids"]
                if self._semantic[key]["node"] in {"symptoms", "mechanisms"}
            ]
            score_rows = [
                dict(self._semantic[key], source_role=score["record_roles"][key],
                     targets=score["record_targets"][key])
                for key in score["ids"]
            ]
            epoch = sha256_hex(canonical_json_bytes(dict(
                all_focus_prefix=neutral,
                scoring_sources=[
                    (row["card_id"], sha256_hex(canonical_json_bytes(row)))
                    for row in score_rows
                ],
            )))
            manifest = build_role_manifest(
                scope, records, score_rows, publication_epoch_sha256=epoch,
            )
            if (
                canonical_json_bytes(prefix["triage_score_prefix"])
                != canonical_json_bytes(score)
                or canonical_json_bytes(prefix["role_source_manifest"])
                != canonical_json_bytes(manifest)
            ):
                raise ValueError("D23 inspection prefix source differs")
        except (KeyError, TypeError) as exc:
            raise ValueError("D23 inspection prefix source differs") from exc
        return extra_fields

    def _current_prefix(
        self, tool: str, scope: list[str], *, focus: str = "all"
    ) -> dict[str, Any]:
        prefix = super()._current_prefix(tool, scope, focus=focus)
        if tool == "compare_candidates" and len(scope) == 2:
            neutral = (
                prefix.copy() if focus == "all"
                else super()._current_prefix(tool, scope, focus="all")
            )
            # Freeze score source identities with this same publication prefix;
            # the ordinary all-focus page still contains exactly its old IDs.
            score = super()._current_prefix(tool, scope, focus="scoring")
            prefix["triage_score_prefix"] = score
            records = [
                dict(
                    self._semantic[key],
                    source_role=prefix["record_roles"][key],
                    targets=prefix["record_targets"][key],
                )
                for key in prefix["overview_ids"]
                if self._semantic[key]["node"] in {"symptoms", "mechanisms"}
            ]
            score_rows = [
                dict(
                    self._semantic[key],
                    source_role=score["record_roles"][key],
                    targets=score["record_targets"][key],
                )
                for key in score["ids"]
            ]
            # A focus switch changes the cursor pin, not the publication epoch.
            # Re-derive the all-focus source prefix so a changed frozen source
            # cannot inherit old read coverage merely because card bytes match.
            epoch = sha256_hex(canonical_json_bytes(dict(
                all_focus_prefix=neutral,
                scoring_sources=[
                    (row["card_id"], sha256_hex(canonical_json_bytes(row)))
                    for row in score_rows
                ],
            )))
            prefix["role_source_manifest"] = build_role_manifest(
                scope, records, score_rows, publication_epoch_sha256=epoch,
            )
        return prefix

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
        value = super()._inspection_page(
            prefix, pin, offset, count, parameters, capacity=capacity
        )
        if prefix["tool"] != "compare_candidates":
            return value
        expanded = expand(value)
        expanded["role_source_manifest"] = prefix["role_source_manifest"]
        if prefix.get("focus") != "all" or offset != 0:
            return compact_record(expanded)
        score_prefix = prefix["triage_score_prefix"]
        score_pairs = self._pair_units(score_prefix)
        score = next(
            row["used_source_contrast"]
            for row in score_pairs
            if row["kind"] == "score_comparison"
        )
        records = [
            dict(
                self._semantic[key],
                source_role=prefix["record_roles"][key],
                targets=prefix["record_targets"][key],
            )
            for key in prefix["overview_ids"]
            if self._semantic[key]["node"] in {"symptoms", "mechanisms"}
        ]
        expanded["role_contrast"] = build_role_contrast(
            prefix["scope"], records, score,
            delivered_records=expanded["page"]["records"],
        )
        return compact_record(expanded)


class OpsSreBalancedLedgerState(OpsSreReceiptReuseLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/13"] = (
        "nurse.v2.sre-investigation-ledger/13"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/14"] = BALANCED_PROFILE  # type: ignore[assignment]
    records: tuple[OpsSreReceiptReuseActionRecord, ...] = Field(max_length=8)


class OpsSreBalancedInvestigationLedger(OpsSreReceiptReuseInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreBalancedLedgerState
    skill_tools = BALANCED_SKILL_TOOLS
    toolbox_type = OpsSreBalancedToolbox


class OpsSreBalancedPublicationReader(OpsSreReceiptReusePublicationReader):
    toolbox_type = OpsSreBalancedToolbox
