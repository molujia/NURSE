"""D23 post-action coverage from actual source sets, actions and deliveries."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_d21_workbench import post_action_state
from nurse_rcl.v2.live.ops_sre_encoding import expand

_NODES = ("symptoms", "mechanisms", "scoring")
_LABELS = ("left", "right")
_KINDS = ("complete", "brief", "partial")


def _manifest(view: dict[str, Any]) -> dict[str, Any] | None:
    value = view.get("role_source_manifest")
    if value is None:
        return None
    if (
        type(value) is not dict
        or set(value) != {
            "schema_version", "scope", "publication_epoch_sha256", "authority", *_LABELS,
        }
        or value["schema_version"] != "nurse.v2.sre-role-source-manifest/1"
        or value["authority"] != "frozen_publication_set_identity_not_delivered_evidence"
        or type(value["publication_epoch_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["publication_epoch_sha256"]) is None
        or type(value["scope"]) is not list
        or len(value["scope"]) != 2
        or len(set(value["scope"])) != 2
    ):
        raise ValueError("D23 source manifest identity differs")
    for label in _LABELS:
        if type(value[label]) is not dict or set(value[label]) != set(_NODES):
            raise ValueError("D23 source manifest role differs")
        for node in _NODES:
            source = value[label][node]
            if (
                type(source) is not dict
                or set(source) != {"source_set_sha256", "total_records", "status"}
                or type(source["source_set_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", source["source_set_sha256"]) is None
                or type(source["total_records"]) is not int
                or source["total_records"] < 0
                or source["status"] not in {
                    "published_source_set", "not_published_in_frozen_prefix",
                }
            ):
                raise ValueError("D23 source manifest count differs")
    return value


def build_investigation_workbench(
    actions: list[dict[str, Any]],
    history: list[dict[str, Any]],
    delivered_index: dict[str, Any],
    post_state: dict[str, Any],
    *,
    role_source_resolver: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Never infer question resolution or transform a navigation count into evidence."""
    if (
        type(actions) is not list or type(history) is not list
        or len(actions) != len(history) or len(actions) > 8
        or any(
            action.get("action_index") != index
            or saved.get("action_index") != index
            or action.get("status") != saved.get("status")
            for index, (action, saved) in enumerate(zip(actions, history, strict=True), 1)
        )
        or post_state != post_action_state(
            actions, delivered_index, actions_available_now=8 - len(actions)
        )
    ):
        raise ValueError("D23 workbench ledger identity differs")
    views = [expand(saved["view"]) for saved in history]
    manifests = [(i, value) for i, view in enumerate(views, 1)
                 if (value := _manifest(view)) is not None]
    source_members: dict[int, dict[str, Any]] = {}
    if role_source_resolver is not None:
        for index, manifest in manifests:
            pin = views[index - 1].get("publication_prefix")
            if type(pin) is not str:
                raise ValueError("D23 role source prefix missing")
            resolved = role_source_resolver(pin)
            if (
                type(resolved) is not dict or set(resolved) != {"manifest", "members"}
                or canonical_json_bytes(resolved["manifest"]) != canonical_json_bytes(manifest)
                or type(resolved["members"]) is not dict
                or set(resolved["members"]) != set(_LABELS)
            ):
                raise ValueError("D23 role source identity differs")
            for label in _LABELS:
                roles = resolved["members"][label]
                if type(roles) is not dict or set(roles) != set(_NODES):
                    raise ValueError("D23 role source members differ")
                for node in _NODES:
                    identities = roles[node]
                    if (
                        type(identities) is not list
                        or any(type(pair) is not list or len(pair) != 2
                               or any(type(part) is not str for part in pair)
                               for pair in identities)
                        or len(identities) != len({tuple(pair) for pair in identities})
                        or len(identities) != manifest[label][node]["total_records"]
                        or sha256_hex(canonical_json_bytes(sorted(identities)))
                        != manifest[label][node]["source_set_sha256"]
                    ):
                        raise ValueError("D23 role source member digest differs")
            source_members[index] = resolved["members"]
    current = manifests[-1][1] if manifests else None
    scope = current["scope"] if current is not None else []
    epoch = current["publication_epoch_sha256"] if current is not None else None
    coverage: dict[str, Any] = {}
    changed = False
    if current is not None:
        manifest_by_index = dict(manifests)
        latest_manifest_index: dict[int, int | None] = {}
        active: int | None = None
        for index in range(1, len(actions) + 1):
            if index in manifest_by_index:
                active = index
            latest_manifest_index[index] = active
        current_members = source_members.get(manifests[-1][0])
        for label, candidate in zip(_LABELS, scope, strict=True):
            coverage[label] = {}
            for node in _NODES:
                source = current[label][node]
                digest = source["source_set_sha256"]
                matching = {
                    index
                    for index, manifest in manifests
                    if manifest["scope"] == scope
                    and manifest["publication_epoch_sha256"] == epoch
                    and manifest[label][node]["source_set_sha256"] == digest
                }
                changed |= any(
                    manifest["scope"] == scope
                    and (
                        manifest["publication_epoch_sha256"] != epoch
                        or manifest[label][node]["source_set_sha256"] != digest
                    )
                    for _, manifest in manifests[:-1]
                )
                selected = []
                for row in delivered_index["records"]:
                    if row["node"] != node or row["targets"] != [candidate]:
                        continue
                    for index in row["action_indices"]:
                        bound = latest_manifest_index[index]
                        if bound not in matching:
                            continue
                        if index not in matching:
                            action = actions[index - 1]
                            parameters = action.get("parameters") or {}
                            if (
                                action["tool"] != "read_node"
                                or action["status"] != "completed"
                                or parameters != dict(
                                    card_id=row["card_id"], section="payload", cursor=None,
                                )
                            ):
                                continue
                        if current_members is not None:
                            identities = current_members[label][node]
                            if row["delivery_kind"] == "complete":
                                admitted = [row["card_id"], row["record_sha256"]] in identities
                            else:
                                admitted = any(pair[0] == row["card_id"] for pair in identities)
                            if not admitted:
                                continue
                        elif index not in matching:
                            continue  # Synthetic legacy projection has no frozen member source.
                        selected.append(row)
                        break
                counts = {
                    kind: len({r["record_sha256"] for r in selected
                               if r["delivery_kind"] == kind})
                    for kind in _KINDS
                }
                complete_cards = {r["card_id"] for r in selected
                                  if r["delivery_kind"] == "complete"}
                total = source["total_records"]
                coverage[label][node] = dict(
                    publication_epoch_sha256=epoch,
                    source_set_sha256=digest,
                    total_records=total,
                    source_status=source["status"],
                    unique_delivered_records=counts,
                    unread_records=total - len(complete_cards)
                    if len(complete_cards) <= total else "unknown",
                    known_child_unread="unknown",
                    meaning="representation_cover_not_evidence_vote_or_complete_domain_check",
                )
    latest = None
    cursor = None
    if actions:
        actual = post_state["last_actual_action"]
        latest = dict(
            action_index=actual["action_index"], tool=actual["tool"],
            status=actual["status"], content_relation=post_state["content_relation"],
            delivered_record_refs=list(delivered_index["latest_record_refs"]),
        )
        view = views[-1]
        page = view.get("page", {})
        parameters = actions[-1].get("parameters") or {}
        if (
            actual["status"] == "completed"
            and type(page) is dict and page.get("next_cursor") is not None
            and view.get("publication_prefix") is not None
        ):
            cursor = dict(
                tool=actual["tool"], scope=view.get("scope", []),
                focus=parameters.get("focus", "all"),
                publication_prefix=view["publication_prefix"],
                next_cursor=page["next_cursor"],
            )
    return dict(
        schema_version="nurse.v2.sre-investigation-workbench/1",
        scope=scope,
        latest_actual_result=latest,
        coverage=coverage,
        source_set_changed=changed,
        cursor_binding=cursor,
        focus_switch_rule=(
            "Change focus only with cursor=null; an old cursor remains bound "
            "to its tool, scope, focus and publication prefix."
        ),
        reachable_role_entries={
            node: dict(tool="compare_candidates", focus=node, cursor=None)
            for node in _NODES
        } if scope else {},
        public_memory_role="pre_action_intent_not_current_receipt",
        authority="ledger_and_frozen_delivery_metadata_not_diagnostic_conclusion",
    )


def validate_investigation_workbench(context: dict[str, Any]) -> None:
    """Bind the model-visible latest receipt and bounded coverage to this turn."""
    work = context.get("investigation_workbench")
    actions = context.get("actions")
    post = context.get("post_action_state")
    if (
        type(work) is not dict or type(actions) is not list or type(post) is not dict
        or set(work) != {
            "schema_version", "scope", "latest_actual_result", "coverage",
            "source_set_changed", "cursor_binding", "focus_switch_rule",
            "reachable_role_entries", "public_memory_role", "authority",
        }
        or work["schema_version"] != "nurse.v2.sre-investigation-workbench/1"
        or work["authority"] != "ledger_and_frozen_delivery_metadata_not_diagnostic_conclusion"
        or work["public_memory_role"] != "pre_action_intent_not_current_receipt"
        or type(work["source_set_changed"]) is not bool
    ):
        raise ValueError("D23 workbench identity differs")
    expected_latest = None
    if actions:
        actual = post["last_actual_action"]
        expected_latest = {
            "action_index": actual["action_index"], "tool": actual["tool"],
            "status": actual["status"], "content_relation": post["content_relation"],
            "delivered_record_refs": context["delivered_record_index"]["latest_record_refs"],
        }
    if work["latest_actual_result"] != expected_latest:
        raise ValueError("D23 workbench latest result differs")
    if type(work["coverage"]) is not dict or set(work["coverage"]) not in (
        set(), set(_LABELS)
    ):
        raise ValueError("D23 workbench coverage differs")
    scope = work["scope"]
    if (not work["coverage"] and scope != []) or (
        work["coverage"] and (
            type(scope) is not list or len(scope) != 2 or len(set(scope)) != 2
        )
    ):
        raise ValueError("D23 workbench scope differs")
    for label in work["coverage"]:
        rows = work["coverage"][label]
        if type(rows) is not dict or set(rows) != set(_NODES):
            raise ValueError("D23 workbench role coverage differs")
        for node in _NODES:
            item = rows[node]
            if type(item) is not dict or set(item) != {
                "publication_epoch_sha256", "source_set_sha256", "total_records", "source_status",
                "unique_delivered_records", "unread_records", "known_child_unread", "meaning",
            }:
                raise ValueError("D23 workbench role item differs")
            counts = item["unique_delivered_records"]
            total = item["total_records"]
            if (
                type(counts) is not dict or set(counts) != set(_KINDS)
                or type(item["publication_epoch_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", item["publication_epoch_sha256"]) is None
                or any(type(counts[kind]) is not int or counts[kind] < 0 for kind in _KINDS)
                or type(total) is not int or total < 0
                or (
                    item["unread_records"] != "unknown"
                    and (
                        type(item["unread_records"]) is not int
                        or not 0 <= item["unread_records"] <= total
                    )
                )
                or item["known_child_unread"] != "unknown"
                or item["meaning"]
                != "representation_cover_not_evidence_vote_or_complete_domain_check"
            ):
                raise ValueError("D23 workbench coverage counts differ")
