"""Source-bound progress, never hidden-card delivery or a diagnostic policy."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_context import _delivered_records
from nurse_rcl.v2.live.ops_sre_d23_workbench import build_investigation_workbench
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_workbench import delivered_record_index, select_public_memory

PROGRESS_SCHEMA = "nurse.v2.sre-investigation-progress/1"
PROGRESS_AUTHORITY = "delivered_source_comparison_not_new_evidence_or_diagnosis"
_FIELDS = (
    "family.value", "selected_term.metric_name", "selected_term.modality",
    "selected_term.local_strength", "selected_term.role_mechanism_ids",
)
_FIELD_BYTES = 2048
_CONTRAST_BYTES = 16 * 1024


def select_replanning_memory(memory: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit input-only omission; original statements stay in the conversation."""
    selected = select_public_memory([
        {k: deepcopy(v) for k, v in row.items() if k not in {"public_summary", "unresolved"}}
        for row in memory
    ])
    for group in selected["question_history"]:
        group.pop("unresolved")
    selected["selection"] = dict(
        omitted_fields=["public_summary", "unresolved"],
        meaning="lossy_input_selection_original_conversation_retained",
        resolution="not_assessed_by_program",
    )
    return selected


def exact_replanning_read(
    value: dict[str, Any], records: Any, *, toolbox: Any,
    decisions: Any, context: dict[str, Any],
) -> dict[str, Any] | None:
    """D21 proof plus D25's publication-epoch and source-set boundary."""
    from nurse_rcl.v2.live.ops_sre_d21_reuse import exact_delivered_ledger_read

    proof = exact_delivered_ledger_read(value, records, toolbox=toolbox)
    if proof is None:
        return None
    prior = next((row for row in decisions
                  if row.action_index == proof["prior_action_index"]), None)
    if prior is None:
        raise ValueError("D25 duplicate proof lacks declared action")
    current = context["investigation_progress"]
    previous = json.loads(prior.context_json)["investigation_progress"]
    def epochs(progress: dict[str, Any]) -> dict[str, Any]:
        return {side: {node: [item["publication_epoch_sha256"], item["source_set_sha256"]]
                       for node, item in roles.items()}
                for side, roles in progress["coverage"].items()}

    if current["scope"] != previous["scope"] or epochs(current) != epochs(previous):
        return None
    return proof


def validate_blocked_read(value: dict[str, Any], proof: dict[str, Any],
                          context: dict[str, Any]) -> None:
    """Validate a non-dispatched paid proposal; it is never a model stop."""
    actions = value.get("actions")
    if type(actions) is not list or len(actions) != 1 or actions[0]["tool"] != "read_node":
        raise ValueError("D25 blocked proposal differs")
    parameters = actions[0]["parameters"]
    if set(proof) != {"reason", "prior_action_index", "card_id", "parameters_sha256",
                      "record_sha256", "source_refs", "representation"} or (
        proof["reason"] != "already_delivered_no_new_content"
        or proof["representation"] != "complete_semantic_record"
        or proof["card_id"] != parameters.get("card_id")
        or parameters.get("section") != "payload" or parameters.get("cursor") is not None
        or proof["parameters_sha256"] != sha256_hex(canonical_json_bytes(parameters))
    ):
        raise ValueError("D25 blocked proof differs")
    matched = [row for row in context["delivered_record_index"]["records"]
               if row["record_sha256"] == proof["record_sha256"]
               and row["card_id"] == proof["card_id"] and row["delivery_kind"] == "complete"
               and proof["prior_action_index"] in row["action_indices"]]
    if len(matched) != 1:
        raise ValueError("D25 blocked proof was not completely delivered")


def _ref(row: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return dict(
        card_id=row["card_id"], record_sha256=sha256_hex(canonical_json_bytes(row)),
        action_indices=sorted(indices), source_refs=deepcopy(row.get("source_refs", [])),
        source_role=row["source_role"],
    )


def _contrast(left: tuple[dict[str, Any], list[int]] | None,
              right: tuple[dict[str, Any], list[int]] | None) -> dict[str, Any]:
    bindings = {side: _ref(item[0], item[1]) if item is not None else None
                for side, item in (("left", left), ("right", right))}
    if left is None or right is None:
        return dict(bindings, status="not_supplied", fields={}, omitted_fields=[])
    lrow, rrow = left[0], right[0]
    lb, rb = lrow["payload"], rrow["payload"]
    if not lrow.get("source_refs") or lrow.get("source_refs") != rrow.get("source_refs"):
        return dict(bindings, status="source_identity_differs", fields={}, omitted_fields=[])
    bodies = (lb, rb)
    if any(
        type(body.get("selected_term")) is not dict
        or body["selected_term"].get("term_id") != body["family"].get("selected_term_id")
        or type(body["family"].get("value")) not in (int, float)
        or type(body["selected_term"].get("local_strength")) not in (int, float)
        for body in bodies
    ):
        return dict(bindings, status="numeric_unknown_or_selection_not_supplied",
                    fields={}, omitted_fields=[])
    fields: dict[str, Any] = {}
    omitted = []
    for path in _FIELDS:
        parent, child = path.split(".")
        values = [body[parent].get(child) for body in bodies]
        available = all(child in body[parent] and body[parent][child] is not None
                        for body in bodies)
        if available and len(canonical_json_bytes(values)) > _FIELD_BYTES:
            available = False
            omitted.append(dict(field=path, reason="capacity"))
        fields[path] = dict(
            left=deepcopy(values[0]) if available else None,
            right=deepcopy(values[1]) if available else None,
            relation=("same" if canonical_json_bytes(values[0]) == canonical_json_bytes(values[1])
                      else "different") if available else "not_supplied",
        )
    return dict(bindings, status="complete_delivered_same_epoch", fields=fields,
                omitted_fields=omitted)


def build_investigation_progress(
    actions: list[dict[str, Any]], history: list[dict[str, Any]],
    delivered_index: dict[str, Any], post_state: dict[str, Any], *,
    role_source_resolver: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Owner admits history; the resolver supplies only verified member identities.

    No payload lookup, source reread, measurement or score calculation is allowed.
    """
    if canonical_json_bytes(delivered_index) != canonical_json_bytes(
        delivered_record_index(history)
    ):
        raise ValueError("D25 progress delivery index differs")
    work = build_investigation_workbench(
        actions, history, delivered_index, post_state, role_source_resolver=role_source_resolver,
    )
    result = dict(
        deepcopy(work), schema_version=PROGRESS_SCHEMA, authority=PROGRESS_AUTHORITY,
        family_contrasts=[], omitted_contrasts=[],
        comparison_meaning="original_delivered_values_not_causal_contribution_or_new_evidence",
    )
    views = [expand(saved["view"]) for saved in history]
    current = next((view for view in reversed(views) if "role_source_manifest" in view), None)
    if current is None or role_source_resolver is None:
        return result
    resolved = role_source_resolver(current["publication_prefix"])
    manifest, members = resolved["manifest"], resolved["members"]
    # Membership and every manifest digest have already been checked by D23 above.
    epoch = manifest["publication_epoch_sha256"]
    scope = manifest["scope"]
    by_digest = {row["record_sha256"]: row for row in delivered_index["records"]}
    supplied: dict[str, dict[str, tuple[dict[str, Any], list[int]]]] = {"left": {}, "right": {}}
    active = None
    for index, (action, view) in enumerate(zip(actions, views, strict=True), 1):
        if "role_source_manifest" in view:
            active = view["role_source_manifest"]
        if action["status"] != "completed" or active is None or (
            active["scope"] != scope or active["publication_epoch_sha256"] != epoch
            or any(active[side]["scoring"] != manifest[side]["scoring"]
                   for side in ("left", "right"))
        ):
            continue
        for row in _delivered_records(view):
            digest = sha256_hex(canonical_json_bytes(row))
            meta = by_digest.get(digest)
            if (
                meta is None or meta["delivery_kind"] != "complete"
                or row.get("node") != "scoring" or row.get("tool_version") != "sre-score-lineage/1"
                or row.get("source_role") != "core_published"
                or row["payload"].get("kind") != "score_family"
                or row["payload"].get("used_in_aggregation") is not True
            ):
                continue
            for side, candidate in zip(("left", "right"), scope, strict=True):
                if (
                    row.get("targets") != [candidate]
                    or [row["card_id"], digest] not in members[side]["scoring"]
                ):
                    continue
                if "role_source_manifest" not in view and (
                    action.get("tool") != "read_node"
                    or action.get("parameters") != dict(
                        card_id=row["card_id"], section="payload", cursor=None,
                    )
                ):
                    continue
                key = row["payload"]["family"]["primary_family"]
                old = supplied[side].get(key)
                if old is not None and canonical_json_bytes(old[0]) != canonical_json_bytes(row):
                    raise ValueError("D25 progress conflicting family source")
                entry = supplied[side].setdefault(key, (row, []))
                if index not in entry[1]:
                    entry[1].append(index)
    for family in sorted(set(supplied["left"]) | set(supplied["right"])):
        item = dict(_contrast(supplied["left"].get(family), supplied["right"].get(family)),
                    primary_family=family, publication_epoch_sha256=epoch)
        projected = [*result["family_contrasts"], item]
        if len(canonical_json_bytes(projected)) > _CONTRAST_BYTES:
            result["omitted_contrasts"].append(dict(
                primary_family=family, reason="capacity", left=item["left"], right=item["right"],
            ))
        else:
            result["family_contrasts"] = projected
    return result


def validate_replanning_context(context: dict[str, Any]) -> None:
    """Preserve existing source/sequence checks, validating the new move-only view.

    Source admission remains with the owner; this is a transport validator only.
    """
    from nurse_rcl.v2.live.ops_sre_d23_workbench import validate_investigation_workbench
    from nurse_rcl.v2.live.ops_sre_workbench import validate_workbench_context

    progress = context.get("investigation_progress")
    memory = context.get("public_working_memory")
    if type(progress) is not dict or (
        progress.get("schema_version") != PROGRESS_SCHEMA
        or progress.get("authority") != PROGRESS_AUTHORITY
        or progress.get("comparison_meaning")
        != "original_delivered_values_not_causal_contribution_or_new_evidence"
    ):
        raise ValueError("D25 progress identity differs")
    if type(memory) is not dict or set(memory) != {"recent", "question_history", "selection"}:
        raise ValueError("D25 selected memory shape differs")
    if memory["selection"] != dict(
        omitted_fields=["public_summary", "unresolved"],
        meaning="lossy_input_selection_original_conversation_retained",
        resolution="not_assessed_by_program",
    ) or any(
        "public_summary" in row or "unresolved" in row
        for row in [*memory["recent"], *memory["question_history"]]
    ):
        raise ValueError("D25 selected memory intent differs")
    legacy_memory = dict(
        recent=deepcopy(memory["recent"]),
        question_history=[dict(row, unresolved=[]) for row in memory["question_history"]],
    )
    legacy_work = {
        key: value for key, value in progress.items()
        if key not in {"family_contrasts", "omitted_contrasts", "comparison_meaning"}
    }
    legacy_work.update(
        schema_version="nurse.v2.sre-investigation-workbench/1",
        authority="ledger_and_frozen_delivery_metadata_not_diagnostic_conclusion",
    )
    legacy = dict(context, public_working_memory=legacy_memory,
                  investigation_workbench=legacy_work)
    validate_workbench_context(legacy)
    validate_investigation_workbench(legacy)
    index = {row["record_sha256"]: row for row in context["delivered_record_index"]["records"]}
    contrasts = progress.get("family_contrasts")
    omitted = progress.get("omitted_contrasts")
    if type(contrasts) is not list or type(omitted) is not list or (
        len(canonical_json_bytes(contrasts)) > _CONTRAST_BYTES
    ):
        raise ValueError("D25 contrast capacity differs")
    seen = set()
    for item in [*contrasts, *omitted]:
        family = item.get("primary_family")
        if type(family) is not str or not family or family in seen:
            raise ValueError("D25 contrast family differs")
        seen.add(family)
        for side in ("left", "right"):
            ref = item.get(side)
            if ref is None:
                continue
            row = index.get(ref.get("record_sha256"))
            if row is None or row["delivery_kind"] != "complete" or (
                row["card_id"] != ref.get("card_id")
                or row["primary_family"] != family
                or row["kind"] != "score_family"
                or type(ref.get("action_indices")) is not list
                or not ref["action_indices"]
                or not set(ref["action_indices"]) <= set(row["action_indices"])
            ):
                raise ValueError("D25 contrast source binding differs")
        if item in omitted:
            if set(item) != {"primary_family", "reason", "left", "right"} or (
                item["reason"] != "capacity"
            ):
                raise ValueError("D25 contrast omission differs")
            continue
        if set(item) != {
            "primary_family", "publication_epoch_sha256", "status", "left", "right",
            "fields", "omitted_fields",
        } or item["status"] not in {
            "not_supplied", "source_identity_differs", "numeric_unknown_or_selection_not_supplied",
            "complete_delivered_same_epoch",
        }:
            raise ValueError("D25 contrast shape differs")
        epochs = {progress["coverage"][side]["scoring"]["publication_epoch_sha256"]
                  for side in ("left", "right")}
        if epochs != {item["publication_epoch_sha256"]}:
            raise ValueError("D25 contrast epoch differs")
        fields = item["fields"]
        if item["status"] != "complete_delivered_same_epoch":
            if fields or item["omitted_fields"]:
                raise ValueError("D25 unsupplied contrast invents values")
            continue
        if item["left"] is None or item["right"] is None or (
            not item["left"]["source_refs"]
            or item["left"]["source_refs"] != item["right"]["source_refs"]
            or type(fields) is not dict or set(fields) != set(_FIELDS)
        ):
            raise ValueError("D25 complete contrast differs")
        for field in fields.values():
            if type(field) is not dict or set(field) != {"left", "right", "relation"}:
                raise ValueError("D25 contrast field differs")
            lvalue, rvalue = field["left"], field["right"]
            expected = "not_supplied" if lvalue is None or rvalue is None else (
                "same" if canonical_json_bytes(lvalue) == canonical_json_bytes(rvalue)
                else "different"
            )
            if field["relation"] != expected or (
                expected == "not_supplied" and (lvalue is not None or rvalue is not None)
            ) or len(canonical_json_bytes([lvalue, rvalue])) > _FIELD_BYTES:
                raise ValueError("D25 contrast field relation differs")
