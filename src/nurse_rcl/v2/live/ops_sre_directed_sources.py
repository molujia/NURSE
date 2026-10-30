"""Source navigation and comparability, not a new observation or diagnostic rank."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def directed_source_binding(
    snapshot: str,
    parameters: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            dict(
                policy="sre-directed-source-binding/1",
                snapshot=snapshot,
                parameters=parameters,
                source_rows=rows,
            )
        )
    )


def directed_source_navigation(
    records: list[dict[str, Any]],
    scope: list[str],
    *,
    limit: int = 16,
    observable: bool = False,
) -> dict[str, Any]:
    """Round-robin actual sources; all candidates retain their ordinary inspection entry."""
    if type(limit) is not int or not 1 <= limit <= 16 or len(scope) != len(set(scope)):
        raise ValueError("directed navigation bound differs")
    queues: dict[str, list[dict[str, Any]]] = {}
    for candidate in scope:
        families: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            if row["node"] == "symptoms" and row["targets"] == [candidate]:
                families.setdefault(str(row["payload"].get("signal_family", "")), []).append(row)
        for bucket in families.values():
            bucket.sort(
                key=lambda row: (
                    str(row["payload"].get("metric_name", "")),
                    row["card_id"],
                )
            )
        queues[candidate] = [
            families[family][index]
            for index in range(max((len(bucket) for bucket in families.values()), default=0))
            for family in sorted(families)
            if index < len(families[family])
        ]
    chosen: list[dict[str, Any]] = []
    for index in range(max((len(rows) for rows in queues.values()), default=0)):
        for candidate in scope:
            if index < len(queues[candidate]) and len(chosen) < limit:
                chosen.append(queues[candidate][index])
        if len(chosen) == limit:
            break
    entries = []
    for row in chosen:
        body = row["payload"]
        entries.append(
            dict(
                card_id=row["card_id"],
                record_sha256=sha256_hex(canonical_json_bytes(row)),
                node=row["node"],
                targets=list(row["targets"]),
                source_role=row["source_role"],
                signal_family=body.get("signal_family"),
                metric_name=body.get("metric_name"),
                request_key=body.get("request_key") or row["parameters"].get("request_key"),
                source_refs=list(row["source_refs"]),
                representation="navigation_reference_not_delivered_detail",
                entry=dict(
                    tool="inspect_sources",
                    parameters=dict(
                        card_ids=[row["card_id"]],
                        candidate_service_keys=list(row["targets"]),
                    ),
                ),
                complete_entry=dict(
                    tool="read_node",
                    parameters=dict(
                        card_id=row["card_id"],
                        section="payload",
                        cursor=None,
                    ),
                ),
            )
        )
        if observable:
            entries[-1].update(record_kind=body.get("kind"), record_version=row["tool_version"])
    total = sum(len(rows) for rows in queues.values())
    return dict(
        schema_version=(
            "nurse.v2.sre-directed-source-navigation/2"
            if observable
            else "nurse.v2.sre-directed-source-navigation/1"
        ),
        authority="navigation_not_delivered_evidence",
        entries=entries,
        total=total,
        omitted=total - len(entries),
        selection="candidate_and_family_round_robin_then_metric_card/1",
        limitations=[
            "order_not_diagnostic_priority",
            "unread_not_absent_or_normal",
            "source_identity_and_summary_fields_not_independent_evidence",
        ],
    )


def directed_source_relation(
    rows: list[dict[str, Any]],
    scope: list[str],
) -> dict[str, Any]:
    """Only actual metric/family/windows can license a matched-source descriptor."""
    missing = [
        candidate for candidate in scope if not any(row["targets"] == [candidate] for row in rows)
    ]
    status = "single_candidate_sources"
    if missing:
        status = "missing_candidate_source"
    elif len(scope) == 2:
        left, right = (next(row for row in rows if row["targets"] == [c]) for c in scope)
        fields = ("metric_name", "signal_family", "baseline_window", "fault_window")
        matched = all(
            left["payload"].get(field) is not None
            and right["payload"].get(field) is not None
            and canonical_json_bytes(left["payload"][field])
            == canonical_json_bytes(right["payload"][field])
            for field in fields
        )
        status = "same_metric_family_windows" if matched else "metric_family_or_window_mismatch"
    return dict(
        authority="source_relationship_not_new_evidence",
        status=status,
        missing_candidate_service_keys=missing,
        limitations=[
            "matching_is_not_independence_or_causality",
            "summary_cannot_prove_episode_or_normality",
        ],
    )
