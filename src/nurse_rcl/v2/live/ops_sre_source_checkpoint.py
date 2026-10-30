"""D16a disk-only source binding. No model-selected paths or evidence authority."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_sre_checkpoint import pack_navigation, unpack_navigation
from nurse_rcl.v2.live.ops_sre_score_cards import SCORE_CARD_VERSION, iter_score_lineage_cards
from nurse_rcl.v2.live.ops_sre_score_lineage import SreScoreLineage
from nurse_rcl.v2.live.ops_tools import card_id
from nurse_rcl.v2.observation_schemas import TimeInterval

STORAGE_ENCODING = "sre-checkpoint-score-source/1"
SOURCE_JSON_BYTES = 64 * 1024 * 1024
SOURCE_CARD_BYTES = 64 * 1024 * 1024
COMPONENT_BYTES = 64 * 1024 * 1024
COMBINED_CARD_BYTES = 128 * 1024 * 1024
LOGICAL_BYTES = 384 * 1024 * 1024
DESCRIPTOR_BYTES = 64 * 1024


def require_capacity(size: int, limit: int, boundary: str, *, lower: bool = False) -> None:
    if size > limit:
        raise ExchangeBoundaryError(
            "json_limit",
            boundary=boundary,
            observed=size,
            limit=limit,
            observed_is_lower_bound=lower,
        )


@dataclass(frozen=True)
class ScoreSource:
    descriptor_json: bytes
    cards: Mapping[str, OpsNodeEvidence]
    expanded_bytes: int

    @property
    def descriptor(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.descriptor_json))


def admit_source(projection: SreScoreLineage, *, interval: TimeInterval) -> ScoreSource:
    """Generate incrementally; failed admission does not publish partial cards."""
    cards: dict[str, OpsNodeEvidence] = {}
    manifest: dict[str, str] = {}
    size = 2
    for card in iter_score_lineage_cards(projection, interval=interval):
        key = card_id(card)
        if key in cards:
            raise ValueError("duplicate source card")
        row = canonical_json_bytes(dict(card_id=key, **card.model_dump(mode="json")))
        size += len(row) + bool(cards)
        require_capacity(size, SOURCE_CARD_BYTES, "checkpoint_source_cards_bytes", lower=True)
        cards[key] = card
        manifest[key] = sha256_hex(row)
    descriptor = dict(
        source_role="terminal_score_lineage",
        source_schema=projection.schema_version,
        projection_sha256=projection.projection_sha256,
        core_identity=projection.core_identity.model_dump(mode="json"),
        core_result_digest=projection.core_result_digest,
        interval=interval.model_dump(mode="json"),
        card_version=SCORE_CARD_VERSION,
        storage_encoding=STORAGE_ENCODING,
        card_count=len(cards),
        expanded_bytes=size,
        card_set_sha256=sha256_hex(canonical_json_bytes(sorted(manifest.items()))),
    )
    raw = canonical_json_bytes(descriptor)
    require_capacity(len(raw), DESCRIPTOR_BYTES, "checkpoint_descriptor_bytes")
    return ScoreSource(raw, MappingProxyType(cards), size)


def _array_size(rows: Any) -> int:
    # Incremental byte measurement, with no duplicate full body allocation.
    total = 2
    for count, row in enumerate(rows):
        total += len(canonical_json_bytes(row)) + bool(count)
    return total


def _check_regions(value: dict[str, Any], source: ScoreSource | None) -> dict[str, int]:
    indexed = value["indexed"]
    cards = indexed["cards"]
    source_ids = set(source.cards) if source else set()
    if source_ids & cards.keys() or any(
        c.get("tool_version") == SCORE_CARD_VERSION for c in cards.values()
    ):
        raise ValueError("source cards cannot occur in checkpoint delta")
    if source_ids and not source_ids <= set(indexed["core_cards"]):
        raise ValueError("source publication set differs")
    if value.get("score_lineage_sha256") != (
        source.descriptor["projection_sha256"] if source else None
    ):
        raise ValueError("source lineage binding differs")
    groups: dict[str, list[dict[str, Any]]] = {}
    for key, card in cards.items():
        groups.setdefault(card["tool_version"], []).append(dict(card_id=key, **card))
    sizes = dict(
        cards=sum(_array_size(rows) for rows in groups.values()),
        selection=sum(_array_size(s["selection"]["requests"]) for s in indexed["core_steps"]),
        directories=sum(_array_size(rows) for rows in indexed["index"]["rows"].values()),
        prefixes=_array_size(
            dict(prefix_sha256=k, **p) for k, p in value["inspection_prefixes"].items()
        ),
    )
    for name, size in sizes.items():
        require_capacity(size, COMPONENT_BYTES, "checkpoint_" + name + "_bytes")
    combined = sizes["cards"] + (source.expanded_bytes if source else 0)
    require_capacity(combined, COMBINED_CARD_BYTES, "checkpoint_combined_cards_bytes")
    remainder = dict(
        value,
        indexed=dict(
            indexed,
            cards={},
            core_steps=[
                dict(s, selection=dict(s["selection"], requests=[])) for s in indexed["core_steps"]
            ],
            index=dict(indexed["index"], rows={}),
        ),
        inspection_prefixes={},
    )
    sizes["remainder"] = len(
        canonical_json_bytes(
            dict(navigation=remainder, source=source.descriptor if source else None)
        )
    )
    require_capacity(sizes["remainder"], COMPONENT_BYTES, "checkpoint_remainder_bytes")
    require_capacity(
        sum(sizes.values()) + combined - sizes["cards"], LOGICAL_BYTES, "checkpoint_logical_bytes"
    )
    return sizes


def pack_source_navigation(value: dict[str, Any], *, source: ScoreSource | None) -> dict[str, Any]:
    require_capacity(
        len(source.descriptor_json) if source else 4,
        DESCRIPTOR_BYTES,
        "checkpoint_descriptor_bytes",
    )
    _check_regions(value, source)
    return dict(
        storage_encoding=STORAGE_ENCODING,
        source=source.descriptor if source else None,
        base=pack_navigation(value),
    )


def validate_source_envelope(value: dict[str, Any]) -> bytes:
    if set(value) != {"storage_encoding", "source", "base"} or (
        value["storage_encoding"] != STORAGE_ENCODING
    ):
        raise ValueError("source checkpoint encoding differs")
    descriptor = canonical_json_bytes(value["source"])
    require_capacity(len(descriptor), DESCRIPTOR_BYTES, "checkpoint_descriptor_bytes")
    if value["source"] is not None and (
        type(value["source"]) is not dict
        or value["source"].get("source_role") != "terminal_score_lineage"
        or value["source"].get("storage_encoding") != STORAGE_ENCODING
    ):
        raise ValueError("checkpoint source role differs")
    return descriptor


def unpack_source_navigation(
    value: dict[str, Any], *, source: ScoreSource | None
) -> dict[str, Any]:
    descriptor = validate_source_envelope(value)
    if descriptor != (source.descriptor_json if source else b"null"):
        raise ValueError("checkpoint source was not independently admitted")
    base = unpack_navigation(value["base"])
    _check_regions(base, source)
    if source:
        base["indexed"]["cards"].update(
            {k: c.model_dump(mode="json") for k, c in source.cards.items()}
        )
    return base


def check_state_capacity(
    state: dict[str, Any], *, file_limit: int, raw_size: int | None = None
) -> None:
    """The fixed partitions do not exempt actions, receipts or wrapper metadata."""
    require_capacity(
        len(canonical_json_bytes(state)) if raw_size is None else raw_size,
        file_limit,
        "checkpoint_file_bytes",
    )
    navigation = state["navigation"]
    base = navigation["base"]
    indexed = base["indexed"]
    remainder = dict(
        base,
        selection_rows={},
        inspection_prefixes={},
        indexed=dict(
            indexed,
            cards={},
            index=dict(indexed["index"], rows={}),
            core_steps=[
                dict(step, selection=dict(step["selection"], requests=[]))
                for step in indexed["core_steps"]
            ],
        ),
    )
    raw = canonical_json_bytes(dict(state, navigation=dict(navigation, base=remainder)))
    require_capacity(len(raw), COMPONENT_BYTES, "checkpoint_remainder_bytes")
