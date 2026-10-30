"""Small model views of owner-admitted metadata; no telemetry or scoring access.

The trusted investigation owner supplies registered rows and persists exported
state under its own lock. Neither publication nor restoration is a model tool.
This component does not prove source provenance or authorize a measurement.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json

INDEX_VERSION = "nurse.v2.ops-index/2"
PAGE_BYTES = 3072  # Leave room for a separately checked 4 KiB tool-view envelope.
PAGE_ITEMS = 16
_ID = {"candidates": "candidate_service_key", "signals": "request_key",
       "nodes": "card_id", "onsets": "onset_id"}
_FIELDS = {
    "candidates": {"candidate_service_key"},
    "signals": {"request_key", "candidate_service_key", "observation_id", "unit", "modality",
                "signal_family", "metric_name", "bound", "coverage", "formally_eligible"},
    # Publication role can change after a local card is registered. It belongs
    # to the read receipt, not an immutable row in a stable page chain.
    "nodes": {"card_id", "node", "candidate_service_key", "availability", "truth_value"},
    "onsets": {"onset_id", "candidate_service_key", "bounds_ns"},
}
_FILTERS = {
    "candidates": {"candidate_service_key"},
    "signals": {"candidate_service_key", "modality", "signal_family"},
    "nodes": {"candidate_service_key", "node"},
    "onsets": {"candidate_service_key"},
}
_NODES = {"boundary", "symptoms", "dependency", "timing", "mechanisms", "scoring"}
_MODALITIES = {"metrics", "traces", "logs"}


def index_parameter_contract() -> dict[str, Any]:
    """Describe the actual allowlists without exposing the full case directory."""
    return dict(
        kind=list(_ID),
        filters_by_kind={kind: sorted(names) for kind, names in _FILTERS.items()},
        filter_values=dict(
            node=sorted(_NODES), modality=sorted(_MODALITIES),
            candidate_service_key="exact registered ID from candidate_entry or candidates page",
            signal_family="exact registered signal_family from a signals page; omit if unknown",
        ),
        filters="object of optional exact-match filters; {} means no filter; no other keys",
        cursor="null for first page; otherwise copy issued next_cursor with identical kind/filters",
        example=dict(kind="nodes", filters={}, cursor=None),
        meaning="directory only, not health evidence; each call consumes one action",
    )


def _pin(value: object) -> str:
    if type(value) is not str or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("index identity requires a SHA256 pin")
    return value


def _identifier(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= 512


def _copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


class OpsIndex:
    """Owner-side immutable rows, stable page chains and issued cursor bindings."""

    def __init__(
        self, *, case_snapshot_id: str, catalog_version: str,
        rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.case_snapshot_id = _pin(case_snapshot_id)
        self.catalog_version = _pin(catalog_version)
        if type(rows) is not dict or set(rows) != set(_ID):
            raise ValueError("index requires the four registered directories")
        self._rows: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in _ID}
        self._prefixes: dict[str, dict[str, Any]] = {}
        self._cursors: dict[str, dict[str, Any]] = {}
        for kind in _ID:  # Candidates are admitted before referencing directories.
            self._rows[kind] = self._validate_rows(kind, rows[kind])

    def _validate_rows(self, kind: str, rows: object) -> dict[str, dict[str, Any]]:
        if type(rows) not in (list, tuple):
            raise ValueError("index rows must be a finite metadata sequence")
        result: dict[str, dict[str, Any]] = {}
        for row in cast(list[object] | tuple[object, ...], rows):
            if type(row) is not dict or set(row) != _FIELDS[kind]:
                raise ValueError("index row fields are not registered")
            detached = json.loads(encode_json(row, ExchangeLimits()))
            key = detached[_ID[kind]]
            if not _identifier(key) or key in result:
                raise ValueError("index row ID is invalid or duplicated")
            candidate = detached["candidate_service_key"]
            if kind != "candidates" and candidate is not None and (
                type(candidate) is not str or candidate not in self._rows["candidates"]
            ):
                raise ValueError("index row candidate is outside the catalog")
            if kind == "signals" and (
                not _identifier(detached["observation_id"])
                or type(detached["modality"]) is not str
                or detached["modality"] not in _MODALITIES
                or not _identifier(detached["signal_family"])
                or (detached["unit"] is not None and type(detached["unit"]) is not str)
                or (detached["metric_name"] is not None
                    and type(detached["metric_name"]) is not str)
                or type(detached["bound"]) is not bool
                or detached["bound"] != (candidate is not None)
                or type(detached["formally_eligible"]) is not bool
                or type(detached["coverage"]) not in (int, float)
                or not 0 <= detached["coverage"] <= 1
            ):
                raise ValueError("signal descriptor has invalid metadata")
            if kind == "nodes" and (
                type(detached["node"]) is not str or detached["node"] not in _NODES
                or type(detached["availability"]) is not str
                or detached["availability"] not in {"available", "partial", "unavailable"}
                or type(detached["truth_value"]) is not str
                or detached["truth_value"] not in {"true", "false", "unknown"}
            ):
                raise ValueError("node descriptor has invalid metadata")
            if kind == "onsets":
                bounds = detached["bounds_ns"]
                if bounds is not None and (
                    type(bounds) is not list or len(bounds) != 2
                    or any(type(n) is not int for n in bounds) or bounds[0] > bounds[1]
                ):
                    raise ValueError("onset descriptor has invalid interval")
            result[key] = detached
        return result

    def publish(self, *, kind: str, rows: object) -> None:
        """Append metadata of actual owner-published cards/onsets, never measurements."""
        if type(kind) is not str or kind not in {"nodes", "onsets"}:
            raise ValueError("only published node/onset directories can grow")
        accepted = self._validate_rows(kind, rows)
        current = self._rows[kind]
        if any(key in current and current[key] != row for key, row in accepted.items()):
            raise ValueError("published metadata cannot change an existing ID")
        current.update(accepted)

    def _filters(self, kind: object, filters: object) -> dict[str, str]:
        if type(kind) is not str or kind not in _ID or type(filters) is not dict:
            raise ValueError("index kind or filters are not registered")
        if not set(filters) <= _FILTERS[kind]:
            raise ValueError("index filter name is not registered")
        for name, value in filters.items():
            if not _identifier(value):
                raise ValueError("index filters require registered text IDs")
            domain = {
                "candidate_service_key": self._rows["candidates"],
                "modality": _MODALITIES, "node": _NODES,
                "signal_family": {row["signal_family"] for row in self._rows["signals"].values()},
            }[name]
            if value not in domain:
                raise ValueError("index filter value is not registered")
        return dict(sorted(filters.items()))

    def _digest(self, value: object) -> str:
        return sha256_hex(canonical_json_bytes(dict(
            schema_version=INDEX_VERSION, case_snapshot_id=self.case_snapshot_id,
            catalog_version=self.catalog_version, value=value,
        )))

    def _matches(self, prefix: dict[str, Any], filters: dict[str, str]) -> list[str]:
        rows = self._rows[prefix["kind"]]
        return [key for key in prefix["ids"]
                if all(rows[key].get(name) == value for name, value in filters.items())]

    def browse_index(
        self, *, kind: str, filters: dict[str, str], cursor: str | None,
    ) -> dict[str, Any]:
        selected = self._filters(kind, filters)
        if cursor is None:
            prefix = dict(kind=kind, ids=sorted(self._rows[kind]))
            prefix_pin = self._digest(prefix)
            self._prefixes[prefix_pin] = prefix
            start = 0
        else:
            if type(cursor) is not str or cursor not in self._cursors:
                raise ValueError("index cursor was not issued in this snapshot/catalog")
            binding = self._cursors[cursor]
            if binding["kind"] != kind or binding["filters"] != selected:
                raise ValueError("index cursor kind or filters differ")
            prefix_pin, start = binding["prefix"], binding["offset"]
            prefix = self._prefixes[prefix_pin]
        keys = self._matches(prefix, selected)

        def assemble(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
            offset = start + len(items)
            following = dict(kind=kind, prefix=prefix_pin, filters=selected, offset=offset)
            more = offset < len(keys)
            return dict(
                schema_version=INDEX_VERSION, case_snapshot_id=self.case_snapshot_id,
                catalog_version=self.catalog_version, kind=kind, filters=selected,
                publication_prefix=prefix_pin, total=len(keys), count=len(items), items=items,
                has_more=more,
                next_cursor="ops_cursor_" + self._digest(following) if more else None,
                limitations=["catalog_not_health_evidence"],
            ), following

        items: list[dict[str, Any]] = []
        for key in keys[start:start + PAGE_ITEMS]:
            row = self._rows[kind][key]
            proposed, _ = assemble([*items, row])
            if len(canonical_json_bytes(proposed)) > PAGE_BYTES:
                if items:
                    break
                row = {_ID[kind]: key, "representation": "oversize",
                       "omitted_detail": sorted(set(row) - {_ID[kind]})}
                proposed, _ = assemble([row])
                if len(canonical_json_bytes(proposed)) > PAGE_BYTES:
                    raise ValueError("index required envelope exceeds context capacity")
            items.append(row)
        page, following = assemble(items)
        encode_json(page, ExchangeLimits(max_bytes=PAGE_BYTES))
        if page["next_cursor"] is not None:
            self._cursors[page["next_cursor"]] = following
        return dict(_copy(page))

    def export_state(self) -> dict[str, Any]:
        """Metadata only; the owner must durably save it before releasing a page."""
        body = dict(
            schema_version=INDEX_VERSION, case_snapshot_id=self.case_snapshot_id,
            catalog_version=self.catalog_version,
            rows={kind: [rows[key] for key in sorted(rows)] for kind, rows in self._rows.items()},
            prefixes=self._prefixes, cursors=self._cursors,
        )
        return dict(_copy(dict(**body, state_sha256=sha256_hex(canonical_json_bytes(body)))))

    @classmethod
    def restore(
        cls, state: dict[str, Any], *, case_snapshot_id: str, catalog_version: str,
    ) -> OpsIndex:
        """Restore trusted persisted metadata, not an arbitrary model-supplied cursor."""
        if type(state) is not dict or set(state) != {
            "schema_version", "case_snapshot_id", "catalog_version", "rows",
            "prefixes", "cursors", "state_sha256",
        }:
            raise ValueError("index state fields differ")
        body = {k: v for k, v in state.items() if k != "state_sha256"}
        if state["schema_version"] != INDEX_VERSION or (
            state["case_snapshot_id"], state["catalog_version"]
        ) != (case_snapshot_id, catalog_version) or (
            state["state_sha256"] != sha256_hex(canonical_json_bytes(body))
        ):
            raise ValueError("index state identity or content differs")
        result = cls(case_snapshot_id=case_snapshot_id, catalog_version=catalog_version,
                     rows=state["rows"])
        if type(state["prefixes"]) is not dict or type(state["cursors"]) is not dict:
            raise ValueError("index state registry differs")
        for pin, prefix in state["prefixes"].items():
            if type(prefix) is not dict or set(prefix) != {"kind", "ids"} or (
                type(prefix["kind"]) is not str or prefix["kind"] not in _ID
                or type(prefix["ids"]) is not list
                or any(not _identifier(key) for key in prefix["ids"])
                or prefix["ids"] != sorted(set(prefix["ids"]))
                or not set(prefix["ids"]) <= set(result._rows[prefix["kind"]])
                or pin != result._digest(prefix)
            ):
                raise ValueError("index state prefix differs")
            result._prefixes[pin] = _copy(prefix)
        for token, binding in state["cursors"].items():
            if type(binding) is not dict or set(binding) != {
                "kind", "prefix", "filters", "offset",
            }:
                raise ValueError("index state cursor fields differ")
            filters = result._filters(binding["kind"], binding["filters"])
            if type(binding["prefix"]) is not str or binding["prefix"] not in result._prefixes:
                raise ValueError("index state cursor prefix is unknown")
            prefix = result._prefixes[binding["prefix"]]
            if prefix["kind"] != binding["kind"] or type(binding["offset"]) is not int or (
                not 0 < binding["offset"] < len(result._matches(prefix, filters))
            ) or token != "ops_cursor_" + result._digest(binding):
                raise ValueError("index state cursor binding differs")
            result._cursors[token] = _copy(binding)
        return result
