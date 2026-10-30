"""Exact finite evidence directory; no observations, paths or execution authority."""

from __future__ import annotations

import json
from typing import Any, TypeVar, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

CATALOG_SCHEMA = "nurse.v2.sre-evidence-catalog/1"
FILTERS = {"candidate_service_keys", "metric_name", "signal_family", "record_kind"}
_T = TypeVar("_T")


def _copy(value: _T) -> _T:
    return cast(_T, json.loads(canonical_json_bytes(value)))


class EvidenceCatalog:
    """Reconstructed from the owner's immutable registered semantic rows only."""

    def __init__(
        self,
        *,
        snapshot: str,
        rows: list[dict[str, Any]],
        requests: dict[str, Any],
        candidates: list[str] | None = None,
    ):
        self.snapshot = snapshot
        self._rows: dict[str, dict[str, Any]] = {}
        self._semantic: dict[str, dict[str, Any]] = {}
        self._prefixes: dict[str, dict[str, Any]] = {}
        self._cursors: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not row.get("targets") or len(row["targets"]) > 2:
                continue
            body = row["payload"]
            attribution = row.get("source_attribution", {})
            fields = attribution.get("fields", {})

            def field(
                name: str, body: dict[str, Any] = body, fields: dict[str, Any] = fields
            ) -> Any:
                if name in body:
                    return body[name]
                cell = fields.get(name, {})
                return cell.get("value") if cell.get("state") == "supplied_value" else None

            request = (
                body.get("request_key")
                or row["parameters"].get("request_key")
                or attribution.get("binding", {}).get("request_key")
            )
            actions = [
                dict(
                    tool="read_node",
                    parameters=dict(card_id=row["card_id"], section="payload", cursor=None),
                )
            ]
            if row["node"] == "symptoms" and len(row["targets"]) == 1:
                actions.append(
                    dict(
                        tool="inspect_sources",
                        parameters=dict(
                            card_ids=[row["card_id"]], candidate_service_keys=row["targets"]
                        ),
                    )
                )
            if request in requests:
                actions.append(dict(tool="measure_signal", parameters=dict(request_key=request)))
            item = dict(
                card_id=row["card_id"],
                node=row["node"],
                targets=row["targets"],
                source_role=row["source_role"],
                source_refs=row["source_refs"],
                record_sha256=sha256_hex(canonical_json_bytes(row)),
                metric_name=field("metric_name"),
                signal_family=field("signal_family"),
                record_kind=body.get("kind"),
                record_version=row["tool_version"],
                request_key=request,
                actions=actions,
                representation="navigation_reference_not_delivered_detail",
            )
            key = row["card_id"]
            if key in self._rows and self._rows[key] != _copy(item):
                raise ValueError("evidence catalog duplicate source differs")
            self._rows[key] = _copy(item)
            self._semantic[key] = _copy(row)
        self._candidates = (
            set(candidates)
            if candidates is not None
            else {c for row in self._rows.values() for c in row["targets"]}
        )

    def _digest(self, value: Any) -> str:
        return sha256_hex(
            canonical_json_bytes(
                dict(schema_version=CATALOG_SCHEMA, snapshot=self.snapshot, value=value)
            )
        )

    def filters(self, value: object) -> dict[str, Any]:
        if type(value) is not dict or not set(value) <= FILTERS:
            raise ValueError("evidence filters are not registered")
        scope = value.get("candidate_service_keys")
        if (
            type(scope) is not list
            or not 1 <= len(scope) <= 2
            or any(type(c) is not str or c not in self._candidates for c in scope)
            or len(scope) != len(set(scope))
        ):
            raise ValueError("evidence scope requires one or two registered candidates")
        if any(
            type(v) is not str or not v or len(v) > 512
            for k, v in value.items()
            if k != "candidate_service_keys"
        ):
            raise ValueError("evidence filters require exact text")
        return _copy(dict(sorted(value.items())))

    def binding(self, *, filters: object, cursor: object) -> str:
        selected = self.filters(filters)
        if cursor is None:
            source = self._rows
        elif type(cursor) is str and cursor in self._cursors:
            bound = self._cursors[cursor]
            if bound["filters"] != selected:
                raise ValueError("evidence cursor filters differ")
            source = self._prefixes[bound["prefix"]]
        else:
            raise ValueError("evidence cursor was not issued")
        return self._digest(dict(filters=selected, cursor=cursor, source=source))

    def browse(self, *, filters: object, cursor: str | None) -> dict[str, Any]:
        selected = self.filters(filters)
        if cursor is None:
            prefix = dict(rows=[self._rows[k] for k in sorted(self._rows)], filters=selected)
            pin, start = self._digest(prefix), 0
            self._prefixes[pin] = prefix
        else:
            self.binding(filters=selected, cursor=cursor)
            binding = self._cursors[cursor]
            pin, start = binding["prefix"], binding["offset"]
            prefix = self._prefixes[pin]
        rows = self._matches(prefix)
        items = _copy(self._page(rows, start))
        self._add_pair(items, selected)
        atoms = sum(len(row["actions"]) for row in items)
        following = start + len(items)
        binding = dict(prefix=pin, filters=selected, offset=following, previous_offset=start)
        token = "sre_evidence_cursor_" + self._digest(binding) if following < len(rows) else None
        if token:
            self._cursors[token] = binding
        return _copy(
            dict(
                schema_version=CATALOG_SCHEMA,
                kind="evidence",
                filters=selected,
                publication_prefix=pin,
                authority="navigation_not_delivered_evidence",
                total=len(rows),
                count=len(items),
                items=items,
                action_atom_count=atoms,
                has_more=bool(token),
                next_cursor=token,
                limitations=[
                    "directory_not_delivered_evidence",
                    "navigation_costs_one_action",
                    "explicit_read_inspect_or_measure_requires_another_action",
                ],
            )
        )

    @staticmethod
    def _matches(prefix: dict[str, Any]) -> list[dict[str, Any]]:
        selected = prefix["filters"]
        rows = [
            row
            for row in prefix["rows"]
            if set(row["targets"]) <= set(selected["candidate_service_keys"])
            and all(row[k] == v for k, v in selected.items() if k != "candidate_service_keys")
        ]
        if len(selected["candidate_service_keys"]) != 2:
            return rows
        queues = [
            [row for row in rows if row["targets"] == [candidate]]
            for candidate in selected["candidate_service_keys"]
        ]
        return [
            queue[i]
            for i in range(max(map(len, queues), default=0))
            for queue in queues
            if i < len(queue)
        ] + [row for row in rows if len(row["targets"]) != 1]

    @staticmethod
    def _page(rows: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        atoms = 0
        # Reserve one concrete atom for a comparison of two actual page cards.
        limit = (
            15
            if sum(row["node"] == "symptoms" and len(row["targets"]) == 1 for row in rows[start:])
            >= 2
            else 16
        )
        for row in rows[start:]:
            cost = len(row["actions"])
            if len(items) == 16 or atoms + cost > limit:
                break
            items.append(row)
            atoms += cost
        return items

    def _add_pair(self, items: list[dict[str, Any]], filters: dict[str, Any]) -> None:
        available = [row for row in items if row["node"] == "symptoms" and len(row["targets"]) == 1]
        if len(available) < 2:
            return
        left = available[0]
        right = next(
            (row for row in available[1:] if row["targets"] != left["targets"]), available[1]
        )
        rows = [left, right]
        scope = [
            candidate
            for candidate in filters["candidate_service_keys"]
            if any(row["targets"] == [candidate] for row in rows)
        ]
        params = dict(card_ids=[row["card_id"] for row in rows], candidate_service_keys=scope)
        from nurse_rcl.v2.live.ops_sre_focused_sources import focused_source_binding

        binding = focused_source_binding(
            self.snapshot, params, [self._semantic[row["card_id"]] for row in rows]
        )
        shared = {
            field: left[field] if left[field] == right[field] else None
            for field in ("metric_name", "signal_family", "record_kind", "record_version")
        }
        left["actions"].append(
            dict(
                tool="inspect_sources",
                parameters=params,
                binding_sha256=binding,
                source_context=dict(
                    node="symptoms",
                    targets=scope,
                    card_ids=params["card_ids"],
                    source_role="selected_sources",
                    source_refs=sorted({ref for row in rows for ref in row["source_refs"]}),
                    representation="navigation_reference_not_delivered_detail",
                    **shared,
                ),
            )
        )

    def export_state(self) -> dict[str, Any]:
        body = dict(
            schema_version=CATALOG_SCHEMA,
            snapshot=self.snapshot,
            rows=[self._rows[k] for k in sorted(self._rows)],
            prefixes=self._prefixes,
            cursors=self._cursors,
        )
        return _copy(dict(body, state_sha256=sha256_hex(canonical_json_bytes(body))))

    def restore_state(self, state: dict[str, Any]) -> None:
        if type(state) is not dict or set(state) != {
            "schema_version",
            "snapshot",
            "rows",
            "prefixes",
            "cursors",
            "state_sha256",
        }:
            raise ValueError("evidence catalog state fields differ")
        body = {k: v for k, v in state.items() if k != "state_sha256"}
        if (
            state["schema_version"] != CATALOG_SCHEMA
            or state["snapshot"] != self.snapshot
            or state["state_sha256"] != sha256_hex(canonical_json_bytes(body))
            or state["rows"] != [self._rows[k] for k in sorted(self._rows)]
        ):
            raise ValueError("evidence catalog state registry differs")
        for pin, prefix in state["prefixes"].items():
            if (
                type(prefix) is not dict
                or set(prefix) != {"rows", "filters"}
                or self._digest(prefix) != pin
                or self.filters(prefix["filters"]) != prefix["filters"]
                or any(self._rows.get(r["card_id"]) != r for r in prefix["rows"])
                or [r["card_id"] for r in prefix["rows"]]
                != sorted({r["card_id"] for r in prefix["rows"]})
            ):
                raise ValueError("evidence catalog prefix differs")
        for token, binding in state["cursors"].items():
            if (
                type(binding) is not dict
                or set(binding) != {"prefix", "filters", "offset", "previous_offset"}
                or binding["prefix"] not in state["prefixes"]
                or binding["filters"] != state["prefixes"][binding["prefix"]]["filters"]
                or type(binding["offset"]) is not int
                or binding["offset"] <= 0
                or type(binding["previous_offset"]) is not int
                or not 0 <= binding["previous_offset"] < binding["offset"]
                or token != "sre_evidence_cursor_" + self._digest(binding)
            ):
                raise ValueError("evidence catalog cursor differs")
            rows = self._matches(state["prefixes"][binding["prefix"]])
            start = binding["previous_offset"]
            if (
                binding["offset"] >= len(rows)
                or start + len(self._page(rows, start)) != binding["offset"]
                or (
                    start
                    and not any(
                        p["prefix"] == binding["prefix"] and p["offset"] == start
                        for p in state["cursors"].values()
                    )
                )
            ):
                raise ValueError("evidence catalog cursor page boundary differs")
        self._prefixes, self._cursors = _copy(state["prefixes"]), _copy(state["cursors"])
