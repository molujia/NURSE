"""Build the fixed paper V1 service topology from normal-window trace spans."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone

from pydantic import Field

from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.schemas import StrictFrozenModel

WINDOW_SECONDS = 480
_NANOSECONDS_PER_SECOND = 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TraceSpan(StrictFrozenModel):
    """Canonical trace fields needed to resolve one service-level parent link."""

    case_id: str = Field(min_length=1)
    timestamp_ns: int
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_span_id: str | None
    service: str = Field(min_length=1)


class TopologyEdge(StrictFrozenModel):
    """Aggregated callee-to-caller edge from resolved normal spans."""

    callee: str = Field(min_length=1)
    caller: str = Field(min_length=1)
    link_count: int = Field(ge=1)


class TopologyCoverage(StrictFrozenModel):
    """Counts proving what entered or was excluded from the fixed topology."""

    input_span_count: int = Field(ge=0)
    in_scope_span_count: int = Field(ge=0)
    excluded_cross_case_span_count: int = Field(ge=0)
    excluded_out_of_window_span_count: int = Field(ge=0)
    unique_trace_count: int = Field(ge=0)
    parent_reference_count: int = Field(ge=0)
    resolved_parent_reference_count: int = Field(ge=0)
    orphan_span_count: int = Field(ge=0)
    parent_link_coverage: float = Field(ge=0.0, le=1.0)


class NormalTopology(StrictFrozenModel):
    """Deterministically ordered service graph for one case's normal window."""

    case_id: str = Field(min_length=1)
    t0: int
    window_start: int
    window_end: int
    nodes: tuple[str, ...]
    edges: tuple[TopologyEdge, ...]
    coverage: TopologyCoverage


def parse_trace_rows(
    *,
    case_id: str,
    rows: Iterable[Mapping[str, object]],
) -> tuple[TraceSpan, ...]:
    """Parse canonical Parquet row fields without admitting unrelated columns."""

    normalized_case_id = _required_text(case_id, field="case_id")
    spans: list[TraceSpan] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        trace_id = _row_text(row, "trace_id")
        span_id = _row_text(row, "span_id")
        identity = (trace_id, span_id)
        if identity in identities:
            raise ValueError(f"duplicate span identity: {trace_id}/{span_id}")
        identities.add(identity)

        raw_parent = row.get("parent_span_id")
        if raw_parent is None:
            parent_span_id = None
        elif isinstance(raw_parent, str):
            parent_span_id = raw_parent.strip() or None
        else:
            raise ValueError("parent_span_id must be a string or null")
        spans.append(
            TraceSpan(
                case_id=normalized_case_id,
                timestamp_ns=_row_timestamp_ns(row),
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                service=canonicalize_service_name(_row_text(row, "service_name")),
            )
        )
    return tuple(
        sorted(
            spans,
            key=lambda span: (
                span.timestamp_ns,
                span.trace_id,
                span.span_id,
                span.service,
                span.parent_span_id or "",
            ),
        )
    )


def build_normal_topology(
    *,
    case_id: str,
    t0: int,
    spans: Iterable[TraceSpan],
) -> NormalTopology:
    """Build only same-case, normal-window, callee-to-caller service edges."""

    normalized_case_id = _required_text(case_id, field="case_id")
    all_spans = tuple(spans)
    window_start_ns = (t0 - WINDOW_SECONDS) * _NANOSECONDS_PER_SECOND
    window_end_ns = t0 * _NANOSECONDS_PER_SECOND
    case_spans = tuple(span for span in all_spans if span.case_id == normalized_case_id)
    scoped_spans = tuple(
        span
        for span in case_spans
        if window_start_ns <= span.timestamp_ns < window_end_ns
    )

    span_index: dict[tuple[str, str], TraceSpan] = {}
    for span in scoped_spans:
        identity = (span.trace_id, span.span_id)
        if identity in span_index:
            raise ValueError(f"duplicate in-scope span identity: {span.trace_id}/{span.span_id}")
        span_index[identity] = span

    edge_counts: Counter[tuple[str, str]] = Counter()
    parent_reference_count = 0
    resolved_parent_reference_count = 0
    orphan_span_count = 0
    for child in scoped_spans:
        if child.parent_span_id is None:
            continue
        parent_reference_count += 1
        parent = span_index.get((child.trace_id, child.parent_span_id))
        if parent is None:
            orphan_span_count += 1
            continue
        resolved_parent_reference_count += 1
        if child.service != parent.service:
            edge_counts[(child.service, parent.service)] += 1

    edges = tuple(
        TopologyEdge(callee=callee, caller=caller, link_count=link_count)
        for (callee, caller), link_count in sorted(edge_counts.items())
    )
    coverage_ratio = (
        resolved_parent_reference_count / parent_reference_count
        if parent_reference_count
        else 0.0
    )
    coverage = TopologyCoverage(
        input_span_count=len(all_spans),
        in_scope_span_count=len(scoped_spans),
        excluded_cross_case_span_count=len(all_spans) - len(case_spans),
        excluded_out_of_window_span_count=len(case_spans) - len(scoped_spans),
        unique_trace_count=len({span.trace_id for span in scoped_spans}),
        parent_reference_count=parent_reference_count,
        resolved_parent_reference_count=resolved_parent_reference_count,
        orphan_span_count=orphan_span_count,
        parent_link_coverage=coverage_ratio,
    )
    return NormalTopology(
        case_id=normalized_case_id,
        t0=t0,
        window_start=t0 - WINDOW_SECONDS,
        window_end=t0,
        nodes=tuple(sorted({span.service for span in scoped_spans})),
        edges=edges,
        coverage=coverage,
    )


def _row_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return _required_text(value, field=field)


def _required_text(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _row_timestamp_ns(row: Mapping[str, object]) -> int:
    value = row.get("time")
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("time must be a timezone-aware datetime")
    delta = value.astimezone(timezone.utc) - _EPOCH
    timestamp_ns = (
        (delta.days * 86_400 + delta.seconds) * _NANOSECONDS_PER_SECOND
        + delta.microseconds * 1_000
    )
    nanosecond = getattr(value, "nanosecond", 0)
    if isinstance(nanosecond, int):
        timestamp_ns += nanosecond
    return timestamp_ns
