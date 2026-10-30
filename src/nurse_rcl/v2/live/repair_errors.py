"""Typed error observations over observed records, with honest denominators.

These private input/result objects are not an outbound model projection.
Source-record replay is necessary but actual file/adapter binding belongs to
the live source boundary. Neither status fractions nor self-consistent hashes
establish population sampling coverage or a causal service failure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import TimeInterval

ERROR_OBSERVATION_SCHEMA_VERSION: Literal["nurse.v2.live-repair-errors/1"] = (
    "nurse.v2.live-repair-errors/1"
)
Outcome = Literal["error", "non_error", "unknown", "conflict", "excluded"]
Status = Literal["error", "non_error", "unknown"]
Kind = Literal["internal", "server", "client", "producer", "consumer", "unknown", "log"]


class _SourceRecord(StrictFrozenModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ref: str = Field(min_length=1)
    entity_key: StrictStr | None
    timestamp_unix_ns: int = Field(ge=0, strict=True)


class TraceErrorRecord(_SourceRecord):
    record_kind: Literal["trace"] = "trace"
    duration_ns: StrictInt | StrictFloat | None
    trace_id: StrictStr | None
    span_id: StrictStr | None
    span_kind: StrictStr | StrictInt | None
    status_code: StrictStr | StrictInt | None
    http_status_code: StrictStr | StrictInt | StrictFloat | None


class LogErrorRecord(_SourceRecord):
    record_kind: Literal["log"] = "log"
    severity: StrictStr | None


Record = TraceErrorRecord | LogErrorRecord


class ErrorMember(StrictFrozenModel):
    member_id: str
    entity_key: str | None
    span_kind: Kind
    source_refs: tuple[str, ...]
    outcome: Outcome
    otel_outcome: Status
    http_outcome: Status
    reasons: tuple[str, ...]


class ErrorGroup(StrictFrozenModel):
    entity_key: str | None
    span_kind: Kind
    statistic: Literal["observed_span_status_fraction", "observed_log_severity_fraction"]
    member_ids: tuple[str, ...]
    evidence_group_id: str
    score_combination_policy: Literal["single_group_not_sum_of_derived_fields"] = (
        "single_group_not_sum_of_derived_fields"
    )
    observed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    non_error_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    known_denominator: int = Field(ge=0)
    error_fraction: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    known_status_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


_STATUS_NAMES: dict[str, Status] = {"error": "error", "ok": "non_error"}
_KINDS: dict[str, Kind] = {
    "internal": "internal",
    "server": "server",
    "client": "client",
    "producer": "producer",
    "consumer": "consumer",
}
_STATUS_NUMBERS: dict[int, Status] = {1: "non_error", 2: "error"}
_KIND_NUMBERS: dict[int, Kind] = {
    1: "internal",
    2: "server",
    3: "client",
    4: "producer",
    5: "consumer",
}
_SEVERITIES = {"TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"}
_ERROR_SEVERITIES = {"ERROR", "FATAL", "CRITICAL"}
ERROR_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": "nurse-v2-live-repair-errors/1",
            "otel_status": {"Ok": "non_error", "Error": "error", "Unset_or_missing": "unknown"},
            "otel_numeric": {"0": "unknown", "1": "non_error", "2": "error"},
            "http": "integral_200_399_non_error_400_599_protocol_error_other_unknown",
            "status_disagreement": "retain_both_and_exclude_from_known_denominator",
            "kind": _KIND_NUMBERS,
            "trace_identity": "trace_id_plus_span_id_not_entity_or_kind",
            "duplicate_conflict": "all_variants_retained_no_vote_or_preference",
            "log_identity": "source_file_hash_plus_row_reference_not_message_content",
            "log_error_severities": sorted(_ERROR_SEVERITIES),
            "log_known_severities": sorted(_SEVERITIES),
            "denominator": "known_nonconflicted_unique_observed_records_of_one_entity_and_kind",
            "aiops25_trace": "preprocessor_materialized_placeholder_not_measured_status",
            "aiops25_logs": "preprocessor_message_heuristic_not_collector_severity",
            "coverage": "unknown_no_request_population_claim",
            "score_combination": "single_source_group_not_sum_of_counts_and_fractions",
        }
    )
)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _kind(value: str | int | None) -> Kind:
    if type(value) is int:
        return _KIND_NUMBERS.get(value, "unknown")
    return _KINDS.get(value.strip().lower(), "unknown") if isinstance(value, str) else "unknown"


def _http_status(value: str | int | float | None) -> Status:
    if isinstance(value, str):
        if not value.isascii() or not value.isdigit():
            return "unknown"
        value = int(value) if len(value) <= 3 else None
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "unknown"
    if not 200 <= value <= 599 or (isinstance(value, float) and not value.is_integer()):
        return "unknown"
    if 200 <= value <= 399:
        return "non_error"
    return "error" if 400 <= value <= 599 else "unknown"


def _trace_outcome(
    record: TraceErrorRecord, dataset: str, interval: TimeInterval
) -> tuple[Outcome, Status, Status, tuple[str, ...]]:
    if any(
        not value or value.strip().lower() in {"nan", "none", "null"}
        for value in (record.trace_id, record.span_id)
    ):
        return "excluded", "unknown", "unknown", ("missing_span_identity",)
    duration = record.duration_ns
    if (
        duration is None
        or duration < 0
        or (isinstance(duration, float) and not math.isfinite(duration))
    ):
        return "excluded", "unknown", "unknown", ("invalid_or_unknown_span_duration",)
    if record.timestamp_unix_ns + duration > interval.end_unix_ns:
        return "excluded", "unknown", "unknown", ("unfinished_at_decision",)
    if dataset == "aiops25":
        return "unknown", "unknown", "unknown", ("preprocessor_status_placeholder",)
    token = record.status_code
    otel: Status = (
        _STATUS_NUMBERS.get(token, "unknown")
        if type(token) is int
        else _STATUS_NAMES.get(token.strip().lower(), "unknown")
        if isinstance(token, str)
        else "unknown"
    )
    http = _http_status(record.http_status_code)
    known = {value for value in (otel, http) if value != "unknown"}
    if len(known) == 2:
        return "conflict", otel, http, ("status_sources_disagree",)
    if known:
        return ("error" if "error" in known else "non_error"), otel, http, ()
    return "unknown", otel, http, ("no_decisive_status",)


def _record_identity(record: Record) -> str:
    if isinstance(record, TraceErrorRecord) and record.trace_id and record.span_id:
        return "span_" + _digest([record.trace_id, record.span_id])
    return "row_" + _digest([record.record_kind, record.source_sha256, record.source_ref])


def _derive(
    records: tuple[Record, ...], *, dataset: str, interval: TimeInterval, modality: str
) -> dict[str, object]:
    if dataset not in {"rcabench", "aiops25"}:
        raise ValueError("unregistered error-source dataset")
    if any(record.record_kind != modality for record in records):
        raise ValueError("mixed or incompatible error source record types")
    indexed: dict[str, list[Record]] = defaultdict(list)
    source_slots: dict[tuple[str, str], bytes] = {}
    for record in records:
        if not interval.start_unix_ns <= record.timestamp_unix_ns < interval.end_unix_ns:
            raise ValueError("error source record is outside its window")
        slot = (record.source_sha256, record.source_ref)
        payload = canonical_json_bytes(record.model_dump(mode="json"))
        if slot in source_slots and source_slots[slot] != payload:
            raise ValueError("conflicting payloads for the same source row")
        source_slots[slot] = payload
        indexed[_record_identity(record)].append(record)
    members = []
    for member_id, variants in sorted(indexed.items()):
        source_refs = tuple(sorted({f"{v.source_sha256}:{v.source_ref}" for v in variants}))
        bodies = {
            canonical_json_bytes(v.model_dump(mode="json", exclude={"source_ref", "source_sha256"}))
            for v in variants
        }
        entities = {v.entity_key for v in variants}
        kinds: set[Kind] = {
            _kind(v.span_kind) if isinstance(v, TraceErrorRecord) else "log" for v in variants
        }
        entity = next(iter(entities)) if len(entities) == 1 else None
        kind: Kind = next(iter(kinds)) if len(kinds) == 1 else "unknown"
        record = variants[0]
        otel: Status = "unknown"
        http: Status = "unknown"
        reasons: tuple[str, ...] = ()
        if len(bodies) > 1:
            outcome: Outcome = "conflict"
            reasons = ("conflicting_duplicate_identity",)
        elif isinstance(record, TraceErrorRecord):
            outcome, otel, http, reasons = _trace_outcome(record, dataset, interval)
        else:
            severity = record.severity.strip().upper() if record.severity else ""
            outcome = (
                "error"
                if severity in _ERROR_SEVERITIES
                else "non_error"
                if severity in _SEVERITIES
                else "unknown"
            )
            if outcome == "unknown":
                reasons = ("unknown_log_severity",)
        members.append(
            ErrorMember(
                member_id=member_id,
                entity_key=entity,
                span_kind=kind,
                source_refs=source_refs,
                outcome=outcome,
                otel_outcome=otel,
                http_outcome=http,
                reasons=reasons,
            )
        )
    by_group: dict[tuple[str | None, Kind], list[ErrorMember]] = defaultdict(list)
    for member in members:
        by_group[(member.entity_key, member.span_kind)].append(member)
    groups = []
    for (entity, kind), values in sorted(
        by_group.items(), key=lambda pair: (pair[0][0] or "", pair[0][1])
    ):
        counts = {
            state: sum(member.outcome == state for member in values)
            for state in ("error", "non_error", "unknown", "conflict", "excluded")
        }
        denominator = counts["error"] + counts["non_error"]
        member_ids = tuple(v.member_id for v in values)
        groups.append(
            ErrorGroup(
                entity_key=entity,
                span_kind=kind,
                statistic="observed_span_status_fraction"
                if modality == "trace"
                else "observed_log_severity_fraction",
                member_ids=member_ids,
                evidence_group_id="errgrp_"
                + _digest(
                    [dataset, modality, entity, kind, interval.model_dump(mode="json"), member_ids]
                ),
                observed_count=len(values),
                error_count=counts["error"],
                non_error_count=counts["non_error"],
                unknown_count=counts["unknown"],
                conflict_count=counts["conflict"],
                excluded_count=counts["excluded"],
                known_denominator=denominator,
                error_fraction=counts["error"] / denominator if denominator else None,
                known_status_fraction=denominator / len(values),
            )
        )
    limitations = [
        "sampling_coverage_unknown",
        "observed_records_not_request_population",
        "error_status_not_root_cause",
        "derived_fields_share_one_evidence_group",
    ]
    severity_basis = "not_applicable"
    if modality == "log":
        severity_basis = (
            "preprocessor_message_heuristic" if dataset == "aiops25" else "exported_severity"
        )
        if dataset == "aiops25":
            limitations.append("heuristic_severity_not_request_success")
    return {
        "input_record_count": len(records),
        "duplicate_record_count": len(records) - len(members),
        "source_records_sha256": _digest([r.model_dump(mode="json") for r in records]),
        "members": tuple(m.model_dump(mode="json") for m in members),
        "groups": tuple(g.model_dump(mode="json") for g in groups),
        "severity_basis": severity_basis,
        "limitations": tuple(sorted(limitations)),
    }


class ErrorObservation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-repair-errors/1"] = ERROR_OBSERVATION_SCHEMA_VERSION
    semantics_version: Literal["nurse-v2-live-repair-errors/1"] = "nurse-v2-live-repair-errors/1"
    semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: Literal["rcabench", "aiops25"]
    modality: Literal["trace", "log"]
    interval: TimeInterval
    source_records: tuple[TraceErrorRecord | LogErrorRecord, ...]
    source_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_record_count: int = Field(ge=0)
    duplicate_record_count: int = Field(ge=0)
    members: tuple[ErrorMember, ...]
    groups: tuple[ErrorGroup, ...]
    sampling_coverage: Literal["unknown"] = "unknown"
    severity_basis: Literal["not_applicable", "exported_severity", "preprocessor_message_heuristic"]
    limitations: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_accounting(self) -> Self:
        if self.semantics_sha256 != ERROR_SEMANTICS_SHA256:
            raise ValueError("error observation semantics mismatch")
        ordered = tuple(
            sorted(
                self.source_records, key=lambda r: canonical_json_bytes(r.model_dump(mode="json"))
            )
        )
        if self.source_records != ordered:
            raise ValueError("error sources must be canonically ordered")
        expected = _derive(
            self.source_records,
            dataset=self.dataset,
            interval=self.interval,
            modality=self.modality,
        )
        if canonical_json_bytes(
            self.model_dump(mode="json", include=set(expected))
        ) != canonical_json_bytes(expected):
            raise ValueError("error accounting is not source-derived")
        return self

    def verify(self) -> bool:
        try:
            document = self.model_dump(mode="json")
            type(self).model_validate(document)
            return self.result_sha256 == _digest(
                {k: v for k, v in document.items() if k != "result_sha256"}
            )
        except (TypeError, ValueError):
            return False

    def verify_source_records(
        self, *, records: Sequence[Record], dataset: str, interval: TimeInterval
    ) -> bool:
        try:
            replayed = _summarize(
                records=records, dataset=dataset, interval=interval, modality=self.modality
            )
            return self.verify() and canonical_json_bytes(
                self.model_dump(mode="json")
            ) == canonical_json_bytes(replayed.model_dump(mode="json"))
        except (TypeError, ValueError):
            return False


def _summarize(
    *,
    records: Sequence[Record],
    dataset: str,
    interval: TimeInterval,
    modality: Literal["trace", "log"],
) -> ErrorObservation:
    record_type = TraceErrorRecord if modality == "trace" else LogErrorRecord
    if any(type(record) is not record_type for record in records):
        raise TypeError("exact error source record types are required")
    validated = tuple(
        record_type.model_validate(record.model_dump(mode="json")) for record in records
    )
    ordered = tuple(
        sorted(validated, key=lambda r: canonical_json_bytes(r.model_dump(mode="json")))
    )
    fields = _derive(ordered, dataset=dataset, interval=interval, modality=modality)
    unsigned = {
        "schema_version": ERROR_OBSERVATION_SCHEMA_VERSION,
        "semantics_version": "nurse-v2-live-repair-errors/1",
        "semantics_sha256": ERROR_SEMANTICS_SHA256,
        "dataset": dataset,
        "modality": modality,
        "interval": interval.model_dump(mode="json"),
        "source_records": tuple(r.model_dump(mode="json") for r in ordered),
        "sampling_coverage": "unknown",
        **fields,
    }
    return ErrorObservation.model_validate({**unsigned, "result_sha256": _digest(unsigned)})


def summarize_trace_errors(
    *, records: Sequence[TraceErrorRecord], dataset: str, interval: TimeInterval
) -> ErrorObservation:
    return _summarize(records=records, dataset=dataset, interval=interval, modality="trace")


def summarize_log_errors(
    *, records: Sequence[LogErrorRecord], dataset: str, interval: TimeInterval
) -> ErrorObservation:
    return _summarize(records=records, dataset=dataset, interval=interval, modality="log")
