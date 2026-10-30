"""One fixed five-attempt budget shared by all logical LLM request failures."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import httpx
from hermes_runtime.retry import parse_retry_after_seconds

MAX_REQUEST_ATTEMPTS = 5
FIXED_RETRY_BACKOFF_SECONDS = 2.0

AttemptCategory = Literal["configuration", "parse", "repair", "success", "transport"]
AttemptOutcome = Literal["exhausted", "fatal", "retry", "success"]
AttemptUsageStatus = Literal["partial", "reported", "unavailable"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AttemptUsage:
    """Provider-reported token usage for one attempt, without inferred values."""

    status: AttemptUsageStatus
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cached_tokens: int | None

    def __post_init__(self) -> None:
        if self.status not in {"partial", "reported", "unavailable"}:
            raise ValueError("attempt usage status is invalid")
        values = (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cached_tokens,
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in values
        ):
            raise ValueError("attempt usage token counts must be non-negative integers")
        available = sum(value is not None for value in values)
        if self.status == "unavailable" and available != 0:
            raise ValueError("unavailable attempt usage must use null token counts")
        if self.status == "reported" and available != len(values):
            raise ValueError("reported attempt usage requires every token count")
        if self.status == "partial" and available in {0, len(values)}:
            raise ValueError("partial attempt usage requires some reported token counts")

    @classmethod
    def unavailable(cls) -> AttemptUsage:
        return cls(
            status="unavailable",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_tokens=None,
        )


class _RetryableResponseError(RuntimeError):
    """Retryable response error that can retain usage already returned by a provider."""

    def __init__(self, message: str, *, usage: AttemptUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage if usage is not None else AttemptUsage.unavailable()

    def bind_usage(self, usage: AttemptUsage) -> None:
        self.usage = usage


class RetryableParsingError(_RetryableResponseError):
    """A response could not be parsed but the logical request may be retried."""


class RetryableRepairError(_RetryableResponseError):
    """A structured response could not be repaired within this attempt."""


@dataclass(frozen=True, slots=True)
class AttemptResult(Generic[T]):
    """A successful attempt value with exactly attributable provider usage."""

    value: T
    usage: AttemptUsage


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Non-secret audit record for one consumed attempt."""

    attempt_number: int
    category: AttemptCategory
    outcome: AttemptOutcome
    latency_seconds: float
    usage: AttemptUsage
    delay_seconds: float | None = None
    status_code: int | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or not 1 <= self.attempt_number <= MAX_REQUEST_ATTEMPTS
        ):
            raise ValueError("attempt number must be between 1 and 5")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, int | float)
            or not math.isfinite(self.latency_seconds)
            or self.latency_seconds < 0
        ):
            raise ValueError("attempt latency must be finite and non-negative")
        if self.delay_seconds is not None and (
            isinstance(self.delay_seconds, bool)
            or not isinstance(self.delay_seconds, int | float)
            or not math.isfinite(self.delay_seconds)
            or self.delay_seconds < 0
        ):
            raise ValueError("attempt delay must be finite and non-negative")
        if self.category not in {"configuration", "parse", "repair", "success", "transport"}:
            raise ValueError("attempt category is invalid")
        if self.outcome not in {"exhausted", "fatal", "retry", "success"}:
            raise ValueError("attempt outcome is invalid")
        if self.outcome == "retry":
            if self.category == "configuration":
                raise ValueError("configuration failures cannot be retried")
            if self.delay_seconds is None:
                raise ValueError("retry attempts require a delay")
        elif self.delay_seconds is not None:
            raise ValueError("only retry attempts may record a delay")
        if self.outcome == "success":
            if self.category != "success":
                raise ValueError("successful attempts require the success category")
            if self.error_type is not None or self.status_code is not None:
                raise ValueError("successful attempts cannot record an error or status")
        else:
            if self.category == "success":
                raise ValueError("failed attempts cannot use the success category")
            if not isinstance(self.error_type, str) or not self.error_type:
                raise ValueError("failed attempts require an error type")
        if self.status_code is not None:
            if (
                isinstance(self.status_code, bool)
                or not isinstance(self.status_code, int)
                or not 100 <= self.status_code <= 599
            ):
                raise ValueError("HTTP status must be between 100 and 599")
            if self.category not in {"configuration", "transport"}:
                raise ValueError("only HTTP transport/configuration failures have status")


@dataclass(frozen=True, slots=True)
class RequestExecution(Generic[T]):
    """Successful logical request plus the complete shared attempt trace."""

    value: T
    records: tuple[AttemptRecord, ...]

    @property
    def attempt_count(self) -> int:
        return len(self.records)


class FatalProviderRequestError(RuntimeError):
    """An authentication, configuration, or other non-retryable failure."""

    def __init__(self, *, cause: Exception, record: AttemptRecord) -> None:
        super().__init__(
            f"fatal provider request failure on attempt {record.attempt_number}: "
            f"{type(cause).__name__}"
        )
        self.attempt_count = record.attempt_number
        self.record = record


class RetryExhaustedError(RuntimeError):
    """All five attempts failed transiently after a checkpoint was requested."""

    def __init__(self, records: tuple[AttemptRecord, ...]) -> None:
        super().__init__(f"provider request exhausted {MAX_REQUEST_ATTEMPTS} attempts")
        self.records = records


class CircuitOpenError(RuntimeError):
    """A logical request was denied because the run circuit is already open."""


class CircuitBreaker:
    """Minimal run-level circuit state with an observable one-way open transition."""

    def __init__(self, *, on_open: Callable[[str], None] | None = None) -> None:
        self._is_open = False
        self._reason: str | None = None
        self._on_open = on_open

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def reason(self) -> str | None:
        return self._reason

    def open(self, reason: str) -> None:
        if self._is_open:
            return
        self._reason = reason
        self._is_open = True
        if self._on_open is not None:
            self._on_open(reason)

    def require_closed(self) -> None:
        if self._is_open:
            raise CircuitOpenError(f"provider circuit is open: {self._reason}")


@dataclass(frozen=True, slots=True)
class _FailureClassification:
    retryable: bool
    category: AttemptCategory
    retry_after_seconds: float | None = None
    status_code: int | None = None


def _classify(error: Exception) -> _FailureClassification:
    if isinstance(error, RetryableParsingError):
        return _FailureClassification(retryable=True, category="parse")
    if isinstance(error, RetryableRepairError):
        return _FailureClassification(retryable=True, category="repair")
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        retryable = status_code == 429 or status_code == 408 or status_code >= 500
        retry_after = (
            parse_retry_after_seconds(error.response.headers) if retryable else None
        )
        return _FailureClassification(
            retryable=retryable,
            category="transport" if retryable else "configuration",
            retry_after_seconds=retry_after,
            status_code=status_code,
        )
    if isinstance(error, httpx.TransportError):
        return _FailureClassification(retryable=True, category="transport")
    return _FailureClassification(retryable=False, category="configuration")


def execute_llm_request(
    attempt: Callable[[int], T | AttemptResult[T]],
    *,
    sleep: Callable[[float], None],
    checkpoint: Callable[[tuple[AttemptRecord, ...]], None],
    circuit_breaker: CircuitBreaker,
    initial_records: tuple[AttemptRecord, ...] = (),
    reserve_attempt: Callable[[int], None] = lambda _: None,
    record_attempt: Callable[[AttemptRecord, T | None], None] = lambda _record, _value: None,
) -> RequestExecution[T]:
    """Execute one logical LLM request with exactly one shared five-attempt budget."""

    circuit_breaker.require_closed()
    if tuple(item.attempt_number for item in initial_records) != tuple(
        range(1, len(initial_records) + 1)
    ):
        raise ValueError("initial attempt records must be contiguous and one-based")
    if len(initial_records) >= MAX_REQUEST_ATTEMPTS:
        raise ValueError("initial attempt records must leave at least one attempt")
    if any(item.outcome != "retry" for item in initial_records):
        raise ValueError("initial attempt records must be retry outcomes")
    records = list(initial_records)
    for attempt_number in range(len(records) + 1, MAX_REQUEST_ATTEMPTS + 1):
        reserve_attempt(attempt_number)
        started = time.monotonic()
        try:
            raw_result = attempt(attempt_number)
        except Exception as error:
            latency_seconds = max(0.0, time.monotonic() - started)
            usage = (
                error.usage
                if isinstance(error, _RetryableResponseError)
                else AttemptUsage.unavailable()
            )
            classification = _classify(error)
            if not classification.retryable:
                record = AttemptRecord(
                    attempt_number=attempt_number,
                    category=classification.category,
                    outcome="fatal",
                    latency_seconds=latency_seconds,
                    usage=usage,
                    status_code=classification.status_code,
                    error_type=type(error).__name__,
                )
                records.append(record)
                record_attempt(record, None)
                raise FatalProviderRequestError(cause=error, record=record) from error

            if attempt_number == MAX_REQUEST_ATTEMPTS:
                record = AttemptRecord(
                    attempt_number=attempt_number,
                    category=classification.category,
                    outcome="exhausted",
                    latency_seconds=latency_seconds,
                    usage=usage,
                    status_code=classification.status_code,
                    error_type=type(error).__name__,
                )
                records.append(record)
                record_attempt(record, None)
                frozen_records = tuple(records)
                checkpoint(frozen_records)
                circuit_breaker.open("retry_exhausted")
                raise RetryExhaustedError(frozen_records) from error

            delay = (
                classification.retry_after_seconds
                if classification.retry_after_seconds is not None
                else FIXED_RETRY_BACKOFF_SECONDS
            )
            record = AttemptRecord(
                attempt_number=attempt_number,
                category=classification.category,
                outcome="retry",
                latency_seconds=latency_seconds,
                usage=usage,
                delay_seconds=delay,
                status_code=classification.status_code,
                error_type=type(error).__name__,
            )
            records.append(record)
            record_attempt(record, None)
            sleep(delay)
        else:
            latency_seconds = max(0.0, time.monotonic() - started)
            if isinstance(raw_result, AttemptResult):
                value = raw_result.value
                usage = raw_result.usage
            else:
                value = raw_result
                usage = AttemptUsage.unavailable()
            record = AttemptRecord(
                attempt_number=attempt_number,
                category="success",
                outcome="success",
                latency_seconds=latency_seconds,
                usage=usage,
            )
            records.append(record)
            record_attempt(record, value)
            return RequestExecution(value=value, records=tuple(records))
    raise AssertionError("fixed attempt loop did not terminate")
