"""Explicit live audit: a single full source document, context and replayable result.

Wire identity alone is not proof of source or score correctness. Every replay
requires independent upstream pins and repeats the complete deterministic chain.
Legacy audit/report readers must continue to reject this separate contract.
"""
from __future__ import annotations

import hmac
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import TypeAdapter

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live import bootstrap_scoring as scoring, scoring_selection as selection_module
from nurse_rcl.v2.live.scoring import LIVE_SCORING_MAX_JSON_BYTES
from nurse_rcl.v2.live_capacity import LIVE_B0_CAPACITY_PROFILE, binding_capacity


@dataclass(frozen=True, slots=True)
class LiveBootstrapAudit:
    sources: selection_module.LiveScoringSources
    context: scoring.LiveBootstrapContext
    result: scoring.LiveBootstrapScoring
    schema_version: Literal['nurse.v2.live-bootstrap-audit/1'] = 'nurse.v2.live-bootstrap-audit/1'


_ADAPTER = TypeAdapter(LiveBootstrapAudit)


def _values(audit: LiveBootstrapAudit) -> dict[str, object]:
    if type(audit) is not LiveBootstrapAudit:
        raise ValueError('live audit requires an exact current-version audit')
    schema = object.__getattribute__(audit, 'schema_version')
    if type(schema) is not str or schema != 'nurse.v2.live-bootstrap-audit/1':
        raise ValueError('live audit schema differs')
    return dict(schema_version=schema,
        sources=selection_module._values(object.__getattribute__(audit, 'sources')),
        context=object.__getattribute__(audit, 'context'),
        result=object.__getattribute__(audit, 'result'))


def _snapshot(audit: LiveBootstrapAudit) -> tuple[bytes, bytes]:
    return scoring._snapshot(_values(audit))


def _unchanged(audit: LiveBootstrapAudit, before: tuple[bytes, bytes]) -> None:
    try:
        same = all(hmac.compare_digest(a, b) for a, b in zip(before, _snapshot(audit), strict=True))
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
        same = False
    if not same:
        raise ValueError('live audit content or representation changed')


def live_bootstrap_audit_bytes(audit: LiveBootstrapAudit) -> bytes:
    """Bounded canonical serialization, not an authoritative replay receipt."""
    original = _snapshot(audit)
    try:
        return original[0]
    finally:
        _unchanged(audit, original)


def load_live_bootstrap_audit(raw: bytes, *, expected_audit_sha256: str) -> LiveBootstrapAudit:
    selection_module._pin(expected_audit_sha256)
    if type(raw) is not bytes or len(raw) > LIVE_SCORING_MAX_JSON_BYTES:
        raise ValueError('live audit exceeds its exact byte budget')
    if not hmac.compare_digest(sha256_hex(raw), expected_audit_sha256):
        raise ValueError('live audit independent pin mismatch')
    selection_module._scan_json(raw)
    with binding_capacity(LIVE_B0_CAPACITY_PROFILE):
        fresh = _ADAPTER.validate_json(raw)
    if not hmac.compare_digest(_snapshot(fresh)[0], raw):
        raise ValueError('live audit bytes are not canonical')
    return fresh


def build_live_bootstrap_audit(*, sources: selection_module.LiveScoringSources,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, selection: selection_module.LiveScoringSelection,
        expected_selection_sha256: str, context: scoring.LiveBootstrapContext,
        expected_context_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveBootstrapAudit:
    result = scoring.build_live_bootstrap_scoring(sources=sources,
        expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256,
        expected_binding_sha256=expected_binding_sha256, selection=selection,
        expected_selection_sha256=expected_selection_sha256, context=context,
        expected_context_sha256=expected_context_sha256, progress=progress)
    # The scoring call protected all inputs/results through both exits. Only a
    # callback-free exact-tree capacity check and plain container construction
    # follow here; no transferred validation token or source cache is created.
    audit = LiveBootstrapAudit(sources=sources, context=context, result=result)
    scoring._preflight(_values(audit))
    return audit


def replay_live_bootstrap_audit(*, audit: LiveBootstrapAudit, expected_audit_sha256: str,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, expected_selection_sha256: str,
        expected_context_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> scoring.LiveBootstrapScoring:
    return cast(scoring.LiveBootstrapScoring, _replay(audit=audit, expected_audit_sha256=expected_audit_sha256,
        expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256,
        expected_binding_sha256=expected_binding_sha256,
        expected_selection_sha256=expected_selection_sha256,
        expected_context_sha256=expected_context_sha256, progress=progress))


def _replay(*, audit: LiveBootstrapAudit, expected_audit_sha256: str,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, expected_selection_sha256: str,
        expected_context_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None,
        _consume: Callable[[selection_module.LiveScoringSources, scoring.LiveBootstrapScoring], Any] | None = None,
        ) -> Any:
    for pin in (expected_audit_sha256, expected_source_records_sha256, expected_rule_set_sha256,
                expected_binding_sha256, expected_selection_sha256, expected_context_sha256):
        selection_module._pin(pin)
    original = _snapshot(audit)
    try:
        if not hmac.compare_digest(sha256_hex(original[0]), expected_audit_sha256):
            raise ValueError('live audit independent pin mismatch')
        expected_result = scoring._snapshot(audit.result)[0]
        def consume(fresh: selection_module.LiveScoringSources, rebuilt: scoring.LiveBootstrapScoring) -> Any:
            if not hmac.compare_digest(scoring._snapshot(rebuilt)[0], expected_result):
                raise ValueError('live audit complete result differs from source and score replay')
            return rebuilt if _consume is None else _consume(fresh, rebuilt)

        return scoring._run_bootstrap(sources=audit.sources,
            expected_source_records_sha256=expected_source_records_sha256,
            expected_rule_set_sha256=expected_rule_set_sha256,
            expected_binding_sha256=expected_binding_sha256, selection=audit.result.evidence.selection,
            expected_selection_sha256=expected_selection_sha256, context=audit.context,
            expected_context_sha256=expected_context_sha256, progress=progress, _consume=consume)
    finally:
        _unchanged(audit, original)
