"""Private normal-owner semantic composition using the existing durable ARK bridge.

Inputs must come from the admitted preparation/paired-result loader. This is not
a new raw-data loader or a source-authentication cache. Only fixed projection,
actual branch-paid tools and explicit follow-up cross this boundary; no B0 rank.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Literal

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.hermes_exchange import PreparedExchange, execute_once
from nurse_rcl.v2.inference_leakage import (
    _reject_forbidden_report_provenance_values,
    audit_inference_boundary,
)
from nurse_rcl.v2.live import ark_bridge
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalQuerySession
from nurse_rcl.v2.live.normal_hybrid_queries import _load, _publish_value
from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession, run_normal_hybrid_tools
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_semantic import (
    NormalSemanticCompilation,
    NormalSemanticResponse,
    compile_normal_semantics,
    parse_normal_semantic_response,
)
from nurse_rcl.v2.live.normal_semantic_compact_exchange import (
    COMPACT_LIMITS,
    prepare_compact_semantic_exchange,
)
from nurse_rcl.v2.live.normal_semantic_context import project_normal_semantic_context
from nurse_rcl.v2.live.normal_semantic_disabled_exchange import (
    DISABLED_SEMANTIC_EXCHANGE_VERSION,
    execute_disabled_semantic_once,
    prepare_disabled_semantic_exchange,
    semantic_policy,
    validate_disabled_semantic_payload,
)
from nurse_rcl.v2.live.normal_semantic_feedback import (
    build_feedback_package,
    project_semantic_feedback,
)
from nurse_rcl.v2.live.normal_semantic_followup import (
    HYBRID_SEMANTIC_EXCHANGE_VERSION,
    NormalSemanticSnapshotSelection,
    _reported_budget_exceeded,
    prepare_semantic_followup_exchange,
    run_normal_semantic_followup,
    validate_hybrid_semantic_payload,
)
from nurse_rcl.v2.live.normal_semantic_indexed_exchange import (
    INDEXED_SEMANTIC_EXCHANGE_VERSION,
    execute_indexed_semantic_once,
    prepare_indexed_semantic_exchange,
    validate_indexed_semantic_payload,
)
from nurse_rcl.v2.live.normal_semantic_wire import (
    build_semantic_wire,
    expand_semantic_response,
    restore_semantic_context,
)
from nurse_rcl.v2.live.repair_driver import publish_json
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest


class NormalSemanticFirstReceipt(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-first-receipt/1"] = (
        "nurse.v2.normal-semantic-first-receipt/1"
    )
    owner_identity_sha256: SHA256
    context_sha256: SHA256
    request_sha256: SHA256 | None
    response_sha256: SHA256 | None
    response: NormalSemanticResponse | None
    compilation: NormalSemanticCompilation | None
    failure_code: str | None
    receipt_sha256: SHA256


class NormalHybridSemanticRun(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-semantic-run/1"] = (
        "nurse.v2.normal-hybrid-semantic-run/1"
    )
    owner_identity_sha256: SHA256
    first: NormalSemanticFirstReceipt
    session: NormalHybridToolSession | None
    selection: NormalSemanticSnapshotSelection | None
    status: Literal["completed", "followup_fallback", "first_fallback"]
    result_sha256: SHA256


def run_normal_hybrid_semantics(
    *, data: RepairPreparationData, block_root: Path, paired_base_session: NormalQuerySession,
    expected_base_session_sha256: str, expected_source_descriptor_sha256: str,
    expected_catalog_sha256: str, identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str, state_root: Path, execution_sha256: str,
    code_sha256: str, protocol: str = HYBRID_SEMANTIC_EXCHANGE_VERSION,
) -> NormalHybridSemanticRun:
    semantic_policy(protocol)
    disabled = protocol == DISABLED_SEMANTIC_EXCHANGE_VERSION
    validate = validate_disabled_semantic_payload if disabled else validate_hybrid_semantic_payload
    execute = execute_disabled_semantic_once if disabled else execute_once
    indexed = protocol == INDEXED_SEMANTIC_EXCHANGE_VERSION
    if indexed:
        validate = validate_indexed_semantic_payload
        execute = execute_indexed_semantic_once
    for pin in (expected_base_session_sha256, expected_source_descriptor_sha256,
                expected_catalog_sha256, expected_identity_manifest_sha256,
                execution_sha256, code_sha256):
        require_pin(pin)
    opaque = data.catalog.opaque_case_id
    if type(identity_manifest) is not InferenceIdentityManifest or (
        not identity_manifest.verify()
        or identity_manifest.manifest_sha256 != expected_identity_manifest_sha256
        or not any((r.dataset, r.opaque_case_id) == (data.diagnostics.dataset, opaque)
                   for r in identity_manifest.records)
        or state_root.name != opaque
        or data.source_descriptor_sha256 != expected_source_descriptor_sha256
        or data.catalog.catalog_sha256 != expected_catalog_sha256
        or paired_base_session.session_sha256 != expected_base_session_sha256
        or _seal(paired_base_session, "session_sha256") != paired_base_session
        or paired_base_session.catalog_sha256 != expected_catalog_sha256
        or paired_base_session.pending_selection is not None
        or paired_base_session.failure_reason is not None
        or paired_base_session.stop_reason is None
    ):
        raise ValueError("semantic owner independent source/case/base identity differs")
    root = require_plain_path(state_root)
    lock = os.open(root / "semantic-owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _module_span("semantic_fixed_projection"):
            # Existing provenance audit for C, not whole-source or score replay.
            legacy = data.diagnostics.legacy
            audit_inference_boundary((legacy.entity_registry, legacy.candidate_universe),
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=expected_identity_manifest_sha256)
            context = project_normal_semantic_context(data)
            # Provenance-like display names may not smuggle an answer field/path.
            _reject_forbidden_report_provenance_values(
                # Check the finite metadata once, including later paid non-Q cards.
                # This does not display non-Q rows or rescan their raw telemetry.
                (context.candidate_service_keys, tuple(s.metric_name for s in data.sources)),
                location="$semantic.fixed_display_names",
            )
        identity = dict(
            schema_version="nurse.v2.normal-semantic-owner-identity/1", opaque_case_id=opaque,
            code_sha256=code_sha256, execution_sha256=execution_sha256,
            source_descriptor_sha256=expected_source_descriptor_sha256,
            source_files_sha256=data.source_files_sha256, catalog_sha256=expected_catalog_sha256,
            fixed_context_sha256=context.context_sha256,
            paired_base_session_sha256=expected_base_session_sha256,
            identity_manifest_sha256=expected_identity_manifest_sha256,
            request_protocol=protocol,
        )
        identity_pin = sha256_hex(canonical_json_bytes(identity))
        publish_json(root / "semantic-identity.json", identity)

        def send(
            payload: bytes, *, expected: PreparedExchange, context_pin: str,
            phase: Literal["semantic_first", "semantic_followup"],
        ) -> bytes:
            # Bind actual complete bytes, phase and version before loading KEY2.
            request = decode_json_object(payload, COMPACT_LIMITS)
            if payload != expected.payload_bytes or (
                validate(request) != phase
            ):
                raise ValueError("semantic outbound differs from owned projection")
            binding = ark_bridge.LiveRequestBinding(
                opaque_case_id=opaque, variant="B2", phase=phase,
                input_sha256=expected_source_descriptor_sha256, context_sha256=context_pin,
                prompt_schema_sha256=sha256_hex(request["messages"][0]["content"].encode()),
                request_sha256=expected.sha256,
            )
            settings = ark.load_ark_settings_from_key_file(
                Path("/home/wangrunzhou/.config/nurse/ark.env"),
            )
            if settings.redact_artifact(request) != request:
                raise ValueError("semantic outbound contains credential material")
            publish_json(root / f"{phase}-binding.json", binding.model_dump(mode="json"))
            publish_json(root / f"{phase}-request.json", request)
            with ark_bridge.LiveArkBridge(
                root=root / "semantic-ark",
                lock_path=Path(
                    "/home/wangrunzhou/NURSE/development/live-rcl-20260918/locks/ark-in-flight.lock"
                ), settings=settings, code_sha256=code_sha256, config_sha256=identity_pin,
                authorized_case_ids=(opaque,), request_protocol=protocol,
            ) as bridge:
                try:
                    return bridge.complete(binding, payload)
                finally:
                    receipt = bridge.receipt(binding)
                    pin = sha256_hex(canonical_json_bytes(receipt))
                    publish_json(root / f"{phase}-ark-{pin[:16]}.json", receipt)

        first_path = root / "semantic-first.json"
        if first_path.exists():
            first = _load(first_path, NormalSemanticFirstReceipt, "receipt_sha256")
            if first.owner_identity_sha256 != identity_pin or (
                first.context_sha256 != context.context_sha256
            ):
                raise ValueError("saved semantic first identity differs")
            if first.compilation is not None and (first.response is None or
                compile_normal_semantics(data=data, context=context, response=first.response)
                != first.compilation
            ):
                raise ValueError("saved semantic first compilation differs")
        else:
            request_pin = response_pin = failure = None
            response, compilation = None, None
            with _module_span("semantic_first"):
                try:
                    package = build_semantic_wire(context)
                    restore_semantic_context(package, context)
                    request = (prepare_indexed_semantic_exchange(context=context) if indexed
                               else prepare_compact_semantic_exchange(context=context))
                    if disabled:
                        request = prepare_disabled_semantic_exchange(request)
                except ExchangeBoundaryError as exc:
                    failure = exc.code
                else:
                    request_pin = request.sha256
                    attempt = execute(request, lambda raw: send(
                        raw, expected=request, context_pin=context.context_sha256,
                        phase="semantic_first",
                    ))
                    response_pin = attempt.response_sha256
                    if attempt.response is None:
                        failure = attempt.failure_code
                    elif _reported_budget_exceeded(attempt.response, protocol=protocol):
                        failure = "reported_semantic_budget_exceeded"
                    else:
                        try:
                            parsed = parse_normal_semantic_response(attempt.response.output_json)
                        except ExchangeBoundaryError as exc:
                            failure = exc.code
                        else:
                            response = expand_semantic_response(parsed, package)
                            compilation = compile_normal_semantics(data=data, context=context,
                                                                   response=response)
            first = _seal(NormalSemanticFirstReceipt(
                owner_identity_sha256=identity_pin, context_sha256=context.context_sha256,
                request_sha256=request_pin, response_sha256=response_pin, response=response,
                compilation=compilation, failure_code=failure, receipt_sha256="0" * 64,
            ), "receipt_sha256")
            _publish_value(first_path, first)
        terminal_path = root / "semantic-result.json"
        if terminal_path.exists():
            saved = _load(terminal_path, NormalHybridSemanticRun, "result_sha256")
            if saved.owner_identity_sha256 != identity_pin or saved.first != first:
                raise ValueError("saved semantic terminal differs")
            return saved
        session, selection = None, None
        if first.compilation is not None:
            assert first.response is not None
            with _module_span("semantic_tools"):
                session = run_normal_hybrid_tools(
                    data=data, ledger_root=root / "tools", block_root=block_root,
                    paired_base_session=paired_base_session,
                    expected_base_session_sha256=expected_base_session_sha256,
                    branch_identity_sha256=first.receipt_sha256,
                    intents=first.compilation.intents,
                    card_candidates=first.compilation.card_candidates,
                )
            with _module_span("semantic_feedback"):
                feedback = project_semantic_feedback(
                    data=data, context=context, first_response=first.response,
                    first_compilation=first.compilation, session=session,
                    expected_session_sha256=session.session_sha256,
                )

            def followup_complete(payload: bytes) -> bytes:
                package = build_feedback_package(feedback)
                request = prepare_semantic_followup_exchange(
                    feedback=feedback, package=package, protocol=protocol,
                )
                return send(payload, expected=request, context_pin=feedback.feedback_sha256,
                            phase="semantic_followup")

            with _module_span("semantic_followup"):
                selection = run_normal_semantic_followup(
                    data=data, feedback=feedback, first_compilation=first.compilation,
                    complete=followup_complete, protocol=protocol,
                )
        result = _seal(NormalHybridSemanticRun(
            owner_identity_sha256=identity_pin, first=first, session=session, selection=selection,
            status=("first_fallback" if session is None else "followup_fallback"
                    if selection is not None and selection.status == "fallback" else "completed"),
            result_sha256="0" * 64,
        ), "result_sha256")
        _publish_value(terminal_path, result)
        return result
    finally:
        os.close(lock)
