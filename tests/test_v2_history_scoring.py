"""Isolated, label-free history scoring for NURSE V2."""

from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError, dataclass
from types import ModuleType

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import (
    CANDIDATE_FREEZE_SCHEMA_VERSION,
    FrozenCandidateUniverse,
)

_TRUSTED_AUTHORITY_MANIFESTS: dict[str, bytes] = {}
_TRUSTED_AUTHORITIES: dict[str, object] = {}
_TRUSTED_AUTHORITY_MATERIALS: dict[str, tuple[dict[str, object], ...]] = {}
_TRUSTED_ENTRY_SOURCES: dict[str, object] = {}


@dataclass(frozen=True, slots=True)
class FrozenHistoryRunManifest:
    history_generation_authority_sha256: str


_TRUSTED_RUN_MANIFESTS: dict[str, FrozenHistoryRunManifest] = {}


def history_scoring_module() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.history_scoring")
    except ModuleNotFoundError:
        pytest.fail("Task 8.7 isolated history scoring is not implemented", pytrace=False)


def _install_digest_collision(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    schema_version: str,
    collision_kind: str,
) -> None:
    real_digest = module._digest

    def collision_digest(value: object) -> str:
        digest = real_digest(value)
        if isinstance(value, dict) and value.get("schema_version") == schema_version:
            if collision_kind == "prefix":
                return "f" * 24 + digest[24:]
            if collision_kind == "full":
                return "f" * 64
            raise AssertionError(f"unsupported collision kind: {collision_kind}")
        return digest

    monkeypatch.setattr(module, "_digest", collision_digest)


def _frozen_candidates() -> FrozenCandidateUniverse:
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_FREEZE_SCHEMA_VERSION,
        "opaque_case_id": "opq_" + "a" * 32,
        "candidate_universe_sha256": "b" * 64,
        "observation_contract_sha256": "c" * 64,
        "candidate_count": 3,
        "candidate_service_keys": ("service-a", "service-b", "service-c"),
        "freeze_stage": "pre_semantic",
        "candidate_set_mutable": False,
    }
    return FrozenCandidateUniverse.model_validate(
        {**unsigned, "freeze_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def _generation_authority(tag: str = "a") -> object:
    return _register_generation_batch(_typed_entry_materials(tag=tag))


def _batch_manifest(tag: str = "a") -> bytes:
    return _typed_batch_manifest_bytes(_typed_entry_materials(tag=tag))


def _manifest_entry_key(case_hex: str) -> str:
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "dataset": "rcabench",
                "source_partition": "dev",
                "source_opaque_case_id": "opq_" + case_hex * 32,
            }
        )
    )
    return f"hment_{digest[:24]}"


def _manifest_base_score_key(manifest_entry_key: str, source_service_key: str) -> str:
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "manifest_entry_key": manifest_entry_key,
                "source_service_key": source_service_key,
            }
        )
    )
    return f"hmbs_{digest[:24]}"


def _safe_external_flags() -> dict[str, object]:
    return {
        "uses_supervision": False,
        "has_solution_data": False,
        "has_failure_metadata": False,
        "has_injected_metadata": False,
        "has_reversible_source_refs": False,
    }


def _typed_entry_materials(
    *,
    case_hex: str = "b",
    group_hex: str = "d",
    completed_at: int = 10,
    embedding: tuple[float, ...] = (1.0, 0.0),
    tag: str = "typed",
    service_scores: tuple[tuple[str, float], ...] | None = None,
    artifact_extra: tuple[str, str, object] | None = None,
) -> dict[str, object]:
    entry_key = _manifest_entry_key(case_hex)
    score_documents: list[dict[str, object]] = []
    score_bytes: list[bytes] = []
    score_memberships: list[dict[str, object]] = []
    score_specs = service_scores or (
        (f"source-{tag}-a", 0.8),
        (f"source-{tag}-b", 0.2),
    )
    for index, (service, score) in enumerate(score_specs):
        suffix = chr(ord("a") + index)
        score_key = _manifest_base_score_key(entry_key, service)
        document: dict[str, object] = {
            "schema_version": "nurse.v2.history-sanitized-base-score-artifact/1",
            "manifest_entry_key": entry_key,
            "manifest_base_score_key": score_key,
            "source_service_key": service,
            "base_history_score": score,
            "source_history_score": 0.0,
            "source_rule_score": 0.0,
            **_safe_external_flags(),
        }
        if artifact_extra is not None and artifact_extra[0] == f"score-{suffix}":
            document[artifact_extra[1]] = artifact_extra[2]
        raw = canonical_json_bytes(document)
        score_documents.append(document)
        score_bytes.append(raw)
        score_memberships.append(
            {
                "schema_version": "nurse.v2.history-generation-base-score-membership/1",
                "manifest_entry_key": entry_key,
                "manifest_base_score_key": score_key,
                "source_service_key": service,
                "base_history_score": score,
                "source_history_score": 0.0,
                "source_rule_score": 0.0,
                "score_artifact_sha256": sha256_hex(raw),
            }
        )
    score_memberships.sort(key=lambda item: item["source_service_key"])

    artifact_documents: dict[str, dict[str, object]] = {}
    artifact_bytes: dict[str, bytes] = {}
    artifact_fields = {
        "semantic": "semantic_signal_sha256s",
        "graph": "graph_edge_sha256s",
        "topology": "topology_relation_sha256s",
    }
    for kind, digest_field in artifact_fields.items():
        document = {
            "schema_version": f"nurse.v2.history-sanitized-{kind}-artifact/1",
            "manifest_entry_key": entry_key,
            "source_opaque_case_id": "opq_" + case_hex * 32,
            "service_keys": tuple(sorted(item["source_service_key"] for item in score_memberships)),
            digest_field: (sha256_hex(canonical_json_bytes({"kind": kind, "tag": tag})),),
            **_safe_external_flags(),
        }
        if artifact_extra is not None and artifact_extra[0] == kind:
            document[artifact_extra[1]] = artifact_extra[2]
        artifact_documents[kind] = document
        artifact_bytes[kind] = canonical_json_bytes(document)

    membership: dict[str, object] = {
        "schema_version": "nurse.v2.history-generation-entry-membership/1",
        "manifest_entry_key": entry_key,
        "dataset": "rcabench",
        "source_partition": "dev",
        "source_opaque_case_id": "opq_" + case_hex * 32,
        "source_group_fingerprint": group_hex * 64,
        "source_completed_at_unix_ns": completed_at,
        "source_history_mode": "off",
        "source_rule_channel_enabled": False,
        **_safe_external_flags(),
        "embedding_policy": "semantic-v1",
        "embedding_model": "embedding-v1",
        "embedding_dimension": len(embedding),
        "embedding": embedding,
        "semantic_artifact_sha256": sha256_hex(artifact_bytes["semantic"]),
        "graph_artifact_sha256": sha256_hex(artifact_bytes["graph"]),
        "topology_artifact_sha256": sha256_hex(artifact_bytes["topology"]),
        "base_scores": tuple(score_memberships),
    }
    return {
        "manifest_entry_key": entry_key,
        "membership": membership,
        "score_documents": tuple(score_documents),
        "score_bytes": tuple(score_bytes),
        "semantic_bytes": artifact_bytes["semantic"],
        "graph_bytes": artifact_bytes["graph"],
        "topology_bytes": artifact_bytes["topology"],
    }


def _typed_batch_manifest_bytes(
    *materials: dict[str, object],
    extra_field: tuple[str, object] | None = None,
) -> bytes:
    document: dict[str, object] = {
        "schema_version": "nurse.v2.history-generation-batch-manifest/1",
        "dataset": "rcabench",
        "source_partition": "dev",
        "generation_policy": "history-base-generation-v1",
        "source_history_mode": "off",
        "source_rule_channel_enabled": False,
        **_safe_external_flags(),
        "entries": tuple(
            material["membership"]
            for material in sorted(
                materials,
                key=lambda item: item["manifest_entry_key"],
            )
        ),
    }
    if extra_field is not None:
        document[extra_field[0]] = extra_field[1]
    return canonical_json_bytes(document)


def _independent_authority_anchor(canonical_batch_manifest: bytes) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "schema_version": "nurse.v2.history-generation-authority/1",
                "dataset": "rcabench",
                "source_partition": "dev",
                "source_history_mode": "off",
                "source_rule_channel_enabled": False,
                "generation_policy": "history-base-generation-v1",
                "canonical_batch_manifest_sha256": sha256_hex(canonical_batch_manifest),
            }
        )
    )


def _freeze_history_run_manifest(
    canonical_batch_manifest: bytes,
) -> FrozenHistoryRunManifest:
    return FrozenHistoryRunManifest(
        history_generation_authority_sha256=_independent_authority_anchor(canonical_batch_manifest)
    )


def _register_generation_batch(*materials: dict[str, object]) -> object:
    canonical_batch_manifest = _typed_batch_manifest_bytes(*materials)
    frozen_run_manifest = _freeze_history_run_manifest(canonical_batch_manifest)
    authority = history_scoring_module().build_history_generation_authority(
        dataset="rcabench",
        source_partition="dev",
        generation_policy="history-base-generation-v1",
        canonical_batch_manifest=canonical_batch_manifest,
    )
    assert (
        authority.generation_authority_sha256
        == frozen_run_manifest.history_generation_authority_sha256
    )
    _TRUSTED_AUTHORITY_MANIFESTS[authority.generation_authority_sha256] = canonical_batch_manifest
    _TRUSTED_AUTHORITIES[authority.generation_authority_sha256] = authority
    _TRUSTED_AUTHORITY_MATERIALS[authority.generation_authority_sha256] = materials
    _TRUSTED_RUN_MANIFESTS[authority.generation_authority_sha256] = frozen_run_manifest
    return authority


def _build_authority_for_typed_manifest(
    canonical_batch_manifest: bytes,
) -> tuple[object, FrozenHistoryRunManifest]:
    frozen_run_manifest = _freeze_history_run_manifest(canonical_batch_manifest)
    authority = history_scoring_module().build_history_generation_authority(
        dataset="rcabench",
        source_partition="dev",
        generation_policy="history-base-generation-v1",
        canonical_batch_manifest=canonical_batch_manifest,
    )
    assert (
        authority.generation_authority_sha256
        == frozen_run_manifest.history_generation_authority_sha256
    )
    return authority, frozen_run_manifest


def _build_entry_from_typed_material(
    material: dict[str, object],
    *,
    canonical_batch_manifest: bytes,
    authority: object,
    external_anchor: FrozenHistoryRunManifest,
) -> tuple[object, tuple[object, ...]]:
    module = history_scoring_module()
    membership = material["membership"]
    score_documents = material["score_documents"]
    score_bytes = material["score_bytes"]
    base_sources = tuple(
        module.HistoryTrustedBaseScoreSource(
            source_service_key=document["source_service_key"],
            base_score_artifact=raw,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    scores = tuple(
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=document["base_history_score"],
            base_score_artifact=raw,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=canonical_batch_manifest,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    entry = module.build_history_library_entry(
        dataset=membership["dataset"],
        source_partition=membership["source_partition"],
        source_opaque_case_id=membership["source_opaque_case_id"],
        source_group_fingerprint=membership["source_group_fingerprint"],
        source_completed_at_unix_ns=membership["source_completed_at_unix_ns"],
        embedding_policy=membership["embedding_policy"],
        embedding_model=membership["embedding_model"],
        embedding=membership["embedding"],
        sanitized_semantic_payload=material["semantic_bytes"],
        sanitized_graph=material["graph_bytes"],
        sanitized_topology=material["topology_bytes"],
        base_candidate_scores=scores,
        generation_authority=authority,
        expected_generation_authority_sha256=(external_anchor.history_generation_authority_sha256),
        canonical_batch_manifest=canonical_batch_manifest,
        trusted_base_score_sources=base_sources,
    )
    return entry, base_sources


def _base_scores(
    tag: str = "a",
    *,
    authority: object | None = None,
) -> tuple[object, ...]:
    module = history_scoring_module()
    generation_authority = _generation_authority() if authority is None else authority
    authority_sha256 = generation_authority.generation_authority_sha256
    manifest = _TRUSTED_AUTHORITY_MANIFESTS[authority_sha256]
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority_sha256]
    expected_services = {f"source-{tag}-a", f"source-{tag}-b"}
    material = next(
        (
            item
            for item in _TRUSTED_AUTHORITY_MATERIALS[authority_sha256]
            if {document["source_service_key"] for document in item["score_documents"]}
            == expected_services
        ),
        _typed_entry_materials(tag=tag),
    )
    return tuple(
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=document["base_history_score"],
            base_score_artifact=raw,
            generation_authority=generation_authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )
        for document, raw in zip(material["score_documents"], material["score_bytes"], strict=True)
    )


def _entry(
    *,
    case_hex: str = "b",
    group_hex: str = "d",
    completed_at: int = 10,
    embedding: tuple[float, ...] = (1.0, 0.0),
    tag: str = "a",
    authority: object | None = None,
    embedding_policy: str = "semantic-v1",
    embedding_model: str = "embedding-v1",
    manifest_embedding: tuple[float, ...] | None = None,
    service_scores: tuple[tuple[str, float], ...] | None = None,
) -> object:
    module = history_scoring_module()
    material = _typed_entry_materials(
        case_hex=case_hex,
        group_hex=group_hex,
        completed_at=completed_at,
        embedding=embedding if manifest_embedding is None else manifest_embedding,
        tag=tag,
        service_scores=service_scores,
    )
    generation_authority = _register_generation_batch(material) if authority is None else authority
    authority_sha256 = generation_authority.generation_authority_sha256
    manifest = _TRUSTED_AUTHORITY_MANIFESTS[authority_sha256]
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority_sha256]
    score_documents = material["score_documents"]
    score_bytes = material["score_bytes"]
    base_sources = tuple(
        module.HistoryTrustedBaseScoreSource(
            source_service_key=document["source_service_key"],
            base_score_artifact=raw,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    scores = tuple(
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=document["base_history_score"],
            base_score_artifact=raw,
            generation_authority=generation_authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    entry = module.build_history_library_entry(
        dataset="rcabench",
        source_partition="dev",
        source_opaque_case_id="opq_" + case_hex * 32,
        source_group_fingerprint=group_hex * 64,
        source_completed_at_unix_ns=completed_at,
        embedding_policy=embedding_policy,
        embedding_model=embedding_model,
        embedding=embedding,
        sanitized_semantic_payload=material["semantic_bytes"],
        sanitized_graph=material["graph_bytes"],
        sanitized_topology=material["topology_bytes"],
        base_candidate_scores=scores,
        generation_authority=generation_authority,
        expected_generation_authority_sha256=(
            frozen_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=manifest,
        trusted_base_score_sources=base_sources,
    )
    _TRUSTED_ENTRY_SOURCES[entry.entry_id] = module.HistoryTrustedEntrySource(
        entry_id=entry.entry_id,
        base_score_sources=base_sources,
        sanitized_semantic_payload=material["semantic_bytes"],
        sanitized_graph=material["graph_bytes"],
        sanitized_topology=material["topology_bytes"],
    )
    return entry


def _shared_entry_batch(
    *entry_specs: dict[str, object],
) -> tuple[object, tuple[object, ...]]:
    materials = tuple(_typed_entry_materials(**spec) for spec in entry_specs)
    authority = _register_generation_batch(*materials)
    entries = tuple(_entry(authority=authority, **spec) for spec in entry_specs)
    return authority, entries


def _trusted_bundle_for_entries(
    authority: object,
    entries: tuple[object, ...],
) -> object:
    module = history_scoring_module()
    return module.HistoryTrustedSourceBundle(
        generation_authority=authority,
        expected_generation_authority_sha256=(
            _TRUSTED_RUN_MANIFESTS[
                authority.generation_authority_sha256
            ].history_generation_authority_sha256
        ),
        canonical_batch_manifest=_TRUSTED_AUTHORITY_MANIFESTS[
            authority.generation_authority_sha256
        ],
        entry_sources=tuple(_TRUSTED_ENTRY_SOURCES[entry.entry_id] for entry in entries),
    )


def _trusted_bundle_for_library(library: object) -> object:
    authority_sha256 = library.generation_authority_sha256
    authority = _TRUSTED_AUTHORITIES[authority_sha256]
    return _trusted_bundle_for_entries(authority, library.entries)


def _install_typed_document_parse_counter(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> dict[tuple[object, bytes], int]:
    parse_counts: dict[tuple[object, bytes], int] = {}
    parse_document = module._parse_canonical_typed_document

    def counting_parse(value: object, model_type: object) -> object:
        assert type(value) is bytes
        key = (model_type, value)
        parse_counts[key] = parse_counts.get(key, 0) + 1
        return parse_document(value, model_type)

    monkeypatch.setattr(module, "_parse_canonical_typed_document", counting_parse)
    return parse_counts


def _assert_trusted_bundle_artifacts_parsed_once(
    module: ModuleType,
    bundle: object,
    parse_counts: dict[tuple[object, bytes], int],
) -> None:
    for entry_source in bundle.entry_sources:
        assert (
            parse_counts[
                (
                    module.HistorySanitizedSemanticArtifact,
                    entry_source.sanitized_semantic_payload,
                )
            ]
            == 1
        )
        assert (
            parse_counts[(module.HistorySanitizedGraphArtifact, entry_source.sanitized_graph)] == 1
        )
        assert (
            parse_counts[(module.HistorySanitizedTopologyArtifact, entry_source.sanitized_topology)]
            == 1
        )
        for base_source in entry_source.base_score_sources:
            assert (
                parse_counts[
                    (
                        module.HistorySanitizedBaseScoreArtifact,
                        base_source.base_score_artifact,
                    )
                ]
                == 1
            )


def _library(*entries: object, authority: object | None = None) -> object:
    generation_authority = (
        _generation_authority()
        if authority is None and not entries
        else _TRUSTED_AUTHORITIES[entries[0].generation_authority_sha256]
        if authority is None
        else authority
    )
    bundle = _trusted_bundle_for_entries(generation_authority, entries)
    return history_scoring_module().build_frozen_history_library(
        dataset="rcabench",
        source_partition="dev",
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        entries=entries,
        generation_authority=generation_authority,
        trusted_source_bundle=bundle,
    )


def _query(
    *,
    case_hex: str = "a",
    group_hex: str = "c",
    decision_time: int = 20,
    embedding: tuple[float, ...] = (1.0, 0.0),
    embedding_policy: str = "semantic-v1",
    embedding_model: str = "embedding-v1",
) -> object:
    return history_scoring_module().build_history_query_embedding(
        current_opaque_case_id="opq_" + case_hex * 32,
        current_group_fingerprint=group_hex * 64,
        dataset="rcabench",
        current_decision_time_unix_ns=decision_time,
        embedding_policy=embedding_policy,
        embedding_model=embedding_model,
        embedding_dimension=2,
        embedding=embedding,
    )


def _mapping(entry: object, *, source: str, current: str) -> object:
    return history_scoring_module().build_history_service_mapping(
        entry_id=entry.entry_id,
        source_service_key=source,
        current_service_key=current,
        mapping_policy="exact-service-alias-v1",
        provenance_ids=("mapping:fixture-v1",),
    )


def _trusted_components(tag: str = "trusted") -> tuple[object, ...]:
    module = history_scoring_module()
    authority = _generation_authority(tag)
    manifest = _TRUSTED_AUTHORITY_MANIFESTS[authority.generation_authority_sha256]
    scores = _base_scores(tag, authority=authority)
    entry = _entry(tag=tag, authority=authority)
    entry_source = _TRUSTED_ENTRY_SOURCES[entry.entry_id]
    bundle = _trusted_bundle_for_entries(authority, (entry,))
    library = module.build_frozen_history_library(
        dataset="rcabench",
        source_partition="dev",
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        entries=(entry,),
        generation_authority=authority,
        trusted_source_bundle=bundle,
    )
    return (
        authority,
        manifest,
        entry_source.base_score_sources,
        scores,
        entry,
        entry_source,
        bundle,
        library,
    )


def test_history_off_is_default_and_never_loads_a_library() -> None:
    frozen = _frozen_candidates()
    result = history_scoring_module().evaluate_history(
        frozen_candidates=frozen,
        library=None,
        query_embedding=None,
        service_mappings=(),
    )

    assert result.verify_structure()
    assert result.mode == "off"
    assert result.library_sha256 is None
    assert result.selected_neighbor_ids == ()
    assert result.hint_references == ()
    assert result.requires_current_case_reverification is False
    assert result.candidate_universe_sha256 == frozen.candidate_universe_sha256
    assert result.candidate_count == frozen.candidate_count
    assert result.candidate_service_keys == frozen.candidate_service_keys
    assert tuple(item.candidate_service_key for item in result.candidate_scores) == (
        "service-a",
        "service-b",
        "service-c",
    )
    assert all(item.history_score == 0.0 for item in result.candidate_scores)


def test_result_schema_rejects_resigned_incomplete_candidate_coverage() -> None:
    module = history_scoring_module()
    result = module.evaluate_history(frozen_candidates=_frozen_candidates())
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["candidate_scores"] = tuple(
        item.model_dump(mode="json") for item in result.candidate_scores[:-1]
    )
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError, match="candidate coverage"):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": f"heval_{digest[:24]}",
                "result_sha256": digest,
            }
        )


def test_verify_structure_rejects_resigned_candidate_subset_with_stale_binding() -> None:
    module = history_scoring_module()
    result = module.evaluate_history(frozen_candidates=_frozen_candidates())
    expected_binding = sha256_hex(
        canonical_json_bytes(
            {
                "candidate_freeze_sha256": result.candidate_freeze_sha256,
                "candidate_universe_sha256": result.candidate_universe_sha256,
                "candidate_count": result.candidate_count,
                "candidate_service_keys": result.candidate_service_keys,
            }
        )
    )
    assert result.candidate_binding_sha256 == expected_binding
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned.update(
        candidate_count=2,
        candidate_service_keys=("service-a", "service-b"),
        candidate_scores=tuple(
            item.model_dump(mode="json") for item in result.candidate_scores[:2]
        ),
    )
    digest = sha256_hex(canonical_json_bytes(unsigned))
    forged = result.model_copy(
        update={
            "candidate_count": 2,
            "candidate_service_keys": ("service-a", "service-b"),
            "candidate_scores": result.candidate_scores[:2],
            "history_evaluation_result_id": f"heval_{digest[:24]}",
            "result_sha256": digest,
        }
    )

    assert not forged.verify_structure()


def test_verify_structure_rejects_resigned_authoritative_input_digest_change() -> None:
    module = history_scoring_module()
    result = module.evaluate_history(frozen_candidates=_frozen_candidates())
    forged_authoritative_sha256 = "f" * 64
    assert forged_authoritative_sha256 != result.authoritative_input_sha256
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["authoritative_input_sha256"] = forged_authoritative_sha256
    digest = sha256_hex(canonical_json_bytes(unsigned))
    forged = result.model_copy(
        update={
            "authoritative_input_sha256": forged_authoritative_sha256,
            "history_evaluation_result_id": f"heval_{digest[:24]}",
            "result_sha256": digest,
        }
    )

    assert not forged.verify_structure()


def test_candidate_authority_resign_is_structural_only_and_external_anchors_win() -> None:
    module = history_scoring_module()
    original = module.evaluate_history(frozen_candidates=_frozen_candidates())
    subset_unsigned = _frozen_candidates().model_dump(mode="json", exclude={"freeze_sha256"})
    subset_unsigned["candidate_count"] = 2
    subset_unsigned["candidate_service_keys"] = ("service-a", "service-b")
    subset = FrozenCandidateUniverse.model_validate(
        {
            **subset_unsigned,
            "freeze_sha256": sha256_hex(canonical_json_bytes(subset_unsigned)),
        }
    )
    resigned = module.evaluate_history(frozen_candidates=subset)

    assert resigned.verify_structure()
    assert not resigned.verify_trusted(
        expected_result_sha256=original.result_sha256,
        expected_authoritative_input_sha256=original.authoritative_input_sha256,
        frozen_candidates=subset,
    )


def test_history_builders_freeze_verified_label_free_nonrecursive_inputs() -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    entry = _entry(authority=authority)
    library = module.build_frozen_history_library(
        dataset="rcabench",
        source_partition="dev",
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        entries=(entry,),
        generation_authority=authority,
        trusted_source_bundle=_trusted_bundle_for_entries(authority, (entry,)),
    )
    query = module.build_history_query_embedding(
        current_opaque_case_id="opq_" + "a" * 32,
        current_group_fingerprint="c" * 64,
        dataset="rcabench",
        current_decision_time_unix_ns=20,
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        embedding=(1.0, 0.0),
    )
    mapping = module.build_history_service_mapping(
        entry_id=entry.entry_id,
        source_service_key="source-a-a",
        current_service_key="service-a",
        mapping_policy="exact-service-alias-v1",
        provenance_ids=("mapping:fixture-v1",),
    )
    config = module.build_history_scoring_config()

    assert entry.verify()
    assert library.verify()
    assert query.verify()
    assert mapping.verify()
    assert config.verify()
    assert entry.source_history_mode == "off"
    assert entry.source_rule_channel_enabled is False
    assert entry.label_free is True
    assert entry.contains_answers is False
    assert entry.contains_fault_metadata is False
    assert entry.contains_injection_metadata is False
    assert entry.contains_original_identifiers_or_paths is False
    assert library.read_only is True
    assert config.k == 3
    assert config.min_similarity == 0.7
    assert config.time_policy == "strict_prior"
    with pytest.raises(ValueError):
        library.entries += (entry,)


def test_generation_authority_recomputes_artifacts_and_is_shared_by_library() -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    scores = _base_scores(authority=authority)
    entry = _entry(authority=authority)
    library = _library(entry, authority=authority)
    authority_sha256 = authority.generation_authority_sha256
    expected_manifest = _TRUSTED_AUTHORITY_MANIFESTS[authority_sha256]
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority_sha256]
    material = _TRUSTED_AUTHORITY_MATERIALS[authority_sha256][0]
    expected_base_artifact = material["score_bytes"][0]
    changed_document = dict(material["score_documents"][0])
    changed_document["base_history_score"] = 0.7

    assert authority.verify()
    assert authority.source_history_mode == "off"
    assert authority.source_rule_channel_enabled is False
    assert authority.canonical_batch_manifest_sha256 == sha256_hex(expected_manifest)
    assert scores[0].base_score_artifact_sha256 == sha256_hex(expected_base_artifact)
    with pytest.raises(ValueError, match="^history base score input verification failed$"):
        module.build_history_base_candidate_score(
            source_service_key="source-a-a",
            base_history_score=0.7,
            base_score_artifact=canonical_json_bytes(changed_document),
            generation_authority=authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=expected_manifest,
        )
    assert all(
        score.generation_authority_sha256 == authority.generation_authority_sha256
        for score in scores
    )
    assert entry.generation_authority_sha256 == authority.generation_authority_sha256
    assert library.generation_authority_sha256 == authority.generation_authority_sha256


def test_generation_authority_verify_trusted_replays_external_manifest_anchor() -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority.generation_authority_sha256]

    assert authority.verify_trusted(
        expected_generation_authority_sha256=(
            frozen_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=_batch_manifest(),
    )
    assert not authority.verify_trusted(
        expected_generation_authority_sha256="0" * 64,
        canonical_batch_manifest=_batch_manifest(),
    )
    assert not authority.verify_trusted(
        expected_generation_authority_sha256=(
            frozen_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=_batch_manifest("changed"),
    )

    class AuthoritySubclass(module.HistoryGenerationAuthority):
        pass

    subclass = AuthoritySubclass.model_validate(authority.model_dump(mode="json"))
    assert not module.HistoryGenerationAuthority.verify_trusted(
        subclass,
        expected_generation_authority_sha256=(
            frozen_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=_batch_manifest(),
    )


def test_typed_manifest_prevents_same_raw_score_authorizing_another_value() -> None:
    module = history_scoring_module()
    material = _typed_entry_materials(tag="score-binding")
    manifest = _typed_batch_manifest_bytes(material)
    authority, external_anchor = _build_authority_for_typed_manifest(manifest)
    document = material["score_documents"][0]
    raw = material["score_bytes"][0]
    accepted = module.build_history_base_candidate_score(
        source_service_key=document["source_service_key"],
        base_history_score=document["base_history_score"],
        base_score_artifact=raw,
        generation_authority=authority,
        expected_generation_authority_sha256=(external_anchor.history_generation_authority_sha256),
        canonical_batch_manifest=manifest,
    )

    assert accepted.base_history_score == document["base_history_score"]
    with pytest.raises(ValueError, match="^history base score input verification failed$"):
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=0.9 if document["base_history_score"] != 0.9 else 0.1,
            base_score_artifact=raw,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )


def test_base_score_membership_is_entry_scoped_when_a_service_repeats_in_batch() -> None:
    module = history_scoring_module()
    first_material = _typed_entry_materials(
        case_hex="b",
        tag="shared-service-a",
        service_scores=(("shared-service", 0.8),),
    )
    second_material = _typed_entry_materials(
        case_hex="e",
        group_hex="f",
        tag="shared-service-b",
        service_scores=(("shared-service", 0.2),),
    )
    manifest = _typed_batch_manifest_bytes(first_material, second_material)
    authority, frozen_run_manifest = _build_authority_for_typed_manifest(manifest)

    scores = tuple(
        module.build_history_base_candidate_score(
            source_service_key=material["score_documents"][0]["source_service_key"],
            base_history_score=material["score_documents"][0]["base_history_score"],
            base_score_artifact=material["score_bytes"][0],
            generation_authority=authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )
        for material in (first_material, second_material)
    )

    assert tuple(score.source_service_key for score in scores) == (
        "shared-service",
        "shared-service",
    )
    assert tuple(score.manifest_entry_key for score in scores) == (
        first_material["manifest_entry_key"],
        second_material["manifest_entry_key"],
    )


def test_library_trusted_replay_parses_each_typed_document_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, entries = _shared_entry_batch(
        {
            "case_hex": "b",
            "group_hex": "d",
            "tag": "parse-count-a",
            "service_scores": (("shared-service", 0.8),),
        },
        {
            "case_hex": "e",
            "group_hex": "f",
            "tag": "parse-count-b",
            "service_scores": (("shared-service", 0.2),),
        },
    )
    bundle = _trusted_bundle_for_entries(authority, entries)
    parse_counts = _install_typed_document_parse_counter(monkeypatch, module)

    module.build_frozen_history_library(
        dataset="rcabench",
        source_partition="dev",
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        entries=entries,
        generation_authority=authority,
        trusted_source_bundle=bundle,
    )

    _assert_trusted_bundle_artifacts_parsed_once(module, bundle, parse_counts)
    assert (
        parse_counts[
            (
                module.HistoryGenerationBatchManifest,
                bundle.canonical_batch_manifest,
            )
        ]
        == 1
    )


def test_evaluation_trusted_replay_parses_each_typed_document_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, entries = _shared_entry_batch(
        {
            "case_hex": "b",
            "group_hex": "d",
            "tag": "evaluation-parse-count-a",
            "service_scores": (("shared-service", 0.8),),
        },
        {
            "case_hex": "e",
            "group_hex": "f",
            "tag": "evaluation-parse-count-b",
            "service_scores": (("shared-service", 0.2),),
        },
    )
    library = _library(*entries, authority=authority)
    bundle = _trusted_bundle_for_entries(authority, entries)
    parse_counts = _install_typed_document_parse_counter(monkeypatch, module)

    module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=bundle,
    )

    _assert_trusted_bundle_artifacts_parsed_once(module, bundle, parse_counts)
    assert (
        parse_counts[
            (
                module.HistoryGenerationBatchManifest,
                bundle.canonical_batch_manifest,
            )
        ]
        == 1
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("source_opaque_case_id", "opq_" + "c" * 32),
        ("source_group_fingerprint", "e" * 64),
        ("source_completed_at_unix_ns", 11),
        ("embedding", (0.8, 0.6)),
    ),
)
def test_typed_manifest_rejects_unlisted_entry_behavior(
    field_name: str,
    changed_value: object,
) -> None:
    module = history_scoring_module()
    material = _typed_entry_materials(tag=f"entry-binding-{field_name}")
    manifest = _typed_batch_manifest_bytes(material)
    authority, external_anchor = _build_authority_for_typed_manifest(manifest)
    membership = material["membership"]
    score_documents = material["score_documents"]
    score_bytes = material["score_bytes"]
    sources = tuple(
        module.HistoryTrustedBaseScoreSource(
            source_service_key=document["source_service_key"],
            base_score_artifact=raw,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    scores = tuple(
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=document["base_history_score"],
            base_score_artifact=raw,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    claims = {
        "dataset": membership["dataset"],
        "source_partition": membership["source_partition"],
        "source_opaque_case_id": membership["source_opaque_case_id"],
        "source_group_fingerprint": membership["source_group_fingerprint"],
        "source_completed_at_unix_ns": membership["source_completed_at_unix_ns"],
        "embedding_policy": membership["embedding_policy"],
        "embedding_model": membership["embedding_model"],
        "embedding": membership["embedding"],
    }
    claims[field_name] = changed_value

    with pytest.raises(ValueError, match="^history library entry input verification failed$"):
        module.build_history_library_entry(
            **claims,
            sanitized_semantic_payload=material["semantic_bytes"],
            sanitized_graph=material["graph_bytes"],
            sanitized_topology=material["topology_bytes"],
            base_candidate_scores=scores,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
            trusted_base_score_sources=sources,
        )


def test_persisted_score_and_entry_carry_exact_manifest_membership_keys() -> None:
    material = _typed_entry_materials(tag="persisted-membership")
    manifest = _typed_batch_manifest_bytes(material)
    authority, external_anchor = _build_authority_for_typed_manifest(manifest)
    entry, _ = _build_entry_from_typed_material(
        material,
        canonical_batch_manifest=manifest,
        authority=authority,
        external_anchor=external_anchor,
    )
    membership = material["membership"]

    assert entry.manifest_entry_key == material["manifest_entry_key"]
    assert tuple(score.manifest_entry_key for score in entry.base_candidate_scores) == (
        material["manifest_entry_key"],
        material["manifest_entry_key"],
    )
    assert tuple(score.manifest_base_score_key for score in entry.base_candidate_scores) == tuple(
        item["manifest_base_score_key"] for item in membership["base_scores"]
    )


def test_library_requires_exact_manifest_entry_bijection() -> None:
    module = history_scoring_module()
    first_material = _typed_entry_materials(case_hex="b", tag="bijection-a")
    second_material = _typed_entry_materials(
        case_hex="e",
        group_hex="f",
        tag="bijection-b",
    )
    manifest = _typed_batch_manifest_bytes(first_material, second_material)
    authority, external_anchor = _build_authority_for_typed_manifest(manifest)
    first, first_sources = _build_entry_from_typed_material(
        first_material,
        canonical_batch_manifest=manifest,
        authority=authority,
        external_anchor=external_anchor,
    )
    first_source = module.HistoryTrustedEntrySource(
        entry_id=first.entry_id,
        base_score_sources=first_sources,
        sanitized_semantic_payload=first_material["semantic_bytes"],
        sanitized_graph=first_material["graph_bytes"],
        sanitized_topology=first_material["topology_bytes"],
    )
    bundle = module.HistoryTrustedSourceBundle(
        generation_authority=authority,
        expected_generation_authority_sha256=(external_anchor.history_generation_authority_sha256),
        canonical_batch_manifest=manifest,
        entry_sources=(first_source,),
    )

    with pytest.raises(ValueError, match="^history library input verification failed$"):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(first,),
            generation_authority=authority,
            trusted_source_bundle=bundle,
        )


@pytest.mark.parametrize(
    "extra_name",
    ("ground-truth", "root_cause", "ground_truth", "answer", "arbitrary_extra"),
)
def test_typed_generation_manifest_forbids_every_extra_field(
    extra_name: str,
) -> None:
    material = _typed_entry_materials(tag=f"manifest-extra-{extra_name}")
    manifest = _typed_batch_manifest_bytes(material, extra_field=(extra_name, "forbidden"))

    with pytest.raises(
        ValueError,
        match="^history generation authority input verification failed$",
    ):
        _build_authority_for_typed_manifest(manifest)


@pytest.mark.parametrize(
    ("artifact_kind", "extra_name"),
    (
        ("score-a", "ground-truth"),
        ("semantic", "root_cause"),
        ("graph", "ground_truth"),
        ("topology", "arbitrary_extra"),
    ),
)
def test_typed_history_artifacts_forbid_every_extra_field(
    artifact_kind: str,
    extra_name: str,
) -> None:
    module = history_scoring_module()
    material = _typed_entry_materials(
        tag=f"artifact-extra-{artifact_kind}",
        artifact_extra=(artifact_kind, extra_name, "forbidden"),
    )
    manifest = _typed_batch_manifest_bytes(material)
    authority, external_anchor = _build_authority_for_typed_manifest(manifest)
    if artifact_kind == "score-a":
        document = material["score_documents"][0]
        with pytest.raises(ValueError, match="^history base score input verification failed$"):
            module.build_history_base_candidate_score(
                source_service_key=document["source_service_key"],
                base_history_score=document["base_history_score"],
                base_score_artifact=material["score_bytes"][0],
                generation_authority=authority,
                expected_generation_authority_sha256=(
                    external_anchor.history_generation_authority_sha256
                ),
                canonical_batch_manifest=manifest,
            )
        return

    score_documents = material["score_documents"]
    score_bytes = material["score_bytes"]
    sources = tuple(
        module.HistoryTrustedBaseScoreSource(
            source_service_key=document["source_service_key"],
            base_score_artifact=raw,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    scores = tuple(
        module.build_history_base_candidate_score(
            source_service_key=document["source_service_key"],
            base_history_score=document["base_history_score"],
            base_score_artifact=raw,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
        )
        for document, raw in zip(score_documents, score_bytes, strict=True)
    )
    membership = material["membership"]
    with pytest.raises(ValueError, match="^history library entry input verification failed$"):
        module.build_history_library_entry(
            dataset=membership["dataset"],
            source_partition=membership["source_partition"],
            source_opaque_case_id=membership["source_opaque_case_id"],
            source_group_fingerprint=membership["source_group_fingerprint"],
            source_completed_at_unix_ns=membership["source_completed_at_unix_ns"],
            embedding_policy=membership["embedding_policy"],
            embedding_model=membership["embedding_model"],
            embedding=membership["embedding"],
            sanitized_semantic_payload=material["semantic_bytes"],
            sanitized_graph=material["graph_bytes"],
            sanitized_topology=material["topology_bytes"],
            base_candidate_scores=scores,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                external_anchor.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
            trusted_base_score_sources=sources,
        )


def test_typed_generation_manifest_requires_one_library_embedding_contract() -> None:
    first_material = _typed_entry_materials(case_hex="b", tag="embedding-contract-a")
    second_material = _typed_entry_materials(
        case_hex="e",
        group_hex="f",
        tag="embedding-contract-b",
    )
    changed_membership = dict(second_material["membership"])
    changed_membership["embedding_model"] = "embedding-v2"
    changed_second_material = dict(second_material)
    changed_second_material["membership"] = changed_membership
    manifest = _typed_batch_manifest_bytes(first_material, changed_second_material)

    with pytest.raises(
        ValueError,
        match="^history generation authority input verification failed$",
    ):
        _build_authority_for_typed_manifest(manifest)


@pytest.mark.parametrize(
    ("document_kind", "model_name"),
    (
        ("batch", "HistoryGenerationBatchManifest"),
        ("entry", "HistoryGenerationEntryMembership"),
        ("membership", "HistoryGenerationBaseScoreMembership"),
        ("score", "HistorySanitizedBaseScoreArtifact"),
        ("semantic", "HistorySanitizedSemanticArtifact"),
        ("graph", "HistorySanitizedGraphArtifact"),
        ("topology", "HistorySanitizedTopologyArtifact"),
    ),
)
def test_every_external_history_schema_has_a_strict_canonical_round_trip(
    document_kind: str,
    model_name: str,
) -> None:
    module = history_scoring_module()
    material = _typed_entry_materials(tag=f"round-trip-{document_kind}")
    membership = material["membership"]
    raw_by_kind = {
        "batch": _typed_batch_manifest_bytes(material),
        "entry": canonical_json_bytes(membership),
        "membership": canonical_json_bytes(membership["base_scores"][0]),
        "score": material["score_bytes"][0],
        "semantic": material["semantic_bytes"],
        "graph": material["graph_bytes"],
        "topology": material["topology_bytes"],
    }
    raw = raw_by_kind[document_kind]
    model_type = getattr(module, model_name)

    parsed, digest = module._parse_canonical_typed_document(raw, model_type)

    assert type(parsed) is model_type
    assert digest == sha256_hex(raw)
    assert canonical_json_bytes(parsed.model_dump(mode="json", warnings="error")) == raw


def test_trusted_source_dataclasses_are_frozen_exact_and_nonpersistent() -> None:
    _, _, base_sources, _, _, entry_source, bundle, library = _trusted_components()

    assert type(base_sources) is tuple
    assert type(entry_source.base_score_sources) is tuple
    assert type(bundle.entry_sources) is tuple
    assert type(base_sources[0].base_score_artifact) is bytes
    assert type(entry_source.sanitized_semantic_payload) is bytes
    assert not hasattr(bundle, "model_dump")
    with pytest.raises(FrozenInstanceError):
        base_sources[0].source_service_key = "changed"
    persisted = json.dumps(library.model_dump(mode="json"), sort_keys=True)
    assert "artifact_token" not in persisted
    assert "history-batch-trusted-v1" not in persisted


@pytest.mark.parametrize(
    "artifact_kind",
    ("base_score", "semantic", "graph", "topology", "manifest"),
)
def test_trusted_source_dataclasses_reject_individually_oversized_bytes(
    artifact_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority = _generation_authority("dataclass-byte-cap")
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority.generation_authority_sha256]
    valid_base_source = module.HistoryTrustedBaseScoreSource(
        source_service_key="source-dataclass-byte-cap",
        base_score_artifact=b"{}",
    )
    oversized = b"12345"
    monkeypatch.setattr(module, "_MAX_HISTORY_ARTIFACT_BYTES", 4)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        if artifact_kind == "base_score":
            module.HistoryTrustedBaseScoreSource(
                source_service_key="source-dataclass-byte-cap",
                base_score_artifact=oversized,
            )
        elif artifact_kind == "manifest":
            module.HistoryTrustedSourceBundle(
                generation_authority=authority,
                expected_generation_authority_sha256=(
                    frozen_run_manifest.history_generation_authority_sha256
                ),
                canonical_batch_manifest=oversized,
                entry_sources=(),
            )
        else:
            artifacts = {
                "semantic": (oversized, b"{}", b"{}"),
                "graph": (b"{}", oversized, b"{}"),
                "topology": (b"{}", b"{}", oversized),
            }[artifact_kind]
            module.HistoryTrustedEntrySource(
                entry_id="hent_" + "a" * 24,
                base_score_sources=(valid_base_source,),
                sanitized_semantic_payload=artifacts[0],
                sanitized_graph=artifacts[1],
                sanitized_topology=artifacts[2],
            )


def test_trusted_entry_source_rejects_base_source_count_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    source = module.HistoryTrustedBaseScoreSource(
        source_service_key="source-entry-source-cap",
        base_score_artifact=b"{}",
    )
    monkeypatch.setattr(module, "_MAX_HISTORY_BASE_SCORES", 2)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.HistoryTrustedEntrySource(
            entry_id="hent_" + "a" * 24,
            base_score_sources=(source,) * 3,
            sanitized_semantic_payload=b"{}",
            sanitized_graph=b"{}",
            sanitized_topology=b"{}",
        )


def test_trusted_source_bundle_rejects_4097_entries_before_iteration() -> None:
    module = history_scoring_module()
    authority = _generation_authority("bundle-entry-cap")
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority.generation_authority_sha256]
    source = module.HistoryTrustedBaseScoreSource(
        source_service_key="source-bundle-entry-cap",
        base_score_artifact=b"{}",
    )
    entry_source = module.HistoryTrustedEntrySource(
        entry_id="hent_" + "a" * 24,
        base_score_sources=(source,),
        sanitized_semantic_payload=b"{}",
        sanitized_graph=b"{}",
        sanitized_topology=b"{}",
    )

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.HistoryTrustedSourceBundle(
            generation_authority=authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=_batch_manifest("bundle-entry-cap"),
            entry_sources=(entry_source,) * 4_097,
        )


@pytest.mark.parametrize(
    "limit_name",
    (
        "_MAX_TRUSTED_BUNDLE_TOTAL_BASE_SOURCES",
        "_MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES",
        "_MAX_TRUSTED_BUNDLE_PROJECTED_WORK",
    ),
)
def test_trusted_source_bundle_rejects_aggregate_budget_before_replay(
    limit_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, _, _, _, entry, _, bundle, _ = _trusted_components("aggregate-cap")
    monkeypatch.setattr(module, limit_name, 1, raising=False)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(entry,),
            generation_authority=authority,
            trusted_source_bundle=bundle,
        )


@pytest.mark.parametrize("trap_kind", ("parser", "callbacks"))
def test_trusted_source_bundle_aggregate_budget_precedes_replay_callbacks(
    trap_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, _, _, _, entry, _, bundle, _ = _trusted_components("aggregate-trap")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> bool:
        nonlocal called
        called = True
        raise AssertionError("trusted bundle replay ran before aggregate budget preflight")

    monkeypatch.setattr(module, "_MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES", 1, raising=False)
    if trap_kind == "parser":
        monkeypatch.setattr(module.json, "loads", forbidden)
    else:
        monkeypatch.setattr(module.HistoryGenerationAuthority, "verify_trusted", forbidden)
        monkeypatch.setattr(module.FrozenHistoryLibrary, "verify", forbidden)
        monkeypatch.setattr(module, "_verify_trusted_library_entry", forbidden)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(entry,),
            generation_authority=authority,
            trusted_source_bundle=bundle,
        )
    assert called is False


def test_score_and_entry_builders_replay_trusted_raw_sources() -> None:
    module = history_scoring_module()
    authority = _generation_authority("trusted")
    manifest = _batch_manifest("trusted")
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority.generation_authority_sha256]
    source = module.HistoryTrustedBaseScoreSource(
        source_service_key="source-trusted-a",
        base_score_artifact=canonical_json_bytes({"artifact_token": "base-trusted-a-v1"}),
    )

    with pytest.raises(ValueError, match="^history base score input verification failed$"):
        module.build_history_base_candidate_score(
            source_service_key=source.source_service_key,
            base_history_score=0.8,
            base_score_artifact=source.base_score_artifact,
            generation_authority=authority,
            expected_generation_authority_sha256="0" * 64,
            canonical_batch_manifest=manifest,
        )

    _, _, base_sources, scores, _, _, _, _ = _trusted_components()
    forged_sources = (
        module.HistoryTrustedBaseScoreSource(
            source_service_key=base_sources[0].source_service_key,
            base_score_artifact=canonical_json_bytes(
                {"artifact_token": "base-trusted-a-forged-v1"}
            ),
        ),
        base_sources[1],
    )
    with pytest.raises(ValueError, match="^history library entry input verification failed$"):
        module.build_history_library_entry(
            dataset="rcabench",
            source_partition="dev",
            source_opaque_case_id="opq_" + "b" * 32,
            source_group_fingerprint="d" * 64,
            source_completed_at_unix_ns=10,
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding=(1.0, 0.0),
            sanitized_semantic_payload=canonical_json_bytes(
                {"artifact_token": "semantic-trusted-v1"}
            ),
            sanitized_graph=canonical_json_bytes({"artifact_token": "graph-trusted-v1"}),
            sanitized_topology=canonical_json_bytes({"artifact_token": "topology-trusted-v1"}),
            base_candidate_scores=scores,
            generation_authority=authority,
            expected_generation_authority_sha256=(
                frozen_run_manifest.history_generation_authority_sha256
            ),
            canonical_batch_manifest=manifest,
            trusted_base_score_sources=forged_sources,
        )


@pytest.mark.parametrize("coverage", ["missing", "extra", "duplicate"])
def test_library_builder_rejects_nonbijective_trusted_entry_coverage(
    coverage: str,
) -> None:
    module = history_scoring_module()
    authority, _, _, _, entry, entry_source, bundle, _ = _trusted_components()
    if coverage == "missing":
        entry_sources = ()
    elif coverage == "duplicate":
        entry_sources = (entry_source, entry_source)
    else:
        entry_sources = (
            entry_source,
            module.HistoryTrustedEntrySource(
                entry_id="hent_" + "f" * 24,
                base_score_sources=entry_source.base_score_sources,
                sanitized_semantic_payload=entry_source.sanitized_semantic_payload,
                sanitized_graph=entry_source.sanitized_graph,
                sanitized_topology=entry_source.sanitized_topology,
            ),
        )
    malformed = module.HistoryTrustedSourceBundle(
        generation_authority=authority,
        expected_generation_authority_sha256=(
            _TRUSTED_RUN_MANIFESTS[
                authority.generation_authority_sha256
            ].history_generation_authority_sha256
        ),
        canonical_batch_manifest=bundle.canonical_batch_manifest,
        entry_sources=entry_sources,
    )

    with pytest.raises(ValueError, match="^history library input verification failed$"):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(entry,),
            generation_authority=authority,
            trusted_source_bundle=malformed,
        )


def test_enabled_evaluation_and_result_trusted_replay_require_source_bundle() -> None:
    module = history_scoring_module()
    _, _, _, _, entry, _, bundle, library = _trusted_components()
    frozen = _frozen_candidates()
    query = _query()
    mapping = _mapping(entry, source="source-trusted-a", current="service-a")
    result = module.evaluate_history(
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(mapping,),
        trusted_source_bundle=bundle,
    )

    assert result.verify_trusted(
        expected_result_sha256=result.result_sha256,
        expected_authoritative_input_sha256=result.authoritative_input_sha256,
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(mapping,),
        trusted_source_bundle=bundle,
    )
    with pytest.raises(ValueError, match="^history scoring input verification failed$"):
        module.evaluate_history(
            frozen_candidates=frozen,
            mode="score",
            library=library,
            query_embedding=query,
            service_mappings=(mapping,),
        )
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "artifact_token" not in serialized
    assert "history-batch-trusted-v1" not in serialized
    assert all("artifact_token" not in repr(item) for item in result.hint_references)


def test_self_signed_library_fails_against_independent_bundle_anchor() -> None:
    module = history_scoring_module()
    legitimate_authority, _, _, _, _, _, _, _ = _trusted_components("legitimate")
    (
        forged_authority,
        _,
        _,
        _,
        forged_entry,
        forged_entry_source,
        forged_bundle,
        forged_library,
    ) = _trusted_components("forged")
    legitimate_run_manifest = _TRUSTED_RUN_MANIFESTS[
        legitimate_authority.generation_authority_sha256
    ]
    independently_anchored_bundle = module.HistoryTrustedSourceBundle(
        generation_authority=forged_authority,
        expected_generation_authority_sha256=(
            legitimate_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=forged_bundle.canonical_batch_manifest,
        entry_sources=(forged_entry_source,),
    )

    with pytest.raises(ValueError, match="^history library input verification failed$"):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(forged_entry,),
            generation_authority=forged_authority,
            trusted_source_bundle=independently_anchored_bundle,
        )
    with pytest.raises(ValueError, match="^history scoring input verification failed$"):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="hint",
            library=forged_library,
            query_embedding=_query(),
            trusted_source_bundle=independently_anchored_bundle,
        )

    forged_result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=forged_library,
        query_embedding=_query(),
        trusted_source_bundle=forged_bundle,
    )
    assert not forged_result.verify_trusted(
        expected_result_sha256=forged_result.result_sha256,
        expected_authoritative_input_sha256=forged_result.authoritative_input_sha256,
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=forged_library,
        query_embedding=_query(),
        trusted_source_bundle=independently_anchored_bundle,
    )


def test_hint_output_keeps_trusted_raw_bytes_out_of_persistence() -> None:
    module = history_scoring_module()
    _, _, _, _, _, _, bundle, library = _trusted_components()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        trusted_source_bundle=bundle,
    )

    assert result.hint_references
    serialized = result.model_dump_json()
    assert "artifact_token" not in serialized
    assert "history-batch-trusted-v1" not in serialized
    assert all("artifact_token" not in repr(item) for item in result.hint_references)


@pytest.mark.parametrize(
    ("model_name", "field_name", "bad_value"),
    [
        ("HistoryScoreContribution", "selected_neighbor_id", "../case"),
        ("HistoryScoreContribution", "mapping_id", "../case"),
        ("HistoryNeighborAudit", "entry_id", "../case"),
        ("HistoryHintReference", "selected_neighbor_id", "../case"),
    ],
)
def test_output_reference_models_reject_noncanonical_identifier_grammar(
    model_name: str,
    field_name: str,
    bad_value: str,
) -> None:
    module = history_scoring_module()
    _, _, _, _, entry, _, bundle, library = _trusted_components()
    mapping = _mapping(entry, source="source-trusted-a", current="service-a")
    mode = "score" if model_name == "HistoryScoreContribution" else "hint"
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode=mode,
        library=library,
        query_embedding=_query(),
        service_mappings=(mapping,) if mode == "score" else (),
        trusted_source_bundle=bundle,
    )
    if model_name == "HistoryScoreContribution":
        original = result.candidate_scores[0].contributions[0]
        id_field, sha_field, prefix = (
            "contribution_id",
            "contribution_sha256",
            "hcon",
        )
    elif model_name == "HistoryNeighborAudit":
        original = result.neighbor_audits[0]
        id_field, sha_field, prefix = "audit_id", "audit_sha256", "haud"
    else:
        original = result.hint_references[0]
        id_field, sha_field, prefix = (
            "hint_reference_id",
            "hint_reference_sha256",
            "hhint",
        )
    unsigned = original.model_dump(mode="json", exclude={id_field, sha_field})
    unsigned[field_name] = bad_value
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError):
        getattr(module, model_name).model_validate(
            {
                **unsigned,
                id_field: f"{prefix}_{digest[:24]}",
                sha_field: digest,
            }
        )


def test_verify_structure_rejects_resigned_path_bearing_output_references() -> None:
    module = history_scoring_module()
    _, _, _, _, _, _, bundle, library = _trusted_components()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        trusted_source_bundle=bundle,
    )
    path_reference = "../case"
    audit_unsigned = result.neighbor_audits[0].model_dump(
        mode="json",
        exclude={"audit_id", "audit_sha256"},
    )
    audit_unsigned["entry_id"] = path_reference
    audit_id, audit_sha256 = module._stable_id("haud", audit_unsigned)
    forged_audit = result.neighbor_audits[0].model_copy(
        update={
            "entry_id": path_reference,
            "audit_id": audit_id,
            "audit_sha256": audit_sha256,
        }
    )
    hint_unsigned = result.hint_references[0].model_dump(
        mode="json",
        exclude={"hint_reference_id", "hint_reference_sha256"},
    )
    hint_unsigned["selected_neighbor_id"] = path_reference
    hint_id, hint_sha256 = module._stable_id("hhint", hint_unsigned)
    forged_hint = result.hint_references[0].model_copy(
        update={
            "selected_neighbor_id": path_reference,
            "hint_reference_id": hint_id,
            "hint_reference_sha256": hint_sha256,
        }
    )
    result_unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    result_unsigned.update(
        neighbor_audits=(forged_audit.model_dump(mode="json"),),
        selected_neighbor_ids=(path_reference,),
        hint_references=(forged_hint.model_dump(mode="json"),),
    )
    result_id, result_sha256 = module._stable_id("heval", result_unsigned)
    forged = result.model_copy(
        update={
            "neighbor_audits": (forged_audit,),
            "selected_neighbor_ids": (path_reference,),
            "hint_references": (forged_hint,),
            "history_evaluation_result_id": result_id,
            "result_sha256": result_sha256,
        }
    )

    assert not forged.verify_structure()


def test_library_rejects_duplicate_dataset_source_case_identity() -> None:
    first_material = _typed_entry_materials(case_hex="b", group_hex="d", tag="duplicate-case-a")
    duplicate_source_case_material = _typed_entry_materials(
        case_hex="b", group_hex="e", tag="duplicate-case-b"
    )

    with pytest.raises(
        ValueError,
        match="^history generation authority input verification failed$",
    ):
        _register_generation_batch(first_material, duplicate_source_case_material)


@pytest.mark.parametrize("mutation", ["reversed", "duplicate"])
def test_verify_structure_rejects_resigned_noncanonical_mapping_digests(
    mutation: str,
) -> None:
    module = history_scoring_module()
    _, _, _, _, entry, _, bundle, library = _trusted_components()
    mappings = (
        _mapping(entry, source="source-trusted-a", current="service-a"),
        _mapping(entry, source="source-trusted-b", current="service-b"),
    )
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=library,
        query_embedding=_query(),
        service_mappings=mappings,
        trusted_source_bundle=bundle,
    )
    if mutation == "reversed":
        mapping_sha256s = tuple(reversed(result.service_mapping_sha256s))
    else:
        mapping_sha256s = (result.service_mapping_sha256s[0],) * 2
    assert mapping_sha256s != result.service_mapping_sha256s
    authoritative_input_sha256 = module._authoritative_input_sha256(
        candidate_binding_sha256=result.candidate_binding_sha256,
        mode=result.mode,
        generation_authority_sha256=result.generation_authority_sha256,
        library_sha256=result.library_sha256,
        query_embedding_sha256=result.query_embedding_sha256,
        service_mapping_sha256s=mapping_sha256s,
        config_sha256=result.config_sha256,
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned.update(
        service_mapping_sha256s=mapping_sha256s,
        authoritative_input_sha256=authoritative_input_sha256,
    )
    result_id, result_sha256 = module._stable_id("heval", unsigned)
    forged = result.model_copy(
        update={
            "service_mapping_sha256s": mapping_sha256s,
            "authoritative_input_sha256": authoritative_input_sha256,
            "history_evaluation_result_id": result_id,
            "result_sha256": result_sha256,
        }
    )

    assert not forged.verify_structure()


@pytest.mark.parametrize(
    "artifact",
    [
        b"[" * 1_100 + b"0" + b"]" * 1_100,
        b'"' + b"x" * 8_193 + b'"',
    ],
)
def test_artifact_lexical_budget_rejects_before_json_parser(
    artifact: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    called = False

    def forbidden_json_loads(_raw: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("json parser ran before lexical budget preflight")

    monkeypatch.setattr(module.json, "loads", forbidden_json_loads)
    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_history_generation_authority(
            dataset="rcabench",
            source_partition="dev",
            generation_policy="history-base-generation-v1",
            canonical_batch_manifest=artifact,
        )
    assert called is False


def test_json_parser_recursion_error_is_redacted() -> None:
    module = history_scoring_module()
    original_loads = module.json.loads

    def recursive_json_loads(_raw: object) -> object:
        raise RecursionError("parser internals must not escape")

    module.json.loads = recursive_json_loads
    try:
        with pytest.raises(
            ValueError,
            match="^history generation authority input verification failed$",
        ):
            module.build_history_generation_authority(
                dataset="rcabench",
                source_partition="dev",
                generation_policy="history-base-generation-v1",
                canonical_batch_manifest=canonical_json_bytes(
                    {"batch_token": "history-batch-parser-v1"}
                ),
            )
    finally:
        module.json.loads = original_loads


@pytest.mark.parametrize(
    "parsed_document",
    [
        "x" * 8_193,
        [[[[[[[[[[0]]]]]]]]]],
    ],
)
def test_artifact_postparse_tree_rechecks_resource_budget(
    parsed_document: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    if type(parsed_document) is list:
        for _ in range(module._SERIALIZATION_PREFLIGHT_MAX_DEPTH):
            parsed_document = [parsed_document]

    monkeypatch.setattr(module.json, "loads", lambda _raw: parsed_document)
    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_history_generation_authority(
            dataset="rcabench",
            source_partition="dev",
            generation_policy="history-base-generation-v1",
            canonical_batch_manifest=b"{}",
        )


@pytest.mark.parametrize(
    "manifest",
    [
        bytearray(canonical_json_bytes({"batch_token": "history-batch-a-v1"})),
        b'{"batch_token": "history-batch-a-v1"}',
        canonical_json_bytes({"answer": "service-a"}),
    ],
)
def test_generation_authority_rejects_noncanonical_or_forbidden_manifest_bytes(
    manifest: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="^history generation authority input verification failed$",
    ):
        history_scoring_module().build_history_generation_authority(
            dataset="rcabench",
            source_partition="dev",
            generation_policy="history-base-generation-v1",
            canonical_batch_manifest=manifest,
        )


def test_generation_authority_rejects_oversized_manifest_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    called = False

    def forbidden_json_loads(_raw: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("json parser ran before the typed artifact byte budget")

    monkeypatch.setattr(module.json, "loads", forbidden_json_loads)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_history_generation_authority(
            dataset="rcabench",
            source_partition="dev",
            generation_policy="history-base-generation-v1",
            canonical_batch_manifest=b"x" * (module._MAX_HISTORY_ARTIFACT_BYTES + 1),
        )
    assert called is False


def test_base_score_rejects_answer_bearing_canonical_artifact_bytes() -> None:
    with pytest.raises(
        ValueError,
        match="^history base score input verification failed$",
    ):
        history_scoring_module().build_history_base_candidate_score(
            source_service_key="source-a-a",
            base_history_score=0.8,
            base_score_artifact=canonical_json_bytes({"answer": "service-a"}),
            generation_authority=_generation_authority(),
        )


def test_entry_and_library_reject_mixed_generation_authority() -> None:
    module = history_scoring_module()
    authority_a = _generation_authority("a")
    authority_b = _generation_authority("b")
    scores_a = _base_scores(authority=authority_a)

    with pytest.raises(
        ValueError,
        match="^history library entry input verification failed$",
    ):
        module.build_history_library_entry(
            dataset="rcabench",
            source_partition="dev",
            source_opaque_case_id="opq_" + "b" * 32,
            source_group_fingerprint="d" * 64,
            source_completed_at_unix_ns=10,
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding=(1.0, 0.0),
            sanitized_semantic_payload=canonical_json_bytes({"artifact_token": "semantic-a-v1"}),
            sanitized_graph=canonical_json_bytes({"artifact_token": "graph-a-v1"}),
            sanitized_topology=canonical_json_bytes({"artifact_token": "topology-a-v1"}),
            base_candidate_scores=scores_a,
            generation_authority=authority_b,
        )

    entry_a = _entry(authority=authority_a)
    entry_b = _entry(tag="b", authority=authority_b)
    with pytest.raises(
        ValueError,
        match="^history library input verification failed$",
    ):
        _library(entry_a, entry_b, authority=authority_a)


def test_hint_mode_exposes_only_sanitized_digest_references_and_never_scores() -> None:
    module = history_scoring_module()
    entry = _entry()
    mapping = _mapping(entry, source="source-a-a", current="service-a")

    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(mapping,),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )

    assert result.verify_structure()
    assert result.claim_scope == "prior_history"
    assert result.selected_neighbor_ids == (entry.entry_id,)
    assert tuple(item.status for item in result.neighbor_audits) == ("selected",)
    assert len(result.hint_references) == 1
    trusted_entry_source = _TRUSTED_ENTRY_SOURCES[entry.entry_id]
    assert result.hint_references[0].semantic_payload_sha256 == sha256_hex(
        trusted_entry_source.sanitized_semantic_payload
    )
    assert result.hint_references[0].requires_current_case_reverification is True
    assert result.requires_current_case_reverification is True
    assert all(item.history_score == 0.0 for item in result.candidate_scores)
    assert all(not item.contributions for item in result.candidate_scores)
    serialized = result.model_dump_json()
    assert "source-a-a" not in serialized
    assert "mapping:fixture-v1" not in serialized


def test_score_mode_uses_normalized_cosine_and_exact_candidate_mappings() -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "embedding": (1.0, 0.0), "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    mappings = (
        _mapping(first, source="source-a-a", current="service-a"),
        _mapping(first, source="source-a-b", current="service-b"),
        _mapping(second, source="source-b-a", current="service-a"),
        _mapping(second, source="source-b-b", current="service-c"),
    )

    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=library,
        query_embedding=_query(),
        service_mappings=mappings,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )

    scores = {item.candidate_service_key: item for item in result.candidate_scores}
    assert result.verify_structure()
    assert result.hint_references == ()
    assert result.requires_current_case_reverification is False
    assert result.selected_neighbor_ids == (first.entry_id, second.entry_id)
    assert scores["service-a"].history_score == pytest.approx(0.8)
    assert scores["service-b"].history_score == pytest.approx(0.2 / 1.8)
    assert scores["service-c"].history_score == pytest.approx(0.2 * 0.8 / 1.8)
    assert all(len(item.contributions) == 2 for item in result.candidate_scores)
    assert tuple(item.mapping_status for item in scores["service-b"].contributions) == (
        "mapped",
        "missing_mapping",
    )
    assert all(0.0 <= item.history_score <= 1.0 for item in result.candidate_scores)


def test_self_and_current_group_are_excluded_before_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, (self_entry, group_entry) = _shared_entry_batch(
        {"case_hex": "a", "group_hex": "d", "tag": "a"},
        {"case_hex": "b", "group_hex": "c", "tag": "b"},
    )
    library = _library(self_entry, group_entry, authority=authority)
    monkeypatch.setattr(
        module,
        "_cosine_similarity",
        lambda *_args: pytest.fail("excluded entry reached cosine computation"),
    )

    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )

    status_by_entry = {item.entry_id: item.status for item in result.neighbor_audits}
    assert status_by_entry[self_entry.entry_id] == "self_excluded"
    assert status_by_entry[group_entry.entry_id] == "incident_group_excluded"
    assert result.selected_neighbor_ids == ()


def test_strict_prior_excludes_equal_time_while_offline_narrows_claim_scope() -> None:
    module = history_scoring_module()
    boundary = _entry(completed_at=20)
    strict = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=_library(boundary),
        query_embedding=_query(decision_time=20),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(_library(boundary)),
    )
    offline = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=_library(boundary),
        query_embedding=_query(decision_time=20),
        service_mappings=(),
        config=module.build_history_scoring_config(time_policy="offline_similarity"),
        trusted_source_bundle=_trusted_bundle_for_library(_library(boundary)),
    )

    assert strict.neighbor_audits[0].status == "time_excluded"
    assert strict.claim_scope == "prior_history"
    assert offline.neighbor_audits[0].status == "selected"
    assert offline.claim_scope == "offline_similarity_only"


def test_group_dedup_threshold_top_k_and_stable_entry_id_ties() -> None:
    module = history_scoring_module()
    authority, (tied_a, tied_b, same_group_lower, below) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "embedding": (1.0, 0.0), "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (1.0, 0.0), "tag": "b"},
        {"case_hex": "1", "group_hex": "d", "embedding": (0.8, 0.6), "tag": "c"},
        {"case_hex": "2", "group_hex": "2", "embedding": (0.6, 0.8), "tag": "d"},
    )
    library = _library(
        below,
        same_group_lower,
        tied_b,
        tied_a,
        authority=authority,
    )
    config = module.build_history_scoring_config(k=1, min_similarity=0.7)

    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )

    expected_winner = min((tied_a, tied_b), key=lambda item: item.entry_id)
    status_by_entry = {item.entry_id: item.status for item in result.neighbor_audits}
    assert result.selected_neighbor_ids == (expected_winner.entry_id,)
    assert status_by_entry[same_group_lower.entry_id] == "same_group_not_selected"
    assert status_by_entry[below.entry_id] == "below_similarity"
    other_tie = tied_b if expected_winner is tied_a else tied_a
    assert status_by_entry[other_tie.entry_id] == "top_k_not_selected"


def test_zero_total_selected_similarity_retains_zero_contributions() -> None:
    module = history_scoring_module()
    entry = _entry(embedding=(0.0, 1.0))
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(_mapping(entry, source="source-a-a", current="service-a"),),
        config=module.build_history_scoring_config(min_similarity=0.0),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )

    assert result.selected_neighbor_ids == (entry.entry_id,)
    assert result.neighbor_audits[0].normalized_similarity == 0.0
    assert all(item.history_score == 0.0 for item in result.candidate_scores)
    assert all(
        contribution.contribution == 0.0
        for score in result.candidate_scores
        for contribution in score.contributions
    )


@pytest.mark.parametrize(
    ("builder", "expected_message"),
    [
        ("entry", "history library entry input verification failed"),
        ("query", "history query input verification failed"),
    ],
)
def test_public_builders_redact_finite_embedding_norm_overflow(
    builder: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{expected_message}$") as captured:
        if builder == "entry":
            _entry(
                embedding=(1e154, 1e154),
                manifest_embedding=(1.0, 0.0),
            )
        else:
            _query(embedding=(1e154, 1e154))

    assert type(captured.value) is ValueError


@pytest.mark.parametrize("artifact_kind", ["entry", "query"])
def test_hostile_nested_embedding_overflow_is_redacted_at_public_boundary(
    artifact_kind: str,
) -> None:
    module = history_scoring_module()

    with pytest.raises(ValueError) as captured:
        if artifact_kind == "entry":
            hostile_entry = _entry().model_copy(update={"embedding": (1e154, 1e154)})
            module.build_frozen_history_library(
                dataset="rcabench",
                source_partition="dev",
                embedding_policy="semantic-v1",
                embedding_model="embedding-v1",
                embedding_dimension=2,
                entries=(hostile_entry,),
                generation_authority=_generation_authority(),
            )
        else:
            hostile_query = _query().model_copy(update={"embedding": (1e154, 1e154)})
            module.evaluate_history(
                frozen_candidates=_frozen_candidates(),
                mode="hint",
                library=_library(_entry()),
                query_embedding=hostile_query,
                service_mappings=(),
                trusted_source_bundle=_trusted_bundle_for_library(_library(_entry())),
            )

    assert type(captured.value) is ValueError
    assert str(captured.value) in {
        "history library input verification failed",
        "history scoring input verification failed",
    }


def test_persisted_embedding_validators_redact_finite_norm_overflow() -> None:
    for artifact in (_entry(), _query()):
        document = artifact.model_dump(mode="json")
        document["embedding"] = (1e154, 1e154)
        with pytest.raises(ValueError) as captured:
            type(artifact).model_validate(document)
        assert isinstance(captured.value, ValueError)


def test_input_reordering_does_not_change_result_or_mutate_the_library() -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    before = library.model_dump(mode="json")
    mappings = [
        _mapping(first, source="source-a-a", current="service-a"),
        _mapping(second, source="source-b-b", current="service-c"),
    ]

    left = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=library,
        query_embedding=_query(),
        service_mappings=mappings,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    right = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=_library(second, first, authority=authority),
        query_embedding=_query(),
        service_mappings=list(reversed(mappings)),
        trusted_source_bundle=_trusted_bundle_for_entries(authority, (second, first)),
    )

    assert left == right
    assert left.result_sha256 == right.result_sha256
    assert library.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "override",
    [
        {"source_partition": "validation"},
        {"source_partition": "final"},
        {"source_history_mode": "score"},
        {"source_rule_channel_enabled": True},
        {"label_free": False},
        {"contains_answers": True},
        {"contains_fault_metadata": True},
        {"contains_injection_metadata": True},
        {"contains_original_identifiers_or_paths": True},
    ],
)
def test_library_entry_rejects_nondevelopment_recursive_or_label_material(
    override: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "dataset": "rcabench",
        "source_partition": "dev",
        "source_opaque_case_id": "opq_" + "b" * 32,
        "source_group_fingerprint": "d" * 64,
        "source_completed_at_unix_ns": 10,
        "embedding_policy": "semantic-v1",
        "embedding_model": "embedding-v1",
        "embedding": (1.0, 0.0),
        "sanitized_semantic_payload": canonical_json_bytes({"artifact_token": "semantic-a-v1"}),
        "sanitized_graph": canonical_json_bytes({"artifact_token": "graph-a-v1"}),
        "sanitized_topology": canonical_json_bytes({"artifact_token": "topology-a-v1"}),
        "base_candidate_scores": _base_scores(),
        "generation_authority": _generation_authority(),
    }
    arguments.update(override)

    with pytest.raises(ValueError, match="history library entry input verification failed"):
        history_scoring_module().build_history_library_entry(**arguments)


def test_evaluation_rejects_contract_mismatch_and_illegal_mappings() -> None:
    module = history_scoring_module()
    entry = _entry()
    foreign = module.build_history_query_embedding(
        current_opaque_case_id="opq_" + "a" * 32,
        current_group_fingerprint="c" * 64,
        dataset="aiops25",
        current_decision_time_unix_ns=20,
        embedding_policy="semantic-v1",
        embedding_model="embedding-v1",
        embedding_dimension=2,
        embedding=(1.0, 0.0),
    )
    external = _mapping(entry, source="source-a-a", current="outside-c")
    duplicate_a = _mapping(entry, source="source-a-a", current="service-a")
    duplicate_b = _mapping(entry, source="source-a-b", current="service-a")

    for query, mappings in (
        (foreign, ()),
        (_query(), (external,)),
        (_query(), (duplicate_a, duplicate_b)),
    ):
        with pytest.raises(ValueError, match="history scoring input verification failed"):
            module.evaluate_history(
                frozen_candidates=_frozen_candidates(),
                mode="score",
                library=_library(entry),
                query_embedding=query,
                service_mappings=mappings,
                trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
            )


def test_hint_result_schema_rejects_a_consistently_resigned_nonzero_score() -> None:
    module = history_scoring_module()
    entry = _entry()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )
    candidate_scores = list(result.candidate_scores)
    score_result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(_mapping(entry, source="source-a-a", current="service-a"),),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )
    candidate_scores[0] = score_result.candidate_scores[0]
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["candidate_scores"] = tuple(item.model_dump(mode="json") for item in candidate_scores)
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": f"heval_{digest[:24]}",
                "result_sha256": digest,
            }
        )


def test_result_schema_rejects_resigned_noncanonical_neighbor_weights() -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "embedding": (1.0, 0.0), "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    audits = list(result.neighbor_audits)
    audit_unsigned = audits[0].model_dump(
        mode="json",
        exclude={"audit_id", "audit_sha256"},
    )
    audit_unsigned["normalized_similarity"] = 0.5
    audit_digest = sha256_hex(canonical_json_bytes(audit_unsigned))
    audits[0] = module.HistoryNeighborAudit.model_validate(
        {
            **audit_unsigned,
            "audit_id": f"haud_{audit_digest[:24]}",
            "audit_sha256": audit_digest,
        }
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["neighbor_audits"] = tuple(item.model_dump(mode="json") for item in audits)
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError, match="normalized similarity"):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": f"heval_{digest[:24]}",
                "result_sha256": digest,
            }
        )


def test_result_schema_rejects_resigned_contribution_detached_from_neighbor_audit() -> None:
    module = history_scoring_module()
    entry = _entry()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(_mapping(entry, source="source-a-a", current="service-a"),),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )
    candidate_scores = list(result.candidate_scores)
    contribution_unsigned = (
        candidate_scores[0]
        .contributions[0]
        .model_dump(
            mode="json",
            exclude={"contribution_id", "contribution_sha256"},
        )
    )
    contribution_unsigned["normalized_similarity"] = 0.5
    contribution_unsigned["contribution"] = 0.4
    contribution_digest = sha256_hex(canonical_json_bytes(contribution_unsigned))
    contribution = module.HistoryScoreContribution.model_validate(
        {
            **contribution_unsigned,
            "contribution_id": f"hcon_{contribution_digest[:24]}",
            "contribution_sha256": contribution_digest,
        }
    )
    score_unsigned = candidate_scores[0].model_dump(
        mode="json",
        exclude={"candidate_history_score_id", "candidate_history_score_sha256"},
    )
    score_unsigned["contributions"] = (contribution.model_dump(mode="json"),)
    score_unsigned["history_score"] = 0.4
    score_digest = sha256_hex(canonical_json_bytes(score_unsigned))
    candidate_scores[0] = module.CandidateHistoryScore.model_validate(
        {
            **score_unsigned,
            "candidate_history_score_id": f"chs_{score_digest[:24]}",
            "candidate_history_score_sha256": score_digest,
        }
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["candidate_scores"] = tuple(item.model_dump(mode="json") for item in candidate_scores)
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError, match="neighbor audit"):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": f"heval_{digest[:24]}",
                "result_sha256": digest,
            }
        )


def test_verify_structure_rejects_resigned_selected_negative_similarity() -> None:
    module = history_scoring_module()
    entry = _entry()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=_library(entry),
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
    )
    audit_unsigned = result.neighbor_audits[0].model_dump(
        mode="json",
        exclude={"audit_id", "audit_sha256"},
    )
    audit_unsigned.update(similarity=-0.5, normalized_similarity=0.0)
    audit_digest = sha256_hex(canonical_json_bytes(audit_unsigned))
    forged_audit = result.neighbor_audits[0].model_copy(
        update={
            "similarity": -0.5,
            "normalized_similarity": 0.0,
            "audit_id": f"haud_{audit_digest[:24]}",
            "audit_sha256": audit_digest,
        }
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["neighbor_audits"] = (forged_audit.model_dump(mode="json"),)
    digest = sha256_hex(canonical_json_bytes(unsigned))
    forged = result.model_copy(
        update={
            "neighbor_audits": (forged_audit,),
            "history_evaluation_result_id": f"heval_{digest[:24]}",
            "result_sha256": digest,
        }
    )

    assert not forged.verify_structure()


def test_verify_structure_rejects_resigned_duplicate_selected_source_group() -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "embedding": (1.0, 0.0), "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    audits = list(result.neighbor_audits)
    audit_unsigned = audits[1].model_dump(
        mode="json",
        exclude={"audit_id", "audit_sha256"},
    )
    audit_unsigned["source_group_fingerprint"] = audits[0].source_group_fingerprint
    audit_digest = sha256_hex(canonical_json_bytes(audit_unsigned))
    audits[1] = audits[1].model_copy(
        update={
            "source_group_fingerprint": audits[0].source_group_fingerprint,
            "audit_id": f"haud_{audit_digest[:24]}",
            "audit_sha256": audit_digest,
        }
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["neighbor_audits"] = tuple(item.model_dump(mode="json") for item in audits)
    digest = sha256_hex(canonical_json_bytes(unsigned))
    forged = result.model_copy(
        update={
            "neighbor_audits": tuple(audits),
            "history_evaluation_result_id": f"heval_{digest[:24]}",
            "result_sha256": digest,
        }
    )

    assert not forged.verify_structure()


def test_result_supports_structural_and_externally_anchored_full_replay() -> None:
    module = history_scoring_module()
    entry = _entry()
    frozen = _frozen_candidates()
    library = _library(entry)
    query = _query()
    mapping = _mapping(entry, source="source-a-a", current="service-a")
    config = module.build_history_scoring_config()
    result = module.evaluate_history(
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(mapping,),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )

    assert result.verify_trusted(
        expected_result_sha256=result.result_sha256,
        expected_authoritative_input_sha256=result.authoritative_input_sha256,
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(mapping,),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    assert not result.verify_trusted(
        expected_result_sha256="0" * 64,
        expected_authoritative_input_sha256=result.authoritative_input_sha256,
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(mapping,),
        config=config,
    )


def test_whole_tree_consistent_resign_fails_against_external_digest_anchors() -> None:
    module = history_scoring_module()
    entry = _entry()
    frozen = _frozen_candidates()
    library = _library(entry)
    query = _query()
    config = module.build_history_scoring_config()
    original_mapping = _mapping(entry, source="source-a-a", current="service-a")
    forged_mapping = _mapping(entry, source="source-a-b", current="service-a")
    original = module.evaluate_history(
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(original_mapping,),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    resigned = module.evaluate_history(
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(forged_mapping,),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )

    assert resigned.verify_structure()
    assert not resigned.verify_trusted(
        expected_result_sha256=original.result_sha256,
        expected_authoritative_input_sha256=original.authoritative_input_sha256,
        frozen_candidates=frozen,
        mode="score",
        library=library,
        query_embedding=query,
        service_mappings=(forged_mapping,),
        config=config,
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )


def test_nested_model_subclass_is_rejected_before_verification_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    entry = _entry()

    class EntrySubclass(module.HistoryLibraryEntry):
        pass

    nested_subclass = EntrySubclass.model_validate(entry.model_dump(mode="json"))
    hostile_library = _library(entry).model_copy(update={"entries": (nested_subclass,)})
    called = False

    def forbidden_verify(_value: object) -> bool:
        nonlocal called
        called = True
        raise AssertionError("library verify callback ran before exact-tree preflight")

    monkeypatch.setattr(module.FrozenHistoryLibrary, "verify", forbidden_verify)
    with pytest.raises(ValueError, match="history scoring input verification failed"):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="hint",
            library=hostile_library,
            query_embedding=_query(),
            service_mappings=(),
            trusted_source_bundle=_trusted_bundle_for_library(hostile_library),
        )
    assert called is False


def test_result_structure_rejects_a_model_subclass() -> None:
    module = history_scoring_module()
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        service_mappings=(),
    )

    class ResultSubclass(module.HistoryEvaluationResult):
        pass

    subclass = ResultSubclass.model_validate(result.model_dump(mode="json"))
    assert not module.HistoryEvaluationResult.verify_structure(subclass)


def test_mapping_quantity_budget_has_a_fixed_redacted_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    entry = _entry()
    mapping = _mapping(entry, source="source-a-a", current="service-a")
    monkeypatch.setattr(module, "_MAX_HISTORY_MAPPINGS", 0)

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ) as captured:
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="score",
            library=_library(entry),
            query_embedding=_query(),
            service_mappings=(mapping,),
            trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
        )
    assert str(captured.value) == "history scoring budget exceeded"


def test_registered_persisted_collections_enforce_schema_max_length() -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    base_score = _base_scores(authority=authority)[0]
    entry = _entry(authority=authority)
    library = _library(entry, authority=authority)
    mapping = _mapping(entry, source="source-a-a", current="service-a")
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="score",
        library=library,
        query_embedding=_query(),
        service_mappings=(mapping,),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    candidate_score = result.candidate_scores[0]
    contribution = candidate_score.contributions[0]
    cases = (
        (
            entry,
            "base_candidate_scores",
            (base_score,) * (module._MAX_HISTORY_BASE_SCORES + 1),
        ),
        (library, "entries", (entry,) * (module._MAX_HISTORY_ENTRIES + 1)),
        (
            mapping,
            "provenance_ids",
            ("mapping:fixture-v1",) * (module._MAX_PROVENANCE_IDS + 1),
        ),
        (
            candidate_score,
            "contributions",
            (contribution,) * (module._MAX_HISTORY_ENTRIES + 1),
        ),
        (
            result,
            "neighbor_audits",
            (result.neighbor_audits[0],) * (module._MAX_HISTORY_ENTRIES + 1),
        ),
        (
            result,
            "hint_references",
            (module._build_hint_reference(entry),) * (module._MAX_HISTORY_ENTRIES + 1),
        ),
        (
            result,
            "selected_neighbor_ids",
            (entry.entry_id,) * (module._MAX_HISTORY_ENTRIES + 1),
        ),
        (
            result,
            "service_mapping_sha256s",
            (mapping.mapping_sha256,) * (module._MAX_HISTORY_MAPPINGS + 1),
        ),
        (
            result,
            "candidate_service_keys",
            ("service-a",) * (module._MAX_HISTORY_CANDIDATES + 1),
        ),
        (
            result,
            "candidate_scores",
            (candidate_score,) * (module._MAX_HISTORY_CANDIDATES + 1),
        ),
    )

    for model, field_name, oversized in cases:
        document = model.model_dump(mode="json")
        document[field_name] = oversized
        with pytest.raises(ValueError) as captured:
            type(model).model_validate(document)
        errors = captured.value.errors()
        assert any(
            error["type"] == "too_long" and error["loc"] == (field_name,) for error in errors
        ), field_name


@pytest.mark.parametrize("collision_kind", ("prefix", "full"))
def test_entry_schema_rejects_base_score_identity_and_hash_collisions(
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    entry = _entry(authority=authority)
    _install_digest_collision(
        monkeypatch,
        module,
        schema_version=module.HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION,
        collision_kind=collision_kind,
    )
    base_scores = _base_scores(authority=authority)

    assert all(score.verify() for score in base_scores)
    assert len({score.base_candidate_score_id for score in base_scores}) == 1
    expected_sha_count = 2 if collision_kind == "prefix" else 1
    assert len({score.base_candidate_score_sha256 for score in base_scores}) == (expected_sha_count)
    unsigned = entry.model_dump(
        mode="json",
        exclude={"entry_id", "entry_sha256"},
    )
    unsigned["base_candidate_scores"] = tuple(
        score.model_dump(mode="json") for score in base_scores
    )
    identifier, digest = module._stable_id("hent", unsigned)

    with pytest.raises(ValueError, match="unique identities and hashes"):
        module.HistoryLibraryEntry.model_validate(
            {
                **unsigned,
                "entry_id": identifier,
                "entry_sha256": digest,
            }
        )


@pytest.mark.parametrize("collision_kind", ("prefix", "full"))
def test_candidate_score_schema_rejects_contribution_collection_collisions(
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    first_audit = module._build_neighbor_audit(
        entry=first,
        status="selected",
        similarity=1.0,
        normalized_similarity=0.5,
    )
    second_audit = module._build_neighbor_audit(
        entry=second,
        status="selected",
        similarity=0.8,
        normalized_similarity=0.5,
    )
    first_mapping = _mapping(first, source="source-a-a", current="service-a")
    second_mapping = _mapping(second, source="source-b-a", current="service-a")
    _install_digest_collision(
        monkeypatch,
        module,
        schema_version=module.HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION,
        collision_kind=collision_kind,
    )
    contributions = (
        module._build_history_contribution(
            entry=first,
            audit=first_audit,
            candidate_service_key="service-a",
            mapping=first_mapping,
        ),
        module._build_history_contribution(
            entry=second,
            audit=second_audit,
            candidate_service_key="service-a",
            mapping=second_mapping,
        ),
    )

    assert all(contribution.verify() for contribution in contributions)
    assert len({contribution.contribution_id for contribution in contributions}) == 1
    expected_sha_count = 2 if collision_kind == "prefix" else 1
    assert len({contribution.contribution_sha256 for contribution in contributions}) == (
        expected_sha_count
    )
    unsigned = {
        "schema_version": module.CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
        "candidate_service_key": "service-a",
        "contributions": tuple(
            contribution.model_dump(mode="json") for contribution in contributions
        ),
        "history_score": sum(contribution.contribution for contribution in contributions),
    }
    identifier, digest = module._stable_id("chs", unsigned)

    with pytest.raises(ValueError, match="unique identities and hashes"):
        module.CandidateHistoryScore.model_validate(
            {
                **unsigned,
                "candidate_history_score_id": identifier,
                "candidate_history_score_sha256": digest,
            }
        )


def test_candidate_score_schema_retains_unique_selected_neighbor_requirement() -> None:
    module = history_scoring_module()
    authority = _generation_authority()
    entry = _entry(authority=authority)
    audit = module._build_neighbor_audit(
        entry=entry,
        status="selected",
        similarity=1.0,
        normalized_similarity=1.0,
    )
    contributions = (
        module._build_history_contribution(
            entry=entry,
            audit=audit,
            candidate_service_key="service-a",
            mapping=_mapping(entry, source="source-a-a", current="service-a"),
        ),
        module._build_history_contribution(
            entry=entry,
            audit=audit,
            candidate_service_key="service-a",
            mapping=_mapping(entry, source="source-a-b", current="service-a"),
        ),
    )

    assert all(contribution.verify() for contribution in contributions)
    unsigned = {
        "schema_version": module.CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
        "candidate_service_key": "service-a",
        "contributions": tuple(
            contribution.model_dump(mode="json") for contribution in contributions
        ),
        "history_score": 1.0,
    }
    identifier, digest = module._stable_id("chs", unsigned)

    with pytest.raises(ValueError, match="unique per neighbor"):
        module.CandidateHistoryScore.model_validate(
            {
                **unsigned,
                "candidate_history_score_id": identifier,
                "candidate_history_score_sha256": digest,
            }
        )


@pytest.mark.parametrize(
    (
        "collection_field",
        "model_name",
        "id_field",
        "sha_field",
        "id_prefix",
        "schema_version_name",
    ),
    (
        (
            "neighbor_audits",
            "HistoryNeighborAudit",
            "audit_id",
            "audit_sha256",
            "haud",
            "HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION",
        ),
        (
            "hint_references",
            "HistoryHintReference",
            "hint_reference_id",
            "hint_reference_sha256",
            "hhint",
            "HISTORY_HINT_REFERENCE_SCHEMA_VERSION",
        ),
        (
            "candidate_scores",
            "CandidateHistoryScore",
            "candidate_history_score_id",
            "candidate_history_score_sha256",
            "chs",
            "CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION",
        ),
    ),
)
@pytest.mark.parametrize("collision_kind", ("prefix", "full"))
def test_result_schema_rejects_nested_identity_and_hash_collisions(
    monkeypatch: pytest.MonkeyPatch,
    collection_field: str,
    model_name: str,
    id_field: str,
    sha_field: str,
    id_prefix: str,
    schema_version_name: str,
    collision_kind: str,
) -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    model_type = getattr(module, model_name)
    _install_digest_collision(
        monkeypatch,
        module,
        schema_version=getattr(module, schema_version_name),
        collision_kind=collision_kind,
    )
    collision_items = []
    for item in getattr(result, collection_field):
        item_unsigned = item.model_dump(mode="json", exclude={id_field, sha_field})
        item_identifier, item_digest = module._stable_id(id_prefix, item_unsigned)
        collision_items.append(
            model_type.model_validate(
                {
                    **item_unsigned,
                    id_field: item_identifier,
                    sha_field: item_digest,
                }
            )
        )

    assert all(item.verify() for item in collision_items)
    assert len({getattr(item, id_field) for item in collision_items}) == 1
    expected_sha_count = len(collision_items) if collision_kind == "prefix" else 1
    assert len({getattr(item, sha_field) for item in collision_items}) == expected_sha_count
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned[collection_field] = tuple(item.model_dump(mode="json") for item in collision_items)
    identifier, digest = module._stable_id("heval", unsigned)

    with pytest.raises(ValueError, match=r"unique identities and hashes"):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": identifier,
                "result_sha256": digest,
            }
        )


def test_result_schema_explicitly_rejects_duplicate_selected_neighbor_ids() -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "embedding": (0.8, 0.6), "tag": "b"},
    )
    library = _library(first, second, authority=authority)
    result = module.evaluate_history(
        frozen_candidates=_frozen_candidates(),
        mode="hint",
        library=library,
        query_embedding=_query(),
        service_mappings=(),
        trusted_source_bundle=_trusted_bundle_for_library(library),
    )
    unsigned = result.model_dump(
        mode="json",
        exclude={"history_evaluation_result_id", "result_sha256"},
    )
    unsigned["selected_neighbor_ids"] = (result.selected_neighbor_ids[0],) * 2
    identifier, digest = module._stable_id("heval", unsigned)

    with pytest.raises(ValueError, match="selected neighbor IDs must be unique"):
        module.HistoryEvaluationResult.model_validate(
            {
                **unsigned,
                "history_evaluation_result_id": identifier,
                "result_sha256": digest,
            }
        )


def test_projected_output_budget_rejects_before_cosine_or_result_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    entry = _entry()
    mapping = _mapping(entry, source="source-a-a", current="service-a")
    monkeypatch.setattr(module, "_SERIALIZATION_PREFLIGHT_MAX_NODES", 1, raising=False)
    monkeypatch.setattr(
        module,
        "_cosine_similarity",
        lambda *_args: pytest.fail("cosine ran before projected-output preflight"),
    )

    with pytest.raises(module.HistoryScoringBudgetExceeded):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="score",
            library=_library(entry),
            query_embedding=_query(),
            service_mappings=(mapping,),
            trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
        )


def test_library_aggregate_budget_rejects_before_nested_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, (first, second) = _shared_entry_batch(
        {"case_hex": "b", "group_hex": "d", "tag": "a"},
        {"case_hex": "e", "group_hex": "f", "tag": "b"},
    )
    bundle = _trusted_bundle_for_entries(authority, (first, second))
    called = False

    def forbidden_model_dump(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("nested model_dump ran before aggregate library preflight")

    monkeypatch.setattr(module, "_SERIALIZATION_PREFLIGHT_MAX_NODES", 60)
    monkeypatch.setattr(module, "_MAX_TRUSTED_BUNDLE_PROJECTED_WORK", 24)
    monkeypatch.setattr(module.HistoryLibraryEntry, "model_dump", forbidden_model_dump)
    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=(first, second),
            generation_authority=authority,
            trusted_source_bundle=bundle,
        )
    assert called is False


def test_trusted_bundle_aggregate_lexical_budget_rejects_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    authority, entries = _shared_entry_batch(
        {
            "case_hex": "b",
            "group_hex": "d",
            "tag": "aggregate-lexical-budget",
            "service_scores": (("shared-service", 0.8),),
        }
    )
    bundle = _trusted_bundle_for_entries(authority, entries)
    monkeypatch.setattr(module, "_MAX_TRUSTED_BUNDLE_PROJECTED_WORK", 13)
    monkeypatch.setattr(
        module.json,
        "loads",
        lambda *_args, **_kwargs: pytest.fail(
            "json.loads ran before aggregate lexical-work preflight"
        ),
    )

    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=entries,
            generation_authority=authority,
            trusted_source_bundle=bundle,
        )


def test_mapping_aggregate_budget_rejects_before_nested_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    entry = _entry()
    mappings = (
        _mapping(entry, source="source-a-a", current="service-a"),
        _mapping(entry, source="source-a-b", current="service-b"),
    )
    library = _library(entry)
    query = _query()
    called = False

    def forbidden_model_dump(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("nested model_dump ran before aggregate mapping preflight")

    monkeypatch.setattr(module, "_SERIALIZATION_PREFLIGHT_MAX_NODES", 20)
    monkeypatch.setattr(module.HistoryServiceMapping, "model_dump", forbidden_model_dump)
    with pytest.raises(
        module.HistoryScoringBudgetExceeded,
        match="^history scoring budget exceeded$",
    ):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="score",
            library=library,
            query_embedding=query,
            service_mappings=mappings,
            trusted_source_bundle=_trusted_bundle_for_library(library),
        )
    assert called is False


def test_public_sequences_reject_one_shot_iterators_with_fixed_errors() -> None:
    module = history_scoring_module()
    entry = _entry()
    with pytest.raises(ValueError, match="history scoring input verification failed"):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="hint",
            library=_library(entry),
            query_embedding=_query(),
            service_mappings=iter(()),
            trusted_source_bundle=_trusted_bundle_for_library(_library(entry)),
        )
    with pytest.raises(ValueError, match="history library input verification failed"):
        module.build_frozen_history_library(
            dataset="rcabench",
            source_partition="dev",
            embedding_policy="semantic-v1",
            embedding_model="embedding-v1",
            embedding_dimension=2,
            entries=iter((entry,)),
            generation_authority=_generation_authority(),
        )


def test_off_rejects_supplied_mapping_without_touching_history_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = history_scoring_module()
    entry = _entry()
    mapping = _mapping(entry, source="source-a-a", current="service-a")
    called = False

    def forbidden_verify(_value: object) -> bool:
        nonlocal called
        called = True
        raise AssertionError("history mapping callback ran while mode was off")

    monkeypatch.setattr(module.HistoryServiceMapping, "verify", forbidden_verify)
    with pytest.raises(ValueError, match="^history scoring input verification failed$"):
        module.evaluate_history(
            frozen_candidates=_frozen_candidates(),
            mode="off",
            library=None,
            query_embedding=None,
            service_mappings=(mapping,),
        )
    assert called is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mapping_policy", "groundtruth-answer-alias-v1"),
        ("mapping_policy", r"C:\\original_cases\\case.json"),
        ("provenance_ids", ("injection-label:database-kill",)),
    ],
)
def test_mapping_builder_rejects_answer_fault_injection_and_original_path_material(
    field: str,
    value: object,
) -> None:
    entry = _entry()
    arguments: dict[str, object] = {
        "entry_id": entry.entry_id,
        "source_service_key": "source-a-a",
        "current_service_key": "service-a",
        "mapping_policy": "exact-service-alias-v1",
        "provenance_ids": ("mapping:fixture-v1",),
    }
    arguments[field] = value

    with pytest.raises(
        ValueError,
        match="^history service mapping input verification failed$",
    ):
        history_scoring_module().build_history_service_mapping(**arguments)


@pytest.mark.parametrize(
    "provenance_id",
    [
        "cases/t001/task.json",
        "../cases/t001/task.json",
        "/cases/t001/task.json",
        r"C:\cases\t001\task.json",
        "https://example.test/cases/t001",
        "mapping:cases/t001/task-v1",
        "mapping:../task-v1",
    ],
)
def test_mapping_builder_rejects_paths_and_uris_outside_opaque_provenance_grammar(
    provenance_id: str,
) -> None:
    entry = _entry()

    with pytest.raises(
        ValueError,
        match="^history service mapping input verification failed$",
    ):
        history_scoring_module().build_history_service_mapping(
            entry_id=entry.entry_id,
            source_service_key="source-a-a",
            current_service_key="service-a",
            mapping_policy="exact-service-alias-v1",
            provenance_ids=(provenance_id,),
        )


def test_mapping_schema_rejects_resigned_relative_path_provenance() -> None:
    module = history_scoring_module()
    mapping = _mapping(_entry(), source="source-a-a", current="service-a")
    unsigned = mapping.model_dump(mode="json", exclude={"mapping_id", "mapping_sha256"})
    unsigned["provenance_ids"] = ("cases/t001/task.json",)
    digest = sha256_hex(canonical_json_bytes(unsigned))

    with pytest.raises(ValueError, match="opaque provenance ID"):
        module.HistoryServiceMapping.model_validate(
            {
                **unsigned,
                "mapping_id": f"hmap_{digest[:24]}",
                "mapping_sha256": digest,
            }
        )


@pytest.mark.parametrize(
    ("boundary", "value", "expected_message"),
    [
        ("generation_policy", "policies/task.json", "history generation authority"),
        ("base_source", "answer-service-v1", "history base score"),
        ("entry_policy", "fault-policy-v1", "history library entry"),
        ("query_model", "../embedding-v1", "history query"),
        ("mapping_source", "injection-service-v1", "history service mapping"),
        ("mapping_policy", "policies/task.json", "history service mapping"),
    ],
)
def test_public_builders_reject_forbidden_or_path_bearing_persisted_identifiers(
    boundary: str,
    value: str,
    expected_message: str,
) -> None:
    module = history_scoring_module()
    authority = _generation_authority()

    with pytest.raises(ValueError, match=f"^{expected_message} input verification failed$"):
        if boundary == "generation_policy":
            module.build_history_generation_authority(
                dataset="rcabench",
                source_partition="dev",
                generation_policy=value,
                canonical_batch_manifest=canonical_json_bytes(
                    {"batch_token": "history-batch-a-v1"}
                ),
            )
        elif boundary == "base_source":
            module.build_history_base_candidate_score(
                source_service_key=value,
                base_history_score=0.8,
                base_score_artifact=canonical_json_bytes({"artifact_token": "base-a-v1"}),
                generation_authority=authority,
            )
        elif boundary == "entry_policy":
            _entry(authority=authority, embedding_policy=value)
        elif boundary == "query_model":
            _query(embedding_model=value)
        elif boundary == "mapping_source":
            module.build_history_service_mapping(
                entry_id=_entry().entry_id,
                source_service_key=value,
                current_service_key="service-a",
                mapping_policy="exact-service-alias-v1",
                provenance_ids=("mapping:fixture-v1",),
            )
        else:
            module.build_history_service_mapping(
                entry_id=_entry().entry_id,
                source_service_key="source-a-a",
                current_service_key="service-a",
                mapping_policy=value,
                provenance_ids=("mapping:fixture-v1",),
            )


def test_persisted_models_reject_forbidden_or_path_bearing_identifiers() -> None:
    authority = _generation_authority()
    models_and_updates = (
        (authority, {"generation_policy": "policies/task.json"}),
        (
            _base_scores(authority=authority)[0],
            {"source_service_key": "answer-service-v1"},
        ),
        (_entry(authority=authority), {"embedding_policy": "fault-policy-v1"}),
        (_query(), {"embedding_model": "../embedding-v1"}),
        (
            _mapping(_entry(), source="source-a-a", current="service-a"),
            {"mapping_policy": "policies/task.json"},
        ),
    )

    for model, update in models_and_updates:
        document = model.model_dump(mode="json")
        document.update(update)
        with pytest.raises(ValueError, match="identifier grammar v1"):
            type(model).model_validate(document)


def test_identifier_grammar_preserves_legal_service_ids() -> None:
    module = history_scoring_module()
    material = _typed_entry_materials(
        tag="legal-service",
        service_scores=(("checkout.api-v2", 0.8),),
    )
    authority = _register_generation_batch(material)
    authority_sha256 = authority.generation_authority_sha256
    frozen_run_manifest = _TRUSTED_RUN_MANIFESTS[authority_sha256]
    score_document = material["score_documents"][0]
    score = module.build_history_base_candidate_score(
        source_service_key="checkout.api-v2",
        base_history_score=0.8,
        base_score_artifact=material["score_bytes"][0],
        generation_authority=authority,
        expected_generation_authority_sha256=(
            frozen_run_manifest.history_generation_authority_sha256
        ),
        canonical_batch_manifest=_TRUSTED_AUTHORITY_MANIFESTS[authority_sha256],
    )
    mapping = module.build_history_service_mapping(
        entry_id=_entry().entry_id,
        source_service_key=score.source_service_key,
        current_service_key="payments_api-v2",
        mapping_policy="exact-service-alias-v1",
        provenance_ids=("mapping:fixture-v1",),
    )

    assert score.source_service_key == "checkout.api-v2"
    assert score.base_history_score == score_document["base_history_score"]
    assert mapping.current_service_key == "payments_api-v2"
