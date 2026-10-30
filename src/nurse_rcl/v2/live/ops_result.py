"""The semantic core result: no model, investigation, report or runtime envelope.

Only the deterministic owner calls the sealing factory. Validation binds saved
bytes and identities, not an expensive independent source/score replay.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_artifacts import NormalEvidence
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity, snapshot_id


class OpsCoreResult(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-core-result/1"] = "nurse.v2.ops-core-result/1"
    identity: OpsCoreIdentity
    evidence: NormalEvidence
    core_result_digest: SHA256

    @model_validator(mode="after")
    def require_complete_binding(self) -> Self:
        evidence, identity = self.evidence, self.identity
        session, ranking = evidence.session, evidence.ranking
        if (identity.dataset, identity.opaque_case_id, identity.case_snapshot_id) != (
            evidence.dataset, evidence.opaque_case_id, snapshot_id(evidence),
        ):
            raise ValueError("core identity differs from case snapshot")
        if session.stop_reason is None or session.failure_reason is not None or (
            session.pending_selection is not None or session.pending_measurement is not None
            or session.catalog_sha256 != evidence.catalog_sha256
            or session.session_sha256 != evidence.session_sha256
            or session.paid_count != len(session.steps)
            or ranking.opaque_case_id != evidence.opaque_case_id
        ):
            raise ValueError("core requires complete bound terminal evidence")
        keys = ranking.candidate_service_keys
        if not keys or len(set(keys)) != len(keys) or (
            tuple(s.candidate_service_key for s in ranking.scores) != keys
            or len(ranking.ordered_service_keys) != len(keys)
            or set(ranking.ordered_service_keys) != set(keys)
            or ranking.top_k_service_keys != ranking.ordered_service_keys[:ranking.top_k]
        ):
            raise ValueError("core ranking does not cover the full candidate domain")
        expected_evidence = sha256_hex(canonical_json_bytes(
            evidence.model_dump(mode="json", exclude={"result_sha256"}),
        ))
        if expected_evidence != evidence.result_sha256:
            raise ValueError("core evidence content digest mismatch")
        expected = sha256_hex(canonical_json_bytes(
            self.model_dump(mode="json", exclude={"core_result_digest"}),
        ))
        if expected != self.core_result_digest:
            raise ValueError("core semantic digest mismatch")
        return self

    @classmethod
    def from_evidence(cls, *, identity: OpsCoreIdentity, evidence: NormalEvidence) -> Self:
        if type(identity) is not OpsCoreIdentity or type(evidence) is not NormalEvidence:
            raise ValueError("exact deterministic identity/evidence required")
        body = dict(
            schema_version="nurse.v2.ops-core-result/1",
            identity=identity.model_dump(mode="json"), evidence=evidence.model_dump(mode="json"),
        )
        body["core_result_digest"] = sha256_hex(canonical_json_bytes(body))
        # One serialization admission copies the complete tree, avoiding mutable
        # aliases; no source files are reloaded and no score is recalculated.
        return cls.model_validate_json(canonical_json_bytes(body))


def seal_core_result(*, identity: OpsCoreIdentity, evidence: NormalEvidence) -> OpsCoreResult:
    return OpsCoreResult.from_evidence(identity=identity, evidence=evidence)
