"""Private, versioned semantic round audit; hashes are not evidence authority."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.safe_serialization import (
    DEFAULT_SERIALIZATION_MAX_DEPTH,
    DEFAULT_SERIALIZATION_MAX_NODES,
    preflight_exact_pydantic_tree,
)

SEMANTIC_RUN_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.semantic-run-audit/1"] = (
    "nurse.v2.semantic-run-audit/1"
)
RUNNER_VERSION: Literal["nurse-v2-semantic-ranking-runner/1"] = "nurse-v2-semantic-ranking-runner/1"
# Terminal audit expands bounded responses into guards, roles and directed edges;
# it is not a model wire message. Use one capacity contract for build/verify/I/O.
MAX_SEMANTIC_AUDIT_BYTES = 64 * 1024 * 1024
_MAX_ATOM_BYTES = 256
_WHITESPACE = b" \t\r\n"
_TOKEN_END = b' \t\r\n,:{}[]"'


def _preflight_audit_json(raw: bytes) -> None:
    """Bound allocation before parsing, without building a second JSON tree.

    This is only a resource scanner; stdlib parsing and exact canonical equality
    still validate syntax/encoding. Keys do not count as value nodes, matching
    the exact-Pydantic-tree contract. Quoted/escaped contents remain opaque.
    """
    if len(raw) > MAX_SEMANTIC_AUDIT_BYTES:
        raise ValueError("semantic audit exceeds byte limit")
    # Canonical UTF-8 JSON has no literal NUL. Reject UTF-16/32 before their
    # escape/quote layout can disagree with this byte-oriented resource scan.
    if b"\x00" in raw:
        raise ValueError("semantic audit requires canonical UTF-8 JSON")
    depth = nodes = index = 0

    def value_node() -> None:
        nonlocal nodes
        nodes += 1
        if nodes > DEFAULT_SERIALIZATION_MAX_NODES or depth > DEFAULT_SERIALIZATION_MAX_DEPTH:
            raise ValueError("semantic audit exceeds JSON resource limit")

    while index < len(raw):
        token = raw[index]
        if token == 34:  # A quoted value or object key; ignore delimiters inside.
            index += 1
            escaped = False
            while index < len(raw):
                char = raw[index]
                index += 1
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    break
            else:
                raise ValueError("unterminated semantic audit JSON string")
            following = index
            while following < len(raw) and raw[following] in _WHITESPACE:
                following += 1
            if following == len(raw) or raw[following] != 58:
                value_node()
        elif token in (123, 91):
            value_node()
            depth += 1
            index += 1
        elif token in (125, 93):
            depth -= 1
            if depth < 0:
                raise ValueError("invalid semantic audit JSON nesting")
            index += 1
        elif token in _WHITESPACE or token in (44, 58):
            index += 1
        else:
            value_node()
            start = index
            while index < len(raw) and raw[index] not in _TOKEN_END:
                index += 1
                if index - start > _MAX_ATOM_BYTES:
                    raise ValueError("semantic audit JSON scalar exceeds resource limit")
    if depth:
        raise ValueError("unclosed semantic audit JSON container")


class SemanticRankingPolicy(StrictFrozenModel):
    """Per-request limits, explicitly separate from provider usage accounting."""

    policy_version: Literal["semantic-ranking-budget/1"] = "semantic-ranking-budget/1"
    configuration_version: str = Field(default="default/1", min_length=1, max_length=128)
    model: str = Field(
        default="deepseek-v4-flash-ga-260731", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$"
    )
    input_token_limit: int = Field(default=12000, gt=0, strict=True)
    output_token_limit: int = Field(default=3000, gt=0, le=65536, strict=True)
    max_ranking_requests: Literal[2] = 2
    temperature: float = Field(default=0.0, ge=0.0, le=0.0, strict=True)
    estimator_version: Literal["request-utf8-bytes-plus-256/1"] = "request-utf8-bytes-plus-256/1"

    @model_validator(mode="after")
    def require_recorded_override(self) -> Self:
        if self.configuration_version == "default/1" and (
            self.input_token_limit != 12000 or self.output_token_limit != 3000
        ):
            raise ValueError("budget override requires an explicit configuration version")
        return self


class SemanticRoundAudit(StrictFrozenModel):
    round_number: Literal[1, 2]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_payload_hex: str = Field(repr=False)
    request_hex: str | None = Field(repr=False)
    estimated_input_tokens: int | None = Field(ge=0)
    status: Literal["prepare_failed", "budget_exhausted", "failed", "response_ready"]
    failure_code: str | None
    raw_response_hex: str | None = Field(repr=False)
    received_bytes: int | None = Field(ge=0)
    semantic_response_hex: str | None = Field(repr=False)
    usage_status: Literal["unavailable", "reported", "missing", "invalid"]
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    compilation_json: str | None = Field(repr=False)


StopReason = Literal[
    "first_budget_exhausted",
    "first_failure",
    "no_revision_dependency",
    "second_budget_exhausted",
    "second_prompt_denied",
    "second_failure",
    "second_snapshot_selected",
]


class SemanticRunAudit(StrictFrozenModel):
    """Retained source projections must be recomputed before using any effects."""

    schema_version: Literal["nurse.v2.semantic-run-audit/1"] = SEMANTIC_RUN_AUDIT_SCHEMA_VERSION
    runner_version: Literal["nurse-v2-semantic-ranking-runner/1"] = RUNNER_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    sources_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: SemanticRankingPolicy
    rounds: tuple[SemanticRoundAudit, ...] = Field(min_length=1, max_length=2, repr=False)
    query_session: CaseQuerySessionResult | None = Field(repr=False)
    query_receipt_json: str | None = Field(repr=False)
    selected_round: Literal[0, 1, 2]
    stop_reason: StopReason
    followup_failure_code: str | None = None
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Structural/hash closure only, not source authenticity or score permission."""
        try:
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(SemanticRunAudit,),
                label="semantic run audit",
                max_string_length=None,
                provenance_id_limit=None,
            )
            if SemanticRunAudit.model_validate(self.model_dump(mode="python")) != self:
                return False
            if tuple(r.round_number for r in self.rounds) != tuple(range(1, len(self.rounds) + 1)):
                return False
            if self.selected_round > len(self.rounds):
                return False
            if self.query_session is not None and not self.query_session.verify():
                return False
            for record in self.rounds:
                for value in (
                    record.prompt_payload_hex,
                    record.request_hex,
                    record.raw_response_hex,
                    record.semantic_response_hex,
                ):
                    if value is not None and bytes.fromhex(value).hex() != value:
                        return False
                ready = record.status == "response_ready"
                if ready != (
                    record.semantic_response_hex is not None and record.compilation_json is not None
                ):
                    return False
                if ready and (record.raw_response_hex is None or record.failure_code is not None):
                    return False
            _preflight_audit_json(canonical_json_bytes(self.model_dump(mode="json")))
            return self.audit_sha256 == sha256_hex(
                canonical_json_bytes(self.model_dump(mode="json", exclude={"audit_sha256"}))
            )
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


def build_semantic_run_audit(**values: object) -> SemanticRunAudit:
    unsigned = (
        dict(
            schema_version=SEMANTIC_RUN_AUDIT_SCHEMA_VERSION,
            runner_version=RUNNER_VERSION,
            followup_failure_code=None,
        )
        | values
    )
    result = SemanticRunAudit.model_validate(
        dict(**unsigned, audit_sha256=sha256_hex(canonical_json_bytes(unsigned)))
    )
    if not result.verify():
        raise ValueError("invalid_semantic_run_audit")
    return result


def require_semantic_run_audit(audit: SemanticRunAudit, *, expected_audit_sha256: str) -> None:
    if (
        type(audit) is not SemanticRunAudit
        or type(expected_audit_sha256) is not str
        or not SemanticRunAudit.verify(audit)
        or audit.audit_sha256 != expected_audit_sha256
    ):
        raise ValueError("semantic run audit digest or structure mismatch")


def persist_semantic_run_audit(path: Path, audit: SemanticRunAudit) -> None:
    """Publish a complete canonical private audit atomically, without overwriting.

    An identical existing file is idempotent. This is a terminal audit publication,
    not the durable attempt/checkpoint mechanism supplied by the experiment runtime.
    """
    if type(audit) is not SemanticRunAudit or not SemanticRunAudit.verify(audit):
        raise ValueError("invalid_semantic_run_audit")
    # verify() has already applied the exact tree and shared terminal byte limits.
    data = canonical_json_bytes(audit.model_dump(mode="json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".semantic-audit-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and unlike replace cannot clobber a
            # concurrent publisher. No overwrite fallback on unsupported filesystems.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or _read_bounded(path) != data:
                raise FileExistsError("semantic audit path contains different content") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_bounded(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_SEMANTIC_AUDIT_BYTES + 1)
    if len(data) > MAX_SEMANTIC_AUDIT_BYTES:
        raise ValueError("semantic audit exceeds byte limit")
    return data


def load_semantic_run_audit(path: Path, *, expected_audit_sha256: str) -> SemanticRunAudit:
    data = _read_bounded(path)
    _preflight_audit_json(data)
    try:
        document = json.loads(data.decode("utf-8", errors="strict"))
        preflight_exact_pydantic_tree(
            document,
            root_model_types=(SemanticRunAudit,),
            label="semantic audit JSON",
            max_string_length=None,
            provenance_id_limit=None,
        )
        canonical = canonical_json_bytes(document)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("invalid_semantic_audit_json") from None
    # Also rejects duplicate keys, NaN/Infinity and noncanonical field ordering.
    if type(document) is not dict or canonical != data:
        raise ValueError("semantic audit is not canonical JSON")
    audit = SemanticRunAudit.model_validate_json(data)
    if canonical_json_bytes(audit.model_dump(mode="json")) != data:
        raise ValueError("semantic audit schema normalization would change pinned bytes")
    require_semantic_run_audit(audit, expected_audit_sha256=expected_audit_sha256)
    return audit
