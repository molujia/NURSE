"""Metadata-only research gates; no telemetry, provider, evaluator or credentials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
from typing import Any
from uuid import uuid4


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_pin(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("research pin shape differs")
    return value


def plain_path(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("research path is not plain")


def write_once(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise ValueError("nonempty original bytes required")
    plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_path(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def read_plain(path: Path, *, limit: int = 64 * 1024**2) -> bytes:
    plain_path(path)
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > limit:
        raise ValueError("research member type or capacity differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("research member is not regular")
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("research member capacity differs")
    return raw


def read_document(path: Path) -> dict[str, Any]:
    raw = read_plain(path)
    value = json.loads(raw)
    if type(value) is not dict or canonical(value) != raw:
        raise ValueError("research document is not canonical")
    return value


def select_case(*, case_ids: list[str], excluded: set[str], seed: str,
                coverage: dict[str, Any]) -> dict[str, Any]:
    if (type(coverage) is not dict or coverage.get("metadata_reviewed") is not True
            or type(coverage.get("source_sha256s")) is not list
            or not coverage["source_sha256s"]):
        raise ValueError("exposure metadata review required")
    for pin in coverage["source_sha256s"]:
        require_pin(pin)
    if (type(seed) is not str or re.fullmatch(r"[0-9a-f]{32}", seed) is None
            or type(case_ids) is not list or not case_ids
            or any(type(row) is not str or not row for row in case_ids)
            or len(set(case_ids)) != len(case_ids) or type(excluded) is not set
            or any(type(row) is not str or not row for row in excluded)):
        raise ValueError("selection metadata shape differs")
    eligible = sorted(set(case_ids) - excluded)
    if not eligible:
        raise ValueError("no eligible uncontacted case")
    index = random.Random(int(seed, 16)).randrange(len(eligible))
    return dict(schema="nurse.research.blind-selection-private/1",
                original_case_id=eligible[index], opaque_case_id="opq_" + uuid4().hex,
                population_count=len(case_ids), eligible_count=len(eligible),
                population_sha256=digest(canonical(sorted(case_ids))),
                exclusion_sha256=digest(canonical(sorted(excluded))),
                eligible_sha256=digest(canonical(eligible)), seed=seed,
                algorithm="sorted-eligible-python-random-randrange/1",
                selected_index=index, selection_count=1,
                coverage=coverage, label_reads=0)


def public_selection(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "nurse.research.blind-selection-private/1":
        raise ValueError("private selection identity differs")
    # Deliberate allowlist: neither caller-supplied fields nor coverage prose escapes.
    names = ("opaque_case_id", "population_count", "eligible_count", "population_sha256",
             "exclusion_sha256", "eligible_sha256", "selection_count", "label_reads")
    return dict(schema="nurse.research.blind-selection-public/1",
                **{name: value[name] for name in names},
                known_exposure_exclusion_applied=True,
                absolute_pretraining_or_missing_history_blindness_not_established=True)


def require_phase(phase: str, *, codex_sealed: bool, nurse_sealed: bool) -> None:
    if type(codex_sealed) is not bool or type(nurse_sealed) is not bool:
        raise ValueError("strict sealed status required")
    if ((phase == "start_nurse" and codex_sealed)
            or (phase == "unblind" and codex_sealed and nurse_sealed)):
        return
    raise ValueError("research phase order not satisfied")


def _member(root: Path, name: str) -> Path:
    if (type(name) is not str or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or name != PurePosixPath(name).as_posix()
            or any(part in {".", "..", ""} for part in name.split("/"))):
        raise ValueError("research member name differs")
    result = root / name
    plain_path(result)
    return result


def _file_pin(path: Path) -> dict[str, Any]:
    plain_path(path)
    total, hasher = 0, hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("sealed file is not regular")
        while block := handle.read(1024**2):
            total += len(block)
            hasher.update(block)
    return dict(bytes=total, sha256=hasher.hexdigest())


def _names(root: Path, actor: str) -> set[str]:
    names = set()
    for path in root.rglob("*"):
        plain_path(path)
        if path.is_file() and path.name != f"{actor}-sealed.json":
            names.add(path.relative_to(root).as_posix())
    return names


def seal_outputs(root: Path, members: list[str], *, actor: str, identity: str) -> dict[str, Any]:
    require_pin(identity)
    if actor not in {"codex", "nurse"} or not members or len(set(members)) != len(members):
        raise ValueError("seal actor or members differ")
    rows = [{"path": name, **_file_pin(_member(root, name))} for name in sorted(members)]
    if _names(root, actor) != set(members):
        raise ValueError("seal complete member set differs")
    body = dict(schema="nurse.research.blind-prediction-seal/1", actor=actor,
                identity_sha256=identity, members=rows, semantic_correctness_not_certified=True)
    result = dict(body, seal_sha256=digest(canonical(body)))
    write_once(root / f"{actor}-sealed.json", canonical(result))
    return result


def verify_seal(root: Path, value: dict[str, Any], *, actor: str, identity: str) -> bool:
    require_pin(identity)
    body = {key: row for key, row in value.items() if key != "seal_sha256"}
    if (value.get("schema") != "nurse.research.blind-prediction-seal/1"
            or value.get("actor") != actor or value.get("identity_sha256") != identity
            or value.get("seal_sha256") != digest(canonical(body))
            or value != read_document(root / f"{actor}-sealed.json")):
        raise ValueError("seal binding differs")
    rows = value.get("members")
    if type(rows) is not list or not rows:
        raise ValueError("sealed inventory missing")
    if _names(root, actor) != {row["path"] for row in rows}:
        raise ValueError("sealed member set differs")
    for row in rows:
        if row != {"path": row["path"], **_file_pin(_member(root, row["path"]))}:
            raise ValueError("sealed original changed")
    return True
