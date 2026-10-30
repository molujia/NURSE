"""Bounded, content-addressed private source blocks; no network or callback stores."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_audit_json import MAX_REPORT_AUDIT_BYTES, scan_audit_json
from nurse_rcl.v2.signal_transforms import SourceWindowInput

from .contracts import NamedSourceWindow, SourceBlockReference
from .safety import preflight, require_pin, require_unchanged, snapshot


def _require_bytes(raw: object) -> bytes:
    if type(raw) is not bytes or len(raw) > MAX_REPORT_AUDIT_BYTES:
        raise ValueError("source block requires bounded exact bytes")
    return raw


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorySourceBlockStore:
    blocks: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        store_snapshot(self)

    def read(self, pin: str) -> bytes:
        require_pin(pin)
        store_snapshot(self)
        for key, raw in self.blocks:
            if key == pin:
                return raw
        raise ValueError("source block is missing")


def _root_path(root: str) -> Path:
    if type(root) is not str or not root or len(root) > 8192:
        raise ValueError("source store requires an exact absolute directory")
    path = Path(root)
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
            raise ValueError("source store root must be an existing canonical absolute directory")
    except (OSError, RuntimeError) as exc:
        raise ValueError("source store root cannot be resolved safely") from exc
    return path


@contextmanager
def _directory_handle(store: DirectorySourceBlockStore) -> Iterator[int]:
    """Anchor relative I/O to an opened directory, not a replaceable path string."""
    state = store_snapshot(store)
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("directory source storage requires no-follow directory-relative I/O")
    root = _root_path(store.root)
    descriptor = None
    try:
        before = root.stat()
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("source store root identity changed while opening")
        try:
            yield descriptor
        finally:
            current = _root_path(store.root).stat()
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("source store root identity changed during I/O")
            require_store_unchanged(store, state)
    except OSError as exc:
        raise ValueError("source directory I/O failed safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectorySourceBlockStore:
    root: str

    def __post_init__(self) -> None:
        store_snapshot(self)
        _root_path(self.root)

    def read(self, pin: str) -> bytes:
        require_pin(pin)
        with _directory_handle(self) as directory:
            name = pin + ".json"
            before = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_REPORT_AUDIT_BYTES:
                raise ValueError("source block is not a bounded regular file")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(name, flags, dir_fd=directory), "rb") as source:
                opened = os.fstat(source.fileno())
                if (
                    opened.st_size > MAX_REPORT_AUDIT_BYTES
                    or not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (
                        before.st_dev,
                        before.st_ino,
                    )
                ):
                    raise ValueError("source block identity changed while opening")
                raw = source.read(MAX_REPORT_AUDIT_BYTES + 1)
                after = os.fstat(source.fileno())
            _require_bytes(raw)
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("source block changed while reading")
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("source store path changed while reading")
            if sha256_hex(raw) != pin:
                raise ValueError("source block digest mismatch")
            return raw

    def put(self, *, raw: bytes) -> str:
        state = store_snapshot(self)
        _require_bytes(raw)
        pin = sha256_hex(raw)
        require_store_unchanged(self, state)
        with _directory_handle(self) as directory:
            try:
                descriptor = os.open(
                    pin + ".json",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                if self.read(pin) != raw:
                    raise ValueError(
                        "existing source block differs from its content address"
                    ) from None
                return pin
            with os.fdopen(descriptor, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(directory)
        return pin


SourceBlockStore = MemorySourceBlockStore | DirectorySourceBlockStore


def store_snapshot(store: SourceBlockStore) -> object:
    """Snapshot exact immutable store configuration before any store methods run."""
    if type(store) is DirectorySourceBlockStore:
        root = object.__getattribute__(store, "root")
        if type(root) is not str or len(root) > 8192:
            raise ValueError("source store requires an exact bounded root string")
        return root
    if type(store) is not MemorySourceBlockStore:
        raise ValueError("only exact code-owned source stores are supported")
    blocks = object.__getattribute__(store, "blocks")
    if type(blocks) is not tuple or len(blocks) * 3 + 1 > 500_000:
        raise ValueError("source store block domain exceeds its exact tuple/node limit")
    total = 0
    keys = set()
    for item in blocks:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("source store block requires an exact key/bytes pair")
        key, raw = item
        require_pin(key)
        _require_bytes(raw)
        total += len(raw)
        if total > MAX_REPORT_AUDIT_BYTES:
            raise ValueError("source store exceeds complete source byte budget")
        if key in keys or sha256_hex(raw) != key:
            raise ValueError("source store has a duplicate or mismatched content address")
        keys.add(key)
    return blocks


def require_store_unchanged(store: SourceBlockStore, before: object) -> None:
    if store_snapshot(store) != before:
        raise ValueError("source store configuration changed during replay")


def _bounded_source_text(raw: bytes) -> str:
    """Source strings use the ordinary 8192-character cap, not report prose limits."""
    scan_audit_json(raw)
    text = raw.decode("utf-8", errors="strict")
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        index += 1
        count = 0
        while index < len(text) and text[index] != '"':
            count += 1
            if count > 8192:
                raise ValueError("source JSON string exceeds character limit")
            if text[index] == "\\":
                index += 1
                if index < len(text) and text[index] == "u":
                    index += 4
            index += 1
        index += 1
    return text


def read_source_window(
    *, store: SourceBlockStore, block: SourceBlockReference
) -> SourceWindowInput:
    if type(block) is not SourceBlockReference:
        raise ValueError("source block requires exact metadata")
    state = store_snapshot(store)
    before = snapshot(block, (SourceBlockReference,))
    try:
        # Revalidate mutated frozen fields before filesystem access.
        reference = SourceBlockReference.model_validate_json(before[0])
        raw = store.read(reference.sha256)
        _require_bytes(raw)
        if len(raw) != reference.size_bytes or sha256_hex(raw) != reference.sha256:
            raise ValueError("source block size or digest mismatch")
        try:
            document = json.loads(_bounded_source_text(raw))
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError("source block is not canonical UTF-8 JSON") from exc
        preflight(document, (NamedSourceWindow,))
        if (
            type(document) is not dict
            or set(document) != {"schema_version", "inputs"}
            or document["schema_version"] != reference.schema_version
            or type(document["inputs"]) is not dict
            or set(document["inputs"]) != {reference.name}
        ):
            raise ValueError("source block schema or named window domain mismatch")
        entry = document["inputs"][reference.name]
        if (
            type(entry) is not dict
            or set(entry) != {"input_kind", "document"}
            or entry["input_kind"] != reference.input_kind
        ):
            raise ValueError("source block typed window tag mismatch")
        if canonical_json_bytes(document) != raw:
            raise ValueError("source block bytes are not canonical")
        named = NamedSourceWindow.model_validate_json(
            canonical_json_bytes(
                {
                    "name": reference.name,
                    "window": entry["document"],
                }
            )
        )
        if type(named.window).__name__ != reference.input_kind:
            raise ValueError("source block typed window differs from its tag")
        return named.window
    finally:
        require_unchanged(block, (SourceBlockReference,), before)
        require_store_unchanged(store, state)
