"""Transport only for the primary Codex's original owner-bound completions."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from nurse_rcl.v2.live.blind_compare_contract import (
    canonical, digest, plain_path, read_document, require_pin, write_once,
)


class DecisionMailbox:
    def __init__(self, root: Path, *, owner_sha256: str, ceiling: int) -> None:
        require_pin(owner_sha256)
        if type(ceiling) is not int or not 1 <= ceiling <= 8:
            raise ValueError("manual ceiling differs")
        plain_path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.owner, self.ceiling = root, owner_sha256, ceiling
        identity = dict(schema="nurse.research.codex-mailbox/1", actor="primary_codex",
                        owner_sha256=owner_sha256, action_ceiling=ceiling,
                        ark_transport=False)
        path = root / "owner.json"
        if path.exists() or path.is_symlink():
            if read_document(path) != identity:
                raise ValueError("manual owner differs")
        else:
            if any(root.iterdir()):
                raise ValueError("unowned manual mailbox occupied")
            write_once(path, canonical(identity))

    def _path(self, index: int, kind: str, correction: int = 0) -> Path:
        if (type(index) is not int or not 1 <= index <= self.ceiling
                or type(correction) is not int or correction not in {0, 1}):
            raise ValueError("manual turn/correction exceeds grant")
        return self.root / f"turn-{index:03d}-c{correction}.{kind}.json"

    def _request(self, index: int, correction: int = 0) -> dict[str, Any]:
        row = read_document(self._path(index, "request", correction))
        wire = base64.b64decode(row["wire_base64"], validate=True)
        if row != self._request_value(index, wire, correction):
            raise ValueError("manual frozen request differs")
        return row

    def _request_value(self, index: int, wire: bytes, correction: int) -> dict[str, Any]:
        if type(wire) is not bytes or not wire or len(wire) > 64 * 1024**2:
            raise ValueError("manual wire bytes differ")
        return dict(schema="nurse.research.codex-request/1", owner_sha256=self.owner,
                    index=index, correction_index=correction, request_sha256=digest(wire),
                    wire_base64=base64.b64encode(wire).decode("ascii"))

    def publish(self, index: int, wire: bytes, *, correction: int = 0) -> dict[str, Any]:
        path = self._path(index, "request", correction)
        row = self._request_value(index, wire, correction)
        if path.exists() or path.is_symlink():
            if self._request(index, correction) != row:
                raise ValueError("manual request cannot change")
            return row
        if index > 1:
            previous = self._request(index - 1)
            previous_wire = base64.b64decode(previous["wire_base64"], validate=True)
            if self.response(index - 1, previous_wire) is None:
                raise ValueError("manual previous turn has no native")
        if correction:
            original = self._request(index)
            original_wire = base64.b64decode(original["wire_base64"], validate=True)
            if self.response(index, original_wire) is None:
                raise ValueError("manual correction has no original native")
            marker = self.root / "unique-correction.json"
            if marker.exists() or marker.is_symlink():
                raise ValueError("manual unique correction already used")
            write_once(marker, canonical(dict(owner_sha256=self.owner, index=index,
                                              request_sha256=row["request_sha256"])))
        write_once(path, canonical(row))
        return row

    def submit(self, index: int, native: bytes, *, request_sha256: str,
               correction: int = 0) -> None:
        require_pin(request_sha256)
        request = self._request(index, correction)
        if (request_sha256 != request["request_sha256"] or type(native) is not bytes
                or not native or len(native) > 1024**2):
            raise ValueError("manual native binding differs")
        row = dict(schema="nurse.research.codex-native/1", owner_sha256=self.owner,
                   index=index, correction_index=correction,
                   request_sha256=request_sha256, native_sha256=digest(native),
                   native_base64=base64.b64encode(native).decode("ascii"))
        write_once(self._path(index, "response", correction), canonical(row))

    def response(self, index: int, wire: bytes, *, correction: int = 0) -> bytes | None:
        response_path = self._path(index, "response", correction)
        request_path = self._path(index, "request", correction)
        if not response_path.exists() and not response_path.is_symlink():
            if request_path.exists() or request_path.is_symlink():
                if self._request(index, correction) != self._request_value(index, wire, correction):
                    raise ValueError("manual recovery request differs")
            return None
        request = self._request(index, correction)
        if request != self._request_value(index, wire, correction):
            raise ValueError("manual response request differs")
        row = read_document(response_path)
        native = base64.b64decode(row["native_base64"], validate=True)
        expected = dict(schema="nurse.research.codex-native/1", owner_sha256=self.owner,
                        index=index, correction_index=correction,
                        request_sha256=request["request_sha256"], native_sha256=digest(native),
                        native_base64=base64.b64encode(native).decode("ascii"))
        if row != expected:
            raise ValueError("manual original native differs")
        return native
