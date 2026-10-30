"""Two derived files, one bounded receipt. Missing members never count as delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_plan import plain_location
from nurse_rcl.v2.live.ops_sre_report_render import RENDER_POLICY, render_sre_report_pair
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path


def inspect_sre_report_pair(directory: Path) -> dict[str, Any]:
    return _inspect_report_pair(directory, policy=RENDER_POLICY)


def _inspect_report_pair(directory: Path, *, policy: str) -> dict[str, Any]:
    require_plain_path(directory)
    try:
        raw = _read_bytes(directory / "report-delivery.json", 4096)
        receipt = json.loads(raw)
        if set(receipt) != {"policy", "binding_sha256", "members"} or (
            receipt["policy"] != policy
            or set(receipt["members"]) != {"report.md", "report-evidence.md"}
            or canonical_json_bytes(receipt) != raw
        ):
            raise ValueError("report delivery binding differs")
        for name, digest in receipt["members"].items():
            member = _read_bytes(directory / name, 16 * 1024 * 1024)
            if sha256_hex(member) != digest or not member.startswith(
                ("<!-- " + policy + " binding="
                 + receipt["binding_sha256"] + " -->").encode()
            ):
                raise ValueError("report delivery binding differs")
    except FileNotFoundError:
        raise ValueError("report delivery incomplete") from None
    return dict(receipt)


def save_sre_report_pair(directory: Path, **kwargs: Any) -> dict[str, Any]:
    require_plain_path(directory)
    pair = render_sre_report_pair(**kwargs)
    return _save_report_pair(directory, pair=pair, policy=RENDER_POLICY)


def _save_report_pair(directory: Path, *, pair: dict[str, str], policy: str) -> dict[str, Any]:
    require_plain_path(directory)
    members = {"report-evidence.md": pair["evidence"].encode(),
               "report.md": pair["report"].encode()}
    if any(len(data) > 16 * 1024 * 1024 for data in members.values()):
        raise ValueError("report delivery resource limit")
    # The receipt is committed last. An interrupted pair cannot pass inspection.
    for name in (*members, "report-delivery.json"):
        plain_location(directory / name)
    for name, data in members.items():
        atomic_write_bytes(directory / name, data)
    receipt = dict(policy=policy, binding_sha256=pair["binding_sha256"],
        members={name: sha256_hex(data) for name, data in members.items()})
    atomic_write_bytes(directory / "report-delivery.json", canonical_json_bytes(receipt))
    _fsync(directory)
    return _inspect_report_pair(directory, policy=policy)
