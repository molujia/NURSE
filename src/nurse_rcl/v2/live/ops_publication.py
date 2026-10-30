"""One-way, incremental subscriber of durable completed core publications."""

from __future__ import annotations

import json
from pathlib import Path

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.normal_artifacts import NormalQueryJournalEntry
from nurse_rcl.v2.live.normal_queries import _Journal, _read_bytes, _seal
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity, OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_tools import OpsToolbox
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES


class OpsPublicationReader:
    """Trusted owner adapter, not a model tool; no core write/lock/ack API.

    Immutable journal links are read once as they become available. Pending
    reservations/measurements are never advertised as completed reusable work.
    No source replay, scoring, blocking wait, or query selection occurs here.
    """

    def __init__(self, *, root: Path, data: RepairPreparationData, identity: OpsCoreIdentity):
        if identity.case_snapshot_id != prepared_snapshot_id(data):
            raise ValueError("publication preparation identity differs")
        self._core = root / "core"
        self._identity = identity
        self._journal = _Journal(data, self._core / "queries")
        self._node_files: set[str] = set()

    @property
    def completed_count(self) -> int:
        return len(self._journal.state.steps)

    def poll(self, toolbox: OpsToolbox) -> None:
        if toolbox.case_snapshot_id != self._identity.case_snapshot_id:
            raise ValueError("publication toolbox identity differs")
        identity_path = self._core / "identity.json"
        if not identity_path.exists() and not identity_path.is_symlink():
            return
        if _read_bytes(identity_path, 16384) != canonical_json_bytes(
            self._identity.model_dump(mode="json"),
        ):
            raise ValueError("publication core identity differs")
        # Admission validates every new receipt once. The core never waits for
        # this reader and the reader never modifies the producer's journal head.
        journal = self._journal
        while journal.state.count < 27:
            path = journal.root / f"{journal.state.count:03d}.json"
            if not path.exists() and not path.is_symlink():
                break
            raw = _read_bytes(path, REPAIR_MAX_JSON_BYTES)
            entry = NormalQueryJournalEntry.model_validate_json(raw)
            if raw != canonical_json_bytes(entry.model_dump(mode="json")) or (
                _seal(entry, "entry_sha256") != entry
            ):
                raise ValueError("publication receipt digest differs")
            successor = journal.transition(entry)
            if len(successor.steps) > len(journal.state.steps):
                toolbox.admit_core(nodes=(), steps=(successor.steps[-1],))
            journal.state = successor
        for name in ("initial-nodes.json", "diagnostics.json"):
            path = self._core / name
            if name in self._node_files or (not path.exists() and not path.is_symlink()):
                continue
            body = json.loads(_read_bytes(path, REPAIR_MAX_JSON_BYTES))
            if type(body) is not dict or type(body.get("nodes")) is not list or (
                body.get("provisional") is not (name == "initial-nodes.json")
            ):
                raise ValueError("publication node envelope differs")
            if name == "initial-nodes.json" and (
                body.get("case_snapshot_id") != self._identity.case_snapshot_id
            ):
                raise ValueError("publication node identity differs")
            nodes = tuple(OpsNodeEvidence.model_validate_json(canonical_json_bytes(n))
                          for n in body["nodes"])
            toolbox.admit_core(nodes=nodes, steps=())
            self._node_files.add(name)
