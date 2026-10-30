"""D16 one-way terminal source publisher, outside the immutable core directory."""

from __future__ import annotations

import json

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live import ops_sre_source_checkpoint as source_limits
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_publication import OpsPublicationReader
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_comparable_tools import OpsSreComparableToolbox
from nurse_rcl.v2.live.ops_sre_score_lineage import (
    load_score_lineage,
    project_score_lineage,
    save_score_lineage,
)
from nurse_rcl.v2.live.ops_tools import OpsToolbox
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path


class OpsSreComparablePublicationReader(OpsPublicationReader):
    """Used only by /9. Existing readers never load the terminal result for tools."""

    _admitted: OpsSreComparableToolbox | None = None
    toolbox_type: type[OpsSreComparableToolbox] = OpsSreComparableToolbox

    def bind_checkpoint(
        self, toolbox: OpsSreComparableToolbox, *, core: OpsCoreResult | None = None
    ) -> None:
        """No I/O yet. Ledger calls this only after state identity/digest admission."""

        def restore() -> None:
            self.admit_terminal(toolbox, core=core, require_existing=True, admit_nodes=True)

        toolbox._restore_score_source = restore

    def publish_frozen(self, toolbox: OpsSreComparableToolbox, core: OpsCoreResult) -> None:
        self.admit_terminal(toolbox, core=core, admit_nodes=True)

    def poll(self, toolbox: OpsToolbox) -> None:
        if type(toolbox) is not self.toolbox_type:
            raise ValueError("comparable publisher requires exact new toolbox")
        super().poll(toolbox)
        if "diagnostics.json" in self._node_files:
            self.admit_terminal(toolbox)

    def admit_terminal(
        self,
        toolbox: OpsSreComparableToolbox,
        *,
        core: OpsCoreResult | None = None,
        require_existing: bool = False,
        admit_nodes: bool = False,
    ) -> None:
        """Use the already-admitted imported core, or admit one published result.

        Called at publication, not per tool/read. Missing projection can be
        recreated without queries; an incompatible existing artifact is retained
        and rejected, never overwritten or used to rewrite the core result.
        """
        if self._admitted is not None:
            if self._admitted is not toolbox:
                raise ValueError("comparable publisher already bound to another owner")
            return
        if type(toolbox) is not self.toolbox_type:
            raise ValueError("comparable publisher requires exact new toolbox")
        path = self._core / "result.json"
        if core is None:
            if not path.exists() and not path.is_symlink():
                return
            core = OpsCoreResult.model_validate_json(_read_bytes(path, REPAIR_MAX_JSON_BYTES))
        if type(core) is not OpsCoreResult or core.identity != self._identity:
            raise ValueError("comparable publication terminal core identity differs")
        diagnostics = json.loads(
            _read_bytes(self._core / "diagnostics.json", REPAIR_MAX_JSON_BYTES)
        )
        if diagnostics.get("core_result_digest") != core.core_result_digest or (
            diagnostics.get("provisional") is not False
        ):
            raise ValueError("comparable publication terminal diagnostics differ")
        target = self._core.parent / "sre-score-lineage.json"
        if target.exists() or target.is_symlink():
            size = require_plain_path(target).stat().st_size
            source_limits.require_capacity(
                size, source_limits.SOURCE_JSON_BYTES, "checkpoint_source_json_bytes"
            )
            projection = load_score_lineage(
                target, core=core, max_bytes=source_limits.SOURCE_JSON_BYTES
            )
        else:
            if require_existing:
                raise ValueError("bound score source is missing")
            projection = project_score_lineage(core)
            source_limits.require_capacity(
                len(canonical_json_bytes(projection.model_dump(mode="json"))),
                source_limits.SOURCE_JSON_BYTES,
                "checkpoint_source_json_bytes",
            )
            save_score_lineage(target, projection)
        if admit_nodes:
            toolbox.admit_core(
                nodes=tuple(OpsNodeEvidence.model_validate(n) for n in diagnostics["nodes"]),
                steps=() if require_existing else core.evidence.session.steps,
            )
        toolbox.admit_score_lineage(projection)
        self._admitted = toolbox
