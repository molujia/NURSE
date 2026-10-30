"""Manual completion over the original owner and codecs; never activates ARK."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from nurse_rcl.v2.live.blind_compare_mailbox import DecisionMailbox
from nurse_rcl.v2.live.blind_compare_contract import (
    canonical, digest, read_document, require_pin, write_once,
)


class ManualCompletion:
    def __init__(self, root: Path, *, owner_sha256: str, case_id: str, snapshot: str,
                 ceiling: int = 8, on_wait: Callable[..., None] | None = None,
                 progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.box = DecisionMailbox(root, owner_sha256=owner_sha256, ceiling=ceiling)
        self.case, self.snapshot, self.ceiling = case_id, snapshot, ceiling
        self.on_wait, self.progress = on_wait, progress

    def _binding(self, index: int, wire: bytes) -> Any:
        from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import bind_entry_decision

        if type(index) is not int or not 1 <= index <= self.ceiling or type(wire) is not bytes:
            raise ValueError("manual completion exceeds exact owner grant")
        return bind_entry_decision(wire, case_id=self.case, snapshot=self.snapshot, index=index)

    @staticmethod
    def _validate(native: bytes, wire: bytes, binding: Any) -> bytes:
        from nurse_rcl.v2.live.ops_sre_turn_validation import (
            bind_workbench_turn, validate_bound_workbench_response,
        )

        return validate_bound_workbench_response(native, bind_workbench_turn(wire, binding)).output_json

    def decision(self, index: int, wire: bytes) -> bytes:
        binding = self._binding(index, wire)
        correction = binding.correction_index or 0
        row = self.box.publish(index, wire, correction=correction)
        last_notice = float("-inf")
        while True:
            native = self.box.response(index, wire, correction=correction)
            if native is not None:
                return self._validate(native, wire, binding)
            now = monotonic()
            if self.progress is not None and now - last_notice >= 30:
                self.progress(dict(event="primary_codex_completion_waiting", index=index,
                                   correction_index=correction,
                                   request_sha256=row["request_sha256"], ark_calls=0))
                last_notice = now
            if self.on_wait is not None:
                self.on_wait(index, wire, self.box, correction)
            else:
                sleep(0.5)

    def saved_decision(self, index: int, wire: bytes) -> bytes | None:
        binding = self._binding(index, wire)
        native = self.box.response(index, wire, correction=binding.correction_index or 0)
        return None if native is None else self._validate(native, wire, binding)


class ManualStageCompletion(ManualCompletion):
    """Exact supervised-stage signature, with explicit non-provider provenance."""

    def __init__(self, *, root: Path, code_sha256: str, execution_sha256: str,
                 approved_execution_sha256: str | None, opaque_case_id: str,
                 case_snapshot_id: str, stage_decision_limit: int,
                 approved_stage_sha256: str | None) -> None:
        for pin in (code_sha256, execution_sha256, case_snapshot_id):
            require_pin(pin)
        grant = dict(schema_version="nurse.v2.sre-stage-grant/1",
                     execution_sha256=execution_sha256,
                     through_decision_index=stage_decision_limit,
                     correction_limit=1, report_limit=0)
        if (type(stage_decision_limit) is not int or not 1 <= stage_decision_limit <= 8
                or approved_execution_sha256 != execution_sha256
                or approved_stage_sha256 != digest(canonical(grant))):
            raise ValueError("manual stage exact plan/grant approval differs")
        import json

        super().__init__(root.parent / "manual-completion", owner_sha256=execution_sha256,
                         case_id=opaque_case_id, snapshot=case_snapshot_id,
                         ceiling=stage_decision_limit,
                         progress=lambda row: print(json.dumps(row), flush=True))
        origin = dict(schema="nurse.research.manual-completion-origin/1",
                      completion_source="primary_codex_manual", ark_calls=0,
                      codex_session_cost_is_zero=False, code_sha256=code_sha256,
                      execution_sha256=execution_sha256, stage_grant_sha256=approved_stage_sha256,
                      opaque_case_id=opaque_case_id, case_snapshot_id=case_snapshot_id)
        path = self.box.root / "origin.json"
        if path.exists() or path.is_symlink():
            if read_document(path) != origin:
                raise ValueError("manual completion origin differs")
        else:
            write_once(path, canonical(origin))


def manual_worker_command(command: list[str]) -> list[str]:
    if len(command) < 3 or command[1:3] != ["-m", "nurse_rcl.v2.live.ops_sre_directed_stage"]:
        raise ValueError("manual supervision original command differs")
    return [*command[:2], "nurse_rcl.v2.live.blind_compare_worker", *command[3:]]


def main() -> int:
    """Use original phase supervisor/stage; only replace the completion source."""
    from nurse_rcl.v2.live import ops_sre_directed_stage as stage
    from nurse_rcl.v2.live.ops_contracts import OpsSreFeedbackFactProfile

    def manual_provider(profile: object) -> type[ManualStageCompletion]:
        if type(profile) is not OpsSreFeedbackFactProfile:
            raise ValueError("manual experiment requires the exact frozen feedback profile")
        return ManualStageCompletion

    original = stage.supervise_phases

    def supervise(command: list[str], **options: Any) -> int:
        return original(manual_worker_command(command), **options)

    stage.stage_provider_type = manual_provider
    stage.supervise_phases = supervise
    return stage.main()


def run_manual_owner(*, root: Path, toolbox: Any, code_sha256: str,
                     opaque_case_id: str, completion: ManualCompletion,
                     ceiling: int = 8) -> Any:
    from nurse_rcl.v2.live.ops_agent import run_ops_agent

    if completion.ceiling != ceiling or completion.case != opaque_case_id:
        raise ValueError("manual owner and completion identity differ")
    return run_ops_agent(
        root=root, toolbox=toolbox, code_sha256=code_sha256,
        complete=completion.decision, recover_complete=completion.saved_decision,
        empirical=True, sre=True, sre_profile="hermes-sre-tools/23",
        opaque_case_id=opaque_case_id, stage_decision_limit=ceiling, feedback_fact=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
