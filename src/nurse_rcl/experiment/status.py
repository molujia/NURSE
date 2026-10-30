"""Offline status inspection and CLI for an observable NURSE run directory."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.experiment.observability import (
    CompletionMarker,
    FatalMarker,
    read_progress_snapshot,
)
from nurse_rcl.models import ProgressSnapshot
from nurse_rcl.schemas import StrictFrozenModel

STALE_AFTER_SECONDS = 90
RunStatusName = Literal[
    "not_started",
    "running",
    "completed",
    "failed",
    "stale",
    "corrupt",
]


class RunStatus(StrictFrozenModel):
    run_root: Path
    status: RunStatusName
    healthy: bool
    run_id: str | None = None
    total: int | None = Field(default=None, ge=0)
    completed: int | None = Field(default=None, ge=0)
    heartbeat_age_seconds: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    detail: str


def inspect_run_status(*, run_root: Path, now_unix: int | None = None) -> RunStatus:
    """Classify terminal markers and heartbeat health without mutating the run."""

    root = run_root.resolve(strict=False)
    timestamp = int(time.time()) if now_unix is None else now_unix
    if timestamp < 0:
        raise ValueError("now_unix cannot be negative")
    progress_path = root / "progress.json"
    completion_path = root / "RUN_COMPLETE"
    fatal_path = root / "fatal.json"
    if completion_path.exists() and fatal_path.exists():
        return _corrupt(root, detail="multiple terminal markers exist")

    progress: ProgressSnapshot | None = None
    if progress_path.exists():
        try:
            progress = read_progress_snapshot(root)
        except (OSError, TypeError, ValueError) as error:
            return _corrupt(root, detail=f"progress is unreadable: {type(error).__name__}")

    if fatal_path.exists():
        try:
            marker = FatalMarker.model_validate_json(fatal_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            return _corrupt(root, detail=f"fatal marker is unreadable: {type(error).__name__}")
        if not marker.verify() or (progress is not None and progress.run_id != marker.run_id):
            return _corrupt(root, detail="fatal marker verification failed")
        return RunStatus(
            run_root=root,
            status="failed",
            healthy=False,
            run_id=marker.run_id,
            total=progress.total if progress is not None else None,
            completed=progress.completed if progress is not None else None,
            error_code=marker.error_code,
            detail=marker.message,
        )

    if completion_path.exists():
        try:
            completion_marker = CompletionMarker.model_validate_json(
                completion_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError) as error:
            return _corrupt(
                root,
                detail=f"completion marker is unreadable: {type(error).__name__}",
            )
        if (
            not completion_marker.verify()
            or progress is None
            or progress.run_id != completion_marker.run_id
            or progress.completed != progress.total
        ):
            return _corrupt(root, detail="completion marker verification failed")
        return RunStatus(
            run_root=root,
            status="completed",
            healthy=True,
            run_id=completion_marker.run_id,
            total=progress.total,
            completed=progress.completed,
            heartbeat_age_seconds=max(0, timestamp - progress.heartbeat_unix),
            detail=completion_marker.message,
        )

    if progress is None:
        return RunStatus(
            run_root=root,
            status="not_started",
            healthy=False,
            detail="progress.json is absent",
        )

    age = max(0, timestamp - progress.heartbeat_unix)
    if age > STALE_AFTER_SECONDS:
        status: RunStatusName = "stale"
        healthy = False
        detail = "heartbeat is stale"
    else:
        status = "running"
        healthy = True
        detail = "run is active"
    return RunStatus(
        run_root=root,
        status=status,
        healthy=healthy,
        run_id=progress.run_id,
        total=progress.total,
        completed=progress.completed,
        heartbeat_age_seconds=age,
        detail=detail,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a NURSE run directory")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--now-unix", type=int, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    result = inspect_run_status(
        run_root=arguments.run_root,
        now_unix=arguments.now_unix,
    )
    print(canonical_json_bytes(result.model_dump(mode="json")).decode("utf-8"))
    return 0 if result.healthy else 1


def _corrupt(root: Path, *, detail: str) -> RunStatus:
    return RunStatus(
        run_root=root,
        status="corrupt",
        healthy=False,
        detail=detail,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through console entry point
    raise SystemExit(main())
