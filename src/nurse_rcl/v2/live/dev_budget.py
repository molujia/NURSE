"""Shared Dev accounting using the existing private phase socket and journal.

Only clean pause boundaries exclude operator idle time. An unclosed parent crash
needs diagnosis rather than inventing a refundable duration or a fresh budget.
"""

from __future__ import annotations

from pathlib import Path

from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.repair_driver import publish_json
from nurse_rcl.v2.live.repair_timing import (
    DEV_TIMING_POLICY,
    CasePhaseClock,
    TimingJournal,
    _time,
)
from nurse_rcl.v2.live.staged_case import require_plain_path


class DevRecoveryRequired(ValueError):
    """The elapsed interval of a crashed live worker has no safe closure."""


class DevBudget(TimingJournal):
    @property
    def warning_path(self) -> Path:
        return self.root.parent / ("method-warning-" + self.identity + ".json")

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        identity: str,
        now: float,
        boot_id: str,
        expected_policy: str = DEV_TIMING_POLICY,
    ) -> DevBudget:
        require_pin(identity)
        _time(now)
        if (
            expected_policy != DEV_TIMING_POLICY
            or type(boot_id) is not str
            or (not boot_id or len(boot_id) > 128)
        ):
            raise ValueError("invalid shared Dev timing policy/epoch")
        require_plain_path(root.parent)
        if root.is_symlink():
            raise ValueError("Dev timing root must not be a symlink")
        root.mkdir(mode=0o700, exist_ok=True)
        require_plain_path(root)
        if not list(root.iterdir()):
            result = cls(
                root,
                identity,
                boot_id,
                CasePhaseClock.start(
                    now=now,
                    policy=DEV_TIMING_POLICY,
                ),
            )
            result._append("start", "method", "orchestration", now)
        else:
            previous = cls._read_existing(
                root, identity=identity, boot_id=boot_id, expected_policy=DEV_TIMING_POLICY
            )
            assert isinstance(previous, cls)
            result = previous
            if result.clock.active:
                raise DevRecoveryRequired("unclosed Dev clock; no allowance reset or idle guess")
            result._record("resume", "method", "orchestration", now)
        result._warn()
        return result

    def _warn(self) -> None:
        if self.clock.used_seconds["method"] > 600:
            publish_json(
                self.warning_path,
                dict(
                    schema_version="nurse.v2.dev-method-warning/1",
                    identity=self.identity,
                    threshold_seconds=600,
                    hard_deadline=None,
                    action="continue_unchanged",
                ),
            )

    def _record(self, action: str, account: str, reason: str, now: float) -> None:
        super()._record(action, account, reason, now)
        self._warn()

    def pause(self, *, now: float) -> None:
        self._record("pause", "method", "orchestration", now)
