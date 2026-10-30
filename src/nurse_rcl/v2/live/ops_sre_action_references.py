"""Separate action names within the original shared append-only reference budget."""

from typing import ClassVar

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_references import PREFIXES, SreReferenceBook, _index


class SreActionReferenceBook(SreReferenceBook):
    schema_version = "nurse.v2.sre-reference-book/2"
    prefixes: ClassVar[dict[str, str]] = {**PREFIXES, "action": "A"}

    def source_prefix(self, *, visible_at: int) -> str:
        """Bind ordinary issuance only; never prune persisted historical actions."""
        _index(visible_at)
        return sha256_hex(
            canonical_json_bytes(
                dict(
                    policy="sre-action-source-prefix/1",
                    snapshot=self.snapshot,
                    run_id=self.run_id,
                    entries=[
                        list(row)
                        for row in self._rows
                        if row[0] != "action" and row[3] <= visible_at
                    ],
                )
            )
        )
