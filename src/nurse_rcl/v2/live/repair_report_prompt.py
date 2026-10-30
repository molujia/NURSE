"""New repair prompt identity over the unchanged bounded reversible wire."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, decode_json_object
from nurse_rcl.v2.live.report_compact import decode_compact_view
from nurse_rcl.v2.live.report_index import INDEXED_INSTRUCTIONS
from nurse_rcl.v2.report_composition_proposals import report_composition_output_schema

REPAIR_PROMPT_VERSION = "nurse-v2-live-evidence-repair-report-prompt/1"
REPAIR_EXCHANGE_VERSION: Final = "nurse.v2.live-repair-report-exchange/1"
EVIDENCE_REPORT_INSTRUCTIONS = """
The readable_index copies EVERY displayed fact exactly, never adds evidence.
Use mechanism conditions, directional event measurements and dependent-vote
limitations to explain the fixed ranking. Accepted mechanisms satisfy observable
contracts, not proven causality. Unknown/false/refuting conditions remain binding.
A window summary does not prove persistence or an onset. A state-zero existence
does not prove outage duration. Paid episodes retain their measured direction.
An error share with unknown statuses is a lower bound over observed records, not
a population request rate. Logs do not establish a request rate. Dependence pools
are not independent events. Scores and local role effects are ranking inferences.
Dependency-aligned symptom support is not confirmed dependency failure or causal
proof. Missing edge coverage or an unmeasured onset prevents propagation credit.
Explain at least the leading evidence, a useful alternative and the actual limits
when supported; don't merely repeat ranks. Never amend ranking or request new data.
Only propose grounded read-only verification; no mitigation is licensed by this
profile. Produce 3-6 concise Chinese claims if evidence permits, not a fixed quota.
"""
REPAIR_INSTRUCTIONS = (
    INDEXED_INSTRUCTIONS
    + """
This is v2-live-evidence-repair, NOT bootstrap or full B2. Dynamic verification is
enabled with at most eight paid queries; pre-ranking model semantics is disabled.
"""
    + EVIDENCE_REPORT_INSTRUCTIONS.lstrip("\n")
)
_LIMITS = ExchangeLimits(max_bytes=524288, max_nodes=100000)


@dataclass(frozen=True, slots=True)
class RepairReportPrompt:
    payload_json: bytes

    def _data(self) -> dict[str, Any]:
        if type(self) is not RepairReportPrompt:
            raise ValueError("exact repair report prompt required")
        return _prompt_data(object.__getattribute__(self, "payload_json"),
                            version=REPAIR_PROMPT_VERSION, instructions=REPAIR_INSTRUCTIONS,
                            profile="v2-live-evidence-repair")

    @property
    def sha256(self) -> str:
        self._data()
        return sha256_hex(self.payload_json)

    def messages(self) -> list[dict[str, str]]:
        data = self._data()
        return [
            dict(
                role="system",
                content=data["instructions"] + "\nOutput schema:\n" + data["output_schema_json"],
            ),
            dict(role="user", content=canonical_json_bytes(data["data"]).decode()),
        ]


def _prompt_data(raw: bytes, *, version: str, instructions: str, profile: str) -> dict[str, Any]:
    """Shared complete wire/index check; callers fix their exact registered identity."""
    data = decode_json_object(raw, _LIMITS)
    if (
        set(data) != {"prompt_version", "instructions", "output_schema_json", "data"}
        or data["prompt_version"] != version or data["instructions"] != instructions
        or data["output_schema_json"]
        != canonical_json_bytes(report_composition_output_schema()).decode()
        or type(data["data"]) is not dict or canonical_json_bytes(data) != raw
    ):
        raise ValueError("repair prompt identity or shape differs")
    public = data["data"]
    if (set(public) != {"prompt_version", "profile", "evidence_view", "readable_index"}
        or public["prompt_version"] != version or public["profile"] != profile):
        raise ValueError("repair prompt public shape differs")
    if len(canonical_json_bytes(public)) > 80000:
        raise ExchangeBoundaryError("json_limit")
    wire = public["evidence_view"]
    view = decode_compact_view(canonical_json_bytes(wire), expected_view_sha256=wire["view_sha256"])
    expected = [dict(f.model_dump(mode="json"), values={v.name: v.value for v in f.values})
                for f in view.facts]
    if canonical_json_bytes(public["readable_index"]) != canonical_json_bytes(expected):
        raise ValueError("repair readable index differs from complete wire")
    return data
