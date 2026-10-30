"""Explicit B2 report protocol over the same grounded fact and output contracts."""

from dataclasses import dataclass
from typing import Any, Final

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json
from nurse_rcl.v2.hermes_exchange import PreparedExchange
from nurse_rcl.v2.live.repair_report_prompt import EVIDENCE_REPORT_INSTRUCTIONS, _prompt_data
from nurse_rcl.v2.live.report_index import INDEXED_INSTRUCTIONS
from nurse_rcl.v2.report_composition_proposals import report_composition_output_schema
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceView
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy

B2_REPORT_EXCHANGE_VERSION: Final = "nurse.v2.normal-b2-report-exchange/1"
B2_REPORT_PROMPT_VERSION = "nurse-v2-normal-b2-report-prompt/1"
B2_REPORT_POLICY = ReportRunPolicy(
    configuration_version="normal-b2-report/1", input_token_limit=100000, output_token_limit=4096,
)
B2_REPORT_INSTRUCTIONS = INDEXED_INSTRUCTIONS + """
This is v2-live-hybrid-normal (B2), with pre-ranking semantics limited to legal
query priorities. The model cannot assign scores, weights or causal authority.
Each branch has at most eight logical tools, paid cards and verifications share
this limit; reuse still costs a logical slot. Semantic failure and baseline
fallback are explicitly recorded in the limit fact. Selected hypotheses do not
prove mechanisms or grant score authority. The unchanged transparent scorer
uses actual branch evidence, including contrary and unknown results. A selected
second snapshot replaces first hypotheses, not the actual measured evidence.
Explain the frozen numeric evidence, never claim an improvement over B0 without
supplied paired measurements. This report cannot change selection or ranking.
""" + EVIDENCE_REPORT_INSTRUCTIONS


@dataclass(frozen=True, slots=True)
class NormalB2ReportPrompt:
    payload_json: bytes

    def _data(self) -> dict[str, Any]:
        if type(self) is not NormalB2ReportPrompt:
            raise ValueError("exact B2 report prompt required")
        return _prompt_data(object.__getattribute__(self, "payload_json"),
                            version=B2_REPORT_PROMPT_VERSION, instructions=B2_REPORT_INSTRUCTIONS,
                            profile="v2-live-hybrid-normal")

    @property
    def sha256(self) -> str:
        self._data()
        return sha256_hex(self.payload_json)

    def messages(self) -> list[dict[str, str]]:
        data = self._data()
        return [dict(role="system", content=data["instructions"] + "\nOutput schema:\n"
                     + data["output_schema_json"]),
                dict(role="user", content=canonical_json_bytes(data["data"]).decode())]


def _b2_public_for_view(view: ReportEvidenceView) -> dict[str, Any]:
    from nurse_rcl.v2.live.repair_report_request import _public_for_view

    return dict(_public_for_view(view), prompt_version=B2_REPORT_PROMPT_VERSION,
                profile="v2-live-hybrid-normal")


def prepare_b2_report_request(
    view: ReportEvidenceView,
) -> tuple[NormalB2ReportPrompt, PreparedExchange]:
    from nurse_rcl.v2.live.report_exchange import prepare_report_exchange

    prompt = NormalB2ReportPrompt(encode_json(dict(
        prompt_version=B2_REPORT_PROMPT_VERSION, instructions=B2_REPORT_INSTRUCTIONS,
        output_schema_json=canonical_json_bytes(report_composition_output_schema()).decode(),
        data=_b2_public_for_view(view),
    ), ExchangeLimits(max_bytes=524288, max_nodes=100000)))
    return prompt, prepare_report_exchange(messages=prompt.messages(),
                                           protocol=B2_REPORT_EXCHANGE_VERSION)
