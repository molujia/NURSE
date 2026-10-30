"""One exact request builder for repair selection and final report composition.

Building bytes grants no source authority and performs no network operation.
"""

from __future__ import annotations

import json
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json
from nurse_rcl.v2.hermes_exchange import PreparedExchange
from nurse_rcl.v2.live.repair_report_prompt import (
    REPAIR_EXCHANGE_VERSION,
    REPAIR_INSTRUCTIONS,
    REPAIR_PROMPT_VERSION,
    RepairReportPrompt,
)
from nurse_rcl.v2.live.report_compact import encode_compact_view
from nurse_rcl.v2.live.report_exchange import prepare_report_exchange
from nurse_rcl.v2.report_composition_proposals import report_composition_output_schema
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceView
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy

REPAIR_REPORT_CONFIGURATION = "live-evidence-repair-report/2"
RUN_POLICY = ReportRunPolicy(
    configuration_version=REPAIR_REPORT_CONFIGURATION,
    input_token_limit=100000,
    output_token_limit=4096,
)


def _public_for_view(view: ReportEvidenceView) -> dict[str, Any]:
    # Navigation copies every displayed fact; the prompt checks exact equality.
    return dict(
        prompt_version=REPAIR_PROMPT_VERSION,
        profile="v2-live-evidence-repair",
        evidence_view=json.loads(encode_compact_view(view)),
        readable_index=[
            dict(f.model_dump(mode="json"), values={v.name: v.value for v in f.values})
            for f in view.facts
        ],
    )


def prepare_repair_report_request(
    view: ReportEvidenceView,
) -> tuple[RepairReportPrompt, PreparedExchange]:
    prompt = RepairReportPrompt(
        encode_json(
            dict(
                prompt_version=REPAIR_PROMPT_VERSION,
                instructions=REPAIR_INSTRUCTIONS,
                output_schema_json=canonical_json_bytes(report_composition_output_schema()).decode(),
                data=_public_for_view(view),
            ),
            ExchangeLimits(max_bytes=524288, max_nodes=100000),
        )
    )
    request = prepare_report_exchange(
        messages=prompt.messages(), protocol=REPAIR_EXCHANGE_VERSION
    )
    return prompt, request
