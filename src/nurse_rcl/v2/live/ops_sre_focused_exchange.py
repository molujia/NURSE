"""New bounded focused wire identity over original source and transport guards."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import (
    ATOMIC_SYSTEM,
    atomic_common_guidance,
)
from nurse_rcl.v2.live.ops_sre_focused_decision import FOCUSED_INPUT_SCHEMA, FOCUSED_TOOLS

FOCUSED_PROTOCOL = "nurse.v2.sre-decision-exchange/26"


def focused_decision_policy() -> dict[str, Any]:
    system = ATOMIC_SYSTEM + (
        "\nEvidence catalog navigation uses catalog_kind=evidence with filters containing "
        "candidate_service_keys (one or two distinct issued services), optional exact "
        "metric_name/signal_family/record_kind, and cursor. Directory items are not delivered "
        "evidence. Their exact read/inspect/measure actions enter the next frozen menu and "
        "require another action. source_relation/2 reports comparability qualifiers only."
    )
    return dict(
        protocol=FOCUSED_PROTOCOL,
        system=system,
        profile=FOCUSED_TOOLS,
        schema=FOCUSED_INPUT_SCHEMA,
        guidance=atomic_common_guidance(),
        prompt_sha256=sha256_hex(
            canonical_json_bytes(
                dict(
                    protocol=FOCUSED_PROTOCOL,
                    system=system,
                    output_policy=OUTPUT_POLICY,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            )
        ),
    )


def prepare_focused_decision(**kwargs: Any) -> bytes:
    from nurse_rcl.v2.live.ops_sre_brief_exchange import prepare_brief_decision

    return prepare_brief_decision(**kwargs)
