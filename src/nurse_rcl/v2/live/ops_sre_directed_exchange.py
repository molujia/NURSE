"""Directed-source policy only; no network, credentials or full live owner."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_empirical_policy import (
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
    SRE_DIRECTED_DECISION_PROTOCOL,
)
from nurse_rcl.v2.live.ops_sre_action_exchange import (
    ACTION_SKILL_GUIDANCE,
    ACTION_SYSTEM,
    action_common_guidance,
)

DIRECTED_DECISION_PROTOCOL = SRE_DIRECTED_DECISION_PROTOCOL
DIRECTED_SOURCE_GUIDANCE = (
    "Source navigation is not delivered evidence. Select inspect_sources with one or two "
    "issued card_ids and one or two issued candidate_service_keys to read the actual "
    "symptom sources for your question. Compare the same metric, family and baseline/fault "
    "windows when available; missing sides and mismatches remain gaps. Read complete "
    "values, windows, unit, coverage and episode status. A prepared summary is not a new "
    "episode measurement and paired same-source fields are not independent support. "
    "Use existing measure_signal only when a registered measurement is actually needed; "
    "reuse matching core results. Do not browse unrelated hash pages to reach a known "
    "issued source. A resource_limit receipt supplies no complete record. These entries "
    "are optional, not a required sequence or a selected root cause."
)
DIRECTED_SYSTEM = ACTION_SYSTEM + "\n" + DIRECTED_SOURCE_GUIDANCE
DIRECTED_SKILL_GUIDANCE = dict(
    ACTION_SKILL_GUIDANCE,
    symptoms=ACTION_SKILL_GUIDANCE["symptoms"] + " " + DIRECTED_SOURCE_GUIDANCE,
)


def directed_common_guidance() -> dict[str, str]:
    text = action_common_guidance()["guidance"] + " " + DIRECTED_SOURCE_GUIDANCE
    return dict(guidance=text, content_sha256=sha256_hex(text.encode()))


def directed_decision_policy() -> dict[str, Any]:
    pinned = dict(protocol=DIRECTED_DECISION_PROTOCOL, system=DIRECTED_SYSTEM,
                  output_policy=OUTPUT_POLICY, max_tokens=MAX_OUTPUT_TOKENS)
    return dict(protocol=DIRECTED_DECISION_PROTOCOL, system=DIRECTED_SYSTEM,
                profile="hermes-sre-tools/18", schema="nurse.v2.sre-investigation-input/19",
                guidance=directed_common_guidance(),
                prompt_sha256=sha256_hex(canonical_json_bytes(pinned)))
