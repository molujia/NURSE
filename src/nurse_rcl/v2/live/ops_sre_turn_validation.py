"""One frozen workbench request determines the allowed native response references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.hermes_response import ParsedExchangeResponse, parse_response
from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding
from nurse_rcl.v2.live.ops_empirical_policy import RESPONSE_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision, brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision


@dataclass(frozen=True)
class BoundWorkbenchTurn:
    binding: LiveRequestBinding
    context: dict[str, Any]


def bind_workbench_turn(frozen_wire: bytes, binding: LiveRequestBinding) -> BoundWorkbenchTurn:
    """Fail closed before reading paid bytes, without consulting later owner state."""
    if type(frozen_wire) is not bytes or type(binding) is not LiveRequestBinding:
        raise ValueError("workbench frozen request types differ")
    if binding.phase not in {"investigation", "investigation_correction"} or (
        binding.decision_index is None
    ):
        raise ValueError("workbench request index missing")
    try:
        expected = bind_brief_decision(
            frozen_wire,
            opaque_case_id=binding.opaque_case_id,
            case_snapshot_id=binding.input_sha256,
            decision_index=binding.decision_index,
        )
    except ExchangeBoundaryError:
        raise ValueError("workbench frozen request invalid") from None
    if expected != binding:
        raise ValueError("workbench frozen request binding differs")
    try:
        context = brief_context_from_wire(
            frozen_wire,
            opaque_case_id=binding.opaque_case_id,
            case_snapshot_id=binding.input_sha256,
            decision_index=binding.decision_index,
        )
    except ExchangeBoundaryError:
        raise ValueError("workbench frozen request invalid") from None
    return BoundWorkbenchTurn(binding=binding, context=context)


def validate_bound_workbench_response(
    raw: bytes, turn: BoundWorkbenchTurn
) -> ParsedExchangeResponse:
    """Check native envelope, output shape and references visible in this exact turn."""
    parsed = parse_response(raw, output_tool_name=None, limits=RESPONSE_LIMITS)
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/26":
        from nurse_rcl.v2.live.ops_sre_focused_decision import parse_focused_decision

        parse_focused_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/24":
        from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import parse_atomic_decision

        parse_atomic_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/23":
        from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import parse_typed_purpose_decision

        parse_typed_purpose_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/22":
        from nurse_rcl.v2.live.ops_sre_single_owner_decision import parse_single_owner_decision

        parse_single_owner_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/21":
        from nurse_rcl.v2.live.ops_sre_entry_decision import parse_observable_decision

        parse_observable_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") == "nurse.v2.sre-investigation-input/20":
        from nurse_rcl.v2.live.ops_sre_entry_decision import parse_entry_decision

        parse_entry_decision(parsed.output_json, context=turn.context)
        return parsed
    if turn.context.get("schema_version") in {
        "nurse.v2.sre-investigation-input/17",
        "nurse.v2.sre-investigation-input/18",
        "nurse.v2.sre-investigation-input/19",
    }:
        from nurse_rcl.v2.live.ops_sre_d26_decision import parse_question_decision

        parse_question_decision(parsed.output_json, context=turn.context)
    elif turn.context.get("schema_version") in {
        "nurse.v2.sre-investigation-input/15",
        "nurse.v2.sre-investigation-input/16",
    }:
        from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

        parse_balanced_decision(parsed.output_json, context=turn.context)
    else:
        parse_feedback_decision(parsed.output_json, context=turn.context)
    return parsed
