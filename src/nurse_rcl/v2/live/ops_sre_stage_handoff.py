"""Read-only product handoff from a complete, externally bound successful stage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_plan import plain_location


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _document(path: Path) -> dict[str, Any]:
    raw = _read_bytes(path, 4 * 1024**2)
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError("stage handoff document is not canonical")
    return value


def validate_stage_success(
    *,
    round_result: dict[str, Any],
    outcome: dict[str, Any],
    identity: dict[str, Any],
    grant_sha256: str,
    worker_invocation_id: str,
    prefix: dict[str, list[str]],
) -> None:
    """Bind the complete signed prefix to the approved plan, grant and invocation."""
    require_pin(grant_sha256)
    try:
        unsigned_round = {k: v for k, v in round_result.items() if k != "round_result_sha256"}
        unsigned_outcome = {k: v for k, v in outcome.items() if k != "stage_result_sha256"}
        count, actions = outcome["decision_count"], outcome["action_count"]
        grant = outcome["approved_grant"]
        ceiling = grant["through_decision_index"]
        if (
            round_result["round_result_sha256"] != _digest(unsigned_round)
            or outcome["stage_result_sha256"] != _digest(unsigned_outcome)
            or outcome["schema_version"] != "nurse.v2.sre-stage-outcome/1"
            or any(k not in round_result or round_result[k] != v for k, v in outcome.items())
            or outcome["identity"] != identity
            or type(worker_invocation_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", worker_invocation_id) is None
            or round_result["worker_invocation_id"] != worker_invocation_id
            or type(round_result["worker_exit_code"]) is not int
            or round_result["worker_exit_code"] != 0
            or grant != round_result["requested_grant"]
            or _digest(grant) != grant_sha256
            or type(ceiling) is not int
            or not 1 <= ceiling <= 8
            or ("plan_sha256" in identity and grant["execution_sha256"] != identity["plan_sha256"])
            or outcome["primary_failure"] is not None
            or round_result.get("failure_type") is not None
            or set(outcome["channels"]) != {"core", "report", "investigation"}
            or outcome["channels"]["core"] != "completed"
            or outcome["channels"]["report"] != "disabled"
            or outcome["channels"]["investigation"] not in {"completed", "partial"}
            or outcome["termination"]
            not in {"stage_pause", "model_stop", "decision_budget", "no_progress_after_correction"}
            or round_result["status"]
            != ("stage_paused" if outcome["termination"] == "stage_pause" else "stage_terminal")
            or type(count) is not int
            or type(actions) is not int
            or not 0 <= actions <= count <= ceiling
            or count < 1
            or type(prefix) is not dict
            or set(prefix) != {"decisions", "actions"}
            or any(type(prefix[k]) is not list for k in ("decisions", "actions"))
            or outcome["prefix_pins"] != prefix
            or len(prefix["decisions"]) != count
            or len(prefix["actions"]) != actions
        ):
            raise ValueError("successful stage external binding differs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("mandatory successful stage source missing") from error
    for pin in (*prefix["decisions"], *prefix["actions"]):
        require_pin(pin)


def optional_worker_diagnostic(
    root: Path,
    *,
    identity: dict[str, Any],
    grant_sha256: str,
    worker_invocation_id: str,
) -> Any:
    """Absence is permitted only after the caller admits the full successful stage."""
    path = plain_location(root / "worker-outcome.json")
    if not path.exists():
        return None
    from nurse_rcl.v2.live.ops_worker_outcome import read_outcome

    worker = read_outcome(
        root,
        dict(identity, stage_grant_sha256=grant_sha256, worker_invocation_id=worker_invocation_id),
    )
    if worker.primary_failure is not None:
        raise ValueError("stage has an actual worker primary failure")
    return worker


def _focused_checkpoint(navigation: dict[str, Any], *, snapshot: str) -> dict[str, Any]:
    """Admit metadata identity without expanding pools or restoring an owner."""
    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import validate_pooled_source_envelope
    from nurse_rcl.v2.live.ops_sre_focused_catalog import CATALOG_SCHEMA

    if type(navigation) is not dict:
        raise ValueError("focused report checkpoint envelope differs")
    validate_pooled_source_envelope(navigation)
    base = navigation["base"]
    if base.get("profile") != "hermes-sre-tools/25":
        raise ValueError("focused report checkpoint profile differs")
    if "source_registry" not in base or "evidence_catalog" not in base:
        raise ValueError("focused report checkpoint metadata is absent")
    catalog = base["evidence_catalog"]
    if (
        type(catalog) is not dict
        or set(catalog)
        != {
            "schema_version",
            "snapshot",
            "rows",
            "prefixes",
            "cursors",
            "state_sha256",
        }
        or (
            catalog["schema_version"] != CATALOG_SCHEMA
            or catalog["snapshot"] != snapshot
            or type(catalog["rows"]) is not list
            or type(catalog["prefixes"]) is not dict
            or type(catalog["cursors"]) is not dict
            or catalog["state_sha256"]
            != _digest({key: value for key, value in catalog.items() if key != "state_sha256"})
        )
    ):
        raise ValueError("focused report catalog checkpoint identity differs")
    return navigation


def retained_sources(
    stage: Any, ledger: Any, core: Any, *, focused: bool = False
) -> tuple[dict[str, Any], ...]:
    """Decode the current /3 checkpoint without restoring an owner or dispatching."""
    from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
    from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION
    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import unpack_pooled_source_navigation
    from nurse_rcl.v2.live.ops_sre_comparable_briefs import project_comparable_brief
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_score_lineage import load_score_lineage
    from nurse_rcl.v2.live.ops_sre_source_checkpoint import admit_source
    from nurse_rcl.v2.live.ops_sre_views import semantic_record
    from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view
    from nurse_rcl.v2.observation_schemas import TimeInterval

    registry: dict[str, Any] = {}
    focused_navigation: dict[str, Any] = {}
    if focused:
        from nurse_rcl.v2.live.ops_sre_focused_sources import (
            enrich_registered,
            validate_source_registry,
        )

        focused_navigation = _focused_checkpoint(
            ledger.navigation, snapshot=core.identity.case_snapshot_id
        )
        registry = validate_source_registry(ledger.navigation["base"]["source_registry"])
    required = []
    for record in ledger.records:
        if record.model_view_json is not None:
            view = working_evidence_view(json.loads(record.model_view_json))
            for row in _delivered_records(view):
                if row.get("representation") == BRIEF_REPRESENTATION:
                    required.append((view, row))
                elif focused and canonical_json_bytes(
                    enrich_registered(row, registry)
                ) != canonical_json_bytes(row):
                    raise ValueError(
                        "focused report delivered attribution differs from saved registry"
                    )
    if not required:
        return ()
    nav, source = focused_navigation if focused else ledger.navigation, None
    if nav["source"] is not None:
        projection = load_score_lineage(stage.state_root / "sre-score-lineage.json", core=core)
        source = admit_source(
            projection, interval=TimeInterval.model_validate(nav["source"]["interval"])
        )
    decoded = unpack_pooled_source_navigation(nav, source=source)
    cards, prefixes = decoded["indexed"]["cards"], decoded["inspection_prefixes"]
    retained = {}
    for view, row in required:
        prefix, key = prefixes.get(view.get("publication_prefix", "")), row["card_id"]
        if prefix is None or key not in prefix["ids"] or key not in cards:
            raise ValueError("G+ complete retained source prefix missing")
        full = semantic_record(
            OpsNodeEvidence.model_validate(cards[key]),
            published=key in prefix["core_cards"],
            origin=decoded["origins"].get(key),
        )
        full.update(
            source_role=prefix["record_roles"][key],
            targets=prefix["record_targets"][key],
            published_by_core=key in prefix["core_cards"],
        )
        if focused:
            full = enrich_registered(full, registry)
        if canonical_json_bytes(project_comparable_brief(full)) != canonical_json_bytes(row):
            raise ValueError("G+ delivered brief differs from complete saved source")
        retained[_digest(row)] = full
    return tuple(retained[key] for key in sorted(retained))


def stage_report_context(
    stage: Any,
    *,
    grant_sha256: str,
    worker_invocation_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Admit saved originals before constructing the explicitly versioned report context."""
    from nurse_rcl.v2.live.ops_agent_state import (
        OpsAtomicSelectorConversation,
        OpsFocusedConversation,
        load_conversation,
    )
    from nurse_rcl.v2.live.ops_case import case_identity, inspect_saved_core
    from nurse_rcl.v2.live.ops_plan import require_ops_code
    from nurse_rcl.v2.live.ops_report_context import build_report_context
    from nurse_rcl.v2.live.ops_sre_atomic_report import (
        AtomicReportSourceIdentity,
        assemble_atomic_report_context,
    )
    from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import OpsSreAtomicSelectorLedgerState
    from nurse_rcl.v2.live.ops_sre_directed_provider import stage_grant
    from nurse_rcl.v2.live.ops_sre_directed_stage import _inspect_stage, _prefix_pins
    from nurse_rcl.v2.live.ops_sre_focused_report import (
        FocusedReportSourceIdentity,
        assemble_focused_report_context,
    )
    from nurse_rcl.v2.live.ops_sre_focused_tools import OpsSreFocusedLedgerState

    require_ops_code(stage.plan)
    focused = stage.plan.profile.profile_id == "v2-ops-sre/15"
    if stage.plan.profile.profile_id not in {"v2-ops-sre/14", "v2-ops-sre/15"}:
        raise ValueError("stage report requires an explicit admitted source profile")
    root, identity = stage.state_root, case_identity(stage)
    outcome = _inspect_stage(root, identity)
    if _document(root / "stage-outcome.json") != outcome:
        raise ValueError("successful stage outcome changed during admission")
    prefix = _prefix_pins(root, identity)
    round_result = _document(root / "stage-round.json")
    validate_stage_success(
        round_result=round_result,
        outcome=outcome,
        identity=identity,
        grant_sha256=grant_sha256,
        worker_invocation_id=worker_invocation_id,
        prefix=prefix,
    )
    if outcome["approved_grant"] != stage_grant(
        stage.plan_sha256, outcome["approved_grant"]["through_decision_index"]
    ):
        raise ValueError("successful stage approved plan/grant differs")
    optional_worker_diagnostic(
        root,
        identity=identity,
        grant_sha256=grant_sha256,
        worker_invocation_id=worker_invocation_id,
    )
    core = inspect_saved_core(root, stage.plan)
    if core.core_result_digest != outcome["core_result_digest"]:
        raise ValueError("successful stage core digest differs")
    conversation = load_conversation(
        root / "investigation/conversation.json",
        snapshot=core.identity.case_snapshot_id,
        code=stage.plan.code_sha256,
        empirical=True,
        opaque_case_id=core.identity.opaque_case_id,
        sre=True,
        sre_profile="hermes-sre-tools/25" if focused else "hermes-sre-tools/23",
    )
    conversation_type = OpsFocusedConversation if focused else OpsAtomicSelectorConversation
    if type(conversation) is not conversation_type:
        raise ValueError("G+ report actual conversation differs")
    assert isinstance(conversation, OpsAtomicSelectorConversation)
    ledger_type = OpsSreFocusedLedgerState if focused else OpsSreAtomicSelectorLedgerState
    ledger = ledger_type.model_validate_json(
        _read_bytes(root / "investigation/state.json", 64 * 1024**2)
    )
    retained = retained_sources(stage, ledger, core, focused=focused)
    formal = build_report_context(core=core, sources=(), investigation=(), soft_targets=True)
    if not formal.formal_projection_available or not formal.facts:
        raise ValueError("G+ report complete formal source unavailable")
    refs = conversation.reference_state
    source_type = FocusedReportSourceIdentity if focused else AtomicReportSourceIdentity
    source = source_type(
        source_plan_sha256=stage.plan_sha256,
        source_code_sha256=stage.plan.code_sha256,
        stage_grant_sha256=grant_sha256,
        stage_round_sha256=round_result["round_result_sha256"],
        worker_invocation_id=worker_invocation_id,
        opaque_case_id=core.identity.opaque_case_id,
        case_snapshot_id=core.identity.case_snapshot_id,
        source_run_id=refs["run_id"],
        core_result_digest=core.core_result_digest,
        formal_context_sha256=formal.context_sha256,
        reference_state_sha256=_digest(refs),
        decision_pins=tuple(prefix["decisions"]),
        action_pins=tuple(prefix["actions"]),
        retained_sources_sha256=_digest(retained),
        termination=outcome["termination"],
        investigation_status=outcome["channels"]["investigation"],
    )
    assemble = assemble_focused_report_context if focused else assemble_atomic_report_context
    context = assemble(
        formal=formal,
        decisions=conversation.decisions,
        records=ledger.records,
        references=refs,
        retained_sources=retained,
        source_identity=source,
        expected_source_sha256=_digest(source.model_dump(mode="json")),
    )
    return context, refs
