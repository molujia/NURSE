"""Product success handoff reads sealed synthetic originals without execution."""

# ruff: noqa: F811 -- pytest injects the shared imported fixture by parameter name.

from copy import deepcopy
from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.sre_feedback_fact_fixtures import deny_effects  # noqa: F401


@pytest.fixture(autouse=True)
def no_dispatch(deny_effects, monkeypatch):
    from tests.sre_atomic_report_fixtures import forbid_dispatch

    forbid_dispatch(monkeypatch, deny_effects)


def digest(value):
    return sha256_hex(canonical_json_bytes(value))


def api():
    name = "nurse_rcl.v2.live.ops_sre_stage_handoff"
    assert util.find_spec(name), "missing product successful-stage handoff"
    return import_module(name)


def signed_success(*, ceiling=8):
    identity = dict(profile="v2-ops-sre/14", code_sha256="c" * 64, plan_sha256="e" * 64)
    grant = dict(
        schema_version="nurse.v2.sre-stage-grant/1",
        execution_sha256="e" * 64,
        through_decision_index=ceiling,
        correction_limit=1,
        report_limit=0,
    )
    prefix = dict(decisions=["d" * 64], actions=["a" * 64])
    outcome = dict(
        schema_version="nurse.v2.sre-stage-outcome/1",
        identity=identity,
        approved_grant=grant,
        channels=dict(core="completed", investigation="partial", report="disabled"),
        decision_count=1,
        action_count=1,
        termination="stage_pause",
        primary_failure=None,
        secondary_failures=[],
        prefix_pins=prefix,
    )
    outcome["stage_result_sha256"] = digest(outcome)
    result = dict(
        outcome,
        requested_grant=grant,
        worker_exit_code=0,
        worker_invocation_id="1" * 32,
        status="stage_paused",
    )
    result["round_result_sha256"] = digest(result)
    return dict(
        round_result=result,
        outcome=outcome,
        identity=identity,
        grant_sha256=digest(grant),
        worker_invocation_id="1" * 32,
        prefix=prefix,
    )


def reseal(value):
    outcome, result = value["outcome"], value["round_result"]
    outcome["stage_result_sha256"] = digest(
        {k: v for k, v in outcome.items() if k != "stage_result_sha256"}
    )
    result.update(outcome)
    result["round_result_sha256"] = digest(
        {k: v for k, v in result.items() if k != "round_result_sha256"}
    )


def test_product_success_admission_allows_absent_failure_only_diagnostic(tmp_path):
    module, value = api(), signed_success()
    module.validate_stage_success(**value)
    assert (
        module.optional_worker_diagnostic(
            tmp_path,
            identity=value["identity"],
            grant_sha256=value["grant_sha256"],
            worker_invocation_id=value["worker_invocation_id"],
        )
        is None
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "damage",
    [
        "exit",
        "status",
        "primary",
        "identity",
        "invocation",
        "grant",
        "prefix",
        "count",
        "ceiling",
        "boolean_ceiling",
        "execution",
        "channels",
        "termination",
    ],
)
def test_resealed_success_does_not_authorize_changed_external_contract(damage):
    value = signed_success()
    outcome, result = value["outcome"], value["round_result"]
    if damage == "exit":
        result["worker_exit_code"] = 1
    elif damage == "status":
        result["status"] = "failed"
    elif damage == "primary":
        outcome["primary_failure"] = dict(failure_type="ActualFailure")
    elif damage == "identity":
        value["identity"] = dict(value["identity"], code_sha256="f" * 64)
    elif damage == "invocation":
        value["worker_invocation_id"] = "2" * 32
    elif damage == "grant":
        value["grant_sha256"] = "f" * 64
    elif damage == "prefix":
        value["prefix"] = dict(decisions=["f" * 64], actions=["a" * 64])
    elif damage == "count":
        outcome["decision_count"] = 2
    elif damage in {"ceiling", "boolean_ceiling"}:
        outcome["approved_grant"]["through_decision_index"] = 0 if damage == "ceiling" else True
        value["grant_sha256"] = digest(outcome["approved_grant"])
    elif damage == "execution":
        outcome["approved_grant"]["execution_sha256"] = "f" * 64
        value["grant_sha256"] = digest(outcome["approved_grant"])
    elif damage == "channels":
        outcome["channels"]["core"] = "failed"
    else:
        outcome["termination"] = "unavailable"
    reseal(value)
    with pytest.raises(ValueError):
        api().validate_stage_success(**value)


def test_approved_through_index_is_checked_even_below_product_eight_limit():
    value = signed_success(ceiling=1)
    value["outcome"].update(decision_count=2, action_count=2)
    prefix = dict(decisions=["d" * 64, "e" * 64], actions=["a" * 64, "b" * 64])
    value["prefix"] = prefix
    value["outcome"]["prefix_pins"] = deepcopy(prefix)
    reseal(value)
    with pytest.raises(ValueError):
        api().validate_stage_success(**value)


@pytest.mark.parametrize("damage", ["failure", "identity", "corrupt", "noncanonical", "symlink"])
def test_present_worker_diagnostic_remains_strict(tmp_path, damage):
    from nurse_rcl.v2.live.ops_failure import failure_detail
    from nurse_rcl.v2.live.ops_worker_outcome import write_outcome

    value = signed_success()
    identity = dict(
        value["identity"],
        stage_grant_sha256=value["grant_sha256"],
        worker_invocation_id=value["worker_invocation_id"],
    )
    if damage == "identity":
        identity["worker_invocation_id"] = "2" * 32
    primary = (
        failure_detail("investigation_tool", ValueError("synthetic"))
        if damage == "failure"
        else None
    )
    write_outcome(tmp_path, identity, primary)
    path = tmp_path / "worker-outcome.json"
    if damage == "corrupt":
        path.write_bytes(b"{}")
    elif damage == "noncanonical":
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "symlink":
        target = tmp_path / "synthetic-diagnostic.json"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError):
        api().optional_worker_diagnostic(
            tmp_path,
            identity=value["identity"],
            grant_sha256=value["grant_sha256"],
            worker_invocation_id=value["worker_invocation_id"],
        )


def nonempty_brief(*, snapshot="a" * 64):
    """Nonempty current /3 checkpoint with a genuine projected mechanism card."""
    import json

    from nurse_rcl.v2.live.ops_node_common import node_card
    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import pack_pooled_source_navigation
    from nurse_rcl.v2.live.ops_sre_comparable_briefs import project_comparable_brief
    from nurse_rcl.v2.live.ops_sre_views import semantic_record
    from nurse_rcl.v2.observation_schemas import TimeInterval

    conditions = [
        dict(
            condition_id="condition-1",
            candidate_service_key="synthetic-service",
            mechanism_key="pressure",
            predicate_kind="persistent_pressure",
            requirement="necessary",
            original_evidence_ids=["b" * 64],
            operator="or",
            truth_value="unknown",
            reason_codes=["unresolved"],
            leaves=[
                dict(
                    evidence_id="b" * 64,
                    dynamic_predicate_sha256=None,
                    measurement_field="value",
                    truth_value="unknown",
                    reason_codes=["unresolved"],
                )
            ],
            non_applicable=[],
        )
    ]
    card = node_card(
        snapshot=snapshot,
        node="mechanisms",
        parameters=dict(candidate_service_key="synthetic-service"),
        evidence_ids=("b" * 64,),
        source_refs=("c" * 64,),
        interval=TimeInterval(start_unix_ns=10, end_unix_ns=20),
        truth="unknown",
        limitations=("not_causal_confirmation",),
        payload=dict(
            decision=dict(
                mechanism_id="mechanism-1",
                candidate_service_key="synthetic-service",
                mechanism_key="pressure",
                mechanism_definition_sha256="d" * 64,
                condition_ids=["condition-1"],
                disposition="unknown",
                permitted_roles=[],
                support_evidence_ids=[],
                reason_codes=["unresolved"],
                causal_status="observable_contract_only",
            ),
            conditions=conditions,
            distinguishing_checks=[],
            score_authority=False,
        ),
    )
    full = json.loads(canonical_json_bytes(semantic_record(card, published=True)))
    key = full["card_id"]
    prefix = dict(
        ids=[key],
        scope=["synthetic-service"],
        core_cards=[key],
        record_roles={key: full["source_role"]},
        record_targets={key: full["targets"]},
    )
    nav = dict(
        profile="hermes-sre-tools/23",
        semantic_encoding="sre-lossless-rows/1",
        score_lineage_sha256=None,
        indexed=dict(
            cards={key: card.model_dump(mode="json")},
            core_steps=[],
            index=dict(rows={}),
            core_cards=[key],
            feedback=[],
            read_cursors={},
            input_view_policy="ops-soft-targets/1",
        ),
        origins={},
        inspection_prefixes={"p": prefix},
        inspection_cursors={},
    )
    brief = project_comparable_brief(full)
    view = dict(publication_prefix="p", semantic_record=brief)
    ledger = SimpleNamespace(
        navigation=pack_pooled_source_navigation(nav, source=None),
        records=(SimpleNamespace(model_view_json=canonical_json_bytes(view).decode()),),
    )
    return ledger, full, brief


def test_formal_gplus_reads_nonempty_current_pooled_retained_source():
    from nurse_rcl.v2.live.ops_sre_gplus_case import _retained_sources

    ledger, full, _ = nonempty_brief()
    before = canonical_json_bytes(ledger.navigation)
    assert canonical_json_bytes(_retained_sources(None, ledger, None)) == canonical_json_bytes(
        [full]
    )
    assert canonical_json_bytes(ledger.navigation) == before


@pytest.mark.parametrize("damage", ["prefix", "card", "targets", "role", "brief", "encoding"])
def test_current_retained_source_rejects_repacked_semantic_damage(damage):
    import json

    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import (
        pack_pooled_source_navigation,
        unpack_pooled_source_navigation,
    )

    ledger, full, _ = nonempty_brief()
    view = json.loads(ledger.records[0].model_view_json)
    nav = unpack_pooled_source_navigation(ledger.navigation, source=None)
    if damage == "prefix":
        view["publication_prefix"] = "absent"
    elif damage == "card":
        nav["indexed"]["cards"] = {}
    elif damage == "targets":
        nav["inspection_prefixes"]["p"]["record_targets"][full["card_id"]] = ["other"]
    elif damage == "role":
        nav["inspection_prefixes"]["p"]["record_roles"][full["card_id"]] = "investigation_measured"
    elif damage == "brief":
        view["semantic_record"]["payload"]["score_authority"] = True
    ledger.navigation = pack_pooled_source_navigation(nav, source=None)
    if damage == "encoding":
        ledger.navigation["storage_encoding"] = "sre-checkpoint-score-source/1"
    ledger.records[0].model_view_json = canonical_json_bytes(view).decode()
    with pytest.raises(ValueError):
        api().retained_sources(None, ledger, None)


def test_no_delivered_brief_returns_before_navigation_read():
    class NoNavigation:
        records = ()

        @property
        def navigation(self):
            pytest.fail("no-brief handoff opened navigation")

    assert api().retained_sources(None, NoNavigation(), None) == ()


def test_product_handoff_has_no_research_imports():
    import ast

    module = api()
    for name in (module.__file__, import_module("nurse_rcl.v2.live.ops_sre_gplus_case").__file__):
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any("blind_compare" in name for name in imports)


def synthetic_core(*, configuration, environment):
    """Build typed synthetic terminal data; execute no core, tools or owner."""
    from nurse_rcl.v2.candidate_freeze import freeze_candidate_universe
    from nurse_rcl.v2.event_grouping import AdapterAssociationSourceField
    from nurse_rcl.v2.graph_snapshots import (
        build_frozen_graph_input_manifest,
        build_graph_adapter_parent_record,
        build_graph_snapshot_pair,
    )
    from nurse_rcl.v2.live.normal_artifacts import NormalEvidence, NormalQuerySession
    from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity, snapshot_id
    from nurse_rcl.v2.live.ops_result import seal_core_result
    from nurse_rcl.v2.live.repair_error_references import RepairReferencedErrorFacts
    from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
    from nurse_rcl.v2.live.repair_local_scoring import RepairLocalScoring
    from nurse_rcl.v2.live.repair_mechanisms import RepairMechanismEvidence
    from nurse_rcl.v2.live.repair_propagation import RepairPropagationEvidence
    from nurse_rcl.v2.live.repair_ranking import RepairRanking, RepairScoreDecomposition
    from nurse_rcl.v2.live.repair_strengths import RepairStrengthEvidence
    from nurse_rcl.v2.observation_schemas import (
        CandidateEntry,
        CaseTimeContract,
        TimeInterval,
        build_candidate_universe,
    )
    from nurse_rcl.v2.reference_ranking import build_reference_scoring_config
    from nurse_rcl.v2.temporal_condensation import (
        case_time_contract_sha256,
        freeze_clock_resolution,
    )
    from nurse_rcl.v2.temporal_entity_registry import (
        build_entity_metadata_record,
        build_temporal_entity_registry,
    )

    def seal(model, **values):
        draft = model(**values, result_sha256="0" * 64)
        return draft.model_copy(
            update=dict(
                result_sha256=digest(draft.model_dump(mode="json", exclude={"result_sha256"}))
            )
        )

    case = "opq_" + "c" * 32
    keys, pin = ("synthetic-service",), "b" * 64
    interval = TimeInterval(start_unix_ns=100, end_unix_ns=300)
    clock = CaseTimeContract(
        opaque_case_id=case,
        t0_unix_ns=200,
        t0_source="shared_detector",
        localization_scope="detector_triggered",
        baseline_window=TimeInterval(start_unix_ns=100, end_unix_ns=200),
        fault_window=TimeInterval(start_unix_ns=200, end_unix_ns=300),
        decision_time_unix_ns=300,
        timezone="UTC",
        expected_cadence_ns=10,
        clock_uncertainty_ns=0,
        missing_modalities=(),
        late_data_observed=False,
    )
    universe = build_candidate_universe(
        opaque_case_id=case,
        observation_contract_sha256=pin,
        candidates=(
            CandidateEntry(
                service_entity_key=keys[0],
                binding_ids=(),
                observation_ids=(),
                event_group_ids=(),
                evidence_state="zero_support",
            ),
        ),
    )
    frozen = freeze_candidate_universe(universe)
    entity = build_entity_metadata_record(
        opaque_case_id=case,
        metadata_source="deployment",
        entity_kind="service",
        observed_entity_key=keys[0],
        relation="explicit_service_binding",
        service_entity_key=keys[0],
        valid_interval=interval,
        source_fingerprint=pin,
        provenance_ids=("deployment:synthetic",),
    )
    registry = build_temporal_entity_registry(
        opaque_case_id=case, decision_time_unix_ns=300, metadata_records=(entity,), observations=()
    )
    coverages = tuple(
        build_graph_adapter_parent_record(
            opaque_case_id=case,
            observation_contract_sha256=pin,
            window_kind=kind,
            interval=window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed",
            valid_sample_count=10,
            expected_sample_count=10,
            source_identity_fields=(
                AdapterAssociationSourceField(
                    field_kind="counter_series_id", raw_value="series:synthetic-" + kind
                ),
            ),
            provenance_fields=(
                AdapterAssociationSourceField(
                    field_kind="export_id", raw_value="export:synthetic-coverage-" + kind
                ),
            ),
        )
        for kind, window in (("normal", clock.baseline_window), ("fault", clock.fault_window))
    )
    manifest = build_frozen_graph_input_manifest(
        opaque_case_id=case, observation_contract_sha256=pin, parent_records=coverages
    )
    pair = build_graph_snapshot_pair(
        time_contract=clock,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    graph = seal(
        RepairLiveGraph,
        semantics_sha256=pin,
        dataset="rcabench",
        opaque_case_id=case,
        input_descriptor_sha256=pin,
        manifest=manifest,
        graph_pair=pair,
        rcabench_exports=(),
        aiops25_reference_windows=(),
        limitations=("collection_coverage_unknown",),
    )
    errors = seal(
        RepairReferencedErrorFacts,
        semantics_sha256=pin,
        source_bundle_sha256=pin,
        observation_sha256s=(),
        facts=(),
    )
    mechanisms = seal(
        RepairMechanismEvidence,
        semantics_sha256=pin,
        opaque_case_id=case,
        catalog_sha256=pin,
        session_sha256=pin,
        candidate_service_keys=keys,
        signal_facts=(),
        error_facts=errors,
        dynamic_predicates=(),
        conditions=(),
        mechanisms=(),
        unbound_evidence_ids=(),
        dynamic_queries_used=0,
    )
    strengths = seal(
        RepairStrengthEvidence,
        semantics_sha256=pin,
        opaque_case_id=case,
        session_sha256=pin,
        candidate_service_keys=keys,
        mechanisms=mechanisms,
        terms=(),
        pools=(),
        dynamic_queries_used=0,
    )
    local = seal(
        RepairLocalScoring,
        semantics_sha256=pin,
        opaque_case_id=case,
        strengths=strengths,
        scoring_config=build_reference_scoring_config(),
        candidate_service_keys=keys,
        candidates=(),
        unbound_term_ids=(),
    )
    resolution = freeze_clock_resolution(
        time_contract=clock,
        expected_time_contract_sha256=case_time_contract_sha256(clock),
        resolution_status="unresolved",
        provenance_ids=("synthetic-clock",),
    )
    propagation = seal(
        RepairPropagationEvidence,
        semantics_sha256=pin,
        opaque_case_id=case,
        local_scoring=local,
        graph_pair=pair,
        clock_resolution=resolution,
        candidate_service_keys=keys,
        witnesses=(),
        paths=(),
        candidates=(),
    )
    score = RepairScoreDecomposition(
        candidate_service_key=keys[0],
        local_evidence_status="no_retained_evidence",
        event_evidence_score=0.0,
        propagation_score=0.0,
        local_score_component=0.0,
        propagation_score_component=0.0,
        total_score=0.0,
        quantized_total_score="0.000000000000",
        quantized_event_evidence_score="0.000000000000",
        quantized_propagation_score="0.000000000000",
        dependent_pool_keys=(),
        selected_term_ids=(),
        selected_family_keys=(),
        selected_path_ids=(),
        unknown_term_ids=(),
        dependent_unused_term_ids=(),
        propagation_reason_codes=("unresolved",),
        score_sha256=pin,
    )
    from nurse_rcl.v2.live.repair_ranking import RepairEvidenceTieGroup

    tie = RepairEvidenceTieGroup(
        tie_group_id=pin,
        quantized_total_score=score.quantized_total_score,
        quantized_event_evidence_score=score.quantized_event_evidence_score,
        quantized_propagation_score=score.quantized_propagation_score,
        member_service_keys=keys,
        member_score_sha256s=(score.score_sha256,),
    )
    ranking = seal(
        RepairRanking,
        semantics_sha256=pin,
        opaque_case_id=case,
        propagation=propagation,
        candidate_service_keys=keys,
        scores=(score,),
        tie_groups=(tie,),
        ordered_service_keys=keys,
        top_k=1,
        top_k_service_keys=keys,
    )
    session = NormalQuerySession(
        semantics_sha256=pin,
        catalog_sha256=pin,
        journal_sha256=pin,
        steps=(),
        submitted_request_keys=(),
        pending_selection=None,
        pending_measurement=None,
        paid_count=0,
        stop_reason="no_admissible_progress",
        failure_reason=None,
        session_sha256=pin,
    )
    evidence = seal(
        NormalEvidence,
        dataset="rcabench",
        opaque_case_id=case,
        source_descriptor_sha256=pin,
        source_files_sha256=pin,
        candidate_freeze_sha256=pin,
        catalog_sha256=pin,
        session_sha256=pin,
        session=session,
        graph_manifest_sha256=manifest.manifest_sha256,
        graph=graph,
        ranking=ranking,
    )
    return seal_core_result(
        identity=OpsCoreIdentity(
            dataset="rcabench",
            opaque_case_id=case,
            case_snapshot_id=snapshot_id(evidence),
            configuration_sha256=configuration,
            environment_sha256=environment,
        ),
        evidence=evidence,
    )


@pytest.fixture
def sealed_stage(tmp_path, monkeypatch, request):
    import json

    from nurse_rcl.v2.live.ops_agent_state import (
        OpsAtomicSelectorConversation,
        OpsReplanningDecisionRecord,
        load_conversation,
        sre_run_id,
    )
    from nurse_rcl.v2.live.ops_case import case_identity
    from nurse_rcl.v2.live.ops_contracts import (
        OpsInvestigationEvidence,
        OpsNodeEvidence,
        OpsSreFeedbackFactProfile,
    )
    from nurse_rcl.v2.live.ops_plan import code_inventory, require_ops_code, runtime_environment
    from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import (
        OpsAtomicSelectorAction,
        OpsAtomicSelectorActionRecord,
        OpsSreAtomicSelectorLedgerState,
    )
    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import unpack_pooled_source_navigation
    from nurse_rcl.v2.live.ops_sre_directed_provider import stage_grant
    from nurse_rcl.v2.live.ops_sre_directed_stage import STAGE_MEMBERS, _prefix_pins
    from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import prepare_entry_decision
    from tests import sre_m3_entry_fixtures as fixture

    focused = getattr(request, "param", None) in {"focused", "focused_attribution"}
    attribution = getattr(request, "param", None) == "focused_attribution"
    if focused:
        from nurse_rcl.v2.live.ops_agent_state import OpsFocusedConversation
        from nurse_rcl.v2.live.ops_contracts import OpsSreFocusedProfile
        from nurse_rcl.v2.live.ops_sre_focused_tools import OpsSreFocusedLedgerState

    root = tmp_path / "sealed-synthetic"
    root.mkdir()
    repository = Path(__file__).resolve().parents[1]
    inventory, environment = code_inventory(repository), runtime_environment()
    plan = SimpleNamespace(
        code_root=str(repository),
        code_files=inventory,
        environment=environment,
        code_sha256=digest(inventory),
        profile=(OpsSreFocusedProfile() if focused else OpsSreFeedbackFactProfile()),
        core_configuration_sha256="1" * 64,
        environment_sha256=digest(environment),
        case=SimpleNamespace(dataset="rcabench", opaque_case_id="opq_" + "c" * 32),
        mode="with-agent",
    )
    stage = SimpleNamespace(
        state_root=root,
        plan=plan,
        plan_sha256=digest(
            dict(code=inventory, environment=environment, profile=plan.profile.profile_id)
        ),
    )
    require_ops_code(plan)
    core = synthetic_core(
        configuration=plan.core_configuration_sha256, environment=plan.environment_sha256
    )
    snapshot = core.identity.case_snapshot_id
    monkeypatch.setattr(fixture, "SNAPSHOT", snapshot)
    monkeypatch.setattr(fixture, "RUN_ID", sre_run_id(root, snapshot, plan.code_sha256))
    ledger, full, brief = nonempty_brief(snapshot=snapshot)
    decoded = unpack_pooled_source_navigation(ledger.navigation, source=None)
    if focused:
        from nurse_rcl.v2.live.ops_sre_checkpoint_pools import pack_pooled_source_navigation
        from nurse_rcl.v2.live.ops_sre_focused_catalog import EvidenceCatalog
        from nurse_rcl.v2.live.ops_sre_focused_sources import enrich_registered, source_registry

        decoded["profile"] = "hermes-sre-tools/25"
        data = SimpleNamespace(
            catalog=SimpleNamespace(requests=[]), sources=[], diagnostics=SimpleNamespace(errors=[])
        )
        if attribution:
            from nurse_rcl.v2.live.ops_node_common import node_card
            from nurse_rcl.v2.live.ops_sre_views import semantic_record
            from nurse_rcl.v2.observation_schemas import TimeInterval
            from tests.test_v2_live_sre_focused_sources import scalar

            row, requests, observations, sources = scalar()
            row["payload"]["candidate_service_key"] = "synthetic-service"
            requests[0]["candidate_service_key"] = sources[0]["candidate_service_key"] = (
                "synthetic-service"
            )
            node = node_card(
                snapshot=snapshot,
                node="symptoms",
                parameters=row["parameters"],
                payload=row["payload"],
                evidence_ids=("b" * 64,),
                source_refs=tuple(row["source_refs"]),
                unit=row["unit"],
                coverage=row["coverage"],
                interval=TimeInterval(start_unix_ns=20, end_unix_ns=30),
            )
            data = SimpleNamespace(
                catalog=SimpleNamespace(requests=requests),
                sources=sources,
                diagnostics=SimpleNamespace(errors=observations),
            )
            full = json.loads(
                canonical_json_bytes(
                    enrich_registered(semantic_record(node, published=True), source_registry(data))
                )
            )
            brief = full  # Non-mechanism records are delivered exactly, not summarized.
            key = full["card_id"]
            decoded["indexed"].update(cards={key: node.model_dump(mode="json")}, core_cards=[key])
            decoded["inspection_prefixes"] = {
                "p": dict(
                    ids=[key],
                    scope=["synthetic-service"],
                    core_cards=[key],
                    record_roles={key: full["source_role"]},
                    record_targets={key: full["targets"]},
                )
            }
        decoded.update(
            evidence_catalog=EvidenceCatalog(
                snapshot=snapshot, rows=[full], requests={}
            ).export_state(),
            source_registry=source_registry(data),
        )
        ledger.navigation = pack_pooled_source_navigation(decoded, source=None)
    node = OpsNodeEvidence.model_validate(decoded["indexed"]["cards"][full["card_id"]])
    params = dict(card_id=full["card_id"], section="payload", cursor=None)
    route = dict(
        skill_id="sre-symptoms/20"
        if attribution
        else "sre-mechanisms/20"
        if focused
        else "sre-mechanisms/19",
        tool="read_node",
        parameters=params,
        binding_sha256=digest(dict(card=full["card_id"], parameters=params)),
        category="current",
        scope=full["targets"],
        card_ids=[full["card_id"]],
        source_refs=full["source_refs"],
        availability="available",
        execution_history=[],
        new_publication=True,
        core_reusable=True,
    )
    extra = dict(
        investigation_routes=dict(
            schema_version="nurse.v2.sre-investigation-routes/3",
            entries=[route],
            recovery_feedback=None,
        )
    )
    if focused:
        extra["investigation_profile_id"] = "hermes-sre-tools/25"
    first, refs, _ = fixture.normal_context(extra=extra)
    entry = next(
        e for e in first["action_entries"]["entries"] if e["action"]["tool"] == "read_node"
    )
    response = dict(
        actions=[entry["entry_ref"]],
        stop_reason=None,
        stop_basis=None,
        public_summary="Read the synthetic saved mechanism",
        unresolved=["Unresolved"],
        responds_to=[],
        latest_result_update=None,
        investigation_note=None,
    )
    if focused:
        from nurse_rcl.v2.live.ops_sre_focused_exchange import prepare_focused_decision

        wire = prepare_focused_decision(
            context=first,
            references=refs,
            opaque_case_id=plan.case.opaque_case_id,
            case_snapshot_id=snapshot,
            decision_index=1,
        )
    else:
        wire = prepare_entry_decision(
            first, references=refs, case_id=plan.case.opaque_case_id, snapshot=snapshot, index=1
        )
    decision = OpsReplanningDecisionRecord(
        index=1,
        context_json=canonical_json_bytes(first).decode(),
        response_json=canonical_json_bytes(response).decode(),
        action_index=1,
        wire_json=wire.decode(),
        wire_sha256=sha256_hex(wire),
        target_exceeded=len(wire) > 16384,
        expansion_reasons=("required_semantics_or_complete_envelope",) if len(wire) > 16384 else (),
        native_response_json=canonical_json_bytes(response).decode(),
        call_started=True,
    )
    record = OpsAtomicSelectorActionRecord(
        index=1,
        input_sha256=digest(response["actions"]),
        action=OpsAtomicSelectorAction(
            action_index=1,
            skill_id=entry["action"]["skill_id"],
            tool="read_node",
            parameters_json=canonical_json_bytes(params).decode(),
            question=None,
        ),
        status="completed",
        result=OpsInvestigationEvidence(
            action_index=1, evidence=node, origin="read_card", reused_core_evidence_ids=()
        ),
        model_view_json=canonical_json_bytes(
            dict(
                action_index=1,
                origin="read_card",
                publication_prefix="p",
                semantic_record=brief,
                section="payload",
                next_cursor=None,
                semantic_encoding="sre-lossless-rows/1",
            )
        ).decode(),
    )
    conversation_type = OpsFocusedConversation if focused else OpsAtomicSelectorConversation
    conversation = conversation_type(
        case_snapshot_id=snapshot,
        code_sha256=plan.code_sha256,
        opaque_case_id=plan.case.opaque_case_id,
        initial_messages=(),
        decisions=(decision,),
        reference_state=refs,
    )
    conv = conversation.model_dump(mode="json")
    conv["state_sha256"] = digest({k: v for k, v in conv.items() if k != "state_sha256"})
    ledger_type = OpsSreFocusedLedgerState if focused else OpsSreAtomicSelectorLedgerState
    saved_ledger = ledger_type(
        case_snapshot_id=snapshot,
        code_sha256=plan.code_sha256,
        records=(record,),
        navigation=ledger.navigation,
        state_sha256="0" * 64,
    ).model_dump(mode="json")
    saved_ledger["state_sha256"] = digest(
        {k: v for k, v in saved_ledger.items() if k != "state_sha256"}
    )
    members = {
        "core/result.json": core.model_dump(mode="json"),
        "investigation/conversation.json": conv,
        "investigation/state.json": saved_ledger,
    }
    for name in STAGE_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(members.get(name, dict(synthetic_member=name))))
    # Establish that genuine saved readers accept the fixture before testing the new handoff.
    load_conversation(
        root / "investigation/conversation.json",
        snapshot=snapshot,
        code=plan.code_sha256,
        empirical=True,
        opaque_case_id=plan.case.opaque_case_id,
        sre=True,
        sre_profile="hermes-sre-tools/25" if focused else "hermes-sre-tools/23",
    )
    prefix = _prefix_pins(root, case_identity(stage))
    grant = stage_grant(stage.plan_sha256, 1)
    outcome = dict(
        schema_version="nurse.v2.sre-stage-outcome/1",
        identity=case_identity(stage),
        approved_grant=grant,
        core_result_digest=core.core_result_digest,
        channels=dict(core="completed", investigation="partial", report="disabled"),
        termination="stage_pause",
        decision_count=1,
        action_count=1,
        primary_failure=None,
        secondary_failures=[],
        prefix_pins=prefix,
        files={
            name: [len((root / name).read_bytes()), sha256_hex((root / name).read_bytes())]
            for name in STAGE_MEMBERS
        },
    )
    outcome["stage_result_sha256"] = digest(outcome)
    result = dict(
        outcome,
        requested_grant=grant,
        worker_exit_code=0,
        worker_invocation_id="1" * 32,
        status="stage_paused",
        failure_type=None,
    )
    result["round_result_sha256"] = digest(result)
    (root / "stage-outcome.json").write_bytes(canonical_json_bytes(outcome))
    (root / "stage-round.json").write_bytes(canonical_json_bytes(result))
    return stage, core, dict(grant_sha256=digest(grant), worker_invocation_id="1" * 32), full


def test_formal_sealed_stage_reaches_context_without_failure_diagnostic(sealed_stage):
    from nurse_rcl.v2.live import ops_sre_gplus_case as formal
    from nurse_rcl.v2.live.ops_report_context import build_report_context
    from nurse_rcl.v2.live.ops_sre_atomic_report import validate_atomic_report_identity
    from tests.sre_atomic_report_fixtures import file_pins

    stage, core, binding, full = sealed_stage
    before = file_pins(stage.state_root)
    assert not (stage.state_root / "worker-outcome.json").exists()
    context, refs = formal.stage_report_context(stage, **binding)
    source = validate_atomic_report_identity(context.metadata)
    assert source.core_result_digest == core.core_result_digest
    assert source.source_plan_sha256 == stage.plan_sha256
    assert source.source_code_sha256 == stage.plan.code_sha256
    assert source.retained_sources_sha256 == digest([full])
    assert (
        source.formal_context_sha256
        == build_report_context(
            core=core, sources=(), investigation=(), soft_targets=True
        ).context_sha256
    )
    assert source.reference_state_sha256 == digest(refs)
    assert len(source.decision_pins) == len(source.action_pins) == 1
    assert context.metadata.steps[-1]["response_state"] == "not_recorded"
    assert context.metadata.steps[-1]["executed"]["status"] == "completed"
    assert file_pins(stage.state_root) == before


def rewrite_stage(stage, *, outcome_changes=None, result_changes=None):
    """Rebind all outer member/hash envelopes, leaving the attacked invariant live."""
    import json

    root = stage.state_root
    outcome = json.loads((root / "stage-outcome.json").read_bytes())
    outcome.update(outcome_changes or {})
    outcome["files"] = {
        name: [len((root / name).read_bytes()), sha256_hex((root / name).read_bytes())]
        for name in outcome["files"]
    }
    outcome["stage_result_sha256"] = digest(
        {k: v for k, v in outcome.items() if k != "stage_result_sha256"}
    )
    result = json.loads((root / "stage-round.json").read_bytes())
    result.update(outcome)
    result.update(result_changes or {})
    result["round_result_sha256"] = digest(
        {k: v for k, v in result.items() if k != "round_result_sha256"}
    )
    (root / "stage-outcome.json").write_bytes(canonical_json_bytes(outcome))
    (root / "stage-round.json").write_bytes(canonical_json_bytes(result))


@pytest.mark.parametrize(
    "damage",
    [
        "core_digest",
        "core_identity",
        "core_inner_digest",
        "code_admission",
        "ledger_identity",
        "ledger_digest",
        "conversation_identity",
        "retained_role",
        "prefix",
        "round_canonical",
        "outcome_canonical",
    ],
)
def test_formal_resealed_stage_damage_is_rejected_before_report(sealed_stage, damage):
    import json

    from nurse_rcl.v2.live.ops_sre_checkpoint_pools import (
        pack_pooled_source_navigation,
        unpack_pooled_source_navigation,
    )
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context

    stage, _, binding, full = sealed_stage
    root, outcome_changes = stage.state_root, {}
    if damage == "core_digest":
        outcome_changes["core_result_digest"] = "f" * 64
    elif damage == "core_identity":
        stage.plan.core_configuration_sha256 = "f" * 64
    elif damage == "core_inner_digest":
        path = root / "core/result.json"
        body = json.loads(path.read_bytes())
        body["evidence"]["ranking"]["scores"][0]["quantized_total_score"] = "0.5"
        body["core_result_digest"] = digest(
            {k: v for k, v in body.items() if k != "core_result_digest"}
        )
        path.write_bytes(canonical_json_bytes(body))
    elif damage == "code_admission":
        stage.plan.code_files = dict(stage.plan.code_files, **{"src/synthetic-drift.py": "f" * 64})
    elif damage in {"ledger_identity", "ledger_digest", "retained_role"}:
        path = root / "investigation/state.json"
        body = json.loads(path.read_bytes())
        if damage == "ledger_identity":
            body["code_sha256"] = "f" * 64
        elif damage == "ledger_digest":
            body["navigation"]["profile"] = "unregistered"
        else:
            nav = unpack_pooled_source_navigation(body["navigation"], source=None)
            nav["inspection_prefixes"]["p"]["record_roles"][full["card_id"]] = (
                "investigation_measured"
            )
            body["navigation"] = pack_pooled_source_navigation(nav, source=None)
        if damage != "ledger_digest":
            body["state_sha256"] = digest({k: v for k, v in body.items() if k != "state_sha256"})
        path.write_bytes(canonical_json_bytes(body))
    elif damage == "conversation_identity":
        path = root / "investigation/conversation.json"
        body = json.loads(path.read_bytes())
        body["code_sha256"] = "f" * 64
        body["state_sha256"] = digest({k: v for k, v in body.items() if k != "state_sha256"})
        path.write_bytes(canonical_json_bytes(body))
    elif damage == "prefix":
        outcome_changes["prefix_pins"] = dict(decisions=["f" * 64], actions=["a" * 64])
    rewrite_stage(stage, outcome_changes=outcome_changes)
    if damage in {"round_canonical", "outcome_canonical"}:
        path = root / ("stage-round.json" if damage == "round_canonical" else "stage-outcome.json")
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        stage_report_context(stage, **binding)


@pytest.mark.parametrize(
    "member",
    [
        "stage-round.json",
        "stage-outcome.json",
        "core/result.json",
        "investigation/conversation.json",
        "investigation/state.json",
    ],
)
def test_formal_stage_missing_mandatory_artifact_is_refused(sealed_stage, member):
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context

    stage, _, binding, _ = sealed_stage
    target = stage.state_root / member
    assert target.is_file()
    assert target.resolve().is_relative_to(stage.state_root.resolve())
    target.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_report_context(stage, **binding)
    assert not target.exists()
