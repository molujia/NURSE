"""D15 failure feedback comes from real guards, not exception text inference."""

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_condition_tools import OpsSreConditionToolbox
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401


def api():
    name = "nurse_rcl.v2.live.ops_sre_recovery_tools"
    assert importlib.util.find_spec(name), "D15 recoverable tool profile is missing"
    return importlib.import_module(name)


def toolbox(complete, *, hidden=()):  # noqa: F811
    tool = api().OpsSreRecoveryToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=complete[2].session.steps)
    bind_visible(tool, hidden=hidden)
    return tool


def bind_visible(tool, *, hidden=()):
    book = SreReferenceBook(snapshot=tool.case_snapshot_id, run_id="1" * 64)
    for key in tool._by_candidate:
        book.issue("service", key, visible_at=1)
    for key in sorted(tool.cards, key=lambda k: k in hidden):
        book.issue("card", key, visible_at=2 if key in hidden else 1)
    # Issue visible cards before future cards to retain append-only ordering.
    tool.bind_visible_references(book=book, request_index=1)
    return book


def scoring_key(tool):
    return next(k for k, c in tool.cards.items() if c.node == "scoring")


def action(tool, params):
    return [dict(skill_id="sre-mechanisms/5", tool=tool, parameters=params, question="核验条件")]


def test_scoring_card_rejection_has_real_kind_without_measurement(complete):  # noqa: F811
    tool = toolbox(complete)
    key = scoring_key(tool)
    before = (dict(tool._feedback), complete[2].model_dump_json())
    with pytest.raises(ValueError) as caught:
        tool.execute("check_mechanism", dict(card_id=key))
    assert caught.value.diagnostic.reason_code == "card_kind_mismatch"
    assert str(caught.value) == "card_kind_mismatch"
    assert (dict(tool._feedback), complete[2].model_dump_json()) == before


@pytest.mark.parametrize("section", ["mechanisms", "SECRET_LABEL/private/path", 13, []])
def test_invalid_section_is_specific_without_echoing_value(complete, section):  # noqa: F811
    tool = toolbox(complete)
    with pytest.raises(ValueError) as caught:
        tool.execute("read_node", dict(card_id=scoring_key(tool), section=section, cursor=None))
    assert caught.value.diagnostic.reason_code == "invalid_section"
    assert "SECRET_LABEL" not in str(caught.value)


def test_unavailable_unknown_foreign_and_future_cards_are_indistinguishable(complete):  # noqa: F811
    tool = toolbox(complete)
    key = scoring_key(tool)
    book = SreReferenceBook(snapshot=tool.case_snapshot_id, run_id="1" * 64)
    book.issue("card", key, visible_at=2)
    tool.bind_visible_references(book=book, request_index=1)
    errors = []
    for ref in (key, "ops_card_" + "f" * 64, "foreign/private/SECRET_LABEL"):
        with pytest.raises(ValueError) as caught:
            tool.execute("check_mechanism", dict(card_id=ref))
        errors.append(caught.value.diagnostic.model_dump())
    assert all(value == errors[0] for value in errors)
    assert errors[0]["reason_code"] == "reference_not_available"
    assert "SECRET_LABEL" not in json.dumps(errors)


@pytest.mark.parametrize("params", [None, {}, dict(cursor=None, extra=True)])
def test_parameter_shape_rejected_by_registered_guard(complete, params):  # noqa: F811
    tool = toolbox(complete)
    with pytest.raises(ValueError) as caught:
        tool.execute("inspect_case", params)
    assert caught.value.diagnostic.reason_code == "invalid_parameters"


def test_cursor_mismatch_and_focus_are_not_transient(complete):  # noqa: F811
    tool = toolbox(complete)
    candidate = complete[0].catalog.candidate_service_keys[0]
    for params, reason in [
        (
            dict(candidate_service_key=candidate, focus="mechanisms", cursor="unissued"),
            "request_binding_mismatch",
        ),
        (dict(candidate_service_key=candidate, focus="bad", cursor=None), "invalid_parameters"),
    ]:
        with pytest.raises(ValueError) as caught:
            tool.execute("inspect_candidate", params)
        assert caught.value.diagnostic.reason_code == reason
        assert caught.value.diagnostic.retry_hint == "change_request"


def test_ledger_preserves_rejection_and_contract_without_refund(complete, tmp_path):  # noqa: F811
    tool = toolbox(complete)
    key = scoring_key(tool)
    with api().OpsSreRecoveryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        for _ in range(2):
            ledger.dispatch(action("check_mechanism", dict(card_id=key)))
        assert ledger.used == 2
        row = ledger.records[-1]
        assert row.status == "rejected"
        view = expand(json.loads(row.model_view_json))
        feedback = view["parameter_feedback"]
        assert feedback["diagnostic"]["reason_code"] == "card_kind_mismatch"
        assert feedback["object_contract"]["node"] == "scoring"
        assert feedback["object_contract"]["targets"]
        assert "check_mechanism" not in feedback["object_contract"]["allowed_tools"]
        assert row.result.evidence.truth_value == "unknown"
        saved = row.model_view_json
    with api().OpsSreRecoveryInvestigationLedger(
        root=tmp_path, toolbox=toolbox(complete), code_sha256="2" * 64
    ) as restored:
        assert restored.used == 2
        assert restored.records[-1].model_view_json == saved


def test_successful_read_exposes_same_card_contract(complete):  # noqa: F811
    tool = toolbox(complete)
    key = scoring_key(tool)
    result = tool.execute("read_node", dict(card_id=key, section="payload", cursor=None))
    view = expand(json.loads(result.model_view_json))
    contract = view["object_contract"]
    assert contract["card_id"] == key
    assert contract["node"] == "scoring"
    assert contract["authority"] == "navigation_not_evidence"
    assert contract["sections"] == [
        "summary",
        "parameters",
        "payload",
        "evidence_ids",
        "source_refs",
        "limitations",
    ]


def test_unknown_internal_exception_is_not_inferred_from_its_text(complete, tmp_path, monkeypatch):  # noqa: F811
    tool = toolbox(complete)

    def fail(*args, **kwargs):
        raise ValueError("registered formal mechanism card required SECRET_LABEL/private")

    monkeypatch.setattr(tool, "_execute", fail)
    with api().OpsSreRecoveryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.dispatch(action("check_mechanism", dict(card_id=scoring_key(tool))))
        view = expand(json.loads(ledger.records[0].model_view_json))
        assert view["parameter_feedback"]["diagnostic"]["reason_code"] == "unclassified_failure"
        assert view["parameter_feedback"]["diagnostic"]["retry_hint"] == "unknown"
        assert "SECRET_LABEL" not in json.dumps(view)


def test_legacy_guard_remains_plain_value_error(complete):  # noqa: F811
    api()
    tool = OpsSreConditionToolbox(data=complete[0], block_root=complete[1])
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=complete[2].session.steps)
    with pytest.raises(ValueError, match="registered formal mechanism card required") as caught:
        tool.execute("check_mechanism", dict(card_id=scoring_key(tool)))
    assert type(caught.value) is ValueError


def test_added_feedback_cannot_hide_its_own_resource_failure(complete, tmp_path, monkeypatch):  # noqa: F811
    tool = toolbox(complete)
    monkeypatch.setattr(tool, "object_contract", lambda key: {"targets": ["x"] * 21000})
    with api().OpsSreRecoveryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        ledger.dispatch(action("check_mechanism", dict(card_id=scoring_key(tool))))
        row = ledger.records[0]
        assert json.loads(row.model_view_json)["status"] == "resource_limit"
        assert row.failure_detail is not None
        assert row.failure_detail.stage == "investigation_tool"
        assert row.failure_detail.observed > row.failure_detail.limit
        assert ledger.used == 1


def test_multitarget_contract_preserves_endpoints_and_hides_unseen_endpoint(complete):  # noqa: F811
    from nurse_rcl.v2.live.ops_node_common import node_card
    from nurse_rcl.v2.live.ops_tools import card_id

    tool = toolbox(complete)
    left, right = complete[0].catalog.candidate_service_keys[:2]
    card = node_card(
        snapshot=tool.case_snapshot_id,
        node="dependency",
        parameters={},
        payload=dict(caller_service_key=left, callee_service_key=right),
        evidence_ids=(),
        source_refs=(),
        interval=complete[0].diagnostics.legacy.sources.adapter.time_contract.fault_window,
    )
    tool.admit_core(nodes=(card,), steps=())
    bind_visible(tool)
    assert tool.object_contract(card_id(card))["targets"] == sorted([left, right])
    book = SreReferenceBook(snapshot=tool.case_snapshot_id, run_id="1" * 64)
    book.issue("card", card_id(card), visible_at=1)
    book.issue("service", left, visible_at=1)
    tool.bind_visible_references(book=book, request_index=1)
    assert tool.object_contract(card_id(card)) is None


def test_eight_rejections_exhaust_same_budget(complete, tmp_path):  # noqa: F811
    tool = toolbox(complete)
    with api().OpsSreRecoveryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        for _ in range(8):
            ledger.dispatch(action("check_mechanism", dict(card_id=scoring_key(tool))))
        assert ledger.used == 8
        with pytest.raises(ValueError, match="budget exhausted"):
            ledger.dispatch(action("check_mechanism", dict(card_id=scoring_key(tool))))


@pytest.mark.parametrize(
    "reason,retry",
    [
        ("invalid_parameters", "unknown"),
        ("unclassified_failure", "change_request"),
    ],
)
def test_diagnostic_cannot_falsely_label_retryability(reason, retry):
    api()
    from nurse_rcl.v2.live.ops_tool_rejection import OpsToolDiagnostic

    with pytest.raises(ValueError):
        OpsToolDiagnostic(reason_code=reason, retry_hint=retry)
