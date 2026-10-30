# ruff: noqa: RUF001 -- Chinese model contract.
"""Bound model-only representation; no credentials, tools, scoring or admission."""

from __future__ import annotations

import json
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_context_view import _fact_tables, expand_report_facts
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_encoding import compact, expand
from nurse_rcl.v2.live.ops_sre_fact_view import pack_fact_records, unpack_fact_records
from nurse_rcl.v2.live.ops_sre_history_view import _nodes, pack_history_view, unpack_history_view
from nurse_rcl.v2.live.ops_sre_references import (
    SreReferenceBook,
    decode_references,
    encode_references,
)
from nurse_rcl.v2.live.ops_sre_report_bounds import METADATA_LIMITS, validate_report_parts

MODEL_VIEW_SCHEMA = "nurse.v2.sre-model-view/1"
REPORT_MODEL_VIEW_SCHEMA = "nurse.v2.sre-model-view/2"
MODEL_VIEW_GUIDANCE = """
本次 context 使用 nurse.v2.sre-model-view/1，history 是本请求内的完整共享表示。
先按 sre_rows_v1 合并表格；history.content 是正文，publications 保存完整发布摘要，
步骤 publication_ref 是该数组的零基索引；parameter_contracts 的数值索引指向
parameter_rules 的原始完整规则。skill_common 合入每个技能，技能工具权限不变。
登记的标识符使用带类型短引用：S服务、C卡片、Q请求、O起点、E证据、P来源、H摘要。
在参数、回应和断言值中原样使用已提供短引用，不猜测长ID、不跨类型使用；f事实与s步骤不变。
短引用只是原ID的名称，不是新证据；reference_prefix 是本次映射身份，不需要复述。
报告 fact_records 取代 fact_tables：records 按完整来源组织；source 是公共来源，
fact_defaults/value_defaults 是本记录公共事实字段/值。tree 的 at 逐层用点连接为
source_path，嵌套默认值只对其子树生效，fact.ref 是原f别名，metadata/values 是差异。
mirror_record_paths 或 fact.mirror_record_path 为 true 时 record_path 等于完整路径。
未共享字段绝非缺测；全部字段合并后才构成事实；条件父子/AND/OR/反证不可分开解释。
不要输出解压后的整张表，只输出任务要求的调查决策或证据约束中文报告。
"""
REPORT_MODEL_VIEW_GUIDANCE = MODEL_VIEW_GUIDANCE.replace(
    "本次 context 使用 nurse.v2.sre-model-view/1，",
    "本次报告 context 使用 nurse.v2.sre-model-view/2，",
)


def _report_references(
    metadata: dict[str, Any],
    facts: list[dict[str, Any]],
    *,
    book: SreReferenceBook,
    case_id: str,
    index: int,
    encode: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One cumulative fact budget, independent of converter wrapper overhead."""
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    validate_report_parts(metadata, facts, layout="view")
    _require_outbound(dict(metadata, facts=facts), case=case_id, snapshot=book.snapshot)
    translate = encode_references if encode else decode_references
    converted = translate(
        metadata, book=book, visible_at=index, case_id=case_id, limits=METADATA_LIMITS
    )
    result: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    # This batching is only for the converter's {facts: [...]} wrapper, not a
    # fresh resource budget. The complete input/output partitions are checked.
    nodes = 3

    def flush() -> None:
        if pending:
            result.extend(
                translate(
                    dict(facts=pending),
                    book=book,
                    visible_at=index,
                    case_id=case_id,
                    limits=REPORT_LIMITS,
                )["facts"]
            )
            pending.clear()

    for fact in facts:
        count = _nodes(fact)
        if nodes + count > REPORT_LIMITS.max_nodes:
            flush()
            nodes = 3
        if count + 3 > REPORT_LIMITS.max_nodes:
            result.append(
                translate(fact, book=book, visible_at=index, case_id=case_id, limits=REPORT_LIMITS)
            )
        else:
            pending.append(fact)
            nodes += count
    flush()
    validate_report_parts(converted, result, layout="view")
    _require_outbound(dict(converted, facts=result), case=case_id, snapshot=book.snapshot)
    return converted, result


def _book(state: dict[str, Any], snapshot: str) -> SreReferenceBook:
    if type(state) is not dict or type(state.get("run_id")) is not str:
        raise ValueError("SRE reference state required")
    if state.get("schema_version") == "nurse.v2.sre-reference-book/2":
        from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook

        return SreActionReferenceBook.from_state(state, snapshot=snapshot, run_id=state["run_id"])
    return SreReferenceBook.from_state(state, snapshot=snapshot, run_id=state["run_id"])


def _prefix(book: SreReferenceBook, index: int) -> tuple[dict[str, Any], SreReferenceBook]:
    if type(index) is not int or not 1 <= index <= 9:
        raise ValueError("SRE reference request index differs")
    prefix = type(book)(snapshot=book.snapshot, run_id=book.run_id)
    for kind, _handle, original, first in book.to_state()["entries"]:
        if first <= index:
            prefix.issue(kind, original, visible_at=first)
    state = prefix.to_state()
    return dict(
        run_id=book.run_id,
        visible_at=index,
        entry_count=len(state["entries"]),
        state_sha256=state["state_sha256"],
    ), prefix


def pack_model_view(
    view: dict[str, Any],
    *,
    references: dict[str, Any],
    case_id: str,
    snapshot: str,
    index: int,
    report: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a packet and new local state; never mutate the caller's state."""
    limits = REPORT_LIMITS if report else CONTEXT_LIMITS
    book = _book(references, snapshot)
    focused = not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/26"
    atomic_selector = focused or (
        not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/24"
    )
    typed_purpose = (
        not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/23"
    )
    single_owner = (
        not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/22"
    )
    observable = (
        (not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/21")
        or single_owner
        or typed_purpose
        or atomic_selector
    )
    entry = (
        not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/20"
    ) or observable
    if (book.schema_version == "nurse.v2.sre-reference-book/2") != entry:
        raise ValueError("SRE model view/reference identity differs")
    if entry:
        from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
        from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries

        assert isinstance(book, SreActionReferenceBook)
        validate_action_entries(
            view["action_entries"],
            book=book,
            request_index=index,
            _observable=observable,
            _single_owner=single_owner,
            _typed_purpose=typed_purpose,
            _atomic_selector=atomic_selector and not focused,
            _focused=focused,
        )
    if report:
        original = {k: v for k, v in view.items() if k != "fact_tables"}
        # Tuple-backed fields come from the admitted context, before JSON wire.
        original = json.loads(encode_json(original, METADATA_LIMITS))
        facts = [f.model_dump(mode="json") for f in expand_report_facts(view)]
        short, rows = _report_references(
            original, facts, book=book, index=index, case_id=case_id, encode=True
        )
        short["fact_records"] = pack_fact_records(
            tuple(OpsReportFact.model_validate_json(canonical_json_bytes(f)) for f in rows)
        )
    else:
        original = expand(view)
        short = encode_references(
            original, book=book, visible_at=index, case_id=case_id, limits=limits
        )
    receipt_reuse = (
        not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/14"
    )
    question_led = not report and view.get("schema_version") in {
        "nurse.v2.sre-investigation-input/17",
        "nurse.v2.sre-investigation-input/18",
        "nurse.v2.sre-investigation-input/19",
        "nurse.v2.sre-investigation-input/20",
        "nurse.v2.sre-investigation-input/21",
        "nurse.v2.sre-investigation-input/22",
        "nurse.v2.sre-investigation-input/23",
        "nurse.v2.sre-investigation-input/24",
        "nurse.v2.sre-investigation-input/26",
    }
    replanning = not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/16"
    balanced = not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/15"
    used_source = not report and view.get("schema_version") == "nurse.v2.sre-investigation-input/13"
    workbench = not report and view.get("schema_version") in {
        "nurse.v2.sre-investigation-input/12",
        "nurse.v2.sre-investigation-input/13",
        "nurse.v2.sre-investigation-input/14",
        "nurse.v2.sre-investigation-input/15",
        "nurse.v2.sre-investigation-input/16",
        "nurse.v2.sre-investigation-input/17",
        "nurse.v2.sre-investigation-input/18",
        "nurse.v2.sre-investigation-input/19",
        "nurse.v2.sre-investigation-input/20",
        "nurse.v2.sre-investigation-input/21",
        "nurse.v2.sre-investigation-input/22",
        "nurse.v2.sre-investigation-input/23",
        "nurse.v2.sre-investigation-input/24",
        "nurse.v2.sre-investigation-input/26",
    }
    if workbench:
        from nurse_rcl.v2.live.ops_sre_workbench_model import split_workbench

        short, block = split_workbench(short, report=False)
    history = pack_history_view(short, limits=limits)
    prefix, _ = _prefix(book, index)
    packet: dict[str, Any] = dict(
        schema_version=REPORT_MODEL_VIEW_SCHEMA if report else MODEL_VIEW_SCHEMA,
        kind="report" if report else "decision",
        reference_prefix=prefix,
        history=history if report else compact(history),
    )
    if workbench:
        packet.update(
            schema_version=(
                "nurse.v2.sre-model-view/24"
                if focused
                else "nurse.v2.sre-model-view/19"
                if atomic_selector
                else "nurse.v2.sre-model-view/18"
                if typed_purpose
                else "nurse.v2.sre-model-view/17"
                if observable
                else "nurse.v2.sre-model-view/16"
                if entry
                else "nurse.v2.sre-model-view/15"
                if view.get("schema_version") == "nurse.v2.sre-investigation-input/19"
                else "nurse.v2.sre-model-view/13"
                if view.get("schema_version") == "nurse.v2.sre-investigation-input/18"
                else "nurse.v2.sre-model-view/11"
                if question_led
                else "nurse.v2.sre-model-view/9"
                if replanning
                else "nurse.v2.sre-model-view/8"
                if balanced
                else "nurse.v2.sre-model-view/7"
                if receipt_reuse
                else "nurse.v2.sre-model-view/6"
                if used_source
                else "nurse.v2.sre-model-view/5"
            ),
            workbench=block,
        )
    if atomic_selector:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_model import selector_choices

        packet["selector_choices"] = (
            [e["entry_ref"] for e in block["action_entries"]["entries"]]
            if focused
            else selector_choices(block["action_entries"])
        )
    encode_json(packet, limits)
    return packet, book.to_state()


def _validate_prefix(
    packet: dict[str, Any],
    *,
    references: dict[str, Any] | None,
    snapshot: str,
    index: int,
    report: bool,
) -> SreReferenceBook | None:
    encode_json(packet, REPORT_LIMITS if report else CONTEXT_LIMITS)
    focused = not report and packet.get("schema_version") == "nurse.v2.sre-model-view/24"
    atomic_selector = focused or (
        not report and packet.get("schema_version") == "nurse.v2.sre-model-view/19"
    )
    receipt_reuse = not report and packet.get("schema_version") == "nurse.v2.sre-model-view/7"
    question_led = not report and packet.get("schema_version") in {
        "nurse.v2.sre-model-view/11",
        "nurse.v2.sre-model-view/13",
        "nurse.v2.sre-model-view/15",
        "nurse.v2.sre-model-view/16",
        "nurse.v2.sre-model-view/17",
        "nurse.v2.sre-model-view/18",
        "nurse.v2.sre-model-view/19",
        "nurse.v2.sre-model-view/24",
    }
    replanning = not report and packet.get("schema_version") == "nurse.v2.sre-model-view/9"
    balanced = not report and packet.get("schema_version") == "nurse.v2.sre-model-view/8"
    used_source = not report and packet.get("schema_version") in {
        "nurse.v2.sre-model-view/6",
        "nurse.v2.sre-model-view/7",
        "nurse.v2.sre-model-view/8",
        "nurse.v2.sre-model-view/9",
        "nurse.v2.sre-model-view/11",
        "nurse.v2.sre-model-view/13",
        "nurse.v2.sre-model-view/15",
        "nurse.v2.sre-model-view/16",
        "nurse.v2.sre-model-view/17",
        "nurse.v2.sre-model-view/18",
        "nurse.v2.sre-model-view/19",
        "nurse.v2.sre-model-view/24",
    }
    workbench = not report and packet.get("schema_version") in {
        "nurse.v2.sre-model-view/5",
        "nurse.v2.sre-model-view/6",
        "nurse.v2.sre-model-view/7",
        "nurse.v2.sre-model-view/8",
        "nurse.v2.sre-model-view/9",
        "nurse.v2.sre-model-view/11",
        "nurse.v2.sre-model-view/13",
        "nurse.v2.sre-model-view/15",
        "nurse.v2.sre-model-view/16",
        "nurse.v2.sre-model-view/17",
        "nurse.v2.sre-model-view/18",
        "nurse.v2.sre-model-view/19",
        "nurse.v2.sre-model-view/24",
    }
    fields = {"schema_version", "kind", "reference_prefix", "history"}
    if atomic_selector:
        fields.add("selector_choices")
    if workbench:
        from nurse_rcl.v2.live.ops_sre_workbench_model import (
            DECISION_FIELDS,
            DECISION_FIELDS_D21,
            DECISION_FIELDS_D23,
            DECISION_FIELDS_D25,
            DECISION_FIELDS_D26,
            DECISION_FIELDS_ENTRY,
            DECISION_FIELDS_OBSERVABLE,
        )

        fields.add("workbench")
        expected_fields = (
            DECISION_FIELDS_OBSERVABLE
            if packet.get("schema_version")
            in {
                "nurse.v2.sre-model-view/17",
                "nurse.v2.sre-model-view/18",
                "nurse.v2.sre-model-view/19",
                "nurse.v2.sre-model-view/24",
            }
            else DECISION_FIELDS_ENTRY
            if packet.get("schema_version") == "nurse.v2.sre-model-view/16"
            else DECISION_FIELDS_D26
            if question_led
            else DECISION_FIELDS_D25
            if replanning
            else DECISION_FIELDS_D23
            if balanced
            else DECISION_FIELDS_D21
            if used_source
            else DECISION_FIELDS
        )
        if type(packet.get("workbench")) is not dict or set(packet["workbench"]) != expected_fields:
            raise ValueError("workbench packet fields differ")
    if (
        type(packet) is not dict
        or set(packet) != fields
        or packet["schema_version"]
        != (
            "nurse.v2.sre-model-view/24"
            if focused
            else "nurse.v2.sre-model-view/19"
            if atomic_selector
            else "nurse.v2.sre-model-view/18"
            if packet.get("schema_version") == "nurse.v2.sre-model-view/18"
            else "nurse.v2.sre-model-view/17"
            if packet.get("schema_version") == "nurse.v2.sre-model-view/17"
            else "nurse.v2.sre-model-view/16"
            if packet.get("schema_version") == "nurse.v2.sre-model-view/16"
            else "nurse.v2.sre-model-view/15"
            if packet.get("schema_version") == "nurse.v2.sre-model-view/15"
            else "nurse.v2.sre-model-view/13"
            if packet.get("schema_version") == "nurse.v2.sre-model-view/13"
            else "nurse.v2.sre-model-view/11"
            if question_led
            else "nurse.v2.sre-model-view/9"
            if replanning
            else "nurse.v2.sre-model-view/8"
            if balanced
            else "nurse.v2.sre-model-view/7"
            if receipt_reuse
            else "nurse.v2.sre-model-view/6"
            if used_source
            else "nurse.v2.sre-model-view/5"
            if workbench
            else REPORT_MODEL_VIEW_SCHEMA
            if report
            else MODEL_VIEW_SCHEMA
        )
        or packet["kind"] != ("report" if report else "decision")
    ):
        raise ValueError("SRE model view identity differs")
    prefix = packet["reference_prefix"]
    if (
        type(prefix) is not dict
        or set(prefix)
        != {
            "run_id",
            "visible_at",
            "entry_count",
            "state_sha256",
        }
        or (
            type(prefix["visible_at"]) is not int
            or prefix["visible_at"] != index
            or type(index) is not int
            or not 1 <= index <= 9
            or type(prefix["entry_count"]) is not int
            or not 0 <= prefix["entry_count"] <= 4096
        )
    ):
        raise ValueError("SRE reference prefix shape differs")
    require_pin(prefix["run_id"])
    require_pin(prefix["state_sha256"])
    if references is None:
        return None  # Transport reads short IDs only; owner must check its local book.
    expected, book = _prefix(_book(references, snapshot), index)
    if (book.schema_version == "nurse.v2.sre-reference-book/2") != (
        not report
        and packet["schema_version"]
        in {
            "nurse.v2.sre-model-view/16",
            "nurse.v2.sre-model-view/17",
            "nurse.v2.sre-model-view/18",
            "nurse.v2.sre-model-view/19",
            "nurse.v2.sre-model-view/24",
        }
    ):
        raise ValueError("SRE model view/reference identity differs")
    if prefix != expected:
        raise ValueError("SRE reference prefix binding differs")
    return book


def unpack_model_view(
    packet: dict[str, Any],
    *,
    references: dict[str, Any] | None,
    case_id: str,
    snapshot: str,
    index: int,
    report: bool,
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    limits = REPORT_LIMITS if report else CONTEXT_LIMITS
    book = _validate_prefix(
        packet, references=references, snapshot=snapshot, index=index, report=report
    )
    view = unpack_history_view(
        packet["history"] if report else expand(packet["history"]), limits=limits
    )
    if packet.get("schema_version") in {
        "nurse.v2.sre-model-view/5",
        "nurse.v2.sre-model-view/6",
        "nurse.v2.sre-model-view/7",
        "nurse.v2.sre-model-view/8",
        "nurse.v2.sre-model-view/9",
        "nurse.v2.sre-model-view/11",
        "nurse.v2.sre-model-view/13",
        "nurse.v2.sre-model-view/15",
        "nurse.v2.sre-model-view/16",
        "nurse.v2.sre-model-view/17",
        "nurse.v2.sre-model-view/18",
        "nurse.v2.sre-model-view/19",
        "nurse.v2.sre-model-view/24",
    }:
        from nurse_rcl.v2.live.ops_sre_workbench_model import join_workbench

        view = join_workbench(view, packet["workbench"], report=False)
    focused = not report and packet.get("schema_version") == "nurse.v2.sre-model-view/24"
    atomic_selector = focused or (
        not report and packet.get("schema_version") == "nurse.v2.sre-model-view/19"
    )
    if not report and atomic_selector != (
        view.get("schema_version")
        in {"nurse.v2.sre-investigation-input/24", "nurse.v2.sre-investigation-input/26"}
    ):
        raise ValueError("atomic model view/input identity differs")
    if focused != (view.get("schema_version") == "nurse.v2.sre-investigation-input/26"):
        raise ValueError("focused model view/input identity differs")
    if atomic_selector:
        from nurse_rcl.v2.live.ops_sre_atomic_selector_model import validate_selector_choices

        if focused:
            if packet["selector_choices"] != [
                e["entry_ref"] for e in view["action_entries"]["entries"]
            ]:
                raise ValueError("focused selector choices differ")
        else:
            validate_selector_choices(packet["selector_choices"], view["action_entries"])
    if report:
        facts = [f.model_dump(mode="json") for f in unpack_fact_records(view.pop("fact_records"))]
        validate_report_parts(view, facts, layout="view")
        if book is not None:
            view, facts = _report_references(
                view, facts, book=book, index=index, case_id=case_id, encode=False
            )
        view["facts"] = facts
    elif book is not None:
        view = decode_references(view, book=book, visible_at=index, case_id=case_id, limits=limits)
        if atomic_selector:
            from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
            from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import validate_atomic_entries

            if focused:
                from nurse_rcl.v2.live.ops_sre_focused_decision import (
                    validate_focused_entries as validate_atomic_entries,
                )

            assert isinstance(book, SreActionReferenceBook)
            validate_atomic_entries(view["action_entries"], book=book, request_index=index)
    _require_outbound(view, case=case_id, snapshot=snapshot)
    if report:
        view["fact_tables"] = _fact_tables(
            tuple(
                OpsReportFact.model_validate_json(canonical_json_bytes(f))
                for f in view.pop("facts")
            ),
            group_by_provenance=True,
        )
        encode_json(view, limits)
        return view
    return view


def resolve_model_response(
    value: dict[str, Any],
    *,
    packet: dict[str, Any],
    references: dict[str, Any],
    case_id: str,
    snapshot: str,
    index: int,
    report: bool,
) -> dict[str, Any]:
    if packet.get("schema_version") == "nurse.v2.sre-model-view/20":
        from nurse_rcl.v2.live.ops_sre_feedback_fact_exchange import resolve_feedback_response

        if report:
            raise ValueError("feedback presentation is decision-only")
        return resolve_feedback_response(
            value,
            packet=packet,
            references=references,
            case_id=case_id,
            snapshot=snapshot,
            index=index,
        )
    book = _validate_prefix(
        packet, references=references, snapshot=snapshot, index=index, report=report
    )
    assert book is not None
    if not report and packet.get("schema_version") in {
        "nurse.v2.sre-model-view/16",
        "nurse.v2.sre-model-view/17",
        "nurse.v2.sre-model-view/18",
        "nurse.v2.sre-model-view/19",
        "nurse.v2.sre-model-view/24",
    }:
        # Action selection names remain native until semantic action admission.
        # Other response references still follow the original visibility codec.
        note = value["investigation_note"]
        core = decode_references(
            {k: v for k, v in value.items() if k not in {"investigation_note", "actions"}},
            book=book,
            visible_at=index,
            case_id=case_id,
            limits=CONTEXT_LIMITS,
        )
        try:
            translated = decode_references(
                dict(investigation_note=note),
                book=book,
                visible_at=index,
                case_id=case_id,
                limits=CONTEXT_LIMITS,
            )["investigation_note"]
        except (ExchangeBoundaryError, ValueError):
            translated = note
        return dict(core, actions=value["actions"], investigation_note=translated)
    if not report and packet.get("schema_version") in {
        "nurse.v2.sre-model-view/13",
        "nurse.v2.sre-model-view/15",
    }:
        # Translate action/feedback independently. A note cannot poison action restoration.
        note = value["investigation_note"]
        core = decode_references(
            {k: v for k, v in value.items() if k != "investigation_note"},
            book=book,
            visible_at=index,
            case_id=case_id,
            limits=CONTEXT_LIMITS,
        )
        try:
            translated = decode_references(
                {"investigation_note": note},
                book=book,
                visible_at=index,
                case_id=case_id,
                limits=CONTEXT_LIMITS,
            )["investigation_note"]
        except (ExchangeBoundaryError, ValueError):
            # Retain the original note locally; frozen-context note admission rejects it.
            # Never guess another source, change an action, or forward this raw rejected note.
            translated = note
        return dict(core, investigation_note=translated)
    return decode_references(
        value,
        book=book,
        visible_at=index,
        case_id=case_id,
        limits=REPORT_LIMITS if report else CONTEXT_LIMITS,
    )
