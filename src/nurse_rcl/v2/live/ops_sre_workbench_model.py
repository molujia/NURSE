"""D18 move-only work blocks, over the existing bounded shared encoding."""

from typing import Any

MODEL_SCHEMA = "nurse.v2.sre-model-view/5"
DECISION_FIELDS = frozenset(
    {
        "latest_result",
        "delivery_feedback",
        "delivered_record_index",
        "public_working_memory",
    }
)
DECISION_FIELDS_D21 = DECISION_FIELDS | {"post_action_state"}
DECISION_FIELDS_D23 = DECISION_FIELDS_D21 | {"investigation_workbench"}
DECISION_FIELDS_D25 = DECISION_FIELDS_D21 | {"investigation_progress"}
DECISION_FIELDS_D26 = DECISION_FIELDS_D25 | {"question_state", "diagnostic_semantics"}
DECISION_FIELDS_ENTRY = DECISION_FIELDS_D26 | {"action_entries"}
DECISION_FIELDS_OBSERVABLE = DECISION_FIELDS_ENTRY | {"factual_reading_aid"}
REPORT_FIELDS = frozenset({"section_reference_guide"})
REPORT_FIELDS_D26 = REPORT_FIELDS | {"question_state", "diagnostic_semantics"}


def report_fields(schema: str) -> frozenset[str]:
    return (
        REPORT_FIELDS_D26
        if schema
        in {
            "nurse.v2.sre-report-input/18",
            "nurse.v2.sre-report-input/19",
            "nurse.v2.sre-report-input/20",
            "nurse.v2.sre-report-input/22",
        }
        else REPORT_FIELDS
    )


def decision_fields(schema: str) -> frozenset[str]:
    if schema in {
        "nurse.v2.sre-investigation-input/21",
        "nurse.v2.sre-investigation-input/22",
        "nurse.v2.sre-investigation-input/23",
        "nurse.v2.sre-investigation-input/24",
        "nurse.v2.sre-investigation-input/26",
    }:
        return DECISION_FIELDS_OBSERVABLE
    if schema == "nurse.v2.sre-investigation-input/20":
        return DECISION_FIELDS_ENTRY
    if schema == "nurse.v2.sre-investigation-input/12":
        return DECISION_FIELDS
    if schema in {"nurse.v2.sre-investigation-input/13", "nurse.v2.sre-investigation-input/14"}:
        return DECISION_FIELDS_D21
    if schema in {
        "nurse.v2.sre-investigation-input/17",
        "nurse.v2.sre-investigation-input/18",
        "nurse.v2.sre-investigation-input/19",
    }:
        return DECISION_FIELDS_D26
    if schema == "nurse.v2.sre-investigation-input/16":
        return DECISION_FIELDS_D25
    if schema == "nurse.v2.sre-investigation-input/15":
        return DECISION_FIELDS_D23
    raise ValueError("workbench decision input identity differs")


def split_workbench(view: dict[str, Any], *, report: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = view.get("schema_version")
    fields = report_fields(str(expected)) if report else decision_fields(str(expected))
    if (
        report
        and expected
        not in {
            "nurse.v2.sre-report-input/13",
            "nurse.v2.sre-report-input/14",
            "nurse.v2.sre-report-input/15",
            "nurse.v2.sre-report-input/16",
            "nurse.v2.sre-report-input/17",
            "nurse.v2.sre-report-input/18",
            "nurse.v2.sre-report-input/19",
            "nurse.v2.sre-report-input/20",
            "nurse.v2.sre-report-input/22",
        }
    ) or not fields <= view.keys():
        raise ValueError("workbench input identity or fields differ")
    return (
        {k: v for k, v in view.items() if k not in fields},
        {k: view[k] for k in sorted(fields)},
    )


def join_workbench(rest: dict[str, Any], block: dict[str, Any], *, report: bool) -> dict[str, Any]:
    fields = (
        report_fields(str(rest.get("schema_version")))
        if report
        else decision_fields(str(rest.get("schema_version")))
    )
    if type(block) is not dict or set(block) != fields or fields & rest.keys():
        raise ValueError("workbench missing, extra or repeated fields")
    value = dict(rest, **block)
    split_workbench(value, report=report)
    return value
