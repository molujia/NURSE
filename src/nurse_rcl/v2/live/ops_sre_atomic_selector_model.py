"""Read-only selector literals derived from the already admitted frozen table."""

from typing import Any, cast

from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import ATOMIC_ENTRY_SCHEMA


def selector_choices(table: dict[str, Any]) -> list[str]:
    if type(table) is not dict or table.get("schema_version") != ATOMIC_ENTRY_SCHEMA:
        raise ValueError("atomic entry table identity differs")
    rows = table.get("entries")
    if type(rows) is not list:
        raise ValueError("atomic entry table rows differ")
    refs = [row.get("entry_ref") for row in rows if type(row) is dict]
    if len(refs) != len(rows) or any(type(ref) is not str or not ref for ref in refs):
        raise ValueError("atomic entry references differ")
    if len(set(refs)) != len(refs):
        raise ValueError("atomic entry references duplicate")
    return cast(list[str], refs)  # Every leaf was checked above; no value conversion.


def validate_selector_choices(value: Any, table: dict[str, Any]) -> None:
    if type(value) is not list or any(type(ref) is not str for ref in value):
        raise ValueError("atomic selector choices shape differs")
    if value != selector_choices(table):
        raise ValueError("atomic selector choices binding differs")
