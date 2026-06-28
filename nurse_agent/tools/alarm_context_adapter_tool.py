"""Alarm Context Adapter tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.alarm_service import get_or_create_alarm
from nurse.raw_adapters import load_raw_incident

from .common import find_case_dir


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload["dataset"])
    case_id = str(payload["case_id"])
    case_dir = find_case_dir(dataset, case_id)
    incident = load_raw_incident(case_dir, dataset=dataset)
    alarm = get_or_create_alarm(incident, write_cache=True)
    if alarm is None:
        raise RuntimeError(f"missed alarm for {dataset}/{case_id}")
    return {
        "dataset": dataset,
        "case_id": incident.case_id,
        "case_dir": str(case_dir),
        "alarm": alarm.to_dict(),
        "available_modalities": list(incident.available_modalities),
        "topology_edges": [list(edge) for edge in incident.topology_edges],
    }
