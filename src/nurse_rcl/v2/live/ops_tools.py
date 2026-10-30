"""Same-snapshot bounded investigation tools, with no formal state writer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NoReturn

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.live.normal_queries import (
    _consume_blocks,
    _feedback_binding,
    _measurement_binding,
    _seal,
)
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence, OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import node_card, prepared_snapshot_id
from nurse_rcl.v2.live.ops_nodes import initial_nodes
from nurse_rcl.v2.live.ops_relations import compare_dependency_timing
from nurse_rcl.v2.live.ops_tool_values import measurement_card, recheck_mechanism
from nurse_rcl.v2.live.repair_measurements import _measure_values
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_queries import RepairQueryFeedback, _feedback_values
from nurse_rcl.v2.live.repair_query_session import RepairQueryCompletedStep

TOOL_PARAMETERS = {
    "read_node": ("card_id",),
    "measure_signal": ("request_key",),
    "compare_timing": ("source_onset_id", "target_onset_id"),
    "check_mechanism": ("card_id",),
    "list_observations": ("candidate_service_key", "modality"),
}

SKILL_TOOLS = {
    "ops-boundary/1": ("read_node", "list_observations"),
    "ops-symptoms/1": ("read_node", "list_observations", "measure_signal"),
    "ops-dependency/1": ("read_node", "compare_timing"),
    "ops-timing/1": ("read_node", "measure_signal", "compare_timing"),
    "ops-mechanisms/1": ("read_node", "measure_signal", "check_mechanism"),
    "ops-scoring/1": ("read_node",),
}


class OpsToolResult(StrictFrozenModel):
    evidence: OpsNodeEvidence
    origin: Literal["measured", "core_reuse", "read_card", "rejected"]
    reused_core_evidence_ids: tuple[str, ...] = ()


def card_id(card: OpsNodeEvidence) -> str:
    return "ops_card_" + sha256_hex(canonical_json_bytes(card.model_dump(mode="json")))


class OpsToolbox:
    """Trusted executor. Agent only receives JSON and the execute allowlist.

    The owner loads preparation once for this channel. It must not hand the
    executor or its data to the language model or expose admit_core as a tool.
    Durable charging belongs to OpsInvestigationLedger, not this pure adapter.
    """

    def _reject_tool(self, reason: str, message: str) -> NoReturn:
        """Legacy error representation; opt-in profiles may expose finite reasons."""
        raise ValueError(message)

    def __init__(self, *, data: RepairPreparationData, block_root: Path):
        self._data, self._block_root = data, block_root
        self.case_snapshot_id = prepared_snapshot_id(data)
        self._requests = {r.request_key: r for r in data.catalog.requests}
        self._sources = {s.source_sha256: s for s in data.sources}
        self._cards: dict[str, OpsNodeEvidence] = {}
        self._core_cards: set[str] = set()
        self._core_steps: dict[str, RepairQueryCompletedStep] = {}
        self._feedback: dict[str, RepairQueryFeedback] = {}
        self._admitted_blocks: set[str] = set()
        for card in initial_nodes(data):
            self._cards[card_id(card)] = card

    @property
    def cards(self) -> Mapping[str, OpsNodeEvidence]:
        return MappingProxyType(self._cards)

    def capabilities(self) -> dict[str, Any]:
        requests = []
        for request in self._requests.values():
            source = self._sources[request.source_sha256]
            requests.append(
                dict(
                    request_key=request.request_key,
                    candidate_service_key=request.candidate_service_key,
                    observation_id=request.observation_id,
                    unit=source.observation.unit,
                    modality=source.observation.modality,
                    signal_family=source.observation.signal_family,
                    metric_name=source.metric_name,
                    bound=request.candidate_service_key is not None,
                    coverage=source.quality.coverage_quality,
                    formally_eligible=source.quality.formal_scoring_eligible,
                )
            )
        return dict(
            case_snapshot_id=self.case_snapshot_id,
            candidate_service_keys=self._data.catalog.candidate_service_keys,
            requests=requests,
            tools=TOOL_PARAMETERS.copy(),
            nodes=[
                dict(
                    card_id=key,
                    node=c.node,
                    parameters=json.loads(c.parameters_json),
                    availability=c.availability,
                    truth_value=c.truth_value,
                    published_by_core=key in self._core_cards,
                )
                for key, c in sorted(self._cards.items())
            ],
            onset_ids=sorted(p.predicate_sha256 for f in self._feedback.values() for p in f.onsets),
            missing_modalities=self._data.diagnostics.legacy.sources.adapter.time_contract.missing_modalities,
        )

    def admit_core(
        self,
        *,
        nodes: tuple[OpsNodeEvidence, ...],
        steps: tuple[RepairQueryCompletedStep, ...],
    ) -> None:
        """Owner-only admission of published COMPLETED receipts, never model input.

        Copy small receipts across the boundary, validate binding/pins once, and
        never share the core's mutable maps, ledger, budget or write paths.
        """
        admitted_nodes = tuple(
            OpsNodeEvidence.model_validate_json(c.model_dump_json()) for c in nodes
        )
        if any(c.case_snapshot_id != self.case_snapshot_id for c in admitted_nodes):
            raise ValueError("core node snapshot mismatch")
        admitted_steps = tuple(
            RepairQueryCompletedStep.model_validate_json(s.model_dump_json()) for s in steps
        )
        for step in admitted_steps:
            request = self._requests.get(step.feedback.request_key)
            if (
                request is None
                or step.selection.selected_request_key != request.request_key
                or (step.selection.catalog_sha256 != self._data.catalog.catalog_sha256)
            ):
                raise ValueError("core completed request binding differs")
            source = self._sources[request.source_sha256]
            _measurement_binding(step.measurement, source, self._data.catalog)
            _feedback_binding(
                step.feedback, step.measurement, source, self._data.catalog, request.request_key
            )
            for obj, field in (
                (step.measurement, "measurement_sha256"),
                (step.feedback, "result_sha256"),
                (step.selection, "result_sha256"),
            ):
                if _seal(obj, field) != obj:
                    raise ValueError("core completed receipt digest differs")
            if any(_seal(p, "result_sha256") != p for p in step.feedback.predicates):
                raise ValueError("core predicate digest differs")
        for card in admitted_nodes:
            key = card_id(card)
            self._cards[key] = card
            self._core_cards.add(key)
        for step in admitted_steps:
            self._remember_feedback(step.feedback)
            self._core_steps[step.feedback.request_key] = step

    def _remember_feedback(self, feedback: RepairQueryFeedback) -> None:
        request = self._requests.get(feedback.request_key)
        if (
            request is None
            or feedback.catalog_sha256 != self._data.catalog.catalog_sha256
            or (
                len(feedback.predicates) != len(request.needs)
                or len(feedback.onsets) != len(request.needs)
                or _seal(feedback, "result_sha256") != feedback
            )
        ):
            raise ValueError("investigation feedback binding differs")
        for need, predicate, onset in zip(
            request.needs, feedback.predicates, feedback.onsets, strict=True
        ):
            if (
                predicate.source_sha256 != request.source_sha256
                or predicate.observation_id != request.observation_id
                or predicate.candidate_service_key != request.candidate_service_key
                or predicate.mechanism_key != need.mechanism_key
                or predicate.predicate_kind != need.predicate_kind
                or predicate.measurement_sha256 != feedback.measurement_sha256
                or onset.need_key != need.need_key
                or onset.predicate_sha256 != predicate.result_sha256
                or _seal(predicate, "result_sha256") != predicate
            ):
                raise ValueError("investigation predicate binding differs")
        previous = self._feedback.get(feedback.request_key)
        if previous is not None and previous != feedback:
            raise ValueError("same snapshot returned conflicting measurement content")
        self._feedback[feedback.request_key] = feedback

    def restore_evidence(self, saved: OpsInvestigationEvidence) -> None:
        """Owner-only recovery from this channel's integrity-checked ledger."""
        card = saved.evidence
        if card.case_snapshot_id != self.case_snapshot_id:
            raise ValueError("investigation evidence snapshot differs")
        if saved.origin in ("measured", "core_reuse"):
            body = json.loads(card.payload_json)
            if body.get("kind") == "supplementary_episode":
                feedback = RepairQueryFeedback.model_validate_json(
                    canonical_json_bytes(body["feedback"]),
                )
                self._remember_feedback(feedback)
        self._cards[card_id(card)] = card

    def execute(self, tool: str, parameters: object) -> OpsToolResult:
        fields = TOOL_PARAMETERS.get(tool)
        if fields is None or type(parameters) is not dict or set(parameters) != set(fields):
            self._reject_tool("invalid_parameters", "tool or parameter domain is not registered")
        if any(
            type(value) is not str or not 1 <= len(value) <= 512 for value in parameters.values()
        ):
            self._reject_tool(
                "invalid_parameters",
                "tool parameters require bounded registered text IDs",
            )
        if tool == "measure_signal":
            return self._measure(parameters["request_key"])
        if tool == "list_observations":
            return self._observations(parameters["candidate_service_key"], parameters["modality"])
        if tool == "compare_timing":
            return self._timing(parameters["source_onset_id"], parameters["target_onset_id"])
        key = parameters["card_id"]
        if key not in self._cards:
            self._reject_tool("reference_not_available", "node ID not in this snapshot")
        card = self._cards[key]
        if tool == "check_mechanism":
            if key not in self._core_cards:
                self._reject_tool(
                    "card_kind_mismatch",
                    "mechanism requires a core-published condition table",
                )
            card = recheck_mechanism(
                card,
                self._data,
                tuple(self._feedback.values()),
                reject=self._reject_tool,
            )
        return OpsToolResult(evidence=card, origin="read_card")

    def _measure(self, key: str) -> OpsToolResult:
        request = self._requests.get(key)
        if request is None or request.candidate_service_key is None:
            raise ValueError("measurement request is not a registered bound source")
        source = self._sources[request.source_sha256]
        step = self._core_steps.get(key)
        reused: tuple[str, ...] = ()
        if step is not None:
            measurement, feedback = step.measurement, step.feedback
            origin: Literal["core_reuse", "measured"] = "core_reuse"
            reused = (measurement.measurement_sha256, feedback.result_sha256)
        else:
            if key in self._feedback:
                raise ValueError("measurement already obtained; read its evidence card")
            _consume_blocks(source, self._data, self._block_root, self._admitted_blocks)
            measurement = _measure_values(source)
            feedback = _feedback_values(self._data.catalog, source, measurement)
            origin = "measured"
        card = measurement_card(self.case_snapshot_id, source, measurement, feedback)
        self._remember_feedback(feedback)
        self._cards[card_id(card)] = card
        return OpsToolResult(evidence=card, origin=origin, reused_core_evidence_ids=reused)

    def _observations(self, candidate: str, modality: str) -> OpsToolResult:
        if candidate not in self._data.catalog.candidate_service_keys or (
            modality not in {"metrics", "traces", "logs"}
        ):
            raise ValueError("candidate/modality domain is not registered")
        rows = [
            r
            for r in self.capabilities()["requests"]
            if r["candidate_service_key"] == candidate and r["modality"] == modality
        ]
        card = node_card(
            snapshot=self.case_snapshot_id,
            node="boundary",
            parameters=dict(candidate_service_key=candidate, modality=modality),
            payload=dict(requests=rows, score_authority=False),
            evidence_ids=(self._data.catalog.catalog_sha256,),
            source_refs=(self._data.source_descriptor_sha256,),
            interval=self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
            availability="available" if rows else "unavailable",
            limitations=("scalar_series_inventory_only_not_health",)
            if rows
            else (
                "no_registered_scalar_series_for_candidate_modality",
                "error_record_summaries_may_exist_separately",
            ),
        )
        return OpsToolResult(evidence=card, origin="read_card")

    def _timing(self, source_id: str, target_id: str) -> OpsToolResult:
        onsets = {
            onset.predicate_sha256: (predicate.candidate_service_key, onset.bounds_ns)
            for feedback in self._feedback.values()
            for predicate, onset in zip(feedback.predicates, feedback.onsets, strict=True)
        }
        if source_id not in onsets or target_id not in onsets:
            raise ValueError("onset ID is not an actually completed measurement")
        source, source_bounds = onsets[source_id]
        target, target_bounds = onsets[target_id]
        card = compare_dependency_timing(
            snapshot=self.case_snapshot_id,
            graph=self._data.graph.graph_pair,
            clock=self._data.clock,
            source_candidate=source,
            target_candidate=target,
            source_bounds=source_bounds,
            target_bounds=target_bounds,
            source_evidence_id=source_id,
            target_evidence_id=target_id,
        )
        return OpsToolResult(evidence=card, origin="read_card")
