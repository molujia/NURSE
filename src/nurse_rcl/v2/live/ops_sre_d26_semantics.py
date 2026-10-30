"""Registered field meanings bound only to already delivered semantic records."""

# ruff: noqa: RUF001 -- model-facing Chinese engineering meanings.

from __future__ import annotations

import re
from copy import deepcopy
from functools import lru_cache
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION
from nurse_rcl.v2.live.ops_sre_condition_briefs import _FORMAL, _RECHECK
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view
from nurse_rcl.v2.mechanism_library import build_mechanism_library

SEMANTICS_SCHEMA = "nurse.v2.sre-diagnostic-semantics/1"
SEMANTICS_AUTHORITY = "registered_meaning_not_diagnostic_evidence"
_SCORE_KINDS = {
    "score_lineage_summary",
    "score_family",
    "score_term",
    "score_propagation",
    "score_path",
}
_NUMERIC_PATHS = {
    "score_family": ("payload.family.value", "payload.selected_term.local_strength"),
    "score_term": ("payload.term.local_strength",),
    "score_propagation": ("payload.propagation.propagation_score",),
    "score_path": ("payload.path.contribution",),
    "score_lineage_summary": (
        "payload.score.total_score",
        "payload.score.event_evidence_score",
        "payload.score.local_score_component",
        "payload.score.propagation_score_component",
    ),
}
_SCORE_FIELDS = {
    "truth_value": "卡头逻辑真值不覆盖payload中已交付的数值，也不代表数值缺失。",
    "payload.family.value": "正式家族评分值；不是根因概率、健康谓词或观测建议前提。",
    "payload.term.local_strength": "该term的局部强度；数值已知与是否实际选用独立。",
    "payload.selected_term.local_strength": "实际代表term的局部强度，不等于其机制被确认。",
    "payload.score.selected_family_keys": "实际选中家族；含cpu时不能说cpu未进入聚合。",
    "payload.score.unknown_term_ids": "候选term数值unknown的ID集合；不表示未读或未测量数量。",
    "payload.family.unknown_term_ids": "本家族term数值unknown集合；不表示未读或未测量数量。",
    "payload.used_in_aggregation": "该来源是否实际参与正式聚合；core_published不等于已用。",
    "payload.selected_term.role_mechanism_ids": "机制追踪入口；ID不同不能证明机制不同。",
    "payload.term.role_mechanism_ids": "仅追踪入口；同机制须核对实供key、definition与条件。",
    "payload.pool": "选择池与选中/未用来源，不是独立证据票数。",
    "payload.family": "家族代表选择和聚合来源；分差不自动成为因果贡献。",
    "payload.propagation": "冻结传播评分与选中路径；零值不证明没有传播或依赖故障。",
    "payload.score": "冻结评分分解及实际选中家族/term；不是根因概率或观测前提。",
    "payload.path.contribution": "该冻结依赖对齐症状路径的传播支持值，不是确认因果。",
}
_CONDITION_FIELDS = {
    "payload.conditions[].requirement": (
        "necessary必要、optional补充、refuting反证角色；与真值独立。"
    ),
    "payload.conditions[].truth_value": (
        "true成立、false未成立、unknown未确定；反证true会削弱解释。"
    ),
    "payload.conditions[].operator": "AND/OR组合；已知决定性叶子可决定父真值，未知叶子仍未知。",
    "payload.conditions[].leaves": "仅实际交付叶子可引用；模板定义不授予未读叶子或条件事实。",
    "payload.conditions[].resolution_state": "既有规则判定的完整观察/决定性叶子/未解决状态。",
    "payload.decision.disposition": "必要条件合取的模板接受状态；不是因果确认或正式评分贡献。",
    "payload.formal_decision": "补充复核中的原正式机制参考，不覆盖正式RCL。",
}
_NODE_FIELDS = {
    "symptoms": {
        "payload.metric_name": "原始指标名称；工程解释须结合单位、方向、窗口和覆盖。",
        "payload.signed_standardized_change": "相对基线有方向变化，不单独证明故障或健康。",
        "payload.predicates": "已测谓词三值；false不自动等于服务健康，unknown不等于未读。",
        "payload.episodes": "已测异常片段；起点和分辨率不证明因果。",
        "payload.episode_status": "摘要未测episode时不能从摘要推断异常起点或正常。",
        "payload.facts": "实际规则事实及原来源；错误记录占比不等于请求总体失败率。",
        "unit": "原测量单位，不能跨单位比较数值。",
        "window_ns": "该来源实际适用时间窗口；不能扩大到未测时段。",
        "coverage": "来源覆盖限制，未显示/缺失不等于正常。",
    },
    "dependency": {
        "payload.edge": "实际caller/callee、方向、relation_type和覆盖；拓扑边不是依赖失败。",
        "payload.edges": "实际交付边集合；没有观察边不证明不存在依赖。",
    },
    "timing": {
        "payload.source_bounds_ns": "已测源起点区间；仅按原时钟不确定性判断先后。",
        "payload.target_bounds_ns": "已测对端起点区间；先后不等于因果。",
        "payload.onset": "实际测量起点，缺起点保持unknown，不从卡片时间猜测。",
        "payload.path": "原依赖对齐症状路径及原覆盖/时序限制；不是确认的失败传播。",
    },
    "boundary": {
        "payload": "准入候选、窗口、模态及可达能力目录，不是健康或故障观测。",
    },
}
_MECHANISM_ZH = {
    "incoming_load_pressure": "入站请求增加伴随局部压力；来源仍待区分。",
    "local_resource_or_execution_pressure": "局部资源或执行压力可能解释服务恶化。",
    "local_storage_or_network_failure": "局部存储或网络操作失败可能解释恶化。",
    "received_dependency_failure": "对依赖调用观察到失败；局部替代解释仍开放。",
    "unclassified_local_anomaly": "已见局部异常，但不足以确定更具体机制。",
    "workload_unavailability": "工作负载不可用的观测可能解释可用性下降。",
}


@lru_cache(maxsize=1)
def _definitions() -> dict[str, dict[str, Any]]:
    # Immutable code-declared templates only, never a cross-case evidence cache.
    return {
        d.mechanism_key: d.model_dump(mode="json") for d in build_mechanism_library().definitions
    }


def _numeric_fields(row: dict[str, Any], paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for path in paths:
        current: Any = row
        for part in path.split("."):
            if type(current) is not dict or part not in current:
                result[path] = "not_supplied"
                break
            current = current[part]
        else:
            result[path] = (
                "unknown"
                if current is None
                else ("known" if type(current) in {int, float} else "not_numeric")
            )
    return result


def _meaning(row: dict[str, Any]) -> tuple[str | None, dict[str, dict[str, Any]], tuple[str, ...]]:
    body = row["payload"]
    kind, version, node = body.get("kind"), row.get("tool_version"), row.get("node")
    if version == "sre-score-lineage/1" and node == "scoring" and kind in _SCORE_KINDS:
        paths = _NUMERIC_PATHS[kind]
        return kind, {kind: dict(card_kind=kind, fields=_SCORE_FIELDS)}, paths
    shape = set(body)
    comparable = (
        row.get("schema_version") == "nurse.v2.sre-evidence-brief/3"
        and row.get("representation") == BRIEF_REPRESENTATION
        and shape
        in (
            _FORMAL | {"disposition_rule", "score_effect_rule"},
            _RECHECK | {"disposition_rule", "score_effect_rule"},
        )
        and body["disposition_rule"] == "necessary_conjunction_only"
        and body["score_effect_rule"] == "requires_actual_term_and_family_selection"
    )
    if (
        node == "mechanisms"
        and version == "ops-mechanisms/1"
        and (shape in (_FORMAL, _RECHECK) or comparable)
    ):
        decision = body.get("decision", body.get("formal_decision", {}))
        key = decision.get("mechanism_key")
        definition = _definitions().get(key)
        if (
            definition is None
            or decision.get("mechanism_definition_sha256") != definition["definition_sha256"]
        ):
            return None, {}, ()
        name = "mechanism." + key
        return (
            name,
            {
                "mechanism_conditions": dict(card_kind="mechanism", fields=_CONDITION_FIELDS),
                name: dict(
                    definition=definition,
                    chinese_meaning=_MECHANISM_ZH[key],
                    limitations="注册模板解释而非候选真值；接受不等于因果；不授未读叶子。",
                ),
            },
            (),
        )
    recognized = (
        node == "symptoms"
        and version in {"ops-symptoms/1", "sre-prepared-summary/1"}
        and kind
        in {
            "fixed_summary",
            "error_summary",
            "paid_episode",
            "supplementary_episode",
            "prepared_summary",
        }
    ) or (node in {"boundary", "dependency", "timing"} and version == "ops-" + node + "/1")
    if recognized:
        assert type(node) is str
        return node, {node: dict(card_kind=node, fields=_NODE_FIELDS[node])}, ()
    return None, {}, ()


def diagnostic_semantics(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Related dictionary entries enter the actual input; bodies are never duplicated."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    if len(history) > 8:
        raise ValueError("diagnostic semantics action limit")
    deliveries: list[tuple[int, dict[str, Any]]] = []
    for index, saved in enumerate(history, 1):
        if saved.get("action_index") != index:
            raise ValueError("diagnostic semantics action sequence differs")
        if saved["status"] != "completed":
            continue
        view = working_evidence_view(expand(saved["view"]))
        if view.get("status") == "resource_limit":
            continue
        deliveries.extend((index, row) for row in _delivered_records(view))
    return semantics_for_deliveries(deliveries)


def semantics_for_deliveries(deliveries: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Shared pure projection of admitted report/decision deliveries, no source reads."""
    entries: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for index, row in deliveries:
        key, meanings, paths = _meaning(row)
        entries.update(meanings)
        identity = sha256_hex(canonical_json_bytes(row))
        rep = row.get("representation")
        binding = bindings.setdefault(
            identity,
            dict(
                card_id=row["card_id"],
                record_sha256=identity,
                action_indices=[],
                delivery_kind="brief"
                if rep == BRIEF_REPRESENTATION
                else "partial"
                if rep
                in {
                    "overview_not_full_judgment",
                    "delivered_section_not_full_judgment",
                }
                else "complete",
                status="registered" if key else "semantics_not_registered",
                entry_keys=sorted(meanings),
                numeric_fields=_numeric_fields(row, paths),
                authority="meaning_only_original_delivery_authority_unchanged",
            ),
        )
        if index not in binding["action_indices"]:
            binding["action_indices"].append(index)
    return deepcopy(
        dict(
            schema_version=SEMANTICS_SCHEMA,
            authority=SEMANTICS_AUTHORITY,
            entries=entries,
            bindings=list(bindings.values()),
        )
    )


def validate_semantic_dictionary(value: Any, *, short_refs: bool = False) -> None:
    """Check declared representation; only the owner authenticates original content.

    Provider model packets contain typed H handles, not original SHA256 values.
    Original/owner callers stay strict and never accept handles as digest pins.
    """
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "authority",
            "entries",
            "bindings",
        }
        or (value["schema_version"], value["authority"])
        != (
            SEMANTICS_SCHEMA,
            SEMANTICS_AUTHORITY,
        )
        or type(value["entries"]) is not dict
        or type(value["bindings"]) is not list
    ):
        raise ValueError("D26 semantic transport shape differs")
    registered = {key: dict(card_kind=key, fields=_SCORE_FIELDS) for key in _SCORE_KINDS}
    registered.update(
        {key: dict(card_kind=key, fields=fields) for key, fields in _NODE_FIELDS.items()}
    )
    registered["mechanism_conditions"] = dict(card_kind="mechanism", fields=_CONDITION_FIELDS)
    registered.update(
        {
            "mechanism." + key: dict(
                definition=definition,
                chinese_meaning=_MECHANISM_ZH[key],
                limitations="注册模板解释而非候选真值；接受不等于因果；不授未读叶子。",
            )
            for key, definition in _definitions().items()
        }
    )
    if any(k not in registered or v != registered[k] for k, v in value["entries"].items()):
        raise ValueError("D26 semantic registry differs")
    seen = set()
    used = set()
    for binding in value["bindings"]:
        if type(binding) is not dict or set(binding) != {
            "card_id",
            "record_sha256",
            "action_indices",
            "delivery_kind",
            "status",
            "entry_keys",
            "numeric_fields",
            "authority",
        }:
            raise ValueError("D26 semantic binding shape differs")
        digest, indices, keys, fields = (
            binding["record_sha256"],
            binding["action_indices"],
            binding["entry_keys"],
            binding["numeric_fields"],
        )
        if (
            type(digest) is not str
            or re.fullmatch(
                r"H[1-9][0-9]{0,3}" if short_refs else r"[a-f0-9]{64}",
                digest,
            )
            is None
            or digest in seen
            or (
                binding["card_id"] is not None
                and (type(binding["card_id"]) is not str or not binding["card_id"])
            )
            or type(indices) is not list
            or not indices
            or len(indices) > 8
            or any(type(i) is not int or not 1 <= i <= 8 for i in indices)
            or sorted(set(indices)) != indices
            or binding["delivery_kind"] not in {"complete", "brief", "partial"}
            or binding["authority"] != "meaning_only_original_delivery_authority_unchanged"
            or type(keys) is not list
            or any(type(k) is not str for k in keys)
            or keys != sorted(set(keys))
            or not set(keys) <= value["entries"].keys()
            or binding["status"] != ("registered" if keys else "semantics_not_registered")
            or type(fields) is not dict
            or any(
                type(v) is not str
                or v
                not in {
                    "known",
                    "unknown",
                    "not_supplied",
                    "not_numeric",
                }
                for v in fields.values()
            )
        ):
            raise ValueError("D26 semantic binding shape differs")
        score_keys = set(keys) & _SCORE_KINDS
        expected = set(_NUMERIC_PATHS[next(iter(score_keys))]) if len(score_keys) == 1 else set()
        if len(score_keys) > 1 or set(fields) != expected:
            raise ValueError("D26 semantic numeric fields differ")
        seen.add(digest)
        used.update(keys)
    if used != set(value["entries"]):
        raise ValueError("D26 semantic dictionary closure differs")


def validate_diagnostic_semantics(
    value: Any,
    *,
    delivered_index: dict[str, Any],
    short_refs: bool = False,
) -> None:
    """Bind static meanings to the frozen delivery index; owner rederives source content."""
    validate_semantic_dictionary(value, short_refs=short_refs)
    index = {r["record_sha256"]: r for r in delivered_index["records"]}
    seen = set()
    used = set()
    numeric = {key: set(paths) for key, paths in _NUMERIC_PATHS.items()}
    for binding in value["bindings"]:
        if type(binding) is not dict or set(binding) != {
            "card_id",
            "record_sha256",
            "action_indices",
            "delivery_kind",
            "status",
            "entry_keys",
            "numeric_fields",
            "authority",
        }:
            raise ValueError("D26 semantic binding shape differs")
        digest = binding["record_sha256"]
        row = index.get(digest)
        if (
            row is None
            or digest in seen
            or any(
                binding[k] != row[k]
                for k in (
                    "card_id",
                    "action_indices",
                    "delivery_kind",
                )
            )
            or binding["authority"] != "meaning_only_original_delivery_authority_unchanged"
        ):
            raise ValueError("D26 semantic delivery binding differs")
        seen.add(digest)
        keys = binding["entry_keys"]
        if (
            type(keys) is not list
            or keys != sorted(set(keys))
            or (
                not set(keys) <= value["entries"].keys()
                or binding["status"] != ("registered" if keys else "semantics_not_registered")
            )
        ):
            raise ValueError("D26 semantic entry binding differs")
        allowed = {row["node"], row["kind"]}
        if row.get("mechanism_key") in _definitions():
            allowed.update({"mechanism_conditions", "mechanism." + row["mechanism_key"]})
        if not set(keys) <= allowed:
            raise ValueError("D26 semantic registered type differs")
        fields = binding["numeric_fields"]
        expected = numeric.get(row["kind"], set()) if keys else set()
        if (
            type(fields) is not dict
            or set(fields) != expected
            or any(
                v not in {"known", "unknown", "not_supplied", "not_numeric"}
                for v in fields.values()
            )
        ):
            raise ValueError("D26 semantic numeric fields differ")
        used.update(keys)
    if seen != set(index) or used != set(value["entries"]):
        raise ValueError("D26 semantic transport closure differs")
