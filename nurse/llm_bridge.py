"""OpenAI-compatible bounded LLM bridge for NURSE evidence proposals."""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Dict, Iterable, List

from .llm_providers import provider_for_model
from .matching import canonical_entity
from .schemas import EvidenceCandidate


DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(number, 1.0))


def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore
    except Exception:
        return ""

    registry_locations = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for root, subkey in registry_locations:
        try:
            with winreg.OpenKey(root, subkey) as key:
                registry_value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        text = str(registry_value or "").strip()
        if text:
            return os.path.expandvars(text)
    return ""


class NurseLLMBridge:
    """Small self-contained bridge used by the Evidence Core.

    The public repository intentionally uses one OpenAI-compatible path. Users
    can point it at OpenAI or another compatible endpoint by setting
    ``NURSE_OPENAI_BASE_URL``; credentials are always read from
    ``OPENAI_API_KEY``.
    """

    def propose_evidence(self, *, task: str, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        config = self._model_config(model)
        api_key = _get_env_var(config["env_key"])
        if not api_key:
            return {
                "status": "skipped",
                "reason": f"Missing {config['env_key']} for model {model}",
                "candidates": [],
            }

        output_contract = str(payload.get("output_contract") or "")
        rank_repair_only = output_contract == "rank_repair_only"
        grounded_user_manager = output_contract == "grounded_rcl_user_proposal"
        system_prompt = self._system_prompt(rank_repair_only=rank_repair_only, grounded_user_manager=grounded_user_manager)
        max_output_tokens = 3600 if rank_repair_only else 1200
        last_json_error = ""
        parsed: Dict[str, Any] = {}
        usage: Dict[str, Any] = {}

        try:
            for attempt in range(2):
                user_payload = {
                    "task": task,
                    "payload": payload,
                    "retry_instruction": (
                        "Previous response was not parseable JSON. Return one complete JSON object only."
                        if attempt
                        else ""
                    ),
                }
                response_text, usage = self._complete_json_text(
                    config=config,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    max_output_tokens=max_output_tokens,
                )
                try:
                    loaded = json.loads(response_text or "{}")
                except JSONDecodeError as exc:
                    last_json_error = str(exc)
                    if attempt == 0:
                        continue
                    raise
                parsed = loaded if isinstance(loaded, dict) else {}
                break

            candidates = parsed.get("candidates")
            if not isinstance(candidates, list):
                candidates = parsed.get("candidate_evidence")
            semantic_facts = parsed.get("semantic_facts")
            if not isinstance(semantic_facts, list):
                semantic_facts = parsed.get("hypotheses")
            hypotheses = parsed.get("hypotheses")
            candidate_rules = parsed.get("candidate_rules")
            rank_repair = parsed.get("rank_repair")
            return {
                "status": "ok",
                "model": config["model_name"],
                "semantic_facts": semantic_facts if isinstance(semantic_facts, list) else [],
                "candidates": candidates if isinstance(candidates, list) else [],
                "hypotheses": hypotheses if isinstance(hypotheses, list) else [],
                "candidate_rules": candidate_rules if isinstance(candidate_rules, list) else [],
                "rank_repair": rank_repair if isinstance(rank_repair, list) else [],
                "proposal": parsed if isinstance(parsed, dict) else {},
                "usage": usage,
                "retry_count": 1 if last_json_error and parsed else 0,
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "candidates": []}

    def required_env_key(self, model: str) -> str:
        return str(self._model_config(model)["env_key"])

    def has_credentials(self, model: str) -> bool:
        return bool(_get_env_var(self.required_env_key(model)))

    def _model_config(self, model: str) -> Dict[str, str]:
        provider = provider_for_model(model)
        normalized = str(model or "").lower().strip()
        if provider.name == "openai":
            model_name = _get_env_var("NURSE_OPENAI_MODEL") or (
                DEFAULT_OPENAI_MODEL if normalized in {"", "gpt5-nano", "gpt-5-nano"} else provider.model
            )
            base_url = _get_env_var("NURSE_OPENAI_BASE_URL") or provider.base_url
        elif provider.name == "volcano":
            model_name = provider.model
            base_url = _get_env_var("NURSE_VOLCANO_BASE_URL") or provider.base_url
        else:
            model_name = provider.model
            base_url = _get_env_var("NURSE_ZHIPU_BASE_URL") or provider.base_url
        return {
            "provider_name": provider.name,
            "uses_openai_client": provider.uses_openai_client,
            "env_key": provider.env_key,
            "base_url": base_url,
            "model_name": model_name,
        }

    def _system_prompt(self, *, rank_repair_only: bool, grounded_user_manager: bool) -> str:
        if rank_repair_only:
            return (
                "You assist NURSE RCA with bounded neural-symbolic rank repair. "
                "Return only valid JSON with this exact shape: "
                '{"rank_repair":[{"promote":"service","to_rank":1,'
                '"supporting_evidence_ids":["evidence-id"],"confidence":0.0,'
                '"reason":"short evidence-grounded reason"}]}. '
                "If no evidence-grounded repair is justified, return {\"rank_repair\":[]}. "
                "Use only services from current_ranking and evidence IDs listed in candidate_cards. "
                "Do not infer from case IDs, labels, answer keys, hidden data, or prior benchmark knowledge. "
                "No prose, markdown, explanations outside JSON, or extra arrays."
            )
        if grounded_user_manager:
            return (
                "You are NURSE's User Manager for evidence-grounded RCA explanation. "
                "The final_ranking was produced by the deterministic Evidence Core and you must not change it, reorder it, or imply that you changed it. "
                "Return one compact strict JSON object with fields: summary, top_candidate_explanation, "
                "evidence_bound_next_steps, advisory_notes, unsupported_claims. "
                "Every causal or diagnostic claim must cite evidence IDs from evidence_ids or verified_evidence. "
                "Put any useful but unsupported idea in unsupported_claims instead of presenting it as fact. "
                "Do not infer from case IDs, labels, answer keys, hidden data, or prior benchmark knowledge. "
                "No markdown or prose outside JSON."
        )
        return (
            "You assist NURSE RCA with bounded neural-symbolic rank fact proposal. "
            "Return one compact strict JSON object with optional arrays: semantic_facts, hypotheses, candidates, candidate_rules. "
            "Use semantic_facts for any fact that may affect ranking. "
            "Use only services from service_candidates/current_ranking and evidence_ids from verified_evidence. "
            "Do not emit rank_repair unless output_contract is rank_repair_only. "
            "Do not infer from case_id, labels, answer keys, or hidden data."
        )

    def _complete_json_text(
        self,
        *,
        config: Dict[str, str],
        system_prompt: str,
        user_payload: Dict[str, Any],
        max_output_tokens: int,
    ) -> tuple[str, Dict[str, Any]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the `openai` package to enable NURSE LLM calls.") from exc

        client = OpenAI(api_key=_get_env_var(config["env_key"]), base_url=config["base_url"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        try:
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=max_output_tokens,
            )
        except Exception as exc:
            if "max_completion_tokens" not in str(exc):
                raise
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_output_tokens,
            )
        text = str(response.choices[0].message.content or "")
        return text, self._usage_to_dict(getattr(response, "usage", None))

    def _usage_to_dict(self, usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return dict(usage.model_dump())
        if isinstance(usage, dict):
            return dict(usage)
        return {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if hasattr(usage, key)
        }


def llm_candidates_to_evidence(
    *,
    case_id: str,
    raw_candidates: List[Dict[str, Any]],
    time_range: Dict[str, Any],
    model: str,
    service_candidates: Iterable[str] | None = None,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    allowed = {
        canonical_entity(str(candidate))
        for candidate in (service_candidates or [])
        if canonical_entity(str(candidate))
    }
    for index, raw in enumerate(raw_candidates or []):
        if not isinstance(raw, dict):
            continue
        entity = canonical_entity(str(raw.get("entity") or raw.get("service") or "").strip())
        signal = str(raw.get("signal") or raw.get("metric") or "").strip()
        kind = str(raw.get("kind") or raw.get("source_type") or "metric").strip().lower()
        if allowed and entity not in allowed:
            continue
        if not entity or not signal:
            continue
        claim = raw.get("claim") if isinstance(raw.get("claim"), dict) else {}
        confidence = _confidence(raw.get("confidence", 0.5))
        candidates.append(
            EvidenceCandidate(
                evidence_id=str(raw.get("evidence_id") or f"{case_id}:llm:{index}:{entity}:{signal}"),
                case_id=case_id,
                entity=entity,
                signal=signal,
                kind=kind,
                time_range=dict(raw.get("time_range") if isinstance(raw.get("time_range"), dict) else time_range),
                claim={
                    **claim,
                    "metric_role": claim.get("metric_role") or "llm_verified_check",
                    "evidence_program_weight": min(
                        float(claim.get("evidence_program_weight", confidence * 0.05) or 0.0),
                        0.05,
                    ),
                    "llm_reason": raw.get("reason") or raw.get("rationale") or "",
                },
                provenance={
                    "source": "llm_proposal",
                    "model": model,
                    "raw_kind": raw.get("kind") or raw.get("source_type") or "",
                },
                confidence=confidence,
            )
        )
    return candidates
