"""OpenAI-compatible LLM connectivity probe for NURSE."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .llm_bridge import DEFAULT_OPENAI_MODEL, NurseLLMBridge


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe NURSE LLM connectivity")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_OPENAI_MODEL],
        help="OpenAI-compatible model names to test, e.g. gpt-5-nano",
    )
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "nurse" / "llm_connectivity.json"))
    parser.add_argument("--require-ok", action="store_true", help="Exit non-zero if any requested model is not ok")
    return parser


def probe_models(*, models: List[str], openai_model: Optional[str], openai_base_url: str) -> Dict:
    if openai_model:
        os.environ["NURSE_OPENAI_MODEL"] = openai_model
    if openai_base_url:
        os.environ["NURSE_OPENAI_BASE_URL"] = openai_base_url

    bridge = NurseLLMBridge()
    results = []
    for model in models:
        started = time.perf_counter()
        if not bridge.has_credentials(model):
            env_key = bridge.required_env_key(model)
            results.append(
                {
                    "model": model,
                    "status": "skipped",
                    "reason": f"Missing {env_key}",
                    "elapsed_seconds": 0.0,
                }
            )
            continue
        response = bridge.propose_evidence(
            task="nurse_evidence_proposal",
            payload={
                "case_id": "llm-connectivity-smoke",
                "alarm": {
                    "case_id": "llm-connectivity-smoke",
                    "dataset": "synthetic",
                    "alert_entity": "checkout",
                    "signals": [
                        {
                            "entity": "checkout",
                            "metric": "error_rate",
                            "value": 0.42,
                            "baseline": 0.01,
                            "score": 8.0,
                        }
                    ],
                },
            },
            model=model,
        )
        results.append(
            {
                "model": model,
                "resolved_model": response.get("model", ""),
                "status": response.get("status"),
                "reason": response.get("reason", ""),
                "candidate_count": len(response.get("candidates", [])),
                "usage": response.get("usage", {}),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "openai_base_url": os.environ.get("NURSE_OPENAI_BASE_URL", ""),
        "results": results,
    }


def failed_probe_results(report: Dict) -> List[Dict]:
    results = report.get("results", [])
    if not isinstance(results, list):
        return [{"model": "<report>", "status": "error", "reason": "missing results list"}]
    return [item for item in results if isinstance(item, dict) and item.get("status") != "ok"]


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = probe_models(
        models=args.models,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[NURSE] LLM connectivity report: {output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = failed_probe_results(report)
    if args.require_ok and failures:
        print(f"[NURSE] LLM connectivity failed for {len(failures)} model(s).")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
