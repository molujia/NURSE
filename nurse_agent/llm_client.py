"""OpenAI-compatible LLM client for the NURSE User Manager."""

from __future__ import annotations

import os
from typing import Any


class OpenAICompatibleClient:
    """Thin wrapper that reads credentials only from environment variables."""

    def __init__(self, *, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("NURSE_OPENAI_MODEL") or "gpt-5-nano"
        self.base_url = base_url or os.environ.get("NURSE_OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for NURSE agent LLM calls.")

    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the `openai` package to enable NURSE agent LLM calls.") from exc
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def complete_text(self, prompt: str, *, max_output_tokens: int = 512) -> str:
        client = self._client()
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            text = self._response_text(response)
            if text:
                return text
        return self._chat_completion_text(client, prompt, max_output_tokens=max_output_tokens)

    def _chat_completion_text(self, client: Any, prompt: str, *, max_output_tokens: int) -> str:
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_output_tokens,
            )
        except Exception as exc:
            if "max_completion_tokens" not in str(exc):
                raise
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_output_tokens,
            )
        return str(response.choices[0].message.content or "")

    def _response_text(self, response: Any) -> str:
        text = getattr(response, "output_text", "")
        if text:
            return str(text)
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", "") or getattr(content, "content", "")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
