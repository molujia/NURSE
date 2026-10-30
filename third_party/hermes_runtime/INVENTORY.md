# Approved HERMES source inventory

This inventory fixes the review boundary for the minimal NURSE runtime before any upstream Python source is vendored. The audited source is HERMES Agent `0.20.5` at commit `8c8193dce5faf71d80bf98eb52bd896ee7be8a27`. File digests and machine-readable selection details are in `SOURCE_MANIFEST.json`.

## Approved source units

| Upstream path | Approved symbols or behavior | NURSE purpose |
| --- | --- | --- |
| `agent/transports/types.py` | `ToolCall`, `Usage`, `NormalizedResponse`, tool-call construction, finish-reason mapping | Normalized contracts |
| `agent/transports/base.py` | `ProviderTransport` | Minimal transport interface |
| `agent/transports/chat_completions.py` | OpenAI-compatible message/tool conversion, request building, response normalization and validation | Single ARK chat-completions transport |
| `agent/message_sanitization.py` | JSON argument repair and deterministic tool-call IDs | Structured-output recovery |
| `agent/conversation_loop.py` | Single-model assistant/tool-result loop pattern | Bounded agent loop |
| `agent/tool_dispatch_helpers.py` | `make_tool_result_message` | Canonical tool-result messages |
| `agent/retry_utils.py` | `parse_retry_after_seconds` | Retry-After handling within one five-attempt budget |
| `agent/bounded_response.py` | Bounded error-body reads and safe close | Safe provider diagnostics |
| `agent/errors.py` | `EmptyStreamError` | Narrow empty-response failure |
| `agent/usage_pricing.py` | `CanonicalUsage`, `normalize_usage` | Token accounting without pricing lookup |
| `agent/redact.py` | `mask_secret`, `redact_sensitive_text` | Secret-safe diagnostics and artifacts |

Approval applies only to the named symbols and behavior. It is not approval to copy an entire module. Each derived destination added in task 6.2 must be recorded in `selected_files` with its upstream inputs and modifications.

## Explicit exclusions

- Provider/model fallback, provider profiles, credential discovery, and credential refresh are excluded. NURSE accepts only an explicitly supplied `ARK_API_KEY2`, the fixed ARK base URL, and `deepseek-v4-flash-ga-260731`.
- Anthropic, Bedrock, Codex, Gemini, OpenRouter, Ollama, and other unrelated transport branches are excluded.
- TUI, gateway, shell, browser, file operations, external search, MCP, memory, delegation, mixture-of-agents, compression, and prompt-hook systems are excluded.
- General parallel tool dispatch, destructive-command detection, filesystem mutation tracking, multimodal tool results, and untrusted-output wrappers are excluded. NURSE uses a closed allowlist and a deterministic state machine.
- Provider pricing tables, billing routes, remote metadata, and network-backed pricing lookup are excluded. Raw usage remains auditable without them.
- Global HERMES configuration and application state are excluded. The embedded runtime must work without an installed `hermes-agent` package.

## Verification rule

Before extraction, the upstream Git HEAD, package version, and every source SHA-256 must match `SOURCE_MANIFEST.json`. The upstream CRLF license and vendored LF license must match their separately recorded byte hashes and remain text-identical after newline normalization. A mismatch invalidates this inventory and requires a new explicit audit rather than silent source substitution.
