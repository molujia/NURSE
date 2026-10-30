# Modifications from upstream HERMES

At the project-foundation commit, no HERMES Python source file has been copied. Only the upstream license and provenance metadata are present. Runtime selection and every subsequent adaptation are recorded here when the selected files are introduced.

The approved derivative is limited to OpenAI-compatible transport, message/tool contracts, the agent/tool-call loop, structured-output repair, retry and error handling, and usage accounting. TUI, shell, browser, delegation, memory, and unrelated provider features are excluded by design.

## Inventory approval (task 6.1)

The HERMES `0.20.5` source tree at commit `8c8193dce5faf71d80bf98eb52bd896ee7be8a27` was audited on 2026-09-01. `SOURCE_MANIFEST.json` now records the exact eleven upstream source files, their SHA-256 digests, the approved symbols, adaptation boundaries, and excluded capability categories.

No upstream Python source is vendored by this inventory-only step, so `selected_files` remains empty. Task 6.2 will record each derived destination file and its local modifications when the approved symbols are extracted and adapted. The project intentionally uses symbol-level extraction rather than copying whole upstream modules because those modules also contain provider discovery, unrelated transports, interactive tools, generic dispatch, and other out-of-scope behavior.

The upstream license uses CRLF line endings while the preserved project copy uses LF. Their byte-level SHA-256 values are recorded separately in the manifest, and normalized text equality is part of the inventory verification.

## Minimal runtime extraction (task 6.2)

Ten symbol-level derivative modules now live in `hermes_runtime/`, plus one project-owned package assembly file. Each derivative has an MIT/commit attribution header and a byte hash in `SOURCE_MANIFEST.json`.

- `contracts.py`: frozen OpenAI-compatible tool/response/usage records with all other provider metadata removed.
- `transport.py`: the abstract data-path contract and one chat-completions normalizer, with provider profiles, discovery, fallback, prompt caching, and provider special cases removed.
- `repair.py`: bounded JSON-object repair and deterministic call IDs, with broad message rewriting and raw argument logging removed.
- `loop.py`: an injected synchronous single-model loop, with credentials, TUI, hooks, persistence, compression, MoA, delegation, fallback, and built-in tools removed.
- `tool_messages.py`: canonical tool-result construction without general dispatch, web/MCP handling, filesystem tracking, multimodal output, or timestamps.
- `retry.py`: `Retry-After` parsing without jitter, adaptive policy, or provider-specific ceilings.
- `bounded_response.py`: bounded error-body reading with explicit non-negative argument checks and no logger coupling.
- `errors.py`: only the empty-response exception.
- `usage.py`: OpenAI-compatible token normalization and aggregation without pricing or routing.
- `redaction.py`: fixed masking for runtime boundaries without plugins, terminal/file/browser paths, or configurable disablement.

`pyproject.toml` packages this top-level `hermes_runtime` alongside `nurse_rcl`; no `hermes-agent` dependency is introduced. Behavioral conformance and clean-environment execution remain gated by tasks 6.3 and 6.4.

## Embedded runtime assembly (task 6.4)

The project-owned `runtime.py` composes the attributed transport, contracts, repair, tool-result, and usage modules behind an explicit `RegisteredTool` registry. Model completion is injected as a callable; the assembly performs no client, credential, model, provider, or tool discovery. A model-requested name outside the registry is converted to a deterministic `tool_not_allowed` result and audit event without executing code.

Structured argument repair now normalizes Python `None`/`True`/`False` tokens only outside quoted JSON strings. Redaction now traverses nested mappings, lists, and tuples while applying the fixed text boundary. Both changes are reflected in their derived-file hashes and modification records.

Mock-provider tests exercise the actual two-turn tool loop, exact function schemas and tool-result messages, argument repair, multi-request usage aggregation, recursive redaction, and unregistered-tool denial. An isolated subprocess blocks all imports rooted at `agent`, `hermes_agent`, and `hermes_cli` while importing the embedded package, proving that the runtime does not depend on an installed HERMES distribution.

## Strict OpenAI tool-result wire compatibility (task 12.6)

A supervised real-model smoke with `deepseek-v4-pro-ga-260813` showed that the ARK
endpoint rejects a second-turn request when a tool-result message contains the local
`name`/`tool_name` extensions or a structured JSON value in `content`. The first turn
and tool execution succeeded, while the follow-up request returned HTTP 400.

`tool_messages.py` now emits the strict OpenAI-compatible tool-result wire shape:
`role`, string `content`, and canonical `tool_call_id` only. Non-string tool results are
serialized as stable compact JSON with sorted keys; existing string results are retained
verbatim. Tool identity remains bound by `tool_call_id` inside the assistant tool call,
so this does not broaden the runtime authority or change tool dispatch. Contract tests
cover both successful registered-tool execution and deterministic unregistered-tool
denial through the same wire representation.

## Bounded read-only investigation (2026-09-27)

`loop.py` adds opt-in `return_at_turn_limit`, returning the actual assistant/tool
trace at the turn cap without an additional model request or a fabricated model
stop. Existing callers retain the original exception at the cap. Application
persistence, decision parsing, tool budgets and credentials remain outside this
embedded runtime. The derived-file hash is updated in `SOURCE_MANIFEST.json`.
