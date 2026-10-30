# Minimal HERMES-derived runtime

This directory contains the strictly selected HERMES runtime surface approved by the OpenSpec design. The importable package is `hermes_runtime/`; it is bundled into the NURSE wheel and does not require `hermes-agent`.

The runtime contains only normalized contracts, one OpenAI-compatible transport boundary, conservative tool-argument repair, a synchronous injected agent loop, canonical tool-result messages, Retry-After parsing, bounded error reads, usage accounting, redaction, and a project-owned allowlisted assembly. It does not register or contain TUI, shell, browser, file, MCP, external-search, delegation, memory, provider discovery, or unrelated provider support.

See `SOURCE_MANIFEST.json` for exact source and derived-file hashes, `INVENTORY.md` for the selection rationale, and `CHANGES.md` for local modifications. Mock-provider tests and an isolated import test establish the supported boundary; this remains deliberately narrower than general HERMES compatibility.
