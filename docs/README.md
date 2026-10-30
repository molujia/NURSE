# NURSE documentation

Start with the English [repository README](../README.md) for installation, input
contracts, basic synthetic checks, prepared-incident execution, and output files.

NURSE combines deterministic root cause localization, bounded investigation, and
G+ report bindings. The investigation preserves the original ranking while
collecting evidence for the incident account. Operational intervention remains
under the SRE's control.

- [Detailed quick start (Chinese)](nurse_final_focused_quickstart.md): Linux setup,
  the fixed verification suite, composition plans, admission checks, and report viewing.
- [Tool catalog (Chinese)](nurse_final_focused_tools.md): evidence navigation,
  comparisons, source attribution, and report constraints.
- [Software verification scope](../tests/final_explainability_0ark.txt): the
  explicit 45-file synthetic suite for the current focused workflow.
- [Release manifest](../release_manifest.json): source-file hashes and the code inventory.

The repository contains source and synthetic fixtures. Real telemetry, identity
mappings, API credentials, and real-run artifacts are supplied separately.
