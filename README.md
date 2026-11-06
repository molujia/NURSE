# NURSE: Interpretable Evidence and Explainable Investigation for Microservice Root Cause Localization

NURSE is a research implementation for microservice root cause localization (RCL). It combines a deterministic localization core, bounded LLM-assisted investigation, and G+ evidence bindings to produce ranked candidates and evidence-backed guidance for site reliability engineers (SREs).

The core exposes the computations behind its evidence and scores. Investigation uses the resulting ranking to examine incident questions, collect further observations, and identify next checks. G+ carries selected evidence, its context, and its permitted use into the final report. Mitigation and recovery verification remain SRE decisions.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [How to use](#how-to-use)
  - [Input format](#input-format)
  - [Try the software without an API key](#try-the-software-without-an-api-key)
  - [Run a prepared incident](#run-a-prepared-incident)
  - [Inspect the outputs](#inspect-the-outputs)
- [Reproducibility](#reproducibility)
- [Path accuracy](#path-accuracy)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)
- [Paper and citation](#paper-and-citation)
- [License and contact](#license-and-contact)

## Overview

```text
Telemetry and incident scope
            |
            v
Deterministic localization core
            |
            v
Frozen candidate ranking and evidence
            |
            v
Bounded investigation: hypotheses -> checks -> evidence review
            |
            v
G+ report with evidence bindings, limitations, and next checks
            |
            v
SRE-controlled mitigation, recovery verification, and handoff
```

| Objective | Support in NURSE |
| --- | --- |
| **Interpretability (I)** | Inspectable telemetry transformations, mechanism conditions, evidence aggregation, and score decomposition. |
| **Explainability (E)** | An incident account connecting candidate support, competing observations, uncertainty, and suggested next checks. |
| **The I/E connection** | G+ binds report selections to source records, context, field values, and permitted uses. |

The current focused workflow performs a fresh deterministic core run, investigation with an explicit budget of **1–8 decisions**, and one final G+ synthesis. Investigation may stop early. Supplementary findings update the incident account while preserving the original ranking.

The tools support candidate and dependency inspection, evidence navigation, record reading, explicit measurements, timing and mechanism checks, and comparison of up to two evidence records. See the [tool catalog](docs/nurse_final_focused_tools.md) for their contracts. An evidence link records provenance; it does not automatically certify the truth of the model's wording.

## Installation

Use **Linux, Bash, and Python 3.10–3.12**. The live workflow and its test harness use POSIX process limits and file locking. Python 3.11 is used in the commands below.

Clone the repository and install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/molujia/NURSE.git
cd NURSE
uv sync --locked --python 3.11 --extra test
source .venv/bin/activate
```

Run from the complete checkout: admission checks require the source tree, registered skill files, and configuration files in addition to the installed Python package.

Before using the workflow entry point, set the numerical-library thread limits:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:third_party/hermes_runtime
```

The process must be able to apply limits of 8 GiB address space, 64 MiB stack, and 4,096 file descriptors. The software test harness also needs a writable `/tmp`.

## How to use

### Input format

NURSE operates on a **prepared incident with explicitly bound data sources**, rather than an untyped time-series file. The preparation and run plans identify:

- the dataset adapter, opaque incident identity, source files, and source hashes;
- telemetry and entity/dependency mappings for the baseline and incident windows;
- the configuration, code inventory, and numerical environment;
- a fresh output location and an investigation decision limit.

The entry point consumes a canonical JSON composition plan with schema `nurse.v2.gplus-one-case-composition/2`, represented by [`GplusFocusedCompositionPlan`](src/nurse_rcl/v2/live/ops_sre_gplus_case.py). It references a pinned `core-only` plan using profile `v2-ops-isolated/2`. The investigation profile is `v2-ops-sre/15`, configured in [`ops_sre_focused_profile.json`](configs/v2/ops_sre_focused_profile.json).

The underlying schemas are [`RepairCaseInput`](src/nurse_rcl/v2/live/repair_stage.py) for case inputs and [`OpsSinglePlan`](src/nurse_rcl/v2/live/ops_plan.py) for the core plan. The [quick-start guide](docs/nurse_final_focused_quickstart.md) describes their preparation and admission requirements. Prepare an incident-specific plan before execution.

### Try the software without an API key

Inspect the command-line interface:

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case --help
```

Then run the software checks for evidence navigation, source comparisons, and G+ binding:

```bash
python scripts/dev/run_final_explainability_0ark.py -q --tb=short \
  tests/test_v2_live_sre_focused_tools.py \
  tests/test_v2_live_sre_focused_sources.py \
  tests/test_v2_live_sre_gplus_graph.py \
  tests/test_v2_live_sre_gplus_contract.py
```

The expected outcome is a successful pytest summary with no failures. The wrapper blocks network access and real credential loading; it does not send LLM requests. These tests check software behavior and do not measure benchmark localization accuracy.

### Run a prepared incident

The current live backend uses the ARK chat-completions provider. Its endpoint and model are fixed by the provider contract and the focused profile. The live entry uses deployment-specific credential and lock paths declared as `KEY_PATH` and `LOCK_PATH` in [`ops_provider.py`](src/nurse_rcl/v2/live/ops_provider.py). For another Linux deployment, update those locations before generating the code-pinned plans. The credential file must contain one `ARK_API_KEY2` assignment and have permission mode `0600`. The lock location must be writable. Credentials are not needed for the software checks above.

For a prepared incident, set `COMPOSITION_PLAN` to the absolute path of your canonical plan and `COMPOSITION_SHA256` to its SHA-256 digest. First check its metadata and dependencies:

```bash
: "${COMPOSITION_PLAN:?Set the absolute path of the composition plan}"
: "${COMPOSITION_SHA256:?Set the SHA-256 digest of the composition plan}"
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" \
  --plan-sha256 "$COMPOSITION_SHA256" \
  --check-only
```

A successful admission response includes `status="admitted"`, `new_ark_calls=0`, and `preparation_load_import_dispatch=0`. This check requires the complete input, configuration, environment, and code metadata; it does not execute the diagnosis.

After reviewing the plan and its model-call budget, execute that exact plan:

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" \
  --plan-sha256 "$COMPOSITION_SHA256" \
  --approved-composition-sha256 "$COMPOSITION_SHA256"
```

The approval argument binds execution to the reviewed plan. This command performs real provider calls. Changing the code, numerical environment, input, or budget requires a corresponding new plan. A completed or started run cannot be restarted to obtain a new investigation budget.

Inspect an existing run with:

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" \
  --plan-sha256 "$COMPOSITION_SHA256" \
  --status
```

### Inspect the outputs

The composition writes the deterministic core under `core-results/` and the derived investigation under `results/` in the plan's run directory.

| Artifact | Purpose |
| --- | --- |
| `composition-result.json` | Overall status, per-stage status, investigation counts, and execution identifiers. |
| `core/result.json` within the core case directory | Frozen candidate ranking and its evidence decomposition. |
| `investigation/` within the investigation case directory | Investigation state, decisions, actions, and acquired observations. |
| `gplus-report/report.md` | The human-readable incident account. |
| `gplus-report/evidence.md` | The report's evidence appendix. |
| `gplus-report/predicted-paths.json` | The complete model-predicted path collection, source pins, and evidence audit. |
| `gplus-report/predicted-paths.md` | A readable view of the same path hypotheses. |
| `gplus-report/context.json` and `references.json` | Retained context and available evidence references. |
| `gplus-report/result.json` | Report completion state, provenance checks, and `semantic_certified` status. |

The report currently links to `report-evidence.md`, while the saved appendix is named `evidence.md`. The [quick-start guide](docs/nurse_final_focused_quickstart.md#保存报告的展示兼容) shows how to create an independent viewing copy with a matching alias, without changing the sealed run directory.

## Reproducibility

The software verification suite for this release is the explicit [45-file test list](tests/final_explainability_0ark.txt). It covers focused tools, source comparison, investigation ownership and handoff, G+ graphs and binding, report storage, and usage accounting.

Run the complete suite from the repository root in Bash:

```bash
set -euo pipefail
mapfile -t selected_tests < tests/final_explainability_0ark.txt
test "${#selected_tests[@]}" -eq 45
python scripts/dev/run_final_explainability_0ark.py \
  -q --tb=short "${selected_tests[@]}"
```

Use this list to reproduce the release verification scope. Additional test modules supply fixtures and dependencies and are not all part of that scope. An interrupted run is incomplete.

The [Software verification workflow](.github/workflows/software-verification.yml) runs the same suite on Linux with Python 3.11. It also checks the release file hashes and the documented command-line entry point. No provider credentials are needed.

Benchmark reproduction additionally requires the original datasets, exact case selection, prepared inputs, configuration and code hashes, and evaluation records. Ground-truth labels remain separate from inference inputs. A software test pass or a completed single-case workflow does not establish aggregate ranking accuracy or certify the semantics of an LLM-generated report.

## Path accuracy

New reports explicitly propose complete service paths from candidate roots to symptoms. The path collection supports multiple roots, paths longer than two hops, and marked inferred edges, while preserving the frozen core ranking. Paths remain available for evaluation even when their evidence annotations or factual report prose are rejected. Older saved reports remain readable without adding a fabricated prediction collection.

The offline calculator implements mABC's Root Cause Path Accuracy, `PA = (Pc - 0.2 * Pi) / Pt`, using complete service-sequence matching and retaining negative values. It requires an independently supplied reference set. A valid explicit abstention differs from a missing or damaged output.

Evaluate a prediction export against an independently prepared reference file:

```bash
python -m nurse_rcl.evaluation.path_accuracy \
  --predictions PREDICTIONS_JSON \
  --reference REFERENCE_JSON
```

Replace the named placeholders with file paths. See the [path accuracy guide](docs/path_accuracy.md) for the output and reference contracts, failure handling, and aggregation.

## Datasets

The source includes adapters selected by the identifiers `rcabench` and `aiops25`. Obtain the corresponding original data releases separately and prepare them using the declared adapter and input contracts. Raw telemetry, private identity mappings, answer keys, and real-run artifacts are not bundled with this repository. Dataset use remains subject to the original release terms.

## Repository layout

```text
NURSE/
  configs/                    Registered execution and investigation profiles
  docs/                       Workflow and tool documentation
  scripts/dev/                Execution and software verification wrappers
  src/nurse_rcl/              Localization, evidence processing, and reporting
    v2/live/                  Current focused investigation and G+ entry points
    v2/live/ops_skills/        Registered investigation skill contracts
  tests/                      Software tests and fixtures
  third_party/hermes_runtime/  Bundled HERMES-derived runtime
  pyproject.toml              Package metadata and dependencies
  uv.lock                     Locked dependency versions
```

Earlier implementations remain accessible through Git history. The current workflow uses the `nurse_rcl` package and the composition entry point documented above.

## Paper and citation

NURSE is associated with ongoing work on interpretable evidence and explainable investigation for microservice RCL. Publication metadata will be added when available. When referring to this implementation, cite the [repository](https://github.com/molujia/NURSE) and record the commit hash used in your experiment.

## License and contact

The NURSE implementation is released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the bundled runtime and its licensing information.

For usage questions and bug reports, open an [issue](https://github.com/molujia/NURSE/issues) with the commit hash, Python version, command, and minimal reproduction steps.
