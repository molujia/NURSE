# NURSE

**NURSE** is an LLM-enabled neural-symbolic agent for microservice root-cause localization (RCL). It turns incident telemetry into verified evidence, builds a typed evidence graph, ranks candidate root causes with an auditable Evidence Core, and uses an LLM-facing User Manager to produce grounded evidence paths and recovery advice.

This public repository focuses on the basic NURSE workflow: install the code, place the benchmark datasets, configure an OpenAI-compatible LLM, run NURSE, and inspect the generated artifacts.

## Highlights

- **Agentic workflow**: an interactive CLI accepts natural-language SRE commands and records session memory.
- **Neural-symbolic guardrails**: LLM outputs are admitted only through evidence and schema checks before they can support ranking.
- **Auditable Evidence Core**: ranked claims are tied to verified evidence, graph relations, score breakdowns, and provenance.
- **Public benchmark workflow**: release commands target AIOps25 retained non-node cases and RCAEval-TT.
- **OpenAI-first quick start**: scripts support a full two-dataset run and a small GPT model-switch consistency check.

## Repository Layout

```text
NURSE/
  docs/                   # Artifact and repository notes
  nurse/                  # Evidence Core: alarms, evidence admission, graph builder, ranker, runner
  nurse_agent/            # Agent console, session memory, tool registry, report writer
  nurse_agent/tools/      # Tools called by the agent workflow
  scripts/                # Basic smoke, connectivity, and reproduction launchers
  tests/                  # Unit, contract, and smoke tests
  outputs/                # Generated locally; ignored by git
```

See [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for a more detailed map.

## Installation

NURSE is tested with Python 3.10+.

### Windows PowerShell

```powershell
git clone <your-nurse-repository-url> NURSE
cd NURSE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone <your-nurse-repository-url> NURSE
cd NURSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure the LLM

Public NURSE runs require an LLM. The default examples use an OpenAI-compatible GPT model.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:NURSE_OPENAI_MODEL = "gpt-5-nano"
```

Linux / macOS:

```bash
export OPENAI_API_KEY="your_api_key"
export NURSE_OPENAI_MODEL="gpt-5-nano"
```

Optional endpoint override:

```bash
export NURSE_OPENAI_BASE_URL="https://api.openai.com/v1"
```

Do not commit API keys, private endpoints, generated logs, or local absolute paths.

## Datasets

This artifact does not bundle raw benchmark data or benchmark-derived answer keys. Full replication requires the original public benchmark releases. Please follow the original dataset release terms when using these benchmarks.

| Dataset ID | Benchmark | Expected use |
| --- | --- | --- |
| `aiops25-retained-nonnode` | AIOps25 | Retained non-node cases used in the public NURSE workflow. |
| `rcaeval-tt` | RCAEval TrainTicket | TrainTicket cases used in the public NURSE workflow. |

Expected local layout:

```text
<lab-root>/
  NURSE/
  datasets/
    AIOps25/
      aiopschallenge2025-main/
    RCAEval-v2/
      data/
```

If your datasets live elsewhere, set `NURSE_LAB_ROOT` to the directory that contains `datasets/`.

PowerShell:

```powershell
$env:NURSE_LAB_ROOT = "D:\research\microservice-rca"
```

Linux / macOS:

```bash
export NURSE_LAB_ROOT="/path/to/microservice-rca"
```

## Quick Availability Check

This path does not require an API key.

```powershell
python -m pytest `
  tests\test_nurse_agent_console.py `
  tests\test_nurse_agent_session_memory.py `
  tests\test_nurse_agent_tool_contracts.py `
  tests\test_nurse_provider_routing.py `
  tests\test_nurse_release_scripts.py `
  tests\test_nurse_aiops25_dataset_paths.py `
  -q --basetemp .pytest_tmp -p no:cacheprovider
```

Expected result: the selected tests pass.

## Run NURSE

### Single-Case Smoke Run

```powershell
python -m nurse.runner `
  --dataset rcaeval-tt `
  --case-limit 1 `
  --llm gpt-5-nano `
  --run-dir outputs\nurse\quickstart `
  --resume
```

Equivalent script:

```powershell
.\scripts\run_nurse_gpt_smoke.ps1 -Dataset rcaeval-tt -CaseLimit 1
```

### Two-Dataset Full Run

For a fast smoke check, keep `-CaseLimit 1`. For full retained-dataset replication, set `-CaseLimit 0`.

```powershell
$env:OPENAI_API_KEY = "your_api_key"
.\scripts\run_nurse_openai_full.ps1 `
  -Model gpt-5-nano `
  -CaseLimit 1 `
  -Workers 1
```

Full retained-dataset run:

```powershell
.\scripts\run_nurse_openai_full.ps1 `
  -Model gpt-5-nano `
  -CaseLimit 0 `
  -Workers 2
```

The report is written under:

```text
outputs/experiments/nurse_openai_full/<timestamp>/openai_full_results.csv
outputs/experiments/nurse_openai_full/<timestamp>/openai_full_results.xlsx
```

### Five-Case Model-Switch Consistency Check

This command samples five cases per dataset and runs the same cases with `gpt-5-nano`, `gpt-5-mini`, and `gpt-5`. It exits with a non-zero status if any sampled case has a Top-5 mismatch.

```powershell
.\scripts\run_nurse_openai_consistency_5cases.ps1 -SampleSize 5 -Workers 3
```

Inspect:

```text
outputs/experiments/nurse_openai_consistency_5cases/<timestamp>/sample_metrics.csv
outputs/experiments/nurse_openai_consistency_5cases/<timestamp>/top5_diff_report.csv
outputs/experiments/nurse_openai_consistency_5cases/<timestamp>/top5_by_model.json
```

Expected result: `top5_diff_report.csv` has no mismatch rows, and `sample_metrics.csv` reports `all_models_equal=True`.

### Connectivity Checks

OpenAI-only:

```powershell
.\scripts\test_nurse_llm_connectivity.ps1 `
  -Models gpt-5-nano `
  -OpenAIModel gpt-5-nano
```

Provider routing can also be checked when optional provider keys are present:

```powershell
.\scripts\test_nurse_all_llm_providers.ps1
```

## Agent Console

Start an interactive NURSE session:

```powershell
python -m nurse_agent.console --output-root outputs\agent_runs --session-id demo
```

Example commands:

```text
NURSE> help
NURSE> Run single case 1.
NURSE> artifacts
NURSE> Run single case 2.
NURSE> memory
NURSE> exit
```

The two bundled single-case manifests are:

| Command | Dataset |
| --- | --- |
| `Run single case 1.` | AIOps25 retained non-node demo case |
| `Run single case 2.` | RCAEval-TT demo case |

Each single-case report contains Top-K ranking, evidence paths, score breakdown, incident-prior status, recovery advice, and an agent trace.

## Output Files

A NURSE run writes artifacts under `outputs/` unless you choose another output root.

| File or directory | Meaning |
| --- | --- |
| `config.json` / `experiment_config.json` | Dataset, model, seed, evidence backend, and ranker configuration. |
| `summary.json` | Run-level status, timing, token usage, and metrics. |
| `paper_table_row.csv` | One-row Acc@1/3/5 summary for paper tables. |
| `case_metrics.csv` | Per-case evaluation records when generated by reporting utilities. |
| `records.jsonl` | Per-case inference status, timing, LLM usage, and evidence counts. |
| `alarms/` | Shared alarm artifacts and incident windows. |
| `graphs/` | Evidence graphs consumed by the ranker. |
| `predictions/` | Per-case ranked candidates and score breakdowns. |
| `contexts/` | Verified RCL Context passed to the LLM-facing path. |
| `llm_artifacts/` | Grounded LLM semantic facts, explanations, and usage metadata. |
| `agent_trace.json` | Agent planning and tool-call trace for console runs. |

Inference artifacts must not contain benchmark labels or answer keys. Labels are loaded only by evaluation utilities after predictions are produced.

## Development and Tests

Run focused tests for the public path:

```powershell
python -m pytest `
  tests\test_nurse_agent_console.py `
  tests\test_nurse_agent_session_memory.py `
  tests\test_nurse_agent_orchestrator.py `
  tests\test_nurse_agent_core_wrappers.py `
  tests\test_nurse_agent_tool_contracts.py `
  tests\test_nurse_context_packager.py `
  tests\test_nurse_llm_bridge.py `
  tests\test_nurse_provider_routing.py `
  tests\test_nurse_semantic_rank_facts.py `
  tests\test_nurse_release_scripts.py `
  tests\test_nurse_aiops25_dataset_paths.py `
  --basetemp .pytest_tmp `
  -p no:cacheprovider `
  -q
```

Run a dry-run agent smoke command without consuming LLM tokens:

```powershell
python -m nurse_agent.console `
  --script "Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT." `
  --output-root outputs\agent_release_smoke `
  --dry-run `
  --allow-no-llm
```

`--allow-no-llm` is an internal test flag. Public NURSE usage should configure `OPENAI_API_KEY` and run the LLM-enabled path.

## Release Hygiene

Before publishing or creating a supplementary artifact:

1. Exclude `outputs/`, virtual environments, caches, local datasets, and private logs.
2. Do not publish API keys, private endpoints, local absolute paths, or human-participant identifiers.
3. Do not include benchmark answer keys in inference prompts or artifacts.
4. Keep generated evidence tools advisory until they pass schema, provenance, forbidden-field, and smoke checks.
5. Review [`docs/release/RELEASE_AUDIT.md`](docs/release/RELEASE_AUDIT.md).
6. For double-anonymous submission, use an anonymous repository/archive or HotCRP upload. Do not submit a URL that reveals author identity.
7. For the public post-review release, archive an immutable snapshot and record the DOI or Software Heritage identifier in the paper.

## License

The NURSE implementation is released under the MIT License. Benchmark data remains subject to the original AIOps25 and RCAEval release terms.

## Citation

Citation metadata will be added after publication.
