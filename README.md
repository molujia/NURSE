# NURSE

**NURSE** is an LLM-enabled neural-symbolic agent for microservice root-cause localization (RCL). It turns incident telemetry into verified evidence, builds a typed evidence graph, ranks candidate root causes with an auditable Evidence Core, and uses an LLM-facing User Manager to produce grounded evidence paths and recovery advice.

NURSE is built for SRE-facing RCA research. Accuracy matters, but the system must also explain why a candidate is root-like rather than only a propagated symptom.

## Highlights

- **Evidence-first ranking**: candidates are scored through evidence support, propagation structure, history priors, and optional SRE rules.
- **LLM under guardrails**: LLM calls can contribute admitted semantic facts before ranking, but the final rank is produced by NURSE rather than rewritten by the LLM after the fact.
- **Auditable artifacts**: every run writes alarms, graphs, predictions, contexts, score breakdowns, LLM artifacts, and table-ready metrics.
- **Paper-style benchmarks**: the public release workflow targets AIOps25 retained non-node cases and RCAEval TrainTicket cases.
- **OpenAI-first release path**: quickstarts and smoke checks default to `gpt-5-nano`.

## Repository Layout

```text
NURSE/
  docs/                   Artifact evaluation, layout, release audit, smoke logs
  nurse/                  Evidence Core: alarms, evidence admission, graph builder, ranker, runner
  nurse_agent/            Agent console, session memory, tool registry, report writer
  nurse_agent/tools/      Tools called by the agent workflow
  scripts/                Smoke, connectivity, matrix, ablation, and aggregation launchers
  tests/                  Unit, contract, and release-entrypoint tests
  outputs/                Generated locally; ignored by git
```

See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) for a more detailed map.

## Installation

NURSE is tested with Python 3.10+.

Windows PowerShell:

```powershell
git clone <your-nurse-repository-url> NURSE
cd NURSE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux / macOS:

```bash
git clone <your-nurse-repository-url> NURSE
cd NURSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure LLM Access

The public quickstarts use an OpenAI-compatible GPT model.

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

Do not commit API keys, private endpoints, generated logs, local absolute paths, or benchmark answer keys.

## Datasets

This repository does not bundle raw benchmark data or benchmark-derived answer keys. Full replication requires the original public benchmark releases and their license terms.

| Dataset ID | Benchmark | Expected Use |
| --- | --- | --- |
| `aiops25-retained-nonnode` | AIOps25 | Retained non-node cases used by the paper workflow. |
| `rcaeval-tt` | RCAEval TrainTicket | TrainTicket cases used by the paper workflow. |

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

If datasets live elsewhere, set `NURSE_LAB_ROOT` to the directory that contains `datasets/`.

```powershell
$env:NURSE_LAB_ROOT = "D:\research\microservice-rca"
```

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
  tests\test_nurse_sota_reporting.py `
  tests\test_nurse_provider_routing.py `
  -q --basetemp .pytest_tmp -p no:cacheprovider
```

Expected result: the selected tests pass.

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

Dry-run smoke without LLM tokens:

```powershell
python -m nurse_agent.console `
  --script "Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT." `
  --output-root outputs\agent_release_smoke `
  --dry-run `
  --allow-no-llm
```

`--allow-no-llm` is for tests and dry-run artifact checks. Public usage should configure an LLM key.

## Direct Runner

Run one LLM-enabled case:

```powershell
python -m nurse.runner `
  --dataset rcaeval-tt `
  --case-limit 1 `
  --llm gpt-5-nano `
  --run-dir outputs\nurse\quickstart `
  --resume
```

Equivalent release smoke script:

```powershell
.\scripts\run_nurse_gpt_smoke.ps1 -Dataset rcaeval-tt -CaseLimit 1
```

## Release Experiment Entrypoints

All commands below default to `gpt-5-nano` and are safe to run as smoke checks with `-CaseLimit 1` or `-SampleSize 1`. Set `-CaseLimit 0` only for full replication.

Two-dataset full matrix smoke:

```powershell
.\scripts\run_nurse_openai_full.ps1 `
  -Model gpt-5-nano `
  -CaseLimit 1 `
  -Workers 1
```

Full paper-style run:

```powershell
.\scripts\run_nurse_openai_full.ps1 `
  -Model gpt-5-nano `
  -CaseLimit 0 `
  -Workers 2
```

Single-model consistency smoke:

```powershell
.\scripts\run_nurse_openai_consistency_5cases.ps1 `
  -Models gpt-5-nano `
  -SampleSize 1 `
  -Workers 1
```

Internal SOTA full matrix smoke:

```powershell
.\scripts\run_nurse_sota_full_parallel.ps1 `
  -Models gpt-5-nano `
  -CaseLimit 1 `
  -Workers 1
```

E/P/H ablation smoke:

```powershell
.\scripts\run_nurse_sota_ablation_parallel.ps1 `
  -Models gpt-5-nano `
  -CaseLimit 1 `
  -Workers 1
```

Model consistency sample smoke:

```powershell
.\scripts\run_nurse_model_consistency_sample.ps1 `
  -Models gpt-5-nano `
  -SampleSize 1 `
  -Workers 1
```

Aggregate an existing run root:

```powershell
python scripts\aggregate_nurse_sota_results.py aggregate `
  --run-root outputs\experiments\nurse_sota_full_parallel\<timestamp> `
  --output outputs\experiments\aggregated\sota_full.csv
```

## Connectivity Checks

OpenAI-only:

```powershell
.\scripts\test_nurse_llm_connectivity.ps1 `
  -Models gpt-5-nano `
  -OpenAIModel gpt-5-nano
```

Provider routing for additional OpenAI-compatible models can be checked by passing `-Models ...` when the required provider keys are present.

## Output Files

NURSE writes generated artifacts under `outputs/` unless you choose another output root.

| File or directory | Meaning |
| --- | --- |
| `config.json` / `experiment_config.json` | Dataset, model, seed, evidence backend, and ranker configuration. |
| `summary.json` | Run-level status, timing, token usage, and metrics. |
| `paper_table_row.csv` | One-row Acc@1/3/5 summary for paper tables. |
| `case_metrics.csv` | Per-case evaluation records. |
| `records.jsonl` | Per-case status, timing, LLM usage, and evidence counts. |
| `alarms/` | Alarm artifacts and incident windows. |
| `graphs/` | Evidence graphs consumed by the ranker. |
| `predictions/` | Ranked candidates and score breakdowns. |
| `contexts/` | Verified RCL context passed to the LLM-facing path. |
| `llm_artifacts/` | Grounded LLM semantic facts, explanations, and usage metadata. |
| `agent_trace.json` | Agent planning and tool-call trace for console runs. |

Inference artifacts must not contain benchmark labels or answer keys. Labels are loaded only by evaluation utilities after predictions are produced.

## Development Tests

Focused release check:

```powershell
python -m pytest `
  tests\test_nurse_agent_console.py `
  tests\test_nurse_agent_session_memory.py `
  tests\test_nurse_agent_tool_contracts.py `
  tests\test_nurse_sota_reporting.py `
  tests\test_nurse_release_scripts.py `
  tests\test_nurse_aiops25_dataset_paths.py `
  -q --basetemp .pytest_tmp -p no:cacheprovider
```

Broader focused suite:

```powershell
python -m pytest `
  tests\test_nurse_llm_bridge.py `
  tests\test_nurse_llm_payload.py `
  tests\test_nurse_semantic_rank_facts.py `
  tests\test_nurse_ranker_ephr.py `
  tests\test_nurse_sota_reporting.py `
  tests\test_nurse_release_scripts.py `
  -q --basetemp .pytest_tmp -p no:cacheprovider
```

## Release Notes

Before publishing or creating a supplementary artifact:

1. Exclude `outputs/`, `tmp/`, virtual environments, caches, local datasets, private experiment logs, and API keys.
2. Keep benchmark data and answer keys outside the repository.
3. Use [docs/release/RELEASE_AUDIT.md](docs/release/RELEASE_AUDIT.md) as the final checklist.
4. See [docs/release/SMOKE_RESULTS.md](docs/release/SMOKE_RESULTS.md) for the latest local smoke validation log.

## License

The NURSE implementation is released under the MIT License. Benchmark data remains subject to the original AIOps25 and RCAEval release terms.

## Citation

Citation metadata will be added after publication.
