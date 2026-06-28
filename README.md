# NURSE

**NURSE** is an LLM-enabled neural-symbolic agent for microservice root-cause localization (RCL). It turns incident telemetry into verified evidence, builds a typed evidence graph, ranks candidate root causes with an auditable Evidence Core, and then uses an LLM-facing User Manager to produce evidence paths and recovery advice under semantic guardrails.

NURSE is designed for SRE-facing RCA research: accuracy matters, but the system must also explain why a candidate is root-like rather than merely a propagated symptom.

## Highlights

- **Agentic workflow**: an interactive CLI accepts natural-language SRE commands and records session memory.
- **Neural-symbolic guardrails**: the LLM is bounded by Evidence Admission, a typed Evidence Graph, ESCPI scoring, and grounded output contracts.
- **Auditable Evidence Core**: every ranked claim is tied to verified evidence, graph relations, score breakdowns, and provenance.
- **Paper-style benchmarks**: the supported release datasets are AIOps25 retained non-node cases and RCAEval-TT.
- **OpenAI-first quick start**: release examples use `gpt-5-nano` through the OpenAI-compatible API path.

## Artifact Evaluation At A Glance

This repository is intended to be usable as an ICSE-style research artifact. A reviewer should be able to inspect the implementation, install the dependencies, run lightweight contract tests, execute a smoke run, and reproduce paper-style runs when the public benchmark datasets and an LLM API key are available.

For double-anonymous review, do not submit an author-identifying GitHub URL. Provide either an anonymous public mirror/archive of this repository or upload the artifact archive to HotCRP. After review, the public release can point to the de-anonymized GitHub repository and a persistent archive such as Zenodo or Software Heritage.

Recommended reviewer path:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/test_nurse_agent_console.py tests/test_nurse_agent_session_memory.py tests/test_nurse_agent_tool_contracts.py -q --basetemp .pytest_tmp -p no:cacheprovider
```

See [`docs/ARTIFACT_EVALUATION.md`](docs/ARTIFACT_EVALUATION.md) for access instructions, expected resources, replication commands, and double-anonymous review notes.

## Repository Layout

```text
NURSE/
  data/                   # Lightweight retained-case metadata and topology summaries
  docs/                   # Artifact evaluation, repository layout, and release-audit notes
  nurse/                  # Evidence Core: alarms, evidence admission, graph builder, ranker, runner
  nurse_agent/            # Agent console, session memory, tool registry, report writer
  nurse_agent/tools/      # Paper-facing tools called by the agent
  scripts/                # Smoke and reproduction launchers
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

Public NURSE runs require an LLM. The default release examples use an OpenAI-compatible GPT model.

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

Optional OpenAI-compatible endpoint override:

```bash
export NURSE_OPENAI_BASE_URL="https://api.openai.com/v1"
```

Do not commit API keys, private endpoints, generated logs, or local absolute paths.

## Datasets

This artifact includes preprocessed retained-case metadata from two public benchmarks. The bundled JSON files are used only for experimental evaluation, smoke tests, retained-case manifests, and topology/candidate-scope checks. Full paper-style replication requires the original public benchmark releases. Please follow the original dataset release terms when using these benchmarks.

| ID | Dataset | Source |
| --- | --- | --- |
| `ds25` | AIOps-25 retained cases | [aiopschallenge2025 benchmark repository](https://www.aiops.cn/gitlab/aiops-live-benchmark/aiopschallenge2025) |
| `tt` | TrainTicket retained cases | [RCAEval](https://github.com/phamquiluan/RCAEval) |

Bundled metadata files:

| File | Contents |
| --- | --- |
| `data/ds25_faults.json` | 318 retained AIOps-25 non-node fault records. |
| `data/ds25_topo_s2s.json` | AIOps-25 service-to-service topology summary. |
| `data/tt_faults.json` | Retained TrainTicket fault metadata used by the public artifact. |
| `data/tt_topo_s2s.json` | TrainTicket service-to-service topology summary. |

### Local Benchmark Layout

For reproducing full paper-style runs from the original benchmark releases, NURSE expects a workspace layout like this:

```text
<lab-root>/
  NURSE/
  datasets/
    AIOps25/
      aiopschallenge2025-main/
    RCAEval-v2/
      data/
```

If your datasets live elsewhere, set `NURSE_LAB_ROOT` to the directory that contains `datasets/`:

PowerShell:

```powershell
$env:NURSE_LAB_ROOT = "D:\research\microservice-rca"
```

Linux / macOS:

```bash
export NURSE_LAB_ROOT="/path/to/microservice-rca"
```

Supported paper dataset IDs:

| Dataset ID | Description |
| --- | --- |
| `aiops25-retained-nonnode` | AIOps25 retained non-node subset used in the final paper experiments. |
| `rcaeval-tt` | RCAEval TrainTicket cases. |

Legacy and diagnostic dataset IDs are still available in code, but the public paper workflow should use the two IDs above.

## Quick Start: Agent Console

Start an interactive NURSE session:

```powershell
python -m nurse_agent.console --output-root outputs\agent_runs --session-id demo
```

Then type English SRE-style commands:

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

## Run Full NURSE Benchmarks

The same console can launch full benchmark runs. Keep the same seed and dataset split when reproducing paper-style results.

```text
NURSE> Run the full AIOps25 benchmark with the default evidence backend.
NURSE> Run the same configuration on RCAEval-TT.
```

For a one-shot scripted run:

```powershell
python -m nurse_agent.console `
  --script "Run the full RCAEval-TT benchmark with the default evidence backend." `
  --output-root outputs\agent_runs
```

NURSE also supports alternative metric-evidence backends:

```text
NURSE> Run the full AIOps25 benchmark with AD reuse (alarm_fixed_window).
NURSE> Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT.
```

Use `status`, `artifacts`, and `memory` inside the console to inspect progress and output paths.

## Direct Runner

Researchers who prefer an explicit CLI can call the Evidence Core runner directly:

```powershell
python -m nurse.runner `
  --dataset rcaeval-tt `
  --case-limit 1 `
  --llm gpt-5-nano `
  --run-dir outputs\nurse\quickstart `
  --resume
```

Run a GPT smoke test through the release script:

```powershell
.\scripts\run_nurse_gpt_smoke.ps1 -Dataset rcaeval-tt -CaseLimit 1
```

## Output Files

A NURSE run writes artifacts under `outputs/` unless you choose another output root.

| File or directory | Meaning |
| --- | --- |
| `config.json` / `experiment_config.json` | Dataset, model, seed, evidence backend, and ranker configuration. |
| `summary.json` | Run-level status, timing, token usage, and metrics. |
| `paper_table_row.csv` | One-row Acc@1/3/5 summary for paper tables. |
| `case_metrics.csv` | Per-case evaluation records when generated by statistics utilities. |
| `records.jsonl` | Per-case inference status, timing, LLM usage, and evidence counts. |
| `alarms/` | Shared alarm artifacts and incident windows. |
| `graphs/` | Evidence graphs consumed by the ranker. |
| `predictions/` | Per-case ranked candidates and score breakdowns. |
| `contexts/` | Verified RCL Context passed to the LLM User Manager. |
| `llm_artifacts/` | Grounded LLM explanations, recovery advice, and usage metadata. |
| `agent_trace.json` | Agent planning and tool-call trace for console runs. |

Inference artifacts must not contain benchmark labels or answer keys. Labels are loaded only by evaluation utilities after predictions are produced.

## Reproduce Paper-Style Results

For the release paper setting, use:

- datasets: `aiops25-retained-nonnode`, `rcaeval-tt`
- seed: `20260621`
- LLM: `gpt-5-nano` or another OpenAI-compatible model configured through environment variables
- metric backend: `evidence_robust` unless running a sensitivity experiment

Example direct runner commands:

```powershell
python -m nurse.runner --dataset aiops25-retained-nonnode --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs\nurse\paper_aiops25 --resume
python -m nurse.runner --dataset rcaeval-tt --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs\nurse\paper_tt --resume
```

After each run, inspect:

```text
outputs/nurse/paper_aiops25/paper_table_row.csv
outputs/nurse/paper_tt/paper_table_row.csv
```

Some scripts under `scripts/` reproduce internal experiments or historical baselines. They are kept for auditability, but the recommended public quick start is the agent console or `python -m nurse.runner`.

## Development and Tests

Run focused tests for the public agent path:

```powershell
python -m pytest `
  tests\test_nurse_agent_console.py `
  tests\test_nurse_agent_session_memory.py `
  tests\test_nurse_agent_orchestrator.py `
  tests\test_nurse_agent_core_wrappers.py `
  tests\test_nurse_agent_tool_contracts.py `
  tests\test_nurse_context_packager.py `
  tests\test_nurse_llm_bridge.py `
  --basetemp .pytest_tmp `
  -p no:cacheprovider `
  -q
```

Run a dry-run smoke command without consuming LLM tokens:

```powershell
python -m nurse_agent.console `
  --script "Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT." `
  --output-root outputs\agent_release_smoke `
  --dry-run `
  --allow-no-llm
```

`--allow-no-llm` is an internal test flag. Public NURSE usage should configure `OPENAI_API_KEY` and run the LLM-enabled path.

## Legacy NexusRCL Compatibility

Older NexusRCL/CBSN entrypoints are preserved under `legacy/nexusrcl/` for reproducibility. They are not the public NURSE agent interface.

```powershell
python legacy\nexusrcl\main.py --help
python main.py --legacy-nexusrcl -- --help
```

## Release Hygiene

Before publishing or creating a supplementary artifact:

1. Exclude `outputs/`, virtual environments, caches, local datasets, and private experiment logs.
2. Do not publish API keys, private endpoints, local absolute paths, or human-participant identifiers.
3. Do not include benchmark answer keys in inference prompts or artifacts.
4. Keep generated evidence tools advisory until they pass schema, provenance, forbidden-field, and smoke checks.
5. Review [`docs/release/RELEASE_AUDIT.md`](docs/release/RELEASE_AUDIT.md).
6. For double-anonymous submission, use an anonymous repository/archive or HotCRP upload. Do not submit a URL that reveals author identity.
7. For the public post-review release, archive an immutable snapshot and record the DOI or Software Heritage identifier in the paper.

## License

The NURSE implementation is released under the MIT License. The bundled benchmark-derived metadata remains subject to the original AIOps-25 and RCAEval release terms.

## Citation

Citation metadata will be added after publication.
