# NURSE Artifact Evaluation Guide

This document is written for ICSE-style artifact availability and verifiability checks. It explains what the artifact contains, how reviewers can access it, what can run quickly, and what is needed for full replication.

## Access Information For The Submission Form

Use the following text after replacing the placeholder with the anonymous artifact URL or HotCRP upload reference:

```text
The artifact is available as an anonymous review archive at <ANONYMOUS_ARTIFACT_URL_OR_HOTCRP_UPLOAD>. It contains the NURSE implementation, sample-case manifests, tests, scripts, and documentation needed to inspect and independently run the basic NURSE workflow. Reviewers can download or clone the archive, install Python dependencies from requirements.txt, run contract tests, and execute smoke runs without author-private resources. Full benchmark replication requires the public AIOps25 and RCAEval benchmark releases and an OpenAI-compatible LLM API key. The artifact documentation explains dataset placement, expected output files, and privacy restrictions. The public de-anonymized repository and persistent archive DOI will be provided after the double-anonymous review period.
```

Do not submit an author-owned GitHub URL during double-anonymous review if the URL reveals identity.

## Artifact Contents

The artifact contains:

- the NURSE Evidence Core under `nurse/`;
- the interactive agent interface under `nurse_agent/`;
- basic smoke, connectivity, and reproduction scripts under `scripts/`;
- unit, contract, and smoke tests under `tests/`;
- documentation under `docs/`;
- a README with installation, dataset, LLM, quickstart, and replication instructions.

The artifact intentionally excludes:

- API keys and private endpoints;
- generated outputs and local experiment logs;
- virtual environments and caches;
- raw benchmark data and benchmark-derived answer keys;
- signed consent forms or any human-participant identifiers.

## Requirements

- Python 3.10 or newer;
- packages listed in `requirements.txt`;
- network access for dependency installation and LLM calls;
- an OpenAI-compatible API key for LLM-enabled runs.

The artifact can still be inspected and tested without an LLM key through dry-run and contract-test paths.

## Quick Availability Check

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/test_nurse_agent_console.py tests/test_nurse_agent_session_memory.py tests/test_nurse_agent_tool_contracts.py tests/test_nurse_provider_routing.py tests/test_nurse_release_scripts.py tests/test_nurse_aiops25_dataset_paths.py -q --basetemp .pytest_tmp -p no:cacheprovider
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests\test_nurse_agent_console.py tests\test_nurse_agent_session_memory.py tests\test_nurse_agent_tool_contracts.py tests\test_nurse_provider_routing.py tests\test_nurse_release_scripts.py tests\test_nurse_aiops25_dataset_paths.py -q --basetemp .pytest_tmp -p no:cacheprovider
```

Expected outcome: the selected tests pass and no API key is required.

## LLM Smoke Run

Configure an OpenAI-compatible key:

```bash
export OPENAI_API_KEY="your_api_key"
export NURSE_OPENAI_MODEL="gpt-5-nano"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:NURSE_OPENAI_MODEL = "gpt-5-nano"
```

Run a single-case smoke test:

```bash
python -m nurse.runner --dataset rcaeval-tt --case-limit 1 --llm gpt-5-nano --run-dir outputs/nurse/quickstart --resume
```

Expected output files include `summary.json`, `paper_table_row.csv`, `records.jsonl`, alarms, graphs, predictions, contexts, and LLM artifacts.

## Basic Reproduction Commands

The following commands use small samples by default and can be scaled to the retained benchmark datasets by setting `-CaseLimit 0`.

```powershell
.\scripts\run_nurse_gpt_smoke.ps1 -Dataset rcaeval-tt -CaseLimit 1
.\scripts\run_nurse_openai_full.ps1 -Model gpt-5-nano -CaseLimit 1 -Workers 1
.\scripts\run_nurse_openai_consistency_5cases.ps1 -SampleSize 5 -Workers 3
```

## Full Benchmark Replication

Full replication uses:

- `aiops25-retained-nonnode`;
- `rcaeval-tt`.

Expected layout:

```text
<lab-root>/
  NURSE/
  datasets/
    AIOps25/
      aiopschallenge2025-main/
    RCAEval-v2/
      data/
```

If datasets are stored elsewhere:

```bash
export NURSE_LAB_ROOT="/path/to/lab-root"
```

PowerShell:

```powershell
$env:NURSE_LAB_ROOT = "D:\path\to\lab-root"
```

Full direct runner commands:

```bash
python -m nurse.runner --dataset aiops25-retained-nonnode --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs/nurse/paper_aiops25 --resume
python -m nurse.runner --dataset rcaeval-tt --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs/nurse/paper_tt --resume
```

Main table rows are written to `paper_table_row.csv` under each run directory.

## Dataset And License Notes

Raw benchmark releases and benchmark-derived metadata are not relicensed or bundled by this repository. Users should follow the original AIOps25 and RCAEval release terms when downloading or using datasets.

The NURSE implementation is released under the MIT License. Benchmark data remains subject to the original benchmark terms.

## Verifiability Checklist

Reviewers can check the following without private materials:

- no required private endpoint or hard-coded API key is committed;
- README explains installation, datasets, LLM configuration, smoke runs, output files, and full replication;
- tests cover the agent console, session memory, tool contracts, LLM bridge, release scripts, and provider routing;
- inference artifacts are written under ignored output directories;
- benchmark labels are loaded only for evaluation after predictions are produced;
- generated outputs, local datasets, and temp files are excluded from release archives.

## Known Scope Limits

- Full paper-style results require the public benchmark releases and an LLM API key.
- LLM wording can vary across providers and API versions. NURSE constrains ranking and report contracts through verified evidence and schema checks, but natural-language text may still differ.
- During double-anonymous review, the artifact access URL must not reveal author identity.
