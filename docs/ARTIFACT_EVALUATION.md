# NURSE Artifact Evaluation Guide

This document is written for ICSE-style artifact availability and verifiability checks. It explains what the artifact contains, how reviewers can access it, what can be run quickly, and what is needed for full replication.

## Access Information For The Submission Form

Use the following information in the "Information about Artifact Availability" field after replacing the placeholder URL with the actual anonymous artifact URL or HotCRP upload reference:

```text
The artifact is available as an anonymous review archive at <ANONYMOUS_ARTIFACT_URL_OR_HOTCRP_UPLOAD>. It contains the NURSE implementation, retained-case metadata, sample-case manifests, tests, scripts, and documentation needed to inspect and independently run the claimed workflow. Reviewers can download or clone the archive, install Python dependencies from requirements.txt, run the contract tests, and execute smoke runs without access to author-private resources. Full benchmark replication requires the public AIOps-25 and RCAEval benchmark releases and an OpenAI-compatible LLM API key. The artifact documentation explains dataset placement, expected output files, and privacy restrictions. The public de-anonymized repository and persistent archive DOI will be provided after the double-anonymous review period.
```

Do not submit an author-owned GitHub URL during double-anonymous review if the URL reveals identity. Use one of these options instead:

- an anonymous public mirror that contains the same files;
- a HotCRP artifact upload;
- a double-anonymous archival service or archive package approved by the track instructions.

After review, the public repository can be de-anonymized and archived on a persistent service such as Zenodo or Software Heritage.

## Artifact Contents

The artifact contains:

- the NURSE Evidence Core under `nurse/`;
- the interactive agent interface under `nurse_agent/`;
- retained-case metadata and topology summaries under `data/`;
- smoke and connectivity scripts under `scripts/`;
- unit, contract, and smoke tests under `tests/`;
- documentation under `docs/`;
- a README with installation, dataset, and reproduction instructions.

The artifact intentionally excludes:

- API keys and private endpoints;
- generated outputs and local experiment logs;
- virtual environments and caches;
- signed consent forms or any human-participant identifiers;
- benchmark answer keys in prompts or inference artifacts.

## Requirements

Minimum environment:

- Python 3.10 or newer;
- packages listed in `requirements.txt`;
- network access only when installing dependencies or calling an LLM;
- an OpenAI-compatible API key for LLM-enabled runs.

The artifact can still be inspected and tested without an LLM key through the dry-run and contract-test paths below.

## Quick Availability Check

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/test_nurse_agent_console.py tests/test_nurse_agent_session_memory.py tests/test_nurse_agent_tool_contracts.py -q --basetemp .pytest_tmp -p no:cacheprovider
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests\test_nurse_agent_console.py tests\test_nurse_agent_session_memory.py tests\test_nurse_agent_tool_contracts.py -q --basetemp .pytest_tmp -p no:cacheprovider
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

Expected output files include:

- `outputs/nurse/quickstart/summary.json`;
- `outputs/nurse/quickstart/paper_table_row.csv`;
- `outputs/nurse/quickstart/records.jsonl`;
- per-case alarm, graph, prediction, context, and LLM artifact files.

## Full Benchmark Replication

Full replication uses two public benchmark settings:

- `aiops25-retained-nonnode`;
- `rcaeval-tt`.

Place the original benchmark releases under a lab root:

```text
<lab-root>/
  NURSE/
  datasets/
    25挑战赛/
      aiopschallenge2025-main/
    RCAEval-v2/
      data/
```

If the datasets are stored elsewhere:

```bash
export NURSE_LAB_ROOT="/path/to/lab-root"
```

PowerShell:

```powershell
$env:NURSE_LAB_ROOT = "D:\path\to\lab-root"
```

Then run:

```bash
python -m nurse.runner --dataset aiops25-retained-nonnode --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs/nurse/paper_aiops25 --resume
python -m nurse.runner --dataset rcaeval-tt --case-limit 0 --llm gpt-5-nano --shuffle --seed 20260621 --run-dir outputs/nurse/paper_tt --resume
```

The main table rows are written to:

- `outputs/nurse/paper_aiops25/paper_table_row.csv`;
- `outputs/nurse/paper_tt/paper_table_row.csv`.

## Dataset And License Notes

The repository includes lightweight retained-case metadata:

- `data/ds25_faults.json`: 318 retained AIOps-25 non-node fault records;
- `data/ds25_topo_s2s.json`: AIOps-25 service topology summary;
- `data/tt_faults.json`: retained TrainTicket fault metadata for the public artifact;
- `data/tt_topo_s2s.json`: TrainTicket service topology summary.

The full raw benchmark releases are not relicensed by this repository. Users should follow the original AIOps-25 and RCAEval release terms when downloading or using the datasets.

The NURSE implementation is released under the MIT License. Benchmark-derived metadata is distributed only for research evaluation and remains subject to the original benchmark terms.

## Verifiability Checklist

Reviewers can check the following without inspecting private materials:

- the code has no required private endpoint or hard-coded API key;
- the README explains installation, datasets, LLM configuration, and outputs;
- tests cover the agent console, session memory, tool contracts, context packaging, LLM bridge, and evidence core wrappers;
- inference artifacts are written under `outputs/` and excluded from source archives;
- benchmark labels are loaded only for evaluation after predictions are produced;
- human-study identifiers and signed consent forms are not included.

## Known Scope Limits

- Full paper-style results require the original public benchmark releases and an LLM API key.
- LLM output wording can vary across providers and API versions. NURSE constrains the ranking and report contracts through verified evidence and schema checks, but natural-language text may still differ.
- During double-anonymous review, the artifact access URL must not reveal author identity.
