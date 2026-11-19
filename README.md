# NURSE

NURSE is a research tool for microservice root cause localization. Given metrics, logs, traces, and service dependencies for an incident, it ranks candidate services and shows how their scores were computed. A bounded investigation then helps a site reliability engineer (SRE) decide what to check next.

- **Localization.** The core scores local anomalies and propagation support, then freezes the ranking with its supporting evidence.
- **Investigation.** An LLM agent reads evidence, checks mechanisms, and compares observations. New measurements update the incident account while the original ranking stays available for inspection. [Investigation tool catalog (Chinese)](docs/nurse_final_focused_tools.md)
- **Reporting.** The semantic evidence graph (G+) binds report statements to their sources, measurement context, and permitted uses. The report distinguishes observations, inferred propagation paths, and checks still to perform.

[Datasets](#datasets) · [Installation](#installation) · [Quick start](#quick-start) · [Run a prepared incident](#run-a-prepared-incident) · [Outputs](#outputs) · [Evaluation](#evaluation)

## Datasets

Download the telemetry from the benchmark maintainers under their respective terms. The paper uses the following service-level evaluation cohorts.

| Dataset | Cases used in NURSE | Where to get the data |
| --- | --- | --- |
| AIOps Challenge 2025 (AIOps25) | 318 HipsterShop incidents with service- or pod-level roots. The cohort excludes the 82 node-root incidents. | [Official AIOps Challenge 2025 dataset repository](https://www.aiops.cn/gitlab/aiops-live-benchmark/aiopschallenge2025) |
| RCAEval TrainTicket (RCAEval-TT) | 120 incidents across RE2-TT (90) and RE3-TT (30). | [RCAEval project](https://github.com/phamquiluan/RCAEval) · [Download RE2-TT.zip](https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1) · [Download RE3-TT.zip](https://zenodo.org/records/14590730/files/RE3-TT.zip?download=1) |

**From data to a run.** The live CLI reads a prepared incident plan. This checkout includes adapters for AIOps25 (`aiops25`) and the separate RCABench dataset (`rcabench`). Running RCAEval-TT with this checkout requires a TrainTicket preparation adapter. Keep root labels separate for offline evaluation. [Incident input contract](src/nurse_rcl/v2/live/repair_stage.py)

## Installation

Use Linux, Bash, and Python 3.10–3.12. The commands below use Python 3.11 and the locked dependencies. Install [uv](https://docs.astral.sh/uv/) first.

```bash
git clone https://github.com/molujia/NURSE.git
cd NURSE
uv sync --locked --python 3.11 --extra test
source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:third_party/hermes_runtime
```

Run the commands from the repository root so NURSE can locate its profiles and tool definitions. The live process needs support for limits of 8 GiB of address space, 64 MiB of stack, and 4,096 open file descriptors. The test harness also uses a writable `/tmp`.

## Quick start

You can inspect the CLI and test evidence inspection and report binding locally without downloading a dataset or configuring any model credentials.

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case --help
python scripts/dev/run_final_explainability_0ark.py -q --tb=short \
  tests/test_v2_live_sre_focused_tools.py \
  tests/test_v2_live_sre_focused_sources.py \
  tests/test_v2_live_sre_gplus_graph.py \
  tests/test_v2_live_sre_gplus_contract.py
```

The first command prints the plan and execution options. The second runs offline and returns a pytest summary.

## Run a prepared incident

### Configure the two models

For investigations using reviewed historical cases, we recommend configuring both a chat LLM and an embedding model.

| Component | Configuration |
| --- | --- |
| Investigation and report writing | The focused live profile uses the Ark chat model `deepseek-v4-flash-ga-260731`. Set the deployment credential and lock locations (`KEY_PATH` and `LOCK_PATH`) before creating plans. Use a mode-`0600` file containing one `ARK_API_KEY2` assignment and a writable lock location. [Provider configuration](src/nurse_rcl/v2/live/ops_provider.py) |
| Optional historical retrieval | Set `NURSE_EMBEDDING_MODEL` to `doubao-embedding-vision-251215` with `ARK_API_KEY`, or to `text-embedding-3-small` with `OPENAI_API_KEY`. With no embedding model selected, NURSE uses its local fixed rule. An API key alone leaves the fixed rule active. [History configuration](docs/history_retrieval_experiment.md) |

Both retrieval methods rank the same reviewed cases using the same evidence text, including change magnitude and observation quality. The embedding model compares whole-case vectors by Euclidean distance. The fixed rule uses weighted Jaccard similarity. By default, the agent receives the text and source links for up to five cases after the core freezes the ranking.

### Check and execute the plan

Prepare a canonical JSON composition plan that identifies the incident, normal and fault windows, telemetry and mappings, core plan, software and configuration hashes, and a fresh output location. Set the investigation budget to 1–8 decisions. Use schema `nurse.v2.gplus-one-case-composition/2` for a focused run or `/5` for a new run with reviewed history. The history-enabled plan records the selected model or the local-rule choice. [Plan definition](src/nurse_rcl/v2/live/ops_sre_gplus_case.py) · [Preparation guide (Chinese)](docs/nurse_final_focused_quickstart.md)

Set `COMPOSITION_PLAN` to the absolute path of the plan and `COMPOSITION_SHA256` to its verified digest. First check the inputs and configuration.

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" --plan-sha256 "$COMPOSITION_SHA256" --check-only
```

A successful check reports `status="admitted"` and makes no model calls. To execute the reviewed plan and allow live provider calls, run the following command.

```bash
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" --plan-sha256 "$COMPOSITION_SHA256" \
  --approved-composition-sha256 "$COMPOSITION_SHA256"
```

To inspect a started run, use `--status` in place of `--check-only` on the first command.

## Outputs

Start with `composition-result.json` in the run directory for stage status. The `core-results/` and `results/` directories hold the core and investigation outputs for the case.

| File or directory within the case | What to inspect |
| --- | --- |
| `core/result.json` | Frozen service ranking, score contributions, and supporting evidence. |
| `investigation/` | Agent decisions, actions, and returned observations. |
| `gplus-report/report.md` and `gplus-report/evidence.md` | Incident findings, alternatives, open questions, and their supporting records. |
| `gplus-report/predicted-paths.json` | Final root-to-symptom path hypotheses, with the evidence and inference status of individual steps. |

The report viewer expects an appendix alias named `report-evidence.md`. The guide shows how to create a viewing copy outside the sealed run directory. [Report viewing instructions](docs/nurse_final_focused_quickstart.md#保存报告的展示兼容)

## Evaluation

**Software verification.** The Linux CI workflow checks the release inventory, the CLI, the 45-file software suite, and the historical retrieval tests. To run the 45-file suite locally from Bash, use the same test list. [CI workflow](.github/workflows/software-verification.yml)

```bash
mapfile -t selected_tests < tests/final_explainability_0ark.txt
python scripts/dev/run_final_explainability_0ark.py -q --tb=short "${selected_tests[@]}"
```

**Path accuracy.** To score the final report paths with mABC's Root Cause Path Accuracy, provide an independent reference set for the same case. The calculator matches complete service sequences and retains negative scores. Set `PREDICTIONS_JSON` to the exported prediction file and `REFERENCE_JSON` to the reference file. [Path format and scoring rules](docs/path_accuracy.md)

```bash
python -m nurse_rcl.evaluation.path_accuracy \
  --predictions "$PREDICTIONS_JSON" --reference "$REFERENCE_JSON"
```

**Historical retrieval.** The evaluation tools build a reviewed library, rank eligible cases, and score rankings against separately supplied labels. [Retrieval and evaluation guide (Chinese)](docs/history_retrieval_experiment.md)

## Citation, license, and contact

When referencing the software, cite the NURSE repository and include the commit used for your run. [Repository](https://github.com/molujia/NURSE)

NURSE is released under the MIT License. For questions or bug reports, open an issue and include your commit, Python version, and the command you ran. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Issues](https://github.com/molujia/NURSE/issues)
