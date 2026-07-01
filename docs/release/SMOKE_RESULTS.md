# Release Smoke Results

Date: 2026-07-01

Model used for LLM-enabled smoke checks: `gpt-5-nano`.

## Completed Checks

| Check | Scope | Result |
| --- | --- | --- |
| Quick availability tests | 20 selected tests | Passed |
| Agent console dry-run | RCAEval-TT generated-tool workflow | Passed |
| Direct runner LLM smoke | `rcaeval-tt`, 1 case | Passed, `llm_status=ok`, `total_tokens=9862` |
| OpenAI full matrix smoke | 2 datasets, 1 case each | Passed |
| GPT smoke script | `rcaeval-tt`, 1 case | Passed, `llm=ok` |
| OpenAI connectivity | `gpt-5-nano` | Passed |
| SOTA full matrix smoke | 2 datasets, 1 case each | Passed |
| E/P/H ablation smoke | 2 datasets, 4 variants, 1 case each | Passed |
| OpenAI consistency smoke | 2 datasets, 1 model, 1 case each | Passed, 0 Top-5 mismatches |
| Model consistency smoke | 2 datasets, 1 model, 1 case each | Passed, 0 Top-5 mismatches |
| Aggregation smoke | SOTA full smoke run root | Passed, CSV and XLSX written |

## Representative Metrics

SOTA full smoke:

| Dataset | Acc@1 | Acc@3 | Acc@5 | Cases | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aiops25-retained-nonnode` | 0.0 | 1.0 | 1.0 | 1 | 9783 |
| `rcaeval-tt` | 1.0 | 1.0 | 1.0 | 1 | 10507 |

E/P/H ablation smoke covered `full`, `wo_E`, `wo_P`, and `wo_H`. The release excludes a rules ablation because the current public datasets have no SRE rule inputs.

Generated output roots are under `outputs/` and are intentionally ignored by git.
