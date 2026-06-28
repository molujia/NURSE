# Repository Layout

```text
NURSE/
  data/
    ds25_faults.json          Retained AIOps-25 fault metadata.
    ds25_topo_s2s.json        AIOps-25 service topology summary.
    tt_faults.json            Retained TrainTicket fault metadata.
    tt_topo_s2s.json          TrainTicket service topology summary.
  docs/
    ARTIFACT_EVALUATION.md    Reviewer-facing artifact instructions.
    REPOSITORY_LAYOUT.md      This file.
    release/
      RELEASE_AUDIT.md        Release and anonymity checklist.
  nurse/
    alarm_*.py                Alarm artifact construction and detection.
    evidence_*.py             Evidence admission, graph building, and ranking.
    runner.py                 Direct Evidence Core runner.
    schemas.py                Typed artifact schemas.
  nurse_agent/
    console.py                Interactive and scripted agent console.
    orchestrator.py           Agent planning and approved tool orchestration.
    session.py                Session memory.
    tools/                    Approved tools called by the agent.
  scripts/
    run_nurse_gpt_smoke.ps1   Single-case GPT smoke run.
    test_nurse_llm_connectivity.ps1
  tests/
    test_nurse_agent_*.py     Agent console, memory, and tool-contract tests.
    test_nurse_*.py           Evidence Core, LLM bridge, and runner tests.
  main.py                     Public entrypoint shim.
  requirements.txt            Python dependencies.
  README.md                   Main user and artifact guide.
```

Generated outputs are written under `outputs/`, which is ignored by git.
