# Repository Layout

```text
NURSE/
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
    sota_reporting.py         Reporting helper used by the basic OpenAI scripts.
  nurse_agent/
    console.py                Interactive and scripted agent console.
    orchestrator.py           Agent planning and approved tool orchestration.
    session.py                Session memory.
    tools/                    Approved tools called by the agent.
  scripts/
    run_nurse_gpt_smoke.ps1
    run_nurse_openai_full.ps1
    run_nurse_openai_consistency_5cases.ps1
    test_nurse_llm_connectivity.ps1
    test_nurse_all_llm_providers.ps1
  tests/
    test_nurse_agent_*.py     Agent console, memory, and tool-contract tests.
    test_nurse_*.py           Evidence Core, LLM bridge, release entrypoint, and runner tests.
  main.py                     Public entrypoint shim.
  requirements.txt            Python dependencies.
  README.md                   Main user and artifact guide.
```

Generated outputs are written under `outputs/`, `tmp/`, or test temp directories. These paths are ignored by git and should not be included in release archives.
