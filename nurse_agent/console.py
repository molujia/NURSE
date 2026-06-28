"""Console/User Manager entry point for agentic NURSE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .orchestrator import NurseAgentOrchestrator
from .session import NurseAgentSession
from .state import AgentRunResult
from .tools import tool_authoring_tool


HELP_LINES = [
    "Commands:",
    "  help                         Show this help message.",
    "  status                       Show the last run remembered by this session.",
    "  artifacts                    Show artifact paths from the last run.",
    "  memory                       Show the last commands remembered by this session.",
    "  reset                        Clear this session memory.",
    "  approve tool <tool_id>       Approve an agent-generated tool draft.",
    "  reject tool <tool_id>        Reject an agent-generated tool draft.",
    "  exit                         Leave the console.",
    "",
    "Example prompts:",
    "  Run single case 1.",
    "  Run single case 2.",
    "  Run the full AIOps25 benchmark with the default evidence backend.",
    "  Run the full RCAEval-TT benchmark with AD reuse (alarm_fixed_window).",
    "  Generate and use a sliding-window 3-sigma AD evidence tool for RCAEval-TT.",
    "  Run the same configuration on RCAEval-TT.",
]


def run_scripted_console(
    script: str,
    *,
    output_root: str | Path = "outputs/agent_runs",
    llm_required: bool = True,
    dry_run: bool = False,
) -> AgentRunResult:
    orchestrator = NurseAgentOrchestrator(output_root=output_root, llm_required=llm_required)
    return orchestrator.run_script(script, dry_run=dry_run)


def handle_tool_authoring_command(command: str, *, output_root: str | Path = "outputs/agent_runs") -> dict:
    parts = command.strip().split()
    if len(parts) < 3 or parts[1].lower() != "tool":
        raise ValueError("Expected command form: approve tool <tool_id> or reject tool <tool_id>")
    action = parts[0].lower()
    if action not in {"approve", "reject"}:
        raise ValueError(f"unsupported tool authoring command: {action}")
    tool_id = parts[2]
    return tool_authoring_tool.run(
        {
            "action": action,
            "tool_id": tool_id,
            "draft_dir": str(Path(output_root) / "agent_tool_drafts"),
        }
    )


def _print_result(result: AgentRunResult) -> None:
    print(f"[NURSE] intent={result.intent} dataset={result.dataset} run_id={result.run_id}")
    if result.experiment_config.get("metric_score_backend"):
        print(f"[NURSE] metric_score_backend={result.experiment_config['metric_score_backend']}")
    if result.report_path:
        print(f"[NURSE] report={result.report_path}")
    if result.report_json_path:
        print(f"[NURSE] report_json={result.report_json_path}")
    for name, path in result.artifact_paths.items():
        print(f"[NURSE] artifact.{name}={path}")


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(f"[NURSE] {line}")


def interactive_console(
    *,
    output_root: str | Path = "outputs/agent_runs",
    session_id: str = "default",
    llm_required: bool = True,
    dry_run: bool = False,
) -> Optional[AgentRunResult]:
    session = NurseAgentSession(
        output_root=output_root,
        session_id=session_id,
        llm_required=llm_required,
    )
    last_result: Optional[AgentRunResult] = None
    print("NURSE agent console. Type `help` for commands, or `exit` to stop.")
    if dry_run:
        print("Dry-run mode is active; benchmark commands only prepare configs.")
    print(f"Session memory: {session.state_path}")
    while True:
        try:
            command = input("NURSE> ").strip()
        except EOFError:
            break
        if not command:
            continue
        lowered = command.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered == "help":
            _print_lines(HELP_LINES)
            continue
        if lowered == "status":
            _print_lines(session.status_lines())
            continue
        if lowered == "artifacts":
            _print_lines(session.artifact_lines())
            continue
        if lowered == "memory":
            _print_lines(session.memory_lines())
            continue
        if lowered == "reset":
            session.reset()
            print("[NURSE] session memory cleared")
            continue
        if lowered.startswith("approve tool ") or lowered.startswith("reject tool "):
            tool_result = handle_tool_authoring_command(command, output_root=output_root)
            print(
                f"[NURSE] tool.{tool_result['tool_id']} "
                f"state={tool_result['state']} "
                f"ranking_status={tool_result['ranking_status']}"
            )
            continue
        last_result = session.run_command(command, dry_run=dry_run)
        _print_result(last_result)
    return last_result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NURSE agent console")
    parser.add_argument("--script", help="Run one natural-language command and exit.")
    parser.add_argument("--output-root", default="outputs/agent_runs")
    parser.add_argument("--session-id", default="default", help="Persistent interactive session identifier.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-no-llm",
        action="store_true",
        help="Internal test flag. Public NURSE runs should configure OPENAI_API_KEY.",
    )
    args = parser.parse_args(argv)
    llm_required = not args.allow_no_llm
    if args.script:
        result = run_scripted_console(
            args.script,
            output_root=args.output_root,
            llm_required=llm_required,
            dry_run=args.dry_run,
        )
        _print_result(result)
    else:
        interactive_console(
            output_root=args.output_root,
            session_id=args.session_id,
            llm_required=llm_required,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())