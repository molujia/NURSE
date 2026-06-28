"""Public entrypoint shim for NURSE.

The supported interface is the interactive NURSE agent console:

    python -m nurse_agent.console
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NURSE repository entrypoint")
    parser.add_argument(
        "--console-help",
        action="store_true",
        help="Show the recommended console command.",
    )
    parser.parse_args(argv)
    print("NURSE public usage:")
    print("  python -m nurse_agent.console --output-root outputs/agent_runs --session-id demo")
    print()
    print("Direct Evidence Core runner:")
    print("  python -m nurse.runner --dataset rcaeval-tt --case-limit 1 --llm gpt-5-nano --run-dir outputs/nurse/quickstart --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())