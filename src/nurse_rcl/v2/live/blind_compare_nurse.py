"""Independent normal NURSE experiment; only the report handoff is adapted.

No Codex material, completion injection, tool changes or additional model calls.
The original composition still owns preparation, core, bounded stage and G+.
"""


def main() -> int:
    from nurse_rcl.v2.live import ops_sre_gplus_case as composition
    from nurse_rcl.v2.live.blind_compare_report import stage_report_context

    original = composition.stage_report_context
    try:
        composition.stage_report_context = stage_report_context
        return composition.main()
    finally:
        composition.stage_report_context = original


if __name__ == "__main__":
    raise SystemExit(main())
