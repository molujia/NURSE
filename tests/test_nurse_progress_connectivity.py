from nurse.llm_connectivity import failed_probe_results
from nurse.progress import format_progress_line
from nurse.runner import (
    _dedupe_records,
    _is_resumable_completion,
    _llm_payload,
    _timing_payload,
    _unsafe_verified_llm_evidence,
    _require_release_llm_configuration,
    build_arg_parser,
)
from nurse.schemas import EvidenceCandidate


def test_progress_line_names_dataset_model_count_and_status():
    line = format_progress_line(
        dataset="rcaeval-tt",
        llm="gpt-5-nano",
        index=3,
        total=10,
        case_id="case-3",
        status="ranked/llm=ok",
        top1="checkout",
    )

    assert "[Nurse][rcaeval-tt][llm=gpt-5-nano]" in line
    assert "3/10" in line
    assert "case=case-3" in line
    assert "status=ranked/llm=ok" in line
    assert "top1=checkout" in line


def test_llm_preflight_marks_skipped_or_error_as_failures():
    report = {
        "results": [
            {"model": "gpt-5-nano", "status": "ok"},
            {"model": "gpt-5-nano", "status": "skipped", "reason": "Missing OPENAI_API_KEY"},
            {"model": "other", "status": "error", "reason": "timeout"},
        ]
    }

    failures = failed_probe_results(report)

    assert [item["model"] for item in failures] == ["gpt-5-nano", "other"]


def test_llm_resume_requires_successful_llm_record():
    assert _is_resumable_completion({"status": "ranked", "llm_status": "ok"}, "gpt-5-nano")
    assert not _is_resumable_completion({"status": "ranked", "llm_status": "error"}, "gpt-5-nano")
    assert _is_resumable_completion({"status": "ranked", "llm_status": "error"}, "none")
    assert _is_resumable_completion({"status": "missed_alarm"}, "gpt-5-nano")


def test_dedupe_records_keeps_latest_record_per_case():
    records = [
        {"case_id": "case-1", "status": "ranked", "llm_status": "error"},
        {"case_id": "case-2", "status": "ranked", "llm_status": "ok"},
        {"case_id": "case-1", "status": "ranked", "llm_status": "ok"},
    ]

    deduped = _dedupe_records(records)

    assert deduped == [
        {"case_id": "case-2", "status": "ranked", "llm_status": "ok"},
        {"case_id": "case-1", "status": "ranked", "llm_status": "ok"},
    ]


def test_runner_timing_payload_splits_non_llm_and_llm_seconds():
    payload = _timing_payload(total_seconds=10.0, llm_seconds=4.25)

    assert payload == {
        "total_seconds": 10.0,
        "llm_seconds": 4.25,
        "non_llm_seconds": 5.75,
    }


def test_runner_llm_payload_records_model_status_calls_and_tokens():
    payload = _llm_payload(
        llm="gpt-5-nano",
        llm_result={
            "status": "ok",
            "model": "gpt-5-nano",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        },
    )

    assert payload["enabled"] is True
    assert payload["status"] == "ok"
    assert payload["model"] == "gpt-5-nano"
    assert payload["calls"] == 1
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 20
    assert payload["total_tokens"] == 30


def test_runner_cli_accepts_gpt_experiment_mode_and_ranker_overrides():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "rcaeval-tt",
            "--llm",
            "gpt-5-nano",
            "--experiment-mode",
            "explanation_only",
            "--ranker-overrides-json",
            '{"metric_tier_size": 3}',
        ]
    )

    assert args.experiment_mode == "explanation_only"
    assert args.ranker_overrides_json == '{"metric_tier_size": 3}'


def test_runner_cli_accepts_metric_score_backend():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "rcaeval-tt",
            "--metric-score-backend",
            "alarm_fixed_window",
        ]
    )

    assert args.metric_score_backend == "alarm_fixed_window"

def test_runner_cli_defaults_to_gpt_for_release_runs():
    args = build_arg_parser().parse_args([])

    assert args.llm == "gpt-5-nano"


def test_runner_cli_rejects_no_llm_release_runs():
    try:
        build_arg_parser().parse_args(["--llm", "none"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("public Nurse CLI must reject --llm none")
def test_runner_runtime_requires_configured_llm():
    class FakeBridge:
        def required_env_key(self, model):
            return "OPENAI_API_KEY"

        def has_credentials(self, model):
            return False

    try:
        _require_release_llm_configuration("gpt-5-nano", FakeBridge())
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Nurse must fail before running cases when the GPT key is missing")


def test_runner_runtime_rejects_no_llm_direct_calls():
    class FakeBridge:
        def required_env_key(self, model):
            return "OPENAI_API_KEY"

        def has_credentials(self, model):
            return True

    try:
        _require_release_llm_configuration("none", FakeBridge())
    except ValueError as exc:
        assert "requires an LLM" in str(exc)
    else:
        raise AssertionError("run_nurse must reject llm='none' in release mode")
def test_unsafe_llm_guardrail_disabled_evidence_is_marked():
    candidate = EvidenceCandidate(
        evidence_id="case-1:llm:0:checkout:cpu",
        case_id="case-1",
        entity="checkout",
        signal="cpu",
        kind="metric",
        time_range={"start": 1, "end": 2},
        claim={"metric_role": "llm_verified_check"},
        provenance={"source": "llm_proposal"},
    )

    verified = _unsafe_verified_llm_evidence(candidate)

    assert verified.verified is True
    assert verified.verification_rule == "llm_guardrail_disabled"
    assert verified.provenance["candidate_source"] == "llm_proposal"
    assert verified.observed["unsafe"] is True


def test_internal_evidence_core_eval_allows_no_llm_runtime():
    class FakeBridge:
        def required_env_key(self, model):
            return "OPENAI_API_KEY"

        def has_credentials(self, model):
            return False

    _require_release_llm_configuration("none", FakeBridge(), allow_no_llm=True)
