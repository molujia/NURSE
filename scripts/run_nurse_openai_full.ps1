param(
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [string]$Model = "",
    [int]$Seed = 20260628,
    [int]$Workers = 2,
    [int]$CaseLimit = 0,
    [int]$MaxRetries = 1,
    [int]$CandidateExpansionRounds = 2,
    [int]$WindowWidthSeconds = 480,
    [string]$OutputRoot = "outputs\experiments\nurse_openai_full",
    [string]$LabRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required before running NURSE OpenAI experiments."
}

if (-not $Model) {
    if ($env:NURSE_OPENAI_MODEL) {
        $Model = $env:NURSE_OPENAI_MODEL
    }
    else {
        $Model = "gpt-5-nano"
    }
}

Push-Location $RepoRoot
try {
    $args = @(
        "-m", "nurse.sota_reporting", "matrix",
        "--datasets"
    ) + $Datasets + @(
        "--models", $Model,
        "--modes", "full",
        "--seed", $Seed,
        "--case-limit", $CaseLimit,
        "--workers", $Workers,
        "--max-retries", $MaxRetries,
        "--candidate-expansion-rounds", $CandidateExpansionRounds,
        "--window-width-seconds", $WindowWidthSeconds,
        "--report-name", "openai_full_results",
        "--output-root", $OutputRoot,
        "--python", $Python
    )
    if ($LabRoot) {
        $args += @("--lab-root", $LabRoot)
    }

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE OpenAI full experiment failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
