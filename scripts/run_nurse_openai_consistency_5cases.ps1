param(
    [string[]]$Models = @("gpt-5-nano", "gpt-5-mini", "gpt-5"),
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [int]$Seed = 20260628,
    [int]$SampleSize = 5,
    [int]$Workers = 3,
    [int]$MaxRetries = 1,
    [int]$CandidateExpansionRounds = 2,
    [int]$WindowWidthSeconds = 480,
    [string]$OutputRoot = "outputs\experiments\nurse_openai_consistency_5cases",
    [string]$LabRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required before running NURSE OpenAI consistency checks."
}

Push-Location $RepoRoot
try {
    $args = @(
        "-m", "nurse.sota_reporting", "sample",
        "--models"
    ) + $Models + @(
        "--datasets"
    ) + $Datasets + @(
        "--seed", $Seed,
        "--sample-size", $SampleSize,
        "--workers", $Workers,
        "--max-retries", $MaxRetries,
        "--candidate-expansion-rounds", $CandidateExpansionRounds,
        "--window-width-seconds", $WindowWidthSeconds,
        "--output-root", $OutputRoot,
        "--python", $Python
    )
    if ($LabRoot) {
        $args += @("--lab-root", $LabRoot)
    }

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE OpenAI consistency check failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
