param(
    [string[]]$Models = @("gpt-5-nano"),
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [int]$Seed = 20260628,
    [int]$SampleSize = 25,
    [int]$Workers = 10,
    [int]$MaxRetries = 2,
    [int]$CandidateExpansionRounds = 2,
    [int]$WindowWidthSeconds = 480,
    [string]$OutputRoot = "outputs\experiments\nurse_model_consistency_sample",
    [string]$LabRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

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
        throw "NURSE model consistency sample failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
