param(
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [string[]]$Models = @("gpt-5-nano"),
    [string[]]$Modes = @("full"),
    [int]$Seed = 20260628,
    [int]$Workers = 10,
    [int]$CaseLimit = 0,
    [int]$MaxRetries = 2,
    [int]$CandidateExpansionRounds = 2,
    [int]$WindowWidthSeconds = 480,
    [string]$OutputRoot = "outputs\experiments\nurse_sota_full_parallel",
    [string]$LabRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

Push-Location $RepoRoot
try {
    $args = @(
        "-m", "nurse.sota_reporting", "matrix",
        "--datasets"
    ) + $Datasets + @(
        "--models"
    ) + $Models + @(
        "--modes"
    ) + $Modes + @(
        "--seed", $Seed,
        "--case-limit", $CaseLimit,
        "--workers", $Workers,
        "--max-retries", $MaxRetries,
        "--candidate-expansion-rounds", $CandidateExpansionRounds,
        "--window-width-seconds", $WindowWidthSeconds,
        "--report-name", "matrix_runs",
        "--output-root", $OutputRoot,
        "--python", $Python
    )
    if ($LabRoot) {
        $args += @("--lab-root", $LabRoot)
    }

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE full parallel matrix run failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
