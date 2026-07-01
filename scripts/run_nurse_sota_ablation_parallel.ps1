param(
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [string[]]$Variants = @("full", "wo_E", "wo_P", "wo_H"),
    [string[]]$Models = @("gpt-5-nano"),
    [int]$Seed = 20260628,
    [int]$Workers = 10,
    [int]$CaseLimit = 0,
    [int]$MaxRetries = 2,
    [int]$CandidateExpansionRounds = 2,
    [int]$WindowWidthSeconds = 480,
    [string]$OutputRoot = "outputs\experiments\nurse_sota_ablation_parallel",
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
    ) + $Variants + @(
        "--seed", $Seed,
        "--case-limit", $CaseLimit,
        "--workers", $Workers,
        "--max-retries", $MaxRetries,
        "--candidate-expansion-rounds", $CandidateExpansionRounds,
        "--window-width-seconds", $WindowWidthSeconds,
        "--report-name", "ablation_results",
        "--output-root", $OutputRoot,
        "--python", $Python
    )
    if ($LabRoot) {
        $args += @("--lab-root", $LabRoot)
    }

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE ablation parallel run failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
