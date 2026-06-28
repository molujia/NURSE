param(
    [string[]]$Datasets = @("aiops25-retained-nonnode", "rcaeval-tt"),
    [string[]]$Models = @(
        "ep-20260623103508-6dkw6",
        "Doubao-Seed-2.1-pro",
        "ep-20260623160517-ckhj8",
        "glm-4.5-air",
        "glm-4.6v",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5"
    ),
    [string[]]$Modes = @("full"),
    [int]$Seed = 20260628,
    [int]$Workers = 10,
    [int]$CaseLimit = 0,
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
