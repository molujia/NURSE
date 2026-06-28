param(
    [ValidateSet("rcaeval-tt", "aiops25-retained-nonnode")]
    [string]$Dataset = "rcaeval-tt",
    [int]$CaseLimit = 1,
    [string]$RunDir = "outputs\nurse\gpt_smoke",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required before running NURSE."
}
if (-not $env:NURSE_OPENAI_MODEL) {
    $env:NURSE_OPENAI_MODEL = "gpt-5-nano"
}

Push-Location $RepoRoot
try {
    & $Python -m nurse.runner `
        --dataset $Dataset `
        --case-limit $CaseLimit `
        --llm $env:NURSE_OPENAI_MODEL `
        --run-dir $RunDir `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE GPT smoke run failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}