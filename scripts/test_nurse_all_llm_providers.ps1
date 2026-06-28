param(
    [string[]]$Models = @(
        "ep-m-20260104103220-qp9n8",
        "ep-20260623103535-v7z57",
        "ep-20260615114505-247zc",
        "ep-20260305161829-hsnbm",
        "ep-20260305161623-sjhkb",
        "ep-m-20260430105117-46zl7"
    ),
    [string]$OpenAIModel = "gpt-5-nano",
    [string]$OpenAIBaseUrl = "",
    [string]$Python = "python",
    [string]$Output = "outputs\nurse\llm_connectivity.json",
    [switch]$AllowFailures
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

Push-Location $RepoRoot
try {
    $args = @(
        "-m", "nurse.llm_connectivity",
        "--models"
    ) + $Models + @(
        "--openai-model", $OpenAIModel,
        "--output", $Output
    )
    if ($OpenAIBaseUrl) {
        $args += @("--openai-base-url", $OpenAIBaseUrl)
    }
    if (-not $AllowFailures) {
        $args += "--require-ok"
    }

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE LLM provider connectivity probe failed."
    }
}
finally {
    Pop-Location
}
