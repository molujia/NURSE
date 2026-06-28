param(
    [string[]]$Models = @("gpt-5-nano"),
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
    $Args = @(
        "-m", "nurse.llm_connectivity",
        "--models"
    ) + $Models + @(
        "--openai-model", $OpenAIModel,
        "--output", $Output
    )
    if ($OpenAIBaseUrl) {
        $Args += @("--openai-base-url", $OpenAIBaseUrl)
    }
    if (-not $AllowFailures) {
        $Args += "--require-ok"
    }

    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "NURSE LLM connectivity probe failed. Use -AllowFailures only when intentionally diagnosing a failing model."
    }
}
finally {
    Pop-Location
}