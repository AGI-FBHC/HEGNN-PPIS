param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $RepositoryRoot
try {
    & $Python "src/evaluate.py"
    & $Python "scripts/verify_results.py" --result-dir "output/test60"
}
finally {
    Pop-Location
}
