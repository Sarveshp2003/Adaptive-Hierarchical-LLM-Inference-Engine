param([string]$Binary = "build\Debug\benchmark_smoke.exe")

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$resolvedBinary = Join-Path $repoRoot $Binary

if (-not (Test-Path $resolvedBinary)) {
    Write-Error "Benchmark binary not found: $resolvedBinary"
    exit 1
}

& $resolvedBinary
