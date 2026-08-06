<#
PowerShell helper to configure and build the native engine on Windows.
Run this from an "x64 Native Tools Command Prompt for VS 20xx" or a PowerShell session where cl.exe is on PATH.
Usage:
  - Open "x64 Native Tools Command Prompt for VS"
  - cd E:\adaptivellm
  - powershell -ExecutionPolicy Bypass -File native-build.ps1

Optional switch: -UseNinja (if Ninja is installed and preferred)
#>
param(
    [switch]$UseNinja
)

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
if (-not $repoRoot) { $repoRoot = Get-Location }
Write-Host "Repo root: $repoRoot"

# Ensure cl.exe is available
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "cl.exe not found. Open 'x64 Native Tools Command Prompt for VS' or run vcvarsall.bat first."
    exit 1
}

# Detect ninja if requested
$ninja = $null
if ($UseNinja) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) { $ninja = "ninja"; Write-Host "Using Ninja" } else { Write-Warning "Ninja requested but not found; falling back to default Visual Studio generator." }
}

Push-Location -LiteralPath "$repoRoot\native-engine"

# Build options
$cmakeArgs = @(
    "-DADAPTIVELLM_ENABLE_CUDA=ON",
    "-DHAVE_LLAMA=ON",
    "-DGGML_CUDA=ON"
)

if ($ninja) {
    $gen = "Ninja"
} else {
    # Prefer Visual Studio 17 2022 generator; CMake will choose the correct installed instance if available
    $gen = "Visual Studio 17 2022"
}
# Build cmake argument array and execute directly to avoid quoting issues
$cmakeCmd = @("cmake", "-S", ".", "-B", "build", "-G", $gen, "-A", "x64") + $cmakeArgsnWrite-Host "Configuring with: $($cmakeCmd -join ' ')"
& $cmakeCmd[0] $cmakeCmd[1..($cmakeCmd.Length - 1)]
# Note: $cmd variable removed; previous code used Invoke-Expression on a constructed string which caused parsing issues in some environments.

Write-Host "Configuring with: $cmd"
Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed. Ensure Visual Studio and required components are installed."
    Pop-Location
    exit $LASTEXITCODE
}

Write-Host "Building (Release) ..."
# Use cmake --build for portability
$buildCmd = "cmake --build build --config Release -- /m"
Invoke-Expression $buildCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    Pop-Location
    exit $LASTEXITCODE
}

Write-Host "Build completed. DLLs live under: $(Join-Path (Get-Location) 'build')\lib or build\bin\Release (check your project layout)"
Pop-Location
exit 0
