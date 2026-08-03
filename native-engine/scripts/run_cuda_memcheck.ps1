# Run cuda-memcheck (memcheck) multiple times and collect output
param(
  [int]$iterations = 3,
  [string]$exe = "E:\\adaptivellm\\native-engine\\build\\Release\\transformer_stack_benchmark.exe",
  [string]$outDir = "E:\\adaptivellm\\native-engine\\out",
  [string]$dumpDir = "E:\\adaptivellm\\native-engine\\dumps"
)

if(-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
if(-not (Test-Path $dumpDir)) { New-Item -ItemType Directory -Path $dumpDir | Out-Null }

$cm = Get-Command cuda-memcheck -ErrorAction SilentlyContinue
if(-not $cm) {
  Write-Error "cuda-memcheck not found. Install CUDA Toolkit (matching driver) and re-run."
  exit 2
}
$cuda = $cm.Source

for($i=0;$i -lt $iterations;$i++) {
  $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
  $log = Join-Path $outDir ("cuda_memcheck_" + $ts + ".txt")
  Write-Host "Iteration $i -> $log"
  & $cuda --tool memcheck --leak-check full --report-api-errors all --show-backtrace all --cache-control off --log-file $log $exe
  if($LASTEXITCODE -ne 0) { Write-Host "cuda-memcheck exitcode: $LASTEXITCODE (see $log)" }
  Start-Sleep -Seconds 2
}

# Package logs and existing dumps
$zip = Join-Path $outDir "cuda_memcheck_artifacts_$(Get-Date -Format yyyyMMdd_HHmmss).zip"
Compress-Archive -Path $outDir\cuda_memcheck_*.txt, $dumpDir\* -DestinationPath $zip -Force
Write-Host "Packaged artifacts: $zip"
