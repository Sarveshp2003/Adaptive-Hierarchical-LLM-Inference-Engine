# Run the Java interactive client with a provided native adaptive_engine.dll
# Usage: .\run_interactive_with_dll.ps1 -DllFolder "E:\path\to\built\folder" -ModelPath "E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf"
param(
    [Parameter(Mandatory=$true)] [string]$DllFolder,
    [Parameter(Mandatory=$false)] [string]$ModelPath = "E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf",
    [Parameter(Mandatory=$false)] [int]$MaxTokens = 256
)

if (-not (Test-Path $DllFolder)) { Write-Error "DllFolder not found: $DllFolder"; exit 1 }

# Find the adaptive_engine DLL
$dll = Get-ChildItem -Path $DllFolder -Filter adaptive_engine.dll -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $dll) { Write-Error "adaptive_engine.dll not found in $DllFolder"; exit 1 }

# Add DLL folder to PATH for native dependencies
$env:PATH = "$($dll.DirectoryName);$env:PATH"
$javaLibPath = $dll.DirectoryName

# Set model path env var expected by the Java client
if ($ModelPath) { $env:LLAMA_MODEL_PATH = $ModelPath }

# Run Java interactive client
$class = "com.adaptivellm.inference.LlamaInferenceClient"
$cp = "target\classes"

Write-Host "Starting Java REPL with native DLL: $($dll.FullName)"
Write-Host "Model: $env:LLAMA_MODEL_PATH"

java -Djava.library.path="$javaLibPath" -cp "$cp" $class
