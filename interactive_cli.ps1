# Interactive CLI for Real Model Inference with Adaptive Scheduler
# Usage: .\interactive_cli.ps1

Write-Host @"
╔════════════════════════════════════════════════════════╗
║   REAL MODEL INTERACTION WITH ADAPTIVE SCHEDULER      ║
║          Using Java Engine + Scheduler                ║
╚════════════════════════════════════════════════════════╝

"@

Write-Host "Model Setup:" -ForegroundColor Cyan
Write-Host "  ✓ Real Llama-3.2-3B-Instruct (6.4GB)"
Write-Host "  ✓ Adaptive Memory Scheduler"
Write-Host "  ✓ GPU Acceleration Support"
Write-Host "  ✓ Real-time Scheduler Decisions`n"

$modelPath = "E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "❌ Model not found: $modelPath" -ForegroundColor Red
    exit 1
}

$env:ADAPTIVELLM_MODEL_PATH = $modelPath
$env:ADAPTIVELLM_ENABLE_GPU = "1"

Write-Host "Starting interactive session..." -ForegroundColor Green
Write-Host "Commands: Enter questions | 'exit' to quit | 'help' for options`n"
Write-Host ("─" * 56)

$interactiveScript = @"
using System;
using System.Diagnostics;

class InteractiveRunner
{
    static void Main()
    {
        Console.WriteLine("\n🤖 Initializing...\n");
        var psi = new ProcessStartInfo
        {
            FileName = "java",
            Arguments = "-cp `"target/classes;target/dependency/*`" com.adaptivellm.inference.LlamaInferenceClient",
            WorkingDirectory = @"E:\adaptivellm",
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = false
        };
        
        psi.EnvironmentVariables["ADAPTIVELLM_MODEL_PATH"] = @"E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf";
        psi.EnvironmentVariables["ADAPTIVELLM_ENABLE_GPU"] = "1";
        
        using (var process = Process.Start(psi))
        {
            // Read initialization output
            string line;
            while ((line = process.StandardOutput.ReadLine()) != null && !line.Contains("Enter prompts"))
            {
                Console.WriteLine(line);
            }
            
            // Interactive loop
            while (true)
            {
                Console.Write("\n🤖 You: ");
                string input = Console.ReadLine();
                
                if (string.IsNullOrEmpty(input))
                    continue;
                
                if (input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                    break;
                
                process.StandardInput.WriteLine(input);
                process.StandardInput.Flush();
                
                // Read response
                bool inResults = false;
                while ((line = process.StandardOutput.ReadLine()) != null)
                {
                    if (line.Contains("INFERENCE RESULTS"))
                        inResults = true;
                    
                    if (inResults)
                        Console.WriteLine(line);
                    
                    if (inResults && line.Contains("════"))
                        break;
                }
            }
            
            process.StandardInput.WriteLine("exit");
            process.StandardInput.Flush();
            process.WaitForExit();
        }
    }
}
"@

# Run the Java REPL directly
Write-Host "`n"
& "E:\adaptivellm\run_interactive.bat"
