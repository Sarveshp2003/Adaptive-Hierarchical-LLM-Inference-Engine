@echo off
REM Interactive CLI for Real Model Inference with Adaptive Scheduler

cd /d E:\adaptivellm

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║   REAL MODEL INTERACTION WITH ADAPTIVE SCHEDULER      ║
echo ║     Using Java Engine + AI Scheduler Integration      ║
echo ╚════════════════════════════════════════════════════════╝
echo.
echo Features:
echo   ✓ Real Llama-3.2-3B-Instruct model inference
echo   ✓ Adaptive memory scheduler making real decisions
echo   ✓ GPU acceleration (if available)
echo   ✓ Real-time scheduler decisions visible in output
echo.
echo Type your questions and see the scheduler optimize memory in real-time!
echo Type 'exit' to quit
echo ─────────────────────────────────────────────────────────
echo.

REM Set environment variables
set ADAPTIVELLM_MODEL_PATH=E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf
set ADAPTIVELLM_ENABLE_GPU=1
set ADAPTIVELLM_DEBUG=false

REM Run the Java application
java -cp "target/classes;target/dependency/*" com.adaptivellm.inference.LlamaInferenceClient

echo.
echo ✓ Session closed
pause
