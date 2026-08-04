# Adaptive Hierarchical LLM Inference Engine

Latest update: 2026-08-04

Overview

This repository provides a prototype runtime for adaptive hierarchical LLM inference and an adaptive AI scheduler. The project is configured for local-first development and testing: model-loaded validation and end-to-end system checks are executed on developer machines or a designated local test host.

Local-first policy

- All model-loaded, end-to-end, and extended system tests are run locally.
- CI is used for building artifacts and dry-run validation only; model-loaded CI tests are optional and skipped unless a model secret is provided.

Build (Windows - MSVC)

1. Open the "x64 Native Tools Command Prompt for VS" or run vcvars64 in PowerShell.
2. Configure and build the llama wrapper:

`powershell
cmake -S native-engine/llama_wrapper -B native-engine/llama_wrapper/build -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build native-engine/llama_wrapper/build --config Release --target adaptive_engine_llama
`

3. Deploy built DLLs to a local lib directory and add it to PATH or use -Djava.library.path when running Java tests.

Run integration test (example)

`powershell
set LLAMA_MODEL_PATH=E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf
java -Djava.library.path=E:\lib -cp bin com.adaptivellm.scheduler.Phase2ProductionIntegrationTest
`

Native tests

Native unit tests are under native-engine/llama_wrapper/tests. Key tests:
- Export/symbol check
- Prefetch/evict behavior
- Pinned-eviction and concurrent pin/unpin stress tests

Troubleshooting

- Ensure the native toolchain (MSVC or g++) and cmake are available.
- Ensure built DLLs are discoverable by the process (PATH or -Djava.library.path).
- Verify LLAMA_MODEL_PATH points to a valid GGUF file and is readable.
- Confirm adaptive_engine_get_api is exported from the built native library (dumpbin /exports or nm -D).

Contributing

- Keep model artifacts and large runtime outputs out of git.
- Run the local end-to-end checklist before merging native integration changes.

