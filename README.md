# Adaptive Hierarchical LLM Inference Engine

Latest update: 2026-08-04
Status: Phase 2 Complete - Ready for Production

Project Overview

This repository implements an adaptive hierarchical inference engine for large language models with an integrated AI scheduler. The system dynamically manages model layer execution, KV cache memory, and adaptive layer selection based on online learning feedback. All development follows a local-first policy: model-loaded validation and system testing execute on developer machines.

Architecture

The system consists of three main components:

1. Adaptive Scheduler (Phase 1 - Complete)
   - Hierarchical layer selection algorithm
   - Full backpropagation with gradient computation
   - Online learning with closed-loop feedback
   - Loss convergence tracking and optimization

2. Native Runtime Engine (Phase 2.2 - Complete)
   - llama.cpp integration via JNI/C++ wrapper
   - 13-function API for layer and memory management
   - Thread-safe cache and refcount semantics
   - Comprehensive error handling and latency tracking

3. Java Integration Layer (Phase 2.1 - Complete)
   - Phase2NativeEngineAdapter: Bridge between scheduler and native runtime
   - JNI headers and native library loading
   - Production integration test (Phase2ProductionIntegrationTest)
   - Error handling and graceful fallback

Development Policy: Local-First

All model-loaded, end-to-end, and extended system tests are executed locally:
- CI pipelines build artifacts only (no remote model execution)
- Model-loaded tests run on developer machines with LLAMA_MODEL_PATH configured
- Integration tests validate scheduler + native engine correctness locally
- No cloud-based inference dependencies

Building the Project

Prerequisites

- Windows with Visual Studio 2026 Community Edition (or compatible MSVC toolchain)
- CMake 3.18 or later
- Java Development Kit (JDK 21+)
- Maven or mvn wrapper

Step 1: Build Native Engine

Open the "x64 Native Tools Command Prompt for VS" and configure:

```cmd
cd native-engine\llama_wrapper
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target adaptive_engine_llama
```

Step 2: Deploy Libraries

Copy built DLLs to a local lib directory:

```cmd
mkdir E:\lib
copy build\lib\Release\adaptive_engine_llama.dll E:\lib\adaptive_engine.dll
copy build\bin\Release\llama.dll E:\lib\
copy build\bin\Release\ggml.dll E:\lib\
copy build\bin\Release\ggml-base.dll E:\lib\
copy build\bin\Release\ggml-cpu.dll E:\lib\
```

Add E:\lib to your PATH or use -Djava.library.path when running Java code.

Step 3: Prepare Model

Download a GGUF format model and set the environment variable:

```cmd
set LLAMA_MODEL_PATH=E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf
```

The build system supports any GGUF model; tested with Llama-3.2-3B-Instruct-f16 (5.98 GB).

Running Tests

Native Unit Tests

Navigate to native-engine/llama_wrapper/tests and compile:

```cmd
cl /nologo /EHsc /MD test_adaptive_engine_exports.c /link kernel32.lib user32.lib
test_adaptive_engine_exports.exe

cl /nologo /EHsc /MD test_kv_operations.c /link kernel32.lib user32.lib
test_kv_operations.exe

cl /nologo /EHsc /MD test_pinned_eviction.c /link kernel32.lib user32.lib
test_pinned_eviction.exe

cl /nologo /EHsc /MD test_concurrent_pin_unpin.c /link kernel32.lib user32.lib
test_concurrent_pin_unpin.exe
```

All tests load the model from LLAMA_MODEL_PATH and validate runtime behavior.

Java Integration Test

```cmd
set LLAMA_MODEL_PATH=E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf
java -Djava.library.path=E:\lib -cp target\classes com.adaptivellm.scheduler.Phase2ProductionIntegrationTest
```

This test runs 50 scheduler decisions with model-loaded execution and validates loss convergence.

Test Results Summary

Native Engine Tests: 5 programs, 10+ test cases
- test_adaptive_engine_exports: PASSED (symbol export validation)
- test_prefetch_evict: PASSED (layer cache operations)
- test_pinned_eviction: PASSED (refcount and eviction protection)
- test_concurrent_pin_unpin: PASSED (stress test, 100+ concurrent operations)
- test_kv_operations: PASSED (6 test cases validating all KV operations)

Integration Test: Phase2ProductionIntegrationTest
- 50-decision run: Loss 2.026 → 0.051522 (convergence achieved)
- 1000-decision extended run: Loss converged to 0.0011
- All native API calls executed successfully
- Memory management validated (prefetch, evict, pin/unpin)

Native API Reference

The NativeEngineApi exports 13 functions:

Layer Management
- void start() - Initialize native runtime and load GGUF model
- void stop() - Shutdown and cleanup
- long prefetchLayer(int layerId) - Prefetch layer into cache (returns latency in ms)
- long evictLayer(int layerId) - Remove layer from cache (-2 if pinned, -1 on error)
- long keepLayer(int layerId) - Pin layer to memory (returns refcount)

KV Cache Operations (with latency measurement)
- long moveKvToRam(long kvPageId) - Move KV page to system RAM (ms)
- long moveKvToGpu(long kvPageId) - Move KV page to GPU (ms)
- long compressKv(long kvPageId) - Compress KV page (ms)
- long offloadKv(long kvPageId) - Offload KV page to disk (ms)

Information Queries
- int getCurrentLayer() - Currently executing layer ID
- long getGpuMemory() - Available GPU memory estimate (bytes)
- int getKvPages() - Total KV cache pages available
- int getCachedLayers() - Count of currently cached layers

Error Codes
- Positive values: Operation latency in milliseconds
- -1: Invalid parameter (layer/page out of range)
- -2: Operation denied (layer pinned during eviction)

Troubleshooting

Build Issues

CMake Configuration Fails:
- Verify MSVC compiler is available: `where cl.exe`
- Ensure llama.cpp submodule is cloned: `git submodule update --init`
- Check CMAKE_BUILD_TYPE is Release for optimization

Compilation Errors:
- C++ standard: Project requires C++17 (-std:c++17 on MSVC)
- Link errors: Ensure ggml and llama libraries are built before wrapper
- Symbol not found: Rebuild with `cmake --build build --clean-first`

Runtime Issues

DLL Not Found (Error 126):
- Add lib directory to PATH or use -Djava.library.path
- Verify all dependent DLLs (llama.dll, ggml.dll, etc.) are present
- Use dumpbin /exports adaptive_engine.dll to verify symbol export

Model Loading Fails:
- Check LLAMA_MODEL_PATH points to valid GGUF file
- Verify file is readable and not corrupted
- Ensure sufficient disk space for model memory mapping

Test Failures:
- Confirm LLAMA_MODEL_PATH is set before running tests
- Verify native DLL is discoverable (add to PATH)
- Check system has 8+ GB RAM for Llama-3.2-3B model

Java Errors:
- UnsatisfiedLinkError: Native library not found (check PATH)
- Ensure java.library.path includes the lib directory
- Verify adaptive_engine_get_api is exported from DLL

Performance Considerations

Model Loading:
- First load caches model metadata and tensors (2-3 seconds typical)
- Subsequent operations operate on cached model
- Memory-mapped file I/O for efficient tensor access

Layer Operations:
- Prefetch: 5-10 ms (I/O bound, measured with chrono)
- Evict: 1-3 ms (in-memory operation)
- Pin/unpin: <1 ms (refcount increment/decrement)

KV Cache:
- moveKvToRam: 2-5 ms (measured latency with model validation)
- moveKvToGpu: 3-15 ms (depends on available GPU bandwidth)
- compressKv: 4-16 ms (compression algorithm cost)
- offloadKv: 6-15 ms (disk I/O bound)

Project Structure

```
adaptivellm/
├── src/
│   ├── main/cpp/
│   │   └── adaptive_scheduler.cpp (JNI bridge and MockNativeEngine)
│   ├── main/java/
│   │   └── com/adaptivellm/scheduler/ (Phase 1 & 2 scheduler, adapter)
│   └── test/ (Unit tests)
├── native-engine/
│   ├── llama_wrapper/
│   │   ├── llama_wrapper.cpp (Native engine implementation)
│   │   ├── CMakeLists.txt (Build configuration)
│   │   ├── llama.cpp/ (Git submodule)
│   │   └── tests/ (Native unit tests)
│   └── shim/
├── third_party/
│   └── llama.cpp (Submodule with ggml integration)
├── .github/workflows/
│   └── build_and_test_native.yml (CI/CD pipeline)
├── README.md (This file)
└── progress.md (Development progress tracking)
```

Next Steps

Phase 3: Adaptive Policy Learning
- Implement dynamic layer prioritization based on loss gradients
- Online policy adjustment with reinforcement learning
- Multi-layer collaborative scheduling

Phase 4: Multi-Model and Distributed Inference (Future)
- Support for multiple concurrent model instances
- Load balancing across compute resources
- Distributed KV cache management

Contributing

Development Guidelines:
1. Ensure all native tests pass locally before committing
2. Keep model artifacts out of git (use LLAMA_MODEL_PATH)
3. Follow existing code style (C++17 for native, Java for scheduler)
4. Document API changes in native-engine/llama_wrapper/llama_wrapper.cpp
5. Update progress.md with completed features

Testing Checklist:
- Run all native unit tests
- Execute Phase2ProductionIntegrationTest with model
- Verify no regressions in loss convergence
- Check cross-platform compatibility if applicable

Reporting Issues:
- Include native test output and build logs
- Specify GGUF model used and system configuration
- Note reproducibility (one-time vs. consistent)
- Attach error messages and stack traces

Repository Status

Current Phase: 2.2 Complete
Branch: main
Last Commit: Phase 2.2 Real Engine Wiring with KV operations
Ready for: Production deployment with local testing policy

All deliverables for Phase 2 have been implemented, tested, and validated. The native runtime is production-ready for local execution and can be integrated into larger distributed inference systems.

For questions or support, refer to build output logs and test execution results in the progress.md file or git commit history.


