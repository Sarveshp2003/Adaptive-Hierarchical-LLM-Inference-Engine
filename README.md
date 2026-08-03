# Adaptive Hierarchical LLM Inference Engine

This repository contains a prototype runtime for adaptive hierarchical LLM inference with:

- GGUF-backed model probing and metadata inspection
- layered model loading and lightweight inference-style execution
- cache and prefetch scaffolding for layer streaming
- a runtime contract with CPU/GPU backend selection
- Java-facing native bridge entry points for control-plane communication
- CMake-based build and smoke-test validation on Windows/MSVC

## Project layout

- `src/main/java/` – Java control-plane and orchestration code, including the Java-native bridge class
- `native/` – JNI bridge and native interface entry points used by Java
- `native-engine/` – C++/CUDA runtime and allocator components
- `runtime/`, `loader/`, `cache/`, `prefetch/`, `scheduler/`, `compression/`, `util/` – core native runtime modules
- `tests/` and `tools/` – regression tests and sample executables
- `samples/` – lightweight model assets used for smoke testing
- `docs/` and `artifacts/` – project documentation and generated outputs

## Build

Use CMake + MSVC on Windows:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target run_sample_model
```

## Run smoke tests

```powershell
.\build\Release\run_sample_model.exe "E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf"
```

## Java <-> native communication

The Java side exposes `com.adaptivellm.nativeengine.NativeEngine`, which calls native entry points through the JNI bridge:

- `startRuntime()`
- `stopRuntime()`
- `requestLayer(int layerId)`

The native side routes these calls through `native/jni_bridge.cpp` into `native/jni_interface.cpp`, so the control-plane path is connected and can be extended to drive the runtime directly.

## Performance and memory highlights

The current prototype is focused on being visible, debuggable, and extensible rather than fully production-optimized. The main areas that are already highlighted in the repository are:

- Execution-time visibility: the sample runner reports layer execution and generation-style output in a reproducible way.
- Memory allocation awareness: the runtime stack includes allocator and tensor lifecycle code, with a clear path for CPU/GPU memory ownership and reuse.
- Cache/prefetch path: the runtime can now exercise a real GGUF-backed prefetch flow and report a non-zero cache population.
- Layer streaming readiness: the loader and runtime contract already separate metadata/probing from layer execution, making future streaming and paging work easier to plug in.

## End-to-end data flow

The current runtime flow follows the architecture below:

```text
User Request
    |
    v
Java Control Layer
    |
    +--> Rule-based / Prototype AI Scheduler
    |        |
    |        v
    |   Layer Cache / KV Pages / Prefetch Queue
    |
    v
JNI Bridge
    |
    v
Native Runtime (C++)
    |
    +--> Tensor Ops / Memory Movement
    |
    v
GPU / CPU Backend
    |
    v
Generated Tokens / Model Output
    |
    v
Java Control Layer (next decision cycle)
```

1. User request enters the Java control layer.
2. The Java scheduler and memory manager decide what layer or KV page to load next.
3. The loader consults the model artifact (currently GGUF-backed) and prepares the required layer data.
4. The prefetch/cache subsystem stages useful layers and pages into RAM.
5. The JNI bridge passes control to the native execution layer.
6. The C++ runtime handles tensor operations and memory movement.
7. The GPU backend executes the active compute path when available.
8. Output tokens or downstream results flow back up to the control plane for the next decision cycle.

This maps to the repository structure as follows:

- `src/main/java/` – Java orchestration, scheduler, and memory decisions
- `native/` – bridge between Java and native execution
- `runtime/`, `loader/`, `cache/`, `prefetch/` – runtime execution, model access, and memory movement
- `native-engine/` – lower-level tensor and allocator work

## Current focus areas and roadmap

The project is currently working through the following areas:

- Java control-plane integration and native bridge wiring
- GGUF model loading and metadata inspection
- Lightweight inference-style execution and token generation smoke tests
- Cache/prefetch flow for layer staging
- Memory allocation and tensor lifecycle visibility
- Future work: richer KV paging/compression, off-heap memory management, and more advanced scheduling

## How others can try it

1. Clone the repo and build the sample runner:
   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release --target run_sample_model
   ```
2. Point it at a local GGUF model:
   ```powershell
   .\build\Release\run_sample_model.exe "C:\path\to\your\model.gguf"
   ```
3. Explore the runtime modules:
   - `runtime/` for backend and execution flow
   - `loader/` for model-file parsing and layer loading
   - `cache/` and `prefetch/` for memory and prefetch behavior
   - `native/` for Java-to-native bridge wiring
   - `src/main/java/com/adaptivellm/scheduler/` for the prototype AI scheduling stack
4. For Java-side integration, compile or run code that uses `com.adaptivellm.nativeengine.NativeEngine` and call the native entry points directly.
5. For scheduler experiments, try the Java scheduler classes directly and evaluate how the prototype predictor behaves on synthetic memory states before wiring it into the live runtime path.
6. For more advanced experiments, swap in other model files, adjust the cache capacity, or extend the runtime contract to add new backend implementations.

## Repository hygiene

- Generated build outputs and temporary files are ignored by `.gitignore`.
- Local model artifacts and large runtime outputs are not tracked by default.
- The active source tree is focused on the runtime, bridge, and smoke-test path.
