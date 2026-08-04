cmake_minimum_required(VERSION 3.15)
project(adaptive_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/native-engine/include)

# Loader test executable
add_executable(loader_test
    tests/loader_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
)

# Custom model loader test
add_executable(custom_loader_test
    tests/custom_loader_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    loader/custom_model_loader.cpp
    loader/ggup_loader.cpp
    loader/gguf_adapter.cpp
    loader/gguf_reader.cpp
    util/logger.cpp
)

# Cache + Prefetch test
add_executable(cache_test
    tests/cache_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    cache/layer_cache.cpp
    prefetch/prefetcher.cpp
    scheduler/scheduler.cpp
    util/logger.cpp
)

# Runtime test (integration)
add_executable(runtime_test
    tests/runtime_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    cache/layer_cache.cpp
    prefetch/prefetcher.cpp
    cache/kv_cache.cpp
    cache/kv_pager.cpp
    compression/kv_compress.cpp
    runtime/inference.cpp
    scheduler/scheduler.cpp
    util/logger.cpp
)

# Metrics test
add_executable(metrics_test
    tests/metrics_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    cache/layer_cache.cpp
    prefetch/prefetcher.cpp
    cache/kv_cache.cpp
    cache/kv_pager.cpp
    compression/kv_compress.cpp
    runtime/inference.cpp
    scheduler/scheduler.cpp
    util/logger.cpp
)

# Runtime contract test
add_executable(runtime_contract_test
    tests/runtime_contract_test.cpp
    runtime/cpu_backend.cpp
    runtime/runtime_adapter.cpp
    runtime/gpu_backend.cpp
    runtime/cpu_runtime.cpp
    runtime/tensor_ops.cpp
    runtime/gpu/cuda_executor_stub.cpp
    util/logger.cpp
)

# GPU backend test
add_executable(gpu_backend_test
    tests/gpu_backend_test.cpp
    runtime/cpu_backend.cpp
    runtime/runtime_adapter.cpp
    runtime/gpu_backend.cpp
    runtime/cpu_runtime.cpp
    runtime/tensor_ops.cpp
    runtime/gpu/cuda_executor_stub.cpp
    util/logger.cpp
)

# Benchmark smoke test
add_executable(benchmark_smoke
    tests/benchmark_smoke.cpp
)

# GEMM micro-benchmark
add_executable(gemm_benchmark
    tests/gemm_benchmark.cpp
    runtime/tensor_ops.cpp
)

# CPU ops unit test
add_executable(cpu_ops_test
    tests/cpu_ops_test.cpp
    runtime/tensor_ops.cpp
)

# Sample model runner
add_executable(run_sample_model
    tools/run_sample_model.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    loader/custom_model_loader.cpp
    loader/ggup_loader.cpp
    loader/gguf_adapter.cpp
    loader/gguf_reader.cpp
    native-engine/cpp/GGUFLoader.cpp
    cache/layer_cache.cpp
    prefetch/prefetcher.cpp
    scheduler/scheduler.cpp
    runtime/cpu_backend.cpp
    runtime/runtime_adapter.cpp
    runtime/gpu_backend.cpp
    runtime/cpu_runtime.cpp
    runtime/tensor_ops.cpp
    runtime/gpu/cuda_executor_stub.cpp
    util/logger.cpp
)

# Generation benchmark (multi-token)
add_executable(generate_benchmark
    tools/generate_benchmark.cpp
    runtime/runtime_adapter.cpp
    runtime/tensor_ops.cpp
    runtime/cpu_backend.cpp
    runtime/gpu_backend.cpp
    runtime/cpu_runtime.cpp
    runtime/gpu/cuda_executor_stub.cpp
    util/logger.cpp
)

# Conversion tool: ggup/gguf -> adaptive archive
add_executable(convert_ggup_to_adaptive
    tools/convert_ggup_to_adaptive.cpp
    loader/custom_model_loader.cpp
    loader/ggup_loader.cpp
    loader/gguf_adapter.cpp
    loader/gguf_reader.cpp
    util/logger.cpp
)

# MMap weight loader
add_library(mmap_weight_loader STATIC
    loader/mmap_weight_loader.cpp
    loader/mmap_weight_loader.h
    loader/mmap_loader.cpp
    loader/weights_index.cpp
)

# Link mmap_weight_loader into runtime-related targets
foreach(tgt IN ITEMS run_sample_model runtime_contract_test gpu_backend_test generate_benchmark)
    if (TARGET ${tgt})
        target_link_libraries(${tgt} PRIVATE mmap_weight_loader)
    endif()
endforeach()

# KV paging test
add_executable(kv_test
    tests/kv_test.cpp
    cache/kv_cache.cpp
    compression/kv_compress.cpp
    util/logger.cpp
)

# KV integration test (runtime uses KVManager)
add_executable(kv_integration_test
    tests/kv_integration_test.cpp
    loader/layer_loader.cpp
    loader/mmap_loader.cpp
    cache/layer_cache.cpp
    prefetch/prefetcher.cpp
    scheduler/scheduler.cpp
    cache/kv_cache.cpp
    runtime/inference.cpp
    cache/kv_pager.cpp
    compression/kv_compress.cpp
    util/logger.cpp
)

# KV pager test
add_executable(kv_pager_test
    tests/kv_pager_test.cpp
    cache/kv_pager.cpp
    cache/kv_cache.cpp
    cache/layer_cache.cpp
    compression/kv_compress.cpp
    scheduler/scheduler.cpp
    util/logger.cpp
)

# On Windows, require filesystem library for older compilers
if (MSVC)
    target_compile_options(loader_test PRIVATE /EHsc)
    target_compile_options(cache_test PRIVATE /EHsc)
    target_compile_options(runtime_test PRIVATE /EHsc)
    target_compile_options(metrics_test PRIVATE /EHsc)
endif()

# Link llama.cpp libraries for GGUF support
set(LLAMA_CPP_BUILD_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build")
set(LLAMA_GGML_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include")
set(LLAMA_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include")

# Find GGML libraries
if (MSVC)
    set(LLAMA_GGML_LIB "${LLAMA_CPP_BUILD_DIR}/ggml/src/Release/ggml.lib")
    set(LLAMA_GGML_BASE_LIB "${LLAMA_CPP_BUILD_DIR}/ggml/src/Release/ggml-base.lib")
else()
    set(LLAMA_GGML_LIB "${LLAMA_CPP_BUILD_DIR}/ggml/src/libggml.a")
    set(LLAMA_GGML_BASE_LIB "${LLAMA_CPP_BUILD_DIR}/ggml/src/libggml-base.a")
endif()

if (EXISTS "${LLAMA_GGML_LIB}" AND EXISTS "${LLAMA_GGML_BASE_LIB}")
    message(STATUS "Found llama.cpp GGML libs for GGUF support")
    set(HAVE_LLAMA_GGUF TRUE)
    add_compile_definitions(ADAPTIVELLM_HAVE_GGUF)
    set(LLAMA_GGUF_LIBS "${LLAMA_GGML_LIB}" "${LLAMA_GGML_BASE_LIB}")
else()
    message(WARNING "llama.cpp GGML libraries not found")
    set(HAVE_LLAMA_GGUF FALSE)
endif()

# Optional safetensors integration
option(ADAPTIVELLM_ENABLE_SAFETENSORS "Enable safetensors reader integration if library is available" OFF)

if (ADAPTIVELLM_ENABLE_SAFETENSORS)
    # Look in SAFETENSORS_ROOT, CMAKE_PREFIX_PATH and common install locations
    find_path(SAFETENSORS_INCLUDE_DIR NAMES safetensors.hh safetensors.hh PATHS ${SAFETENSORS_ROOT} ${CMAKE_PREFIX_PATH} "${CMAKE_INSTALL_PREFIX}" "${CMAKE_SOURCE_DIR}/third_party/safetensors-install/include" NO_DEFAULT_PATH)
    find_library(SAFETENSORS_LIB NAMES safetensors_cpp safetensors_cpp PATHS ${SAFETENSORS_ROOT} ${CMAKE_PREFIX_PATH} "${CMAKE_INSTALL_PREFIX}" "${CMAKE_SOURCE_DIR}/third_party/safetensors-install/lib" NO_DEFAULT_PATH)
    if (SAFETENSORS_INCLUDE_DIR)
        message(STATUS "safetensors headers found: ${SAFETENSORS_INCLUDE_DIR}")
        add_compile_definitions(ADAPTIVELLM_HAVE_SAFETENSORS)
        # Link safetensors into the core targets that may parse models
        foreach(tgt IN ITEMS runtime_contract_test gpu_backend_test run_sample_model custom_loader_test convert_ggup_to_adaptive)
            if (TARGET ${tgt})
                target_include_directories(${tgt} PRIVATE ${SAFETENSORS_INCLUDE_DIR})
                if (SAFETENSORS_LIB)
                    target_link_libraries(${tgt} PRIVATE ${SAFETENSORS_LIB})
                endif()
            endif()
        endforeach()
    else()
        message(WARNING "ADAPTIVELLM_ENABLE_SAFETENSORS requested but safetensors headers not found. Falling back to lightweight parser.")
    endif()
endif()

# Link llama.cpp GGUF libraries into model-loading targets
if (HAVE_LLAMA_GGUF)
    foreach(tgt IN ITEMS runtime_contract_test gpu_backend_test run_sample_model custom_loader_test convert_ggup_to_adaptive)
        if (TARGET ${tgt})
            target_include_directories(${tgt} PRIVATE ${LLAMA_GGML_INCLUDE} ${LLAMA_INCLUDE})
            target_link_libraries(${tgt} PRIVATE ${LLAMA_GGUF_LIBS})
        endif()
    endforeach()
endif()

# Native JNI stubs
add_library(native_shim STATIC
    native/jni_interface.cpp
    native/jni_bridge.cpp
    native/jni_interface.h
    native/jni_bridge.h
)

# Native bridge CLI for JNI/RPC smoke tests
add_executable(runtime_bridge_cli
    native/runtime_bridge_cli.cpp
    native/jni_interface.cpp
)

# Optional CUDA/cuBLAS acceleration
option(ADAPTIVELLM_ENABLE_CUDA "Enable CUDA/cuBLAS acceleration" OFF)
# Auto-detect CUDA toolkit if available
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit: enabling ADAPTIVELLM_ENABLE_CUDA")
    set(ADAPTIVELLM_ENABLE_CUDA ON CACHE BOOL "Enable CUDA/cuBLAS acceleration" FORCE)
endif()

if (ADAPTIVELLM_ENABLE_CUDA)
    message(STATUS "Building with CUDA/cuBLAS support")
    add_compile_definitions(ADAPTIVELLM_HAVE_CUDA)
    foreach(tgt IN ITEMS run_sample_model runtime_contract_test gpu_backend_test generate_benchmark)
        if (TARGET ${tgt})
            # Add CUDA source file to targets
            target_sources(${tgt} PRIVATE runtime/gpu/cuda_executor.cu)
            # Link against CUDA runtime and cuBLAS
            target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cublas)
            # Ensure NVCC flags are reasonable (optional arch flag omitted)
        endif()
    endforeach()

    # NVCC-based cuda_diag target omitted in this build to avoid toolchain detection issues

    # Add a loader-based CUDA diagnostic tool that locates cudart DLL and calls cudaGetDeviceCount
    add_executable(cuda_diag_loader tools/cuda_diag_loader.cpp)
    if (TARGET cuda_diag_loader)
        target_link_libraries(cuda_diag_loader PRIVATE )
    endif()
else()
    message(STATUS "CUDA not enabled; using CPU fallback for GPU executor")
endif()

add_subdirectory(native-engine)

# ============================================================================
# JNI Native Library (Phase 2.1)
# ============================================================================
# Find Java for JNI support
find_package(Java REQUIRED)
find_package(JNI REQUIRED)

# Build the adaptive_scheduler JNI library
add_library(adaptive_scheduler SHARED
    src/main/cpp/adaptive_scheduler.cpp
)

# Link JNI libraries
target_include_directories(adaptive_scheduler PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(adaptive_scheduler PRIVATE)

# Set output directory for the native library
set_target_properties(adaptive_scheduler PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Platform-specific settings
if(MSVC)
    set_target_properties(adaptive_scheduler PROPERTIES SUFFIX ".dll")
    set_target_properties(adaptive_scheduler PROPERTIES PREFIX "")
elseif(APPLE)
    set_target_properties(adaptive_scheduler PROPERTIES SUFFIX ".dylib")
    set_target_properties(adaptive_scheduler PROPERTIES PREFIX "lib")
else()
    set_target_properties(adaptive_scheduler PROPERTIES SUFFIX ".so")
    set_target_properties(adaptive_scheduler PROPERTIES PREFIX "lib")
endif()

message(STATUS "Building Phase 2.1 JNI native library: adaptive_scheduler")
message(STATUS "  JNI_INCLUDE_DIRS: ${JNI_INCLUDE_DIRS}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/lib")

# Ensure runtime executables can find the GGML runtime DLLs when the native build outputs are available.
foreach(tgt IN ITEMS run_sample_model runtime_contract_test gpu_backend_test generate_benchmark)
    if (TARGET ${tgt})
        foreach(dll IN ITEMS ggml-base.dll ggml-cpu.dll ggml.dll)
            if (EXISTS "${CMAKE_SOURCE_DIR}/native-engine/build/bin/Debug/${dll}" OR EXISTS "${CMAKE_SOURCE_DIR}/native-engine/build/bin/Release/${dll}")
                add_custom_command(TARGET ${tgt} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/native-engine/build/bin/Debug/${dll},${CMAKE_SOURCE_DIR}/native-engine/build/bin/Release/${dll}>"
                        "$<TARGET_FILE_DIR:${tgt}>"
                    COMMENT "Copying ${dll} to ${tgt} output directory")
            endif()
        endforeach()
    endif()
endforeach()
