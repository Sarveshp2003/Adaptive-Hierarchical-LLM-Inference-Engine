cmake_minimum_required(VERSION 3.18)
project(adaptive_native_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA requirement retained for other native-engine components
find_package(CUDAToolkit REQUIRED)

add_subdirectory(minirepro)

# Add a lightweight shim library that exposes adaptive_engine_get_api
# This shim is built for CI to validate Phase 2.2 wiring and falls back to
# a simple mock implementation.
add_subdirectory(shim)

# Optionally build llama.cpp wrapper as a real-engine backend for Phase 2.2
add_subdirectory(llama_wrapper)


