cmake_minimum_required(VERSION 3.18)
project(minirepro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(minimal_allocator_repro
    ../tests/MinimalAllocatorRepro.cpp
    ../cpp/RuntimeMemory.cpp
    ../cpp/AllocationTracker.cpp
    ../cpp/GPUMemoryPool.cpp
)

target_include_directories(minimal_allocator_repro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(minimal_allocator_repro PRIVATE CUDA::cudart)
