cmake_minimum_required(VERSION 3.15)
project(adaptive_engine_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(adaptive_engine SHARED adaptive_engine_shim.cpp)

# Ensure symbol is exported on Unix
if (UNIX)
    set_target_properties(adaptive_engine PROPERTIES PUBLIC_HEADER "adaptive_engine_shim.h")
    target_compile_options(adaptive_engine PRIVATE -fvisibility=hidden)
endif()

# Install to lib output dir
set_target_properties(adaptive_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
