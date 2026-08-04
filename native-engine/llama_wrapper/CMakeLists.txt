cmake_minimum_required(VERSION 3.18)
project(adaptive_engine_llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(adaptive_engine_llama SHARED llama_wrapper.cpp)

# Install into native-engine build output dir
set_target_properties(adaptive_engine_llama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
