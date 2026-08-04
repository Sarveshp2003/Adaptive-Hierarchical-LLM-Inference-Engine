cmake_minimum_required(VERSION 3.18)
project(adaptive_engine_llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build llama.cpp submodule and link against its library if available
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" "${CMAKE_BINARY_DIR}/llama_build")
    add_library(adaptive_engine_llama SHARED llama_wrapper.cpp)
    # Link against llama target provided by llama.cpp (llama) and ggml
    target_link_libraries(adaptive_engine_llama PRIVATE llama ggml::ggml)
    target_include_directories(adaptive_engine_llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include")
    # Define HAVE_LLAMA for conditional compilation in the wrapper
    target_compile_definitions(adaptive_engine_llama PRIVATE HAVE_LLAMA=1)
else()
    add_library(adaptive_engine_llama SHARED llama_wrapper.cpp)
endif()

# Install into native-engine build output dir
set_target_properties(adaptive_engine_llama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
