cmake_minimum_required(VERSION 3.18)
project(adaptive_engine_llama CXX)
find_package(JNI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable CUDA for now to avoid dependency issues
# Users can rebuild with GGML_CUDA=ON if they have CUDA toolkit
option(ADAPTIVELLM_ENABLE_CUDA "Enable CUDA/cuBLAS acceleration for llama.cpp" OFF)

# Build llama.cpp submodule and link against its library if available
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    # Disable llama.cpp examples and tests to speed up build
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server" FORCE)
    
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" "${CMAKE_BINARY_DIR}/llama_build")
    
    # Create shared library with static linking to avoid DLL dependencies
    add_library(adaptive_engine_llama SHARED llama_wrapper.cpp native_jni.cpp)
    target_include_directories(adaptive_engine_llama PRIVATE ${JNI_INCLUDE_DIRS})
    target_include_directories(adaptive_engine_llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include")
    
    # Link llama and ggml statically into the DLL to avoid dependency chain
    if (MSVC)
        # On Windows with MSVC, statically link object files
        target_link_libraries(adaptive_engine_llama PRIVATE llama ggml::ggml)
        set_target_properties(adaptive_engine_llama PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
        )
    else()
        target_link_libraries(adaptive_engine_llama PRIVATE llama ggml::ggml)
    endif()
    
    target_compile_definitions(adaptive_engine_llama PRIVATE HAVE_LLAMA=1)
else()
    add_library(adaptive_engine_llama SHARED llama_wrapper.cpp native_jni.cpp)
    target_include_directories(adaptive_engine_llama PRIVATE ${JNI_INCLUDE_DIRS})
endif()

# Install into native-engine build output dir
set_target_properties(adaptive_engine_llama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PREFIX "adaptive_engine"
    SUFFIX ".dll"
)
