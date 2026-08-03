#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "safetensors-cpp::safetensors_cpp" for configuration "Release"
set_property(TARGET safetensors-cpp::safetensors_cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(safetensors-cpp::safetensors_cpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/safetensors_cpp.lib"
  )

list(APPEND _cmake_import_check_targets safetensors-cpp::safetensors_cpp )
list(APPEND _cmake_import_check_files_for_safetensors-cpp::safetensors_cpp "${_IMPORT_PREFIX}/lib/safetensors_cpp.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
