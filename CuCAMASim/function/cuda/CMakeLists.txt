find_package(CUDAToolkit)
aux_source_directory(. FUNCTION_CUDA_DIR)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)
# yaml package
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

add_library(lib_function_cuda ${FUNCTION_CUDA_DIR})

set_target_properties(lib_function_cuda PROPERTIES CUDA_ARCHITECTURES 61)

# Linking libraries
target_link_libraries(lib_function_cuda PRIVATE CUDA::cudart)
target_link_libraries(lib_function_cuda PUBLIC yaml-cpp::yaml-cpp)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(lib_function_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(lib_function_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)