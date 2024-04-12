find_package(CUDAToolkit)
aux_source_directory(. FUNCTION_DIR)
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

add_library(function ${FUNCTION_DIR})

# Adjusting CUDA flags
set_target_properties(function PROPERTIES CUDA_ARCHITECTURES 61)
add_subdirectory(./cuda)

# Linking libraries
target_link_libraries(function PRIVATE CUDA::cudart)
target_link_libraries(function PUBLIC yaml-cpp::yaml-cpp function_cuda)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(function PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(function PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(function PRIVATE /W4 /WX)
else()
    target_compile_options(function PRIVATE -Wall -Wextra -Werror)
endif()