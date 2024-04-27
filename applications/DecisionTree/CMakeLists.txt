find_package(CUDAToolkit)
aux_source_directory(. DECISIONTREE_DIR)
set(CMAKE_CUDA_STANDARD 17)
add_library(DecisionTree ${DECISIONTREE_DIR})

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

# Adjusting C/CPP flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# Adjusting CUDA flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -Xptxas -O0 -lineinfo -O0 -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set_target_properties(DecisionTree PROPERTIES CUDA_ARCHITECTURES 61)

include_directories(/usr/local/include)
target_link_libraries(DecisionTree PUBLIC yaml-cpp::yaml-cpp)

# setting debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(DecisionTree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(DecisionTree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(DecisionTree PRIVATE /W4 /WX)
else()
    target_compile_options(DecisionTree PRIVATE -Wall -Wextra -Werror)
endif()