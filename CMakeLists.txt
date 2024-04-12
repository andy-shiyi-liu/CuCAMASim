cmake_minimum_required(VERSION 3.0.0)
project(CuCAMASim_runner VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

include(CTest)
enable_testing()

set(CMAKE_CUDA_STANDARD 17)
# Set C++ standard to C++17 or later for std::filesystem support
set(CMAKE_CXX_STANDARD 17)

# Define the flag
# add_definitions(-D__MAIN__)

# Include header files from specific directories
include_directories(
    ${CMAKE_SOURCE_DIR}/CuCAMASim/include/
    ${CMAKE_SOURCE_DIR}/applications/DecisionTree/
)

add_subdirectory(./CuCAMASim)
add_subdirectory(./applications/DecisionTree)

add_executable(CuCAMASim_runner main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Adjusting CUDA flags
set_target_properties(CuCAMASim_runner PROPERTIES CUDA_ARCHITECTURES 61)

# Linking libraries
target_link_libraries(CuCAMASim_runner PRIVATE CUDA::cudart)
target_link_libraries(CuCAMASim_runner PUBLIC CuCAMASim DecisionTree)

# Link against filesystem library if using GCC version < 9
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(CuCAMASim_runner PRIVATE stdc++fs)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CuCAMASim_runner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(CuCAMASim_runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(CuCAMASim_runner PRIVATE /W4 /WX)
else()
    target_compile_options(CuCAMASim_runner PRIVATE -Wall -Wextra -Werror)
endif()
