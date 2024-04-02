cmake_minimum_required(VERSION 3.0.0)
project(CuCAMASim VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 11)
include_directories(./../ref/include/)

add_executable(CuCAMASim CuCAMASim.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -Xcompiler -O0 -Xptxas -O0 -lineinfo -O0 -Xcompiler=-Werror")
# For CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set_target_properties(CuCAMASim PROPERTIES CUDA_ARCHITECTURES 61)

target_link_libraries(CuCAMASim PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CuCAMASim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(CuCAMASim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
