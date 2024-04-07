find_package(CUDAToolkit)
aux_source_directory(. DECISIONTREE_DIR)
set(CMAKE_CUDA_STANDARD 17)
add_library(lib_DecisionTree ${DECISIONTREE_DIR})


# Adjusting C/CPP flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# Adjusting CUDA flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -Xptxas -O0 -lineinfo -O0 -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set_target_properties(lib_CuCAMASim PROPERTIES CUDA_ARCHITECTURES 61)
# Linking libraries
# target_link_libraries(lib_CuCAMASim PRIVATE CUDA::cudart)
# target_link_libraries(lib_CuCAMASim PUBLIC lib_***)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(lib_CuCAMASim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(lib_CuCAMASim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(lib_CuCAMASim PRIVATE /W4 /WX)
else()
    target_compile_options(lib_CuCAMASim PRIVATE -Wall -Wextra -Werror)
endif()