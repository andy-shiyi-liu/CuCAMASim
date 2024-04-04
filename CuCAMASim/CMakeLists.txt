aux_source_directory(. CUCAMASIM_DIR)
add_subdirectory(./util)
add_subdirectory(./function)

add_library(lib_CuCAMASim ${CUCAMASIM_DIR})
target_link_libraries(lib_CuCAMASim lib_util lib_function)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(lib_CuCAMASim PRIVATE /W4 /WX)
else()
    target_compile_options(lib_CuCAMASim PRIVATE -Wall -Wextra -Werror)
endif()