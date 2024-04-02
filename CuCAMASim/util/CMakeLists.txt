aux_source_directory(. UTIL_DIR)

add_library(lib_util ${UTIL_DIR})

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(lib_util PRIVATE /W4 /WX)
else()
    target_compile_options(lib_util PRIVATE -Wall -Wextra -Werror)
endif()