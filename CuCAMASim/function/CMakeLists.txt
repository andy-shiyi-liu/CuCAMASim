aux_source_directory(. FUNCTION_DIR)

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

add_library(lib_function ${FUNCTION_DIR})

# Linking libraries
target_link_libraries(lib_function PUBLIC yaml-cpp::yaml-cpp)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(lib_function PRIVATE /W4 /WX)
else()
    target_compile_options(lib_function PRIVATE -Wall -Wextra -Werror)
endif()