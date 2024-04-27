aux_source_directory(. UTIL_DIR)

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

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_library(util ${UTIL_DIR})
target_link_libraries(util PUBLIC yaml-cpp::yaml-cpp matio)

# Treat warnings as errors for C and C++ files
if(MSVC)
    target_compile_options(util PRIVATE /W4 /WX)
else()
    target_compile_options(util PRIVATE -Wall -Wextra -Werror)
endif()