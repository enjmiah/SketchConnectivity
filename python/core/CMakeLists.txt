project(sketching-python)

if(NOT PYBIND11_DIR)
    set(PYBIND11_DIR "../../3rdparty/pybind11")
endif()
add_subdirectory("${PYBIND11_DIR}" pybind11)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS *.h)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)

pybind11_add_module(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
# So we can do `import _sketching` from Python.
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME _sketching)
target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
if (MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_SNPRINTF)
    target_compile_options(
        ${PROJECT_NAME} PRIVATE
        /W4 # Warning level
        /we4834 # ignoring nodiscard is an error
        )
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:DEBUG>:/Ob1>)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wno-ignored-qualifiers)
endif()
target_precompile_headers(${PROJECT_NAME} PRIVATE
    <pybind11/eigen.h>
    <pybind11/pybind11.h>
    <pybind11/stl.h>
    <Eigen/Core>
    <nonstd/span.hpp>
    <spdlog/spdlog.h>
    <algorithm>
    <map>
    <memory>
    <set>
    <string>
    <unordered_map>
    <unordered_set>
    <vector>)

target_link_libraries(${PROJECT_NAME} PRIVATE sketching)
