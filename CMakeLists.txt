cmake_minimum_required(VERSION 3.15)

project(sketching)

option(WITH_GUROBI "Include features which depend on Gurobi (requires Gurobi to be installed)" ON)
option(SPDLOG_HEADER_ONLY "Use the header-only version of spdlog" OFF)
option(ASAN "Enable address sanitizer" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS src/*.h)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp)
set(NATVIS_FILES src/sketching/sketching.natvis)

if (MSVC)
    string(REPLACE "/Ob0" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

add_library(${PROJECT_NAME}
            ${HEADER_FILES} ${SOURCE_FILES} ${NATVIS_FILES})
source_group(TREE ${PROJECT_SOURCE_DIR}/src FILES ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src)
set_property(TARGET ${PROJECT_NAME} PROPERTY PROJECT_LABEL lib${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
if (MSVC)
    target_compile_options(
        ${PROJECT_NAME} PRIVATE
        /W4 # Warning level
        /we4834 # ignoring nodiscard is an error
        )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
                               _CRT_SECURE_NO_WARNINGS
                               _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:DEBUG>:/Ob1>)
    if (ASAN)
        target_compile_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
    endif()
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wno-ignored-qualifiers)
    if (ASAN)
        target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
    endif()
endif()
if (UNIX AND NOT APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)
endif()
target_precompile_headers(${PROJECT_NAME} PRIVATE
    <Eigen/Core>
    <nonstd/span.hpp>
    <spdlog/spdlog.h>
    <algorithm>
    <map>
    <memory>
    <set>
    <string>
    <unordered_map>
    <vector>)

find_package(Eigen3 3.3.7 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)

find_package(spdlog CONFIG REQUIRED)
if (SPDLOG_HEADER_ONLY)
    target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog_header_only)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)
endif()

if (WITH_GUROBI)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_GUROBI)

    find_package(StrokeStrip QUIET)
    if (NOT StrokeStrip_FOUND)
        add_subdirectory(3rdparty/Cornucopia) # Dependency of StrokeStrip.
        set_target_properties(Cornucopia PROPERTIES FOLDER "3rdparty")

        add_subdirectory(3rdparty/StrokeStrip)
        set_target_properties(StrokeStrip PROPERTIES FOLDER "3rdparty")
    endif()
    target_link_libraries(${PROJECT_NAME} PUBLIC StrokeStrip)
endif()

add_subdirectory(3rdparty/polyclipping)
set_target_properties(polyclipping PROPERTIES FOLDER "3rdparty")
target_link_libraries(${PROJECT_NAME} PUBLIC polyclipping)

add_subdirectory(3rdparty/GraphicsGems)
set_target_properties(GraphicsGems PROPERTIES FOLDER "3rdparty")
target_link_libraries(${PROJECT_NAME} PUBLIC GraphicsGems)

add_subdirectory(3rdparty/span-lite)
target_link_libraries(${PROJECT_NAME} PUBLIC span_lite)
target_compile_definitions(${PROJECT_NAME} PUBLIC span_CONFIG_NO_EXCEPTIONS)

add_subdirectory(3rdparty/Triangle)
set_target_properties(Triangle PROPERTIES FOLDER "3rdparty")
target_link_libraries(${PROJECT_NAME} PUBLIC Triangle)

find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)

add_subdirectory(3rdparty/nlohmann_json)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)

find_package(Cairo QUIET)
if (Cairo_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Cairo::Cairo)
else()
    # Try to use the vcpkg version of Cairo.
    find_package(unofficial-cairo CONFIG QUIET)
    if (unofficial-cairo_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC unofficial::cairo::cairo)
    else()
        # No Cairo, so trigger an error.
        find_package(Cairo REQUIRED)
    endif()
endif()

find_package(pugixml CONFIG QUIET)
if (NOT pugixml_FOUND)
    add_subdirectory(3rdparty/pugixml)
    set_target_properties(pugixml PROPERTIES FOLDER "3rdparty")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE pugixml)

# We require a version of stb_printf with nothings/stb#613 merged.
# So we specify NO_DEFAULT_PATH to avoid including a potentially non-libc-
# compliant snprintf implementation.
find_path(STB_INCLUDE_DIRS "stb_sprintf.h" HINTS "3rdparty/stb" NO_DEFAULT_PATH)
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

add_subdirectory(3rdparty/mapbox)
target_link_libraries(${PROJECT_NAME} PUBLIC mapbox)

add_subdirectory(python/core)
