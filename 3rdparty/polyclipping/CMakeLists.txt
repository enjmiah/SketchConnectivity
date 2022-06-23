project(polyclipping)

set(HEADER_FILES
    polyclipping/clipper.hpp
)
set(SOURCE_FILES
    polyclipping/clipper.cpp
)

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC .)
if (UNIX AND NOT APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()
