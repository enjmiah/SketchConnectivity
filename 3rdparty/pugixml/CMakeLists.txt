project(pugixml)

set(HEADER_FILES
    pugiconfig.hpp
    pugixml.hpp
)
set(SOURCE_FILES
    pugixml.cpp
)

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC .)
if (UNIX AND NOT APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()
