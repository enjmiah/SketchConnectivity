# Cornucopia library CMakeLists.txt

FILE(GLOB Cornucopia_CPP "*.cpp")
FILE(GLOB Cornucopia_H "*.h")
LIST(APPEND Cornucopia_Sources ${Cornucopia_CPP} ${Cornucopia_H})

ADD_LIBRARY( Cornucopia STATIC ${Cornucopia_Sources} )
TARGET_INCLUDE_DIRECTORIES(Cornucopia PUBLIC ..)
if (UNIX AND NOT APPLE)
    target_compile_options(Cornucopia PRIVATE "-fPIC")
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
target_precompile_headers(${PROJECT_NAME} PRIVATE
    <Eigen/Core>
    <Eigen/Cholesky>
    <iostream>
    <map>
    <string>
    <vector>)

find_package(Eigen3 3.3.7 CONFIG REQUIRED)
target_link_libraries(Cornucopia PUBLIC Eigen3::Eigen)

INSTALL( TARGETS Cornucopia ARCHIVE DESTINATION lib )

INSTALL( DIRECTORY . DESTINATION include FILES_MATCHING PATTERN "*.h"  )
