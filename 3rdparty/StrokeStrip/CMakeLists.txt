cmake_minimum_required(VERSION 3.3)

project(StrokeStrip)

set(CMAKE_CXX_STANDARD 14)

set(HEADER_FILES
    src/StrokeStrip/Cluster.h
    src/StrokeStrip/Context.h
    src/StrokeStrip/FittingEigenSparse.h
    src/StrokeStrip/Parameterization.h
    src/StrokeStrip/Polyline2D.h
    src/StrokeStrip/SketchInfo.h
    src/StrokeStrip/StrokeCutting.h
    src/StrokeStrip/StrokeOrientation.h
    src/StrokeStrip/SvgUtils.h
    src/StrokeStrip/Utils.h)
set(SOURCE_FILES
    src/StrokeStrip/Cluster.cpp
    src/StrokeStrip/Context.cpp
    src/StrokeStrip/FittingEigenSparse.cpp
    src/StrokeStrip/Parameterization.cpp
    src/StrokeStrip/StrokeCutting.cpp
    src/StrokeStrip/StrokeOrientation.cpp
    src/StrokeStrip/Utils.cpp)

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(GUROBI REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${GUROBI_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${GUROBI_LIBRARY})
target_link_libraries(${PROJECT_NAME} PUBLIC optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

find_package(glm CONFIG REQUIRED)
if (TARGET glm::glm)
    target_link_libraries(${PROJECT_NAME} PUBLIC glm::glm)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC glm)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Cornucopia)
