cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

file(GLOB OPENCV_MOD_SOURCES *.cc)
add_library(opencv_module STATIC ${OPENCV_MOD_SOURCES})


target_include_directories(opencv_module  
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/3rdparty
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(opencv_module PRIVATE
    ${OpenCV_LIBS}
    database_module
)