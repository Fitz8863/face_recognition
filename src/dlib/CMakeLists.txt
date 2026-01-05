cmake_minimum_required(VERSION 3.10)

find_package(dlib REQUIRED)

find_package(OpenCV REQUIRED)

file(GLOB DLIB_MOD_SOURCES *.cc)
add_library(dlib_module STATIC ${DLIB_MOD_SOURCES})

target_include_directories(dlib_module    
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/3rdparty
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dlib_module PRIVATE
    dlib::dlib
    ${OpenCV_LIBS}
    database_module
)
