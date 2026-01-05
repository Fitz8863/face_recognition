cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

find_library(INSPIREFACE_LIB
    NAMES InspireFace
    PATHS ${PROJECT_SOURCE_DIR}/3rdparty/InspireFace/lib
    REQUIRED
)

file(GLOB INSPIREFACE_MOD_SOURCES *.cc)
add_library(inspireface_module STATIC ${INSPIREFACE_MOD_SOURCES})

target_include_directories(inspireface_module  
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/3rdparty/InspireFace/include
        ${PROJECT_SOURCE_DIR}/3rdparty
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)



target_link_libraries(inspireface_module PRIVATE
    ${OpenCV_LIBS}
    ${INSPIREFACE_LIB}
    database_module
)

