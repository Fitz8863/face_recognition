cmake_minimum_required(VERSION 3.10)

file(GLOB DATABASE_SOURCES *.cc)

# 查找 SQLite3
find_package(SQLite3 REQUIRED)

find_package(OpenCV REQUIRED)

# 定义一个库名为 database_lib
add_library(database_module STATIC ${DATABASE_SOURCES})


target_include_directories(database_module    
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/3rdparty
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(database_module PRIVATE SQLite::SQLite3)
