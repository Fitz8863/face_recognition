cmake_minimum_required(VERSION 3.10)
project(FaceComparison)

set(USE_AVX_INSTRUCTIONS ON)
set(USE_SSE4_INSTRUCTIONS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

# 查找 SQLite3
find_package(SQLite3 REQUIRED)

include_directories(
    ${dlib_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(face_test 
src/main.cpp 
src/FaceCode.cc 
src/FaceDatabase.cc 
src/FaceManager.cc)

target_link_libraries(face_test
    PRIVATE
        ${dlib_LIBRARIES}
        ${OpenCV_LIBS}
        SQLite::SQLite3
)
