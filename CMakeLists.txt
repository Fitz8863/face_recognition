cmake_minimum_required(VERSION 3.10)
project(FaceComparison)

set(USE_AVX_INSTRUCTIONS ON)
set(USE_SSE4_INSTRUCTIONS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 SQLite3
find_package(SQLite3 REQUIRED)

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})


file(GLOB DLIB_SOURCES
    src/dlib/*.cc
)

file(GLOB OPENCV_SOURCES 
  src/opencv/*.cc)  

file(GLOB SQLITE_SOURCES
  src/database/*.cc)

add_executable(face
  main.cc
  src/FaceRecognizer.cc
  ${OPENCV_SOURCES}
  ${SQLITE_SOURCES}
  ${DLIB_SOURCES}
)

target_include_directories(face
PRIVATE 
	${OpenCV_INCLUDE_DIRS}
  ${dlib_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(face
    PRIVATE
        ${OpenCV_LIBS}
        ${dlib_LIBRARIES}
        SQLite::SQLite3
)
