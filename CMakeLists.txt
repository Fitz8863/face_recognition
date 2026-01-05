cmake_minimum_required(VERSION 3.10)
project(FaceComparison)

set(USE_AVX_INSTRUCTIONS ON)
set(USE_SSE4_INSTRUCTIONS ON)

# if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
# 	set(CMAKE_BUILD_TYPE Release)
# endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED)

option(USE_DLIB "Use Dlib backend" OFF)
option(USE_OPENCV "Use OpenCV backend" OFF)
option(USE_INSPIREFACE "Use InspireFace backend" OFF)

# 如果都没有选择，设置默认为 InspireFace
if(NOT USE_DLIB AND NOT USE_OPENCV AND NOT USE_INSPIREFACE)
    message(WARNING "No backend selected, defaulting to InspireFace backend")
    # 使用 FORCE 确保变量被强制写入缓存并立即生效
    set(USE_INSPIREFACE ON CACHE BOOL "Use InspireFace backend" FORCE)
endif()

# 必须先添加被依赖的模块
add_subdirectory(src/database)

# 再添加依赖别人的模块
if (USE_DLIB)
    add_subdirectory(src/dlib)
endif()

if (USE_OPENCV)
    add_subdirectory(src/opencv)
endif()

if (USE_INSPIREFACE)
    add_subdirectory(src/inspireface)
endif()


add_executable(face
	main.cc
	src/FaceRecognizer.cc
)  

target_include_directories(face
	PRIVATE
	${PROJECT_SOURCE_DIR}/src
	${PROJECT_SOURCE_DIR}/3rdparty
	${PROJECT_SOURCE_DIR}/3rdparty/InspireFace/include
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(face
    PRIVATE
    ${OpenCV_LIBS}
)  


if (USE_DLIB)
    target_compile_definitions(face PRIVATE FACE_BACKEND_DLIB)
    target_link_libraries(face PRIVATE dlib_module)
endif()

if (USE_OPENCV)
    target_compile_definitions(face PRIVATE FACE_BACKEND_OPENCV)
    target_link_libraries(face PRIVATE opencv_module)
endif()

if (USE_INSPIREFACE)
    target_compile_definitions(face PRIVATE FACE_BACKEND_INSPIREFACE)
    target_link_libraries(face PRIVATE inspireface_module)
endif()


