# cmake_minimum_required(VERSION 3.10)

# project(camera)

find_package(OpenCV 4.5.4 REQUIRED)

set(CAMERA_INCLUDE_DIR ${OpenCV_INCLUDE_DIRS} ./MVSDK ./)

include_directories(CAMERA_INCLUDE_DIR)

add_library(camera MindVision.cpp)

target_link_libraries(camera ${OpenCV_LIBS})

target_link_libraries(camera libMVSDK.so)
