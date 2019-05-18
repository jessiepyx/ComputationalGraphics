cmake_minimum_required(VERSION 3.13)
project(Panorama)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Panorama main.cpp my_pa.cpp my_pa.h hw6_pa.h)
target_link_libraries(Panorama ${OpenCV_LIBS})