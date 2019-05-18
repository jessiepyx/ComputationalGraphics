cmake_minimum_required(VERSION 3.13)
project(ImageFilters)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageFilters main.cpp)
target_link_libraries(ImageFilters ${OpenCV_LIBS})