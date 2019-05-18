cmake_minimum_required(VERSION 3.13)
project(GSMethod)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(GSMethod main.cpp MySparseMatrix.h GaussSeidel.h)
target_link_libraries(GSMethod ${OpenCV_LIBS})