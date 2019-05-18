cmake_minimum_required(VERSION 3.13)
project(GNMethod)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(GNMethod main.cpp hw3_gn.h Solverxxxx.cpp Solverxxxx.h ResidualFunctionxxxx.cpp ResidualFunctionxxxx.h)
target_link_libraries(GNMethod ${OpenCV_LIBS})