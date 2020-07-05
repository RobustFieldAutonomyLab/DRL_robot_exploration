cmake_minimum_required(VERSION 2.6)

project(exploration)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
include_directories(/usr/include/eigen3)

set(PYTHON_EXECUTABLE /usr/bin/python)

find_package(pybind11 REQUIRED)

pybind11_add_module(inverse_sensor_model src/inverse_sensor_model.cpp)
