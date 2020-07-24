cmake_minimum_required(VERSION 2.6)

project(DRL_robot_exploration)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
include_directories(/usr/include/eigen3)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Force to use Python 2.7/3.5
    # set(PYTHON_EXECUTABLE /usr/local/bin/python)
    set(PYTHON_EXECUTABLE /usr/local/bin/python3)
else ()
    # set(PYTHON_EXECUTABLE /usr/bin/python)
    set(PYTHON_EXECUTABLE /usr/bin/python3)
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

find_package(pybind11 REQUIRED)

pybind11_add_module(inverse_sensor_model src/inverse_sensor_model.cpp)
pybind11_add_module(astar src/astar.cpp)
