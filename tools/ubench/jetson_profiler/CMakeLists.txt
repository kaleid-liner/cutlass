cmake_minimum_required(VERSION 3.13)
project(jetson_profiler)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_library(jetson_profiler SHARED
  ${PROJECT_SOURCE_DIR}/jetson_profiler.cc
)
target_include_directories(jetson_profiler PUBLIC 
  ${PROJECT_SOURCE_DIR}/
)
