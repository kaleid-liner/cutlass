cmake_minimum_required(VERSION 3.13)
project(cuda_energy_profiler)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set (CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

option(ENABLE_THREAD_POOL "Enable threading pool based on ctpl library" ON)
if(ENABLE_THREAD_POOL)
    add_compile_definitions(USE_CTPL_THREAD_POOL)
endif(ENABLE_THREAD_POOL)

add_library(cuda_energy_profiler SHARED
    ${PROJECT_SOURCE_DIR}/cuda_energy_profiler.cc
    ${PROJECT_SOURCE_DIR}/timer.cc
)
target_include_directories(cuda_energy_profiler PUBLIC 
    ${PROJECT_SOURCE_DIR}/
    /usr/local/cuda/include
)
target_link_libraries(cuda_energy_profiler PRIVATE nvidia-ml pthread)
