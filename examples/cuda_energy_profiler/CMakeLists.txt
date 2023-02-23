cmake_minimum_required(VERSION 3.13)
project(cuda_energy_profiler)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set (CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(USE_CUDA)

option(ENABLE_THREAD_POOL "Enable threading pool based on ctpl library" OFF)
if(ENABLE_THREAD_POOL)
    add_library(cuda_energy_profiler ${PROJECT_SOURCE_DIR}/with_ctpl_thread_pool/cuda_energy_profiler.cc)
    target_include_directories(cuda_energy_profiler PUBLIC ${PROJECT_SOURCE_DIR}/with_ctpl_thread_pool/)
else()
    add_library(cuda_energy_profiler ${PROJECT_SOURCE_DIR}/no_thread_pool/cuda_energy_profiler.cc)
    target_include_directories(cuda_energy_profiler PUBLIC ${PROJECT_SOURCE_DIR}/no_thread_pool/)
endif(ENABLE_THREAD_POOL)

target_link_libraries(cuda_energy_profiler nvidia-ml pthread cudart)
