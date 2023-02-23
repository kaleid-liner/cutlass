#include "cuda_energy_profiler.h"

namespace onnxruntime {

namespace profiling {

//////////////////////////////////////////////////////////////////////////////
// Timer.cpp
// =========
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system 
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2003-01-13
// UPDATED: 2017-03-30
//
// Copyright (c) 2003 Song Ho Ahn
//////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// constructor
///////////////////////////////////////////////////////////////////////////////
Timer::Timer()
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceFrequency(&frequency);
    startCount.QuadPart = 0;
    endCount.QuadPart = 0;
#else
    startCount.tv_sec = startCount.tv_usec = 0;
    endCount.tv_sec = endCount.tv_usec = 0;
#endif

    stopped = 0;
    startTimeInMicroSec = 0;
    endTimeInMicroSec = 0;
}

///////////////////////////////////////////////////////////////////////////////
// distructor
///////////////////////////////////////////////////////////////////////////////
Timer::~Timer()
{
}

///////////////////////////////////////////////////////////////////////////////
// start timer.
// startCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::start()
{
    stopped = 0; // reset stop flag
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&startCount);
#else
    gettimeofday(&startCount, NULL);
#endif
}

///////////////////////////////////////////////////////////////////////////////
// stop the timer.
// endCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::stop()
{
    stopped = 1; // set timer stopped flag

#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&endCount);
#else
    gettimeofday(&endCount, NULL);
#endif
}

///////////////////////////////////////////////////////////////////////////////
// compute elapsed time in micro-second resolution.
// other getElapsedTime will call this first, then convert to correspond resolution.
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMicroSec()
{
#if defined(WIN32) || defined(_WIN32)
    if(!stopped)
        QueryPerformanceCounter(&endCount);

    startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
    endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
#else
    if(!stopped)
        gettimeofday(&endCount, NULL);

    startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
    endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
#endif

    return endTimeInMicroSec - startTimeInMicroSec;
}

///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMilliSec()
{
    return this->getElapsedTimeInMicroSec() * 0.001;
}

///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInSec()
{
    return this->getElapsedTimeInMicroSec() * 0.000001;
}

///////////////////////////////////////////////////////////////////////////////
// same as getElapsedTimeInSec()
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTime()
{
    return this->getElapsedTimeInSec();
}

}  // namespace profiling
}  // namespace onnxruntime

#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

namespace onnxruntime {

namespace profiling {

GPUInspector& GPUInspector::Instance()
{
    static GPUInspector instance;
    return instance;
}

bool GPUInspector::Init(int gpu_id, double sampling_interval)
{
    if(pthread_inspect_)
    {
        running_inspect_ = false;
        pthread_inspect_->join();
        pthread_inspect_ = nullptr;
        recordings_.clear();
    }

    // get device handle
    devices_.clear();
    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    if(gpu_id < 0)
    {
        for(unsigned int i = 0; i < deviceCount; i++)
        {
            devices_[i] = 0;
        }
    }
    else if(gpu_id < deviceCount)
    {
        devices_[gpu_id] = 0;
    }
    else
    {
        std::cerr << "Error: invalid gpu_id for initialization." << std::endl;
        return false;
    }

    recordings_.clear();
    for(auto& it : devices_)
    {
        nvmlDeviceGetHandleByIndex(it.first, &it.second);
        // recordings_[it.first] = std::vector<GPUInfo_t>();
    }

    initialized_ = true;
    sampling_interval_micro_second_ = sampling_interval * 1000000;

    std::cout << "GPUInspector initialized, numDevices = " << deviceCount << ", sampling interval = " << sampling_interval << std::endl;
    std::cout << "Inspected GPU Id:";
    for(auto& it : devices_)
    {
        std::cout << " " << it.first;
    }
    std::cout << std::endl;

    return true;
}

GPUInfo_t GPUInspector::GetGPUInfo(const nvmlDevice_t& device)
{
    // CheckInit();
    GPUInfo_t gpu_info;
    gpu_info.time_stamp = 0.0;
    // get momory info
    nvmlMemory_t mem_info;
    nvmlDeviceGetMemoryInfo(device, &mem_info);
    gpu_info.used_memory_percent = static_cast<double>(mem_info.used) / static_cast<double>(mem_info.total) * 100.0;
    // get power usage
    unsigned int power_mw = 0;
    nvmlDeviceGetPowerUsage(device, &power_mw);
    gpu_info.power_watt = static_cast<double>(power_mw) / 1000.0;
    // get temperature
    unsigned int temp;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    gpu_info.temperature = static_cast<double>(temp);
    // get utils
    nvmlUtilization_t utils;
    nvmlDeviceGetUtilizationRates(device, &utils);
    gpu_info.memory_util = static_cast<double>(utils.memory);
    gpu_info.gpu_util = static_cast<double>(utils.gpu);
    return gpu_info;
}

GPUInfo_t GPUInspector::GetGPUInfo(unsigned int gpu_id)
{
    CheckInit();
    if(devices_.count(gpu_id) == 0)
    {
        throw std::runtime_error("Invalid gpu_id while getting GPU Info.");
    }
    return GetGPUInfo(devices_.at(gpu_id));
}

void GPUInspector::StartInspect()
{
    CheckInit();
    cudaDeviceSynchronize();
    pthread_inspect_ = std::shared_ptr<std::thread>(
        new std::thread(&GPUInspector::Run, this));
    while(!running_inspect_)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds(5));
    }
}

void GPUInspector::StopInspect()
{
    if(!running_inspect_)
    {
        throw std::runtime_error("GPUInspector not started while requested to stop.");
    }
    cudaDeviceSynchronize();
    running_inspect_ = false;
    pthread_inspect_->join();
    pthread_inspect_ = nullptr;
}

void GPUInspector::ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    if(recordings_.count(gpu_id))
    {
        readings = recordings_.at(gpu_id);
    }
    else
    {
        throw std::runtime_error("Invalid gpu_id while exporting readings.");
    }
}

void GPUInspector::ExportAllReadings(std::unordered_map<int, std::vector<GPUInfo_t>>& all_readings) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    all_readings = recordings_;
}

unsigned int GPUInspector::NumDevices() const
{
    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    return deviceCount;
}

unsigned int GPUInspector::NumInspectedDevices() const
{
    CheckInit();
    return devices_.size();
}

void GPUInspector::InspectedDeviceIds(std::vector<unsigned int>& device_ids) const
{
    CheckInit();
    device_ids.clear();
    for(const auto& it : devices_)
    {
        device_ids.push_back(it.first);
    }
}

double GPUInspector::CalculateEnergy(const std::vector<GPUInfo_t>& readings)
{
    double result = 0.0;
    for(size_t i = 1; i < readings.size(); i++)
    {
        double dt = readings[i].time_stamp - readings[i - 1].time_stamp;
        result += 0.5 * (readings[i - 1].power_watt + readings[i].power_watt) * dt;
    }
    return result;
}

double GPUInspector::CalculateEnergy(unsigned int gpu_id) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    if(devices_.count(gpu_id) == 0)
    {
        throw std::runtime_error("Invalid gpu_id while calculating energy.");
    }
    return CalculateEnergy(recordings_.at(gpu_id));
}

void GPUInspector::CalculateEnergy(std::vector<double>& energies) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    energies.clear();
    for(const auto& it : recordings_)
    {
        energies.push_back(CalculateEnergy(it.second));
    }
}

double GPUInspector::GetDurationInSec()
{
    return timer_.getElapsedTimeInSec();
}

GPUInspector::GPUInspector(bool auto_init)
{
    nvmlInit();

    initialized_ = false;
    running_inspect_ = false;
    loop_repeat_ = 100000;
    sampling_interval_micro_second_ = 0.05 * 1000000;
    pthread_inspect_ = nullptr;

    if(auto_init)
    {
        Init(0, 0.01);
    }
}

GPUInspector::~GPUInspector()
{
    nvmlShutdown();
}

inline void GPUInspector::CheckInit() const
{
    if(!initialized_)
    {
        throw std::runtime_error("GPUInspector not initialized.");
    }
}

void GPUInspector::Run()
{
    recordings_.clear();
    
    timer_.start();
    running_inspect_ = true;

    Timer local_timer;
    while(running_inspect_)
    {
        local_timer.start();

        // get readings
        for(const auto& it : devices_)
        {
            GPUInfo_t info = GetGPUInfo(it.second);
            info.time_stamp = timer_.getElapsedTimeInSec();
            recordings_[it.first].push_back(info);
        }
        
        local_timer.stop();
        int sleep_time = static_cast<int>(sampling_interval_micro_second_ - local_timer.getElapsedTimeInMicroSec());
        if(sleep_time > 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
        }
        // else
        // {
        //     std::cerr << "Exceeded sampling interval, time cost = " << local_timer.getElapsedTimeInSec() << std::endl;
        // }
    }
    timer_.stop();
}

}  // namespace profiling
}  // namespace onnxruntime
