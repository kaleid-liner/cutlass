#pragma once
#ifndef CUDA_ENERGY_PROFILER_H
#define CUDA_ENERGY_PROFILER_H

#define GPU_ENERGY_PROFILE

//////////////////////////////////////////////////////////////////////////////
// Timer.h
// =======
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

#ifndef CUDA_ENERGY_PROFILER_TIMER_H
#define CUDA_ENERGY_PROFILER_TIMER_H
#if defined(WIN32) || defined(_WIN32)   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif

#include <stdlib.h>

namespace onnxruntime {

namespace profiling {

class Timer
{
public:
    Timer();                                    // default constructor
    ~Timer();                                   // default destructor

    void   start();                             // start timer
    void   stop();                              // stop the timer
    double getElapsedTime();                    // get elapsed time in second
    double getElapsedTimeInSec();               // get elapsed time in second (same as getElapsedTime)
    double getElapsedTimeInMilliSec();          // get elapsed time in milli-second
    double getElapsedTimeInMicroSec();          // get elapsed time in micro-second


protected:


private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag 
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER frequency;                    // ticks per second
    LARGE_INTEGER startCount;                   //
    LARGE_INTEGER endCount;                     //
#else
    timeval startCount;                         //
    timeval endCount;                           //
#endif
};

}  // namespace profiling
}  // namespace onnxruntime

#endif // CUDA_ENERGY_PROFILER_TIMER_H


#include <vector>
#include <unordered_map>
#include <thread>
#include <memory>
#include <nvml.h>

namespace ctpl
{
  class thread_pool;
}

namespace onnxruntime {

namespace profiling {

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete
#define DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete
#define DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  DISALLOW_COPY(TypeName);                     \
  DISALLOW_ASSIGNMENT(TypeName)
#define DISALLOW_MOVE(TypeName)     \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete
#define DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  DISALLOW_MOVE(TypeName)

class GPUInspector
{
 public:
  struct GPUInfo_t
  {
    double time_stamp{};
    double used_memory_percent{};
    double power_watt{};
    double temperature{};
    double memory_util{};
    double gpu_util{};
  };

  ~GPUInspector();
  static GPUInspector& Instance();
  bool Init(int gpu_id = -1, double sampling_interval = 0.05, bool parallel_reading = false);
  static GPUInfo_t GetGPUInfo(const nvmlDevice_t& device);
  GPUInfo_t GetGPUInfo(unsigned int gpu_id);
  void StartInspect();
  void StopInspect();
  void ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings) const;
  void ExportAllReadings(std::unordered_map<int, std::vector<GPUInfo_t>>& all_readings) const;
  unsigned int NumDevices() const;
  unsigned int NumInspectedDevices() const;
  void InspectedDeviceIds(std::vector<unsigned int>& device_ids) const;
  static double CalculateEnergy(const std::vector<GPUInfo_t>& readings);
  double CalculateEnergy(unsigned int gpu_id) const;
  void CalculateEnergy(std::vector<double>& energies) const;
  double GetDurationInSec();
  unsigned int GetLoopRepeat() const { return loop_repeat_; }
  void SetLoopRepeat(unsigned int repeat) { loop_repeat_ = repeat; }

 private:
  DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUInspector);
  GPUInspector(bool auto_init = true);
  inline void CheckInit() const;
  void Run();

  bool initialized_{false};
  bool running_inspect_{false};
  bool parallel_reading_{false};
  unsigned int loop_repeat_{1};
  double sampling_interval_micro_second_{0.05 * 1000000};
  std::unique_ptr<ctpl::thread_pool> pthread_pool_{nullptr};

  // std::vector<nvmlDevice_t> devices_;
  std::unordered_map<int, nvmlDevice_t> devices_;
  Timer timer_;
  // std::vector<std::vector<GPUInfo_t>> recordings_;
  std::unordered_map<int, std::vector<GPUInfo_t>> recordings_;

//   // for debugging
//  public:
//   struct Log_t
//   {
//     int thread_id;
//     int gpu_id;
//     double start_time;
//     double end_time;
//     Log_t() : thread_id(0), gpu_id(0), start_time(0), end_time(0) {}
//     Log_t(int thread_id, int gpu_id, double start_time, double end_time) : 
//         thread_id(thread_id), gpu_id(gpu_id), start_time(start_time), end_time(end_time) {}
//   };
//   std::vector<std::vector<Log_t>> running_logs_;
};

using GPUInfo_t = GPUInspector::GPUInfo_t;

}  // namespace profiling
}  // namespace onnxruntime

#endif  // CUDA_ENERGY_PROFILER_H