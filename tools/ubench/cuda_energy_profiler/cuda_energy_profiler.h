#pragma once
#ifndef CUDA_ENERGY_PROFILER_H
#define CUDA_ENERGY_PROFILER_H

#include <vector>
#include <unordered_map>
#include <memory>

#ifdef USE_CTPL_THREAD_POOL
namespace ctpl
{
  class thread_pool;
}
#else
#include <thread>
#endif

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

class Timer;
struct GPUInfoContainer;

class GPUInspector final
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
    double energy_since_boot{};
  };

  ~GPUInspector();
  
  static unsigned int NumTotalDevices();
  static unsigned int NumInspectedDevices();
  static void InspectedDeviceIds(std::vector<unsigned int>& device_ids);
  static GPUInfo_t GetGPUInfo(unsigned int gpu_id);

  static void StartInspect();
  static void StopInspect();
  static void ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings);
  static void ExportAllReadings(std::unordered_map<unsigned int, std::vector<GPUInfo_t>>& all_readings);

  static double CalculateEnergy(const std::vector<GPUInfo_t>& readings);
  static double CalculateEnergy(unsigned int gpu_id);
  static void CalculateEnergy(std::unordered_map<unsigned int, double>& energies);
  static double GetDurationInSec();

  static void Initialize() { Instance(); }
  static bool Reset(std::vector<unsigned int> gpu_ids = {}, double sampling_interval = 0.05);

 private:
  // sigleton
  DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUInspector);
  GPUInspector();
  static GPUInspector& Instance();
  // implementation
  bool _init(std::vector<unsigned int> gpu_ids = {}, double sampling_interval = 0.05);
  void _run();
  void _start_inspect();
  void _stop_inspect();

  bool running_inspect_{false};
  double sampling_interval_micro_second_{0.05 * 1000000};

#ifdef USE_CTPL_THREAD_POOL
  std::unique_ptr<ctpl::thread_pool> pthread_pool_{nullptr};
  void _thread_pool_wait_ready();
#else
  std::shared_ptr<std::thread> pthread_inspect_{nullptr};
#endif

  std::shared_ptr<Timer> timer_{nullptr};
  std::shared_ptr<GPUInfoContainer> recording_container_{nullptr};

};

using GPUInfo_t = GPUInspector::GPUInfo_t;

}  // namespace profiling
}  // namespace onnxruntime

#endif  // CUDA_ENERGY_PROFILER_H