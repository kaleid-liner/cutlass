#ifndef JETSON_PROFILER_H
#define JETSON_PROFILER_H

#include <string>
#include <chrono>
#include "ctpl_stl.h"

namespace onnxruntime {

namespace profiling {

class JetsonProfiler {
public:
  JetsonProfiler(int interval) 
    : interval_(interval),
      profiling_(false),
      cpu_energy_(0),
      gpu_energy_(0) {
    thread_pool_ = std::make_shared<ctpl::thread_pool>(1);
  }
  JetsonProfiler() : JetsonProfiler(50) {}

  void start();
  void end();

  unsigned long long get_cpu_energy() {
    return cpu_energy_;
  }

  unsigned long long get_gpu_energy() {
    return gpu_energy_;
  }

  int64_t get_elapsed_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
  }

private:
  static const std::string gpu_power_filename_;
  static const std::string cpu_power_filename_;
  int interval_;  // interval in millisecond

  bool profiling_;

  void _run();

  using clock_t = std::chrono::high_resolution_clock;
  using time_t = clock_t::time_point;
  time_t start_time_;
  time_t end_time_;

  std::shared_ptr<ctpl::thread_pool> thread_pool_;

  unsigned long long cpu_energy_;
  unsigned long long gpu_energy_;
};

}
}

#endif