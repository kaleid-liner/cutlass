#include "jetson_profiler.h"
#include <fstream>

namespace onnxruntime {

namespace profiling {

const std::string JetsonProfiler::gpu_power_filename_ = "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input";
const std::string JetsonProfiler::cpu_power_filename_ = "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input";

void JetsonProfiler::start() {
  auto run_func = [this](int thread_id) { _run(); };
  thread_pool_->push(run_func);
  start_time_ = clock_t::now();
  cpu_energy_ = 0;
  gpu_energy_ = 0;
  profiling_ = true;
}

void JetsonProfiler::end() {
  end_time_ = clock_t::now();
  profiling_ = false;
  if(thread_pool_ == nullptr) return;
  while(thread_pool_->n_idle() != thread_pool_->size())
  {
      std::this_thread::sleep_for(std::chrono::microseconds(5));
  }
}

void JetsonProfiler::_run() {
  std::ifstream cpu_power_in, gpu_power_in;
  unsigned long long cpu_power, gpu_power;

  auto last = start_time_;
  while (profiling_) {
    cpu_power_in.open(cpu_power_filename_);
    gpu_power_in.open(gpu_power_filename_);
    cpu_power_in >> cpu_power;
    gpu_power_in >> gpu_power;
    
    auto now = clock_t::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
    last = now;
    cpu_energy_ += duration * cpu_power;
    gpu_energy_ += duration * gpu_power;
    std::this_thread::sleep_for(std::chrono::microseconds(interval_ * 1000));
  }
}

}
}