// Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <stdio.h>
#include <stdlib.h>
// #include <cutil.h>
//  Includes
// #include <stdio.h>

// includes, project
// #include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
// #include <shrQATest.h>
// #include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#include <nvml.h>

#include "cuda_energy_profiler.h"

using namespace onnxruntime::profiling;

using dtype = unsigned;

// #define ITERATIONS 40
// #include "../include/ContAcq-IntClk.h"

// Variables
dtype *h_A;
dtype *h_B;
dtype *h_C;
dtype *d_A;
dtype *d_B;
dtype *d_C;
// bool noprompt = false;
// unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(dtype *, int);
// void ParseArguments(int, char**);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

// end of CUDA Helper Functions
__global__ void PowerKernal2(dtype *A, dtype *B, dtype *C, unsigned cgma, unsigned elements)
{
  int id = (blockDim.x * blockIdx.x + threadIdx.x);
  // Do Some Computation
  dtype sum;
  for(unsigned k = 0; k < cgma; ++k) {

    sum = A[id] + B[id] + id;
    for (unsigned j = 0; j < elements; j++) {
      sum += id + A[id + j];
    }
    A[id] = sum;

    sum = A[id] + B[id] + id;
    for (unsigned j = 0; j < elements; j++) {
      sum += id + B[id + j];
    }
    B[id] = sum;
  // }
  // for (unsigned j = 0; j < elements; j++) {
  //   unsigned I1 = A[i + j];
  //   unsigned I2 = B[i + j];
  //   // Excessive Addition access
  //   #pragma unroll 100
  //   for (unsigned k = 0; k < cgma; k++)
  //   {
  //     Value1= __fmaf_rn(I1,I2,Value1);
  //     Value3= __fmaf_rn(I1,I2,Value2);
  //     Value1= __fmaf_rn(Value1,Value2,Value1);
  //     Value1= __fmaf_rn(Value1,Value2,Value3);
  //     Value4= __fmaf_rn(Value1,Value2,Value3);
  //     Value5= __fmaf_rn(Value1,Value2,Value4);
  //     Value6= __fmaf_rn(Value1,Value2,Value6);
  //     Value2= __fmaf_rn(Value3,Value1,Value5);
  //     Value1= __fmaf_rn(Value2,Value3,Value3);
  //   }
  // }
  }
  C[id] = sum;
}

int main(int argc, char **argv)
{
  unsigned warmups = 10;
  unsigned repeats = 1000;
  if (argc != 3){
    fprintf(stderr,"usage: %s #warmups #repeats\n",argv[0]);
    exit(1);
  }
  else {
    warmups = atoi(argv[1]);
    repeats = atoi(argv[2]);
  }

  cudaSetDevice(1);
  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props, 1));
  nvmlInit();
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(1, &device);
  unsigned long long start_energy, end_energy;
  // GPUInspector::Initialize();
  // GPUInspector::Reset({1}, 0.02);

  printf("Blocks,Elements per thread,CGMA,Threads per block,Latency,Energy,Power\n");
  for (unsigned cgma = 1000; cgma <= 1000; cgma += 10) {
  for (unsigned blocks = 24; blocks <= 256; blocks += 1) {
  // for (unsigned elements = 10; elements <= 10; elements += 1) {
  for (unsigned threads_per_block = 512; threads_per_block <= 512; threads_per_block += 32) {
    unsigned total_elements = 480;
    unsigned elements = total_elements / blocks;
    if (blocks != total_elements / elements) {
      continue;
    }
    unsigned threads_per_block_after_fix = threads_per_block;
    // if (elements <= 200) {
    //   threads_per_block_after_fix = elements * threads_per_block / 200;
    //   elements = 200;
    // }
    int N = threads_per_block_after_fix * blocks * elements;
    size_t size = N * sizeof(dtype);
    // Allocate input vectors h_A and h_B in host memory
    h_A = (dtype *)malloc(size);
    if (h_A == 0)
      CleanupResources();
    h_B = (dtype *)malloc(size);
    if (h_B == 0)
      CleanupResources();
    h_C = (dtype *)malloc(size);
    if (h_C == 0)
      CleanupResources();

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    checkCudaErrors(cudaMalloc((void **)&d_A, size));
    checkCudaErrors(cudaMalloc((void **)&d_B, size));
    checkCudaErrors(cudaMalloc((void **)&d_C, size));

    cudaEvent_t start, stop;
    float elapsedTime = 0;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Copy vectors from host memory to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    dim3 dimGrid(blocks, 1);
    dim3 dimBlock(threads_per_block_after_fix, 1);
    dim3 dimGrid2(1, 1);
    dim3 dimBlock2(1, 1);

    for (unsigned i = 0; i < warmups; i++) {
      PowerKernal2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, cgma, elements);
    }

    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(start));
    nvmlDeviceGetTotalEnergyConsumption(device, &start_energy);
    // GPUInspector::StartInspect();
    for (unsigned i = 0; i < repeats; i++) {
      PowerKernal2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, cgma, elements);
    }
    checkCudaErrors(cudaEventRecord(stop));

    checkCudaErrors(cudaEventSynchronize(stop));
    nvmlDeviceGetTotalEnergyConsumption(device, &end_energy);
    // GPUInspector::StopInspect();
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    unsigned long long energy = end_energy - start_energy;
    //double energy = GPUInspector::CalculateEnergy(1);
    float power = energy / (float)elapsedTime;
    printf("%d,%d,%d,%d,%.6f,%.6f,%.2f\n", blocks, elements, cgma, threads_per_block_after_fix, elapsedTime / repeats, (float)energy / repeats, power);

    getLastCudaError("kernel launch failure");
    cudaThreadSynchronize();

    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    CleanupResources();
  } // threads_per_block
  // } // elements
  } // blocks
  } // cgma

  return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
}

// Allocates an array with random float entries.
void RandomInit(dtype *data, int n)
{
  for (int i = 0; i < n; ++i)
  {
    data[i] = rand() / RAND_MAX;
  }
}
