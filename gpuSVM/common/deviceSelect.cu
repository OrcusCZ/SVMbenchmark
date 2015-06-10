#include "deviceSelect.h"

void chooseLargestGPU(bool verbose) {
  int cudaDeviceCount;
  cudaGetDeviceCount(&cudaDeviceCount);
  int cudaDevice = 0;
  int maxSps = 0;
  struct cudaDeviceProp dp;
  for (int i = 0; i < cudaDeviceCount; i++) {
    cudaGetDeviceProperties(&dp, i);
    if (dp.multiProcessorCount > maxSps) {
      maxSps = dp.multiProcessorCount;
      cudaDevice = i;
    }
  }
  cudaGetDeviceProperties(&dp, cudaDevice);
  if (verbose) {
    printf("Using cuda device %i: %s\n", cudaDevice, dp.name);
  }
  cudaSetDevice(cudaDevice);
}
