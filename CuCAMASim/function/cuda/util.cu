#include "function/cuda/util.cuh"

void initDevice(int devNum) {
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using GPU device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
};