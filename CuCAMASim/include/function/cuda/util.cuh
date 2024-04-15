#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#define CHECK(call)                                                    \
  {                                                                    \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  }

void initDevice(int devNum);

#endif