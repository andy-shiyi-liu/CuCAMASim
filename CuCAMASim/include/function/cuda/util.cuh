#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#include "util/consts.h"
#include "util/data.h"

#define CHECK(call)                                                    \
  {                                                                    \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  }

#define CHECK_KERNEL                                                   \
  {                                                                    \
    const cudaError_t error = cudaGetLastError();                      \
    if (error != cudaSuccess) {                                        \
      printf("\033[0;31mERROR: %s:%d,\033[0m", __FILE__, __LINE__);    \
      printf("\033[0;31mcode:%d,reason:%s\n\033[0m", error, cudaGetErrorString(error)); \
      exit(-1);                                                         \
    }                                                                  \
  }

#define getIx uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x
#define getIy uint64_t iy = threadIdx.y + blockIdx.y * blockDim.y
#define getIdx2D uint64_t idx = ix + iy * nx;
#define outOfRangeReturn2D    \
  if (ix >= nx || iy >= ny) { \
    return;                   \
  }

extern "C" {
__device__ __host__ inline uint64_t getCamIdx(const uint32_t rowIdx,
                                              const uint32_t colIdx,
                                              const uint32_t bdIdx,
                                              const CAMArrayDim camDim) {
  assert(rowIdx < camDim.nRows);
  assert(colIdx < camDim.nCols);
  assert(bdIdx < camDim.nBoundaries);
  return bdIdx + camDim.nBoundaries * (colIdx + camDim.nCols * rowIdx);
};

__device__ inline uint64_t getQueryIdx(const uint32_t vectorIdx,
                                       const uint32_t featureIdx,
                                       const InputDataDim queryDim) {
  assert(vectorIdx < queryDim.nVectors);
  assert(featureIdx < queryDim.nFeatures);
  return featureIdx + queryDim.nFeatures * vectorIdx;
};

__device__ inline uint64_t getDistIdx(const uint32_t vectorIdx,
                                      const uint32_t camRowIdx,
                                      const CAMArrayDim camDim,
                                      const InputDataDim queryDim) {
  assert(vectorIdx < queryDim.nVectors);
  assert(camRowIdx < camDim.nRows);
  return camRowIdx + camDim.nRows * vectorIdx;
};

__device__ inline uint64_t getMatchResultIdx(const uint32_t vectorIdx,
                                             const uint32_t matchedRowCnt,
                                             const uint32_t colCamIdx,
                                             const uint32_t colCam) {
  assert(colCamIdx < colCam);
  assert(matchedRowCnt < MAX_MATCHED_ROWS);
  return (matchedRowCnt + MAX_MATCHED_ROWS * colCamIdx) +
         MAX_MATCHED_ROWS * colCam * vectorIdx;
};
}

void initDevice(int devNum);

inline void checkGridBlockSize(const dim3 grid, const dim3 block) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(int(block.x) <= prop.maxThreadsDim[0] && block.x > 0);
  assert(int(block.y) <= prop.maxThreadsDim[1] && block.y > 0);
  assert(int(block.z) <= prop.maxThreadsDim[2] && block.z > 0);
  assert(int(grid.x) <= prop.maxGridSize[0] && grid.x > 0);
  assert(int(grid.y) <= prop.maxGridSize[1] && grid.y > 0);
  assert(int(grid.z) <= prop.maxGridSize[2] && grid.z > 0);
  assert(int(block.x * block.y * block.z) <= prop.maxThreadsPerBlock);
};

#endif