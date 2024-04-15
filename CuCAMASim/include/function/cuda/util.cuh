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

extern "C" {
__device__ inline uint64_t getCamIdx(const uint32_t rowIdx,
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

#endif