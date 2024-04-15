#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <limits>

#include "function/cuda/distance.cuh"
#include "util/data.h"

__global__ void helloWorld() { printf("Hello World from GPU!\n"); }

__device__ inline uint64_t getCamIdx(const uint32_t rowIdx,
                                     const uint32_t colIdx,
                                     const uint32_t bdIdx,
                                     const CAMArrayDim camDim) {
  assert(rowIdx < camDim.nRows);
  assert(colIdx < camDim.nCols);
  assert(bdIdx < camDim.nBoundaries);
  return bdIdx + camDim.nBoundaries * (colIdx + camDim.nCols * rowIdx);
}

__device__ inline uint64_t getQueryIdx(const uint32_t vectorIdx,
                                       const uint32_t featureIdx,
                                       const InputDataDim queryDim) {
  assert(vectorIdx < queryDim.nVectors);
  assert(featureIdx < queryDim.nFeatures);
  return featureIdx + queryDim.nFeatures * vectorIdx;
}

__global__ void rangeQueryPairwise(const double* rawCamData,
                                   const double* rawQueryData,
                                   double* distanceArray,
                                   const CAMArrayDim camDim,
                                   const InputDataDim queryDim) {
  // printf("in Range Query Pairwise\n");

  assert(camDim.nCols == queryDim.nFeatures);
  assert(camDim.nBoundaries == 2 &&
         "Range distance requires ACAM, with 2 boundaries per cell!");

  const uint32_t nx = camDim.nRows;
  const uint32_t ny = queryDim.nVectors;
  assert(nx != 0 && ny != 0);

  uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t iy = threadIdx.y + blockIdx.y * blockDim.y;
  uint64_t idx = ix + iy * nx;
  if (ix >= nx || iy >= ny) {
    // printf("ix: %llu, nx: %u, iy: %llu, ny: %u, returned.\n", ix, nx, iy,
    // ny);
    return;
  }

  uint64_t camRowIdx = ix;
  uint64_t vectorIdx = iy;

  // calculate the distance between the vector[vectorIdx] and cam row[camRowIdx]
  double rowDist = 0.0;
  for (uint32_t colIdx = 0; colIdx < camDim.nCols; colIdx++) {
    double queryValue = rawQueryData[getQueryIdx(vectorIdx, colIdx, queryDim)];
    // printf("%lf",queryValue);
    double camLowerBd = rawCamData[getCamIdx(camRowIdx, colIdx, 0, camDim)];
    double camUpperBd = rawCamData[getCamIdx(camRowIdx, colIdx, 1, camDim)];
    if (queryValue >= camLowerBd && queryValue <= camUpperBd) {
    } else {
      rowDist += 1.0;
    }
  }

  // for debug
  if (rowDist != 20 && camRowIdx > 29) {
    for (uint32_t colIdx = 0; colIdx < camDim.nCols; colIdx++) {
      double queryValue =
          rawQueryData[getQueryIdx(vectorIdx, colIdx, queryDim)];
      // printf("%lf",queryValue);
      double camLowerBd = rawCamData[getCamIdx(camRowIdx, colIdx, 0, camDim)];
      double camUpperBd = rawCamData[getCamIdx(camRowIdx, colIdx, 1, camDim)];
      if (queryValue >= camLowerBd && queryValue <= camUpperBd) {
      } else {
        rowDist += 1.0;
      }
    }
    printf(
        "queryIdx: %llu, camRowIdx: %llu, rowDist: %lf\nwrite to ix %llu iy "
        "%llu "
        "distanceArray[%llu]\n*************************************************"
        "*******************\n",
        vectorIdx, camRowIdx, rowDist, ix, iy, idx);
  }
  // write the distance to the distance array
  distanceArray[idx] = rowDist;
}
