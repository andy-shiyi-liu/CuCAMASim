#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>
#include <limits>

#include "function/cuda/distance.cuh"
#include "function/cuda/util.cuh"
#include "util/data.h"

__global__ void helloWorld() { printf("Hello World from GPU!\n"); }

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

  getIx;
  getIy;
  assert(threadIdx.z == 0);
  getIdx2D;
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
  // if (rowDist != 20 && camRowIdx > 29) {
  //   for (uint32_t colIdx = 0; colIdx < camDim.nCols; colIdx++) {
  //     double queryValue =
  //         rawQueryData[getQueryIdx(vectorIdx, colIdx, queryDim)];
  //     // printf("%lf",queryValue);
  //     double camLowerBd = rawCamData[getCamIdx(camRowIdx, colIdx, 0,
  //     camDim)]; double camUpperBd = rawCamData[getCamIdx(camRowIdx, colIdx,
  //     1, camDim)]; if (queryValue >= camLowerBd && queryValue <= camUpperBd)
  //     { } else {
  //       rowDist += 1.0;
  //     }
  //   }
  //   printf(
  //       "queryIdx: %llu, camRowIdx: %llu, rowDist: %lf\nwrite to ix %llu iy "
  //       "%llu "
  //       "distanceArray[%llu]\n*************************************************"
  //       "*******************\n",
  //       vectorIdx, camRowIdx, rowDist, ix, iy, idx);
  // }
  // write the distance to the distance array
  distanceArray[idx] = rowDist;
}

__device__ inline double lowerSoftBd(double queryValue, double camLowerBd,
                                     double softness) {
  return 1 / (1 + exp(softness * (queryValue - camLowerBd) + 3));
}

__device__ inline double upperSoftBd(double queryValue, double camUpperBd,
                                     double softness) {
  return 1 / (1 + exp(-softness * (queryValue - camUpperBd) + 3));
}

__global__ void softRangePairwise(const double* rawCamData,
                                  const double* rawQueryData,
                                  double* distanceArray, const double softness,
                                  const CAMArrayDim camDim,
                                  const InputDataDim queryDim) {
  // printf("in Range Query Pairwise\n");

  assert(camDim.nCols == queryDim.nFeatures);
  assert(camDim.nBoundaries == 2 &&
         "Range distance requires ACAM, with 2 boundaries per cell!");

  const uint32_t nx = camDim.nRows;
  const uint32_t ny = queryDim.nVectors;
  assert(nx != 0 && ny != 0);

  getIx;
  getIy;
  assert(threadIdx.z == 0);
  getIdx2D;
  if (ix >= nx || iy >= ny) {
    // printf("ix: %llu, nx: %u, iy: %llu, ny: %u, returned.\n", ix, nx, iy,
    // ny);
    return;
  }

  uint64_t camRowIdx = ix;
  uint64_t vectorIdx = iy;

  // calculate the distance between the vector[vectorIdx] and cam row[camRowIdx]
  double rowDist = 0.0;
  bool hasValidCell = false;  // check whether the row contains valid cells
  for (uint32_t colIdx = 0; colIdx < camDim.nCols; colIdx++) {
    double queryValue = rawQueryData[getQueryIdx(vectorIdx, colIdx, queryDim)];
    // printf("%lf",queryValue);
    double camLowerBd = rawCamData[getCamIdx(camRowIdx, colIdx, 0, camDim)];
    double camUpperBd = rawCamData[getCamIdx(camRowIdx, colIdx, 1, camDim)];
    if (camLowerBd != 0 || camUpperBd != 0) {
      hasValidCell = true;
      // dist caused by the lower bound
      rowDist += lowerSoftBd(queryValue, camLowerBd, softness);
      // dist caused by the upper bound
      rowDist += upperSoftBd(queryValue, camUpperBd, softness);
      // printf(
      //     "queryValue: %lf, camLowerBd: %lf, camUpperBd: %lf, rowDist: %lf\n",
      //     queryValue, camLowerBd, camUpperBd, rowDist);
    }
  }
  if (!hasValidCell) {
    rowDist = 1e50;  // set to a large number for invalid rows
    // printf("Invalid row detected at camRowIdx: %llu\n", camRowIdx);
  }

  // // for debug
  // if (rowDist != 20 && camRowIdx > 29) {
  //   for (uint32_t colIdx = 0; colIdx < camDim.nCols; colIdx++) {
  //     double queryValue =
  //         rawQueryData[getQueryIdx(vectorIdx, colIdx, queryDim)];
  //     // printf("%lf",queryValue);
  //     double camLowerBd = rawCamData[getCamIdx(camRowIdx, colIdx, 0,
  //     camDim)]; double camUpperBd = rawCamData[getCamIdx(camRowIdx, colIdx,
  //     1, camDim)]; if (queryValue >= camLowerBd && queryValue <= camUpperBd)
  //     { } else {
  //       rowDist += 1.0;
  //     }
  //   }
  //   printf(
  //       "queryIdx: %llu, camRowIdx: %llu, rowDist: %lf\nwrite to ix %llu iy "
  //       "%llu "
  //       "distanceArray[%llu]\n*************************************************"
  //       "*******************\n",
  //       vectorIdx, camRowIdx, rowDist, ix, iy, idx);
  // }

  // write the distance to the distance array
  distanceArray[idx] = rowDist;
}
