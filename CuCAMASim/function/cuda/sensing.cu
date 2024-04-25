#include <stdio.h>

#include "function/cuda/sensing.cuh"
#include "function/cuda/util.cuh"
#include "util/consts.h"

__global__ void getArrayExactResults(
    const double *distanceArray_d, uint32_t *matchIdx_d, double *matchIdxDist_d,
    const CAMArrayDim camDim, const InputDataDim queryDim,
    const uint32_t rowCamIdx, const uint32_t colCamIdx, const uint32_t colCam,
    uint32_t *errorCode) {
  assert(camDim.nCols == queryDim.nFeatures);

  const uint32_t nx = queryDim.nVectors;
  assert(nx != 0);

  getIx;
  getIy;
  assert(iy == 0 &&
         "We use 1d block and 1d grid for this kernel, iy should be 0");
  if (ix >= nx) {
    return;
  }

  uint64_t vectorIdx = ix;
  uint32_t matchedRowCnt = 0;
  for (uint32_t inCamRowIdx = 0; inCamRowIdx < camDim.nRows; inCamRowIdx++) {
    double dist =
        distanceArray_d[getDistIdx(vectorIdx, inCamRowIdx, camDim, queryDim)];
    if (dist == 0) {
      if (matchedRowCnt >= MAX_MATCHED_ROWS) {
        *errorCode = 1;
        printf(
            "Error: matchedRowCnt >= MAX_MATCHED_ROWS! vectorIdx: %llu, "
            "colCamIdx: %u, rowCamIdx: %u, inCamRowIdx: %u\n",
            vectorIdx, colCamIdx, rowCamIdx, inCamRowIdx);
        return;
      }
      uint32_t totalCamRowIdx = inCamRowIdx + rowCamIdx * camDim.nRows;
      matchIdx_d[getMatchResultIdx(vectorIdx, matchedRowCnt, colCamIdx,
                                   colCam)] = totalCamRowIdx;
      matchIdxDist_d[getMatchResultIdx(vectorIdx, matchedRowCnt, colCamIdx,
                                       colCam)] = dist;
      matchedRowCnt++;
    }
  }
}