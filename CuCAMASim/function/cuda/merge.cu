#include <stdio.h>

#include "function/cuda/merge.cuh"

__global__ void exactMerge(const uint32_t *matchIdx_d,
                           const double *matchIdxDist_d, uint32_t *result_d,
                           const uint32_t nVectors, const uint32_t colCams) {
  const uint32_t nx = nVectors;
  assert(nx != 0);

  uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t iy = threadIdx.y + blockIdx.y * blockDim.y;
  assert(iy == 0 &&
         "We assume 1d block and 1d grid for this kernel, iy should be 0");
  if (ix >= nx) {
    return;
  }
  uint64_t vectorIdx = ix;
  uint32_t resultMatchedRowCnt = 0;
  for (uint32_t i = 0; i < MAX_MATCHED_ROWS; i++) {
    assert(resultMatchedRowCnt < MAX_MATCHED_ROWS);
    uint32_t matchedRowIdx =
        matchIdx_d[getMatchResultIdx(vectorIdx, i, 0, colCams)];
    if (matchedRowIdx == uint32_t(-1)) {
      continue;
    }
    bool rowIdxInAllCams = true;
    for (uint32_t colCamIdx = 1; colCamIdx < colCams; colCamIdx++) {
      bool rowIdxInColCam = false;
      for (uint32_t j = 0; j < MAX_MATCHED_ROWS; j++) {
        uint32_t matchedRowIdxInColCam =
            matchIdx_d[getMatchResultIdx(vectorIdx, j, colCamIdx, colCams)];
        if (matchedRowIdxInColCam == matchedRowIdx) {
          rowIdxInColCam = true;
          break;
        }
      }
      if (!rowIdxInColCam) {
        rowIdxInAllCams = false;
        break;
      }
    }
    if (rowIdxInAllCams) {
      result_d[getMatchResultIdx(vectorIdx, resultMatchedRowCnt, 0, 1)] =
          matchedRowIdx;  // here, the size of result matrix is equavalent to
                          // when colCam = 1
      resultMatchedRowCnt++;
    }
  }
}