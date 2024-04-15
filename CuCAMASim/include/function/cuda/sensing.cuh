#ifndef SENSING_CUH
#define SENSING_CUH

#include <cuda_runtime.h>

#include "util/data.h"

extern "C" {
__global__ void getArrayExactResults(
    const double *distanceArray_d, uint32_t *matchIdx_d, double *matchIdxDist_d,
    const CAMArrayDim camDim, const InputDataDim queryDim,
    const uint32_t rowCamIdx, const uint32_t colCamIdx, const uint32_t colCam,
    uint32_t *errorCode);
}

#endif