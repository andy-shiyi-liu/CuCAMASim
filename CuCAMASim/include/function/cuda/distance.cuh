#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>

#include "util/data.h"

extern "C" {
__global__ void helloWorld();
__global__ void rangeQueryPairwise(const double* rawCamData,
                                   const double* rawQueryData,
                                   double* distanceArray,
                                   const CAMArrayDim camDim,
                                   const InputDataDim queryDim);
}

#endif