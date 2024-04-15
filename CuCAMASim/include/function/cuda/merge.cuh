#ifndef MERGE_CUH
#define MERGE_CUH

#include <cuda_runtime.h>

#include "function/cuda/util.cuh"
#include "function/search.h"
#include "util/data.h"

extern "C" {
__global__ void exactMerge(const uint32_t *matchIdx_d,
                  const double *matchIdxDist_d, uint32_t *result_d,
                  const uint32_t nVectors, const uint32_t colCams);
}

#endif