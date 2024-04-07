#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>
#include "util/data.h"

extern "C" double* rangeQueryPairwise(CAMData *camData, QueryData *queryData);

#endif