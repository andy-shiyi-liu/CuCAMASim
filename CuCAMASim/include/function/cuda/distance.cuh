#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>
#include "util/data.h"

extern "C" double* rangeQueryPairwise(ACAMArray *camArray, QueryData *queryData);
extern "C" double* softRangePairwise(ACAMArray *camArray, QueryData *queryData);

#endif