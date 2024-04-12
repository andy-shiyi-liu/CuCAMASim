#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>
#include "util/data.h"

extern "C" double* rangeQueryPairwise(ACAMArray *camData, QueryData *queryData);
extern "C" double* softRangePairwise(ACAMArray *camData, QueryData *queryData);

#endif