#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>
#include "util/data.h"

extern "C" double* rangeQueryPairwise(ACAMData *camData, QueryData *queryData);
extern "C" double* softRangePairwise(ACAMData *camData, QueryData *queryData);

#endif