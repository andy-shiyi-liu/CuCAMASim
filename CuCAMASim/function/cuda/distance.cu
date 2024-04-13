#include <stdio.h>
#include <cuda_runtime.h>
#include "util/data.h"
#include "function/cuda/distance.cuh"

__global__ void helloWorld(){
    printf("Hello World from GPU!\n");
}

double* rangeQueryPairwise(ACAMArray *camArray, QueryData *queryData){
    printf("in Range Query Pairwise\n");
    throw std::runtime_error("NotImplementedError: Range distance is not implemented yet");
    return (double*)nullptr;
}

double* softRangePairwise(ACAMArray *camArray, QueryData *queryData){
    printf("in SoftRange Pairwise\n");
    throw std::runtime_error("NotImplementedError: Soft range distance is not implemented yet");
    return (double*)nullptr;
}
