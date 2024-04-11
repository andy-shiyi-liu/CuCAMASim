#ifndef DISTANCE_H
#define DISTANCE_H

#include <stdio.h>
#include <cuda_runtime.h>
#include "util/data.h"
#include "function/cuda/distance.cuh"

__global__ void helloWorld(){
    printf("Hello World from GPU!\n");
}

double* rangeQueryPairwise(CAMData *camData, QueryData *queryData){
    printf("in Range Query Pairwise\n");
    return (double*)nullptr;
}

#endif