#ifndef SEARCH_CUH
#define SEARCH_CUH

#include <cuda_runtime.h>

#include "function/cuda/util.cuh"
#include "function/search.h"
#include "util/data.h"

void CAMSearchCUDA(CAMSearch *camSearch, const CAMDataBase *camData,
                   const QueryData *queryData, SimResult *simResult);

void arraySearch(const CAMSearch *camSearch, const CAMDataBase *camData,
                 const QueryData *queryData, uint32_t *matchIdx_d,
                 double *matchIdxDist_d, double **rawCamData_d,
                 double **rawQueryData_d, double **distanceArray_d,
                 const uint32_t rowCamIdx, const uint32_t colCamIdx,
                 const cudaStream_t &stream, uint32_t **errorCode_d);

void mergeIndices(const CAMSearch *camSearch, const uint32_t *matchIdx_d,
                  const double *matchIdxDist_d, uint32_t *result_d,
                  const uint32_t nVectors, const uint32_t colCams);

template <typename T>
T **new2DArray(const uint32_t row, const uint32_t col, T initValue) {
  T **array = new T *[row];
  for (uint32_t i = 0; i < row; i++) {
    array[i] = new T[col];
    for (uint32_t j = 0; j < col; j++) {
      array[i][j] = initValue;
    }
  }
  return array;
}

template <typename T>
void delete2DArray(T **array, const uint32_t row) {
  for (uint32_t i = 0; i < row; i++) {
    delete[] array[i];
  }
  delete[] array;
}

#endif