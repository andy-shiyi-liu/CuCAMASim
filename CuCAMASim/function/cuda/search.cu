#include <limits>

#include "function/cuda/search.cuh"
#include "function/cuda/util.cuh"
#include "util/consts.h"

void CAMSearchCUDA(CAMSearch *CAMSearch, const CAMDataBase *camData,
                   const QueryData *queryData) {
  initDevice(0);
  const uint32_t nVectors = queryData->getNVectors();
  const uint32_t colCams = camData->getColCams();
  const uint32_t rowSize = camData->getRowSize();

  uint64_t matchIdxMaxCols = MAX_MATCHED_ROWS * colCams;
  uint32_t **matchIdx =
      new2DArray<uint32_t>(nVectors, matchIdxMaxCols, uint32_t(-1));
  double **matchIdxDist = new2DArray<double>(
      nVectors, matchIdxMaxCols, std::numeric_limits<double>::quiet_NaN());

  // 1. Search in multiple arrays
  for (uint32_t rowCamIdx = 0; rowCamIdx < camData->getRowCams(); rowCamIdx++) {
    for (uint32_t colCamIdx = 0; colCamIdx < camData->getColCams();
         colCamIdx++) {
      arraySearch(CAMSearch, camData, queryData, matchIdx, matchIdxDist,
                  rowCamIdx, colCamIdx);
    }
  }

  delete2DArray<uint32_t>(matchIdx, nVectors);
  delete2DArray<double>(matchIdxDist, nVectors);
  cudaDeviceReset();
  std::cerr << "\033[33mWARNING: CAMSearchCUDA() is still under "
               "development\033[0m"
            << CAMSearch << camData << queryData << std::endl;
}

void arraySearch(const CAMSearch *CAMSearch, const CAMDataBase *camData,
                 const QueryData *queryData, uint32_t **matchIdx,
                 double **matchIdxDist, const uint32_t rowCamIdx,
                 const uint32_t colCamIdx) {
  uint32_t rowSize = camData->getRowSize(), colSize = camData->getColSize();
  double **distanceArray = new2DArray<double>(
      rowSize, colSize, std::numeric_limits<double>::quiet_NaN());

  std::cerr << "\033[33mWARNING: arraySearch() is still under "
               "development\033[0m"
            << CAMSearch << camData << queryData << std::endl;
}