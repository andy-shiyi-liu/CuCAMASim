#include <limits>

#include "function/cuda/distance.cuh"
#include "function/cuda/merge.cuh"
#include "function/cuda/search.cuh"
#include "function/cuda/sensing.cuh"
#include "function/cuda/util.cuh"
#include "util/consts.h"
#include "util/data.h"

void CAMSearchCUDA(CAMSearch *camSearch, const CAMDataBase *camData,
                   const QueryData *queryData, SimResult *simResult) {
  initDevice(0);
  const uint32_t nVectors = queryData->getNVectors();
  const uint32_t colCams = camData->getColCams();
  const uint32_t rowSize = camData->getRowSize();

  assert(colCams == camSearch->getColCams());
  assert(camData->getRowCams() == camSearch->getRowCams());

  uint64_t matchIdxMaxCols = MAX_MATCHED_ROWS * colCams;

  uint32_t *matchIdx_d;
  uint64_t nBytes = nVectors * matchIdxMaxCols * sizeof(uint32_t);
  CHECK(cudaMalloc((void **)&matchIdx_d, nBytes));
  CHECK(cudaMemset(
      matchIdx_d, 255,
      nBytes));  // set each bit in the array to 1, then if there is not a
                 // matched row, the index would be uint32_t(-1)

  double *matchIdxDist_d;
  nBytes = nVectors * matchIdxMaxCols * sizeof(double);
  CHECK(cudaMalloc((void **)&matchIdxDist_d, nBytes));
  CHECK(
      cudaMemset(matchIdxDist_d, 255,
                 nBytes));  // set each bit in the array to 1, then if there is
                            // not a matched row, the distance would be -nan

  // 1. Search in multiple arrays
  for (uint32_t rowCamIdx = 0; rowCamIdx < camData->getRowCams(); rowCamIdx++) {
    for (uint32_t colCamIdx = 0; colCamIdx < camData->getColCams();
         colCamIdx++) {
      arraySearch(camSearch, camData, queryData, matchIdx_d, matchIdxDist_d,
                  rowCamIdx, colCamIdx);
    }
  }

  // 2. Merge results from multiple arrays
  uint32_t *result_d;
  nBytes = nVectors * MAX_MATCHED_ROWS * sizeof(uint32_t);
  CHECK(cudaMalloc((void **)&result_d, nBytes));
  CHECK(cudaMemset(
      result_d, 255,
      nBytes));  // set each bit in the array to 1, then if there is not a
                 // matched row, the index would be uint32_t(-1)

  mergeIndices(camSearch, matchIdx_d, matchIdxDist_d, result_d, nVectors,
               colCams);

  cudaDeviceReset();
  std::cerr << "\033[33mWARNING: CAMSearchCUDA() is still under "
               "development\033[0m"
            << camSearch << camData << queryData << std::endl;
}

void mergeIndices(const CAMSearch *camSearch, const uint32_t *matchIdx_d,
                  const double *matchIdxDist_d, uint32_t *result_d,
                  const uint32_t nVectors, const uint32_t colCams) {
  dim3 block4Merging(MERGING_THREAD_X);
  dim3 grid4Merging((nVectors - 1) / block4Merging.x +
                    1);  // we need #nVectors threads

  if (camSearch->getSearchScheme() == "exact") {
    exactMerge<<<grid4Merging, block4Merging>>>(matchIdx_d, matchIdxDist_d,
                                                result_d, nVectors, colCams);
  } else if (camSearch->getSearchScheme() == "knn") {
    throw std::runtime_error(
        "NotImplementedError: KNN sensing is not implemented yet");
  } else if (camSearch->getSearchScheme() == "threshold") {
    throw std::runtime_error(
        "NotImplementedError: Threshold sensing is not implemented yet");
  } else {
    throw std::runtime_error("NotImplementedError: Unknown merge scheme");
  }

  // for debug
  // export result_d to csv file
  uint32_t *result_h = new uint32_t[nVectors * MAX_MATCHED_ROWS * 1];
  CHECK(cudaMemcpy(result_h, result_d,
                   nVectors * MAX_MATCHED_ROWS * 1 * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  std::ofstream file5("/workspaces/CuCAMASim/result.csv");
  file5 << ",";
  for (uint32_t i = 0; i < MAX_MATCHED_ROWS * 1; i++) {
    file5 << i << ",";
  }
  file5 << std::endl;
  for (uint32_t i = 0; i < nVectors; i++) {
    file5 << i << ",";
    for (uint32_t j = 0; j < MAX_MATCHED_ROWS * 1; j++) {
      file5 << result_h[j + MAX_MATCHED_ROWS * 1 * i] << ",";
    }
    file5 << std::endl;
  }
  file5.close();

  std::cerr
      << "\033[33mWARNING: mergeIndices() is still under development\033[0m"
      << matchIdx_d << matchIdxDist_d << result_d << nVectors << colCams
      << std::endl;
}

// for each CAM subarray, search and give the matched index and distance
void arraySearch(const CAMSearch *camSearch, const CAMDataBase *camData,
                 const QueryData *queryData, uint32_t *matchIdx_d,
                 double *matchIdxDist_d, const uint32_t rowCamIdx,
                 const uint32_t colCamIdx) {
  // get and check data dimensions
  const uint32_t rowSize = camData->getRowSize(),
                 colSize = camData->getColSize(),
                 nVectors = queryData->getNVectors();
  const CAMArrayDim camDim = camData->at(rowCamIdx, colCamIdx)->getDim();
  const InputDataDim queryDim = queryData->at(colCamIdx)->getDim();

  assert(camDim.nCols == queryDim.nFeatures);
  assert(camDim.nCols == rowSize);
  assert(queryDim.nVectors == nVectors);
  assert(camDim.nCols == colSize);

  // init data for cuda kernel
  const double *rawCamData_h =
      camData->at(rowCamIdx, colCamIdx)->getData(FOR_CUDA_MEM_CPY);
  double *rawCamData_d;
  uint64_t nBytes =
      camDim.nRows * camDim.nCols * camDim.nBoundaries * sizeof(double);
  CHECK(cudaMalloc((void **)&rawCamData_d, nBytes));
  CHECK(cudaMemcpy(rawCamData_d, rawCamData_h, nBytes, cudaMemcpyHostToDevice));

  const double *rawQueryData_h =
      queryData->at(colCamIdx)->getData(FOR_CUDA_MEM_CPY);
  double *rawQueryData_d;
  nBytes = nVectors * colSize * sizeof(double);
  CHECK(cudaMalloc((void **)&rawQueryData_d, nBytes));
  CHECK(cudaMemcpy(rawQueryData_d, rawQueryData_h, nBytes,
                   cudaMemcpyHostToDevice));

  // cuda grid and block size
  nBytes = nVectors * rowSize * sizeof(double);
  double *distanceArray_d;
  CHECK(cudaMalloc((void **)&distanceArray_d, nBytes));
  dim3 block4Dist(DIST_FUNC_THREAD_X, DIST_FUNC_THREAD_Y);
  dim3 grid4Dist((long long int)(rowSize - 1) / block4Dist.x + 1,
                 (long long int)(nVectors - 1) / block4Dist.y + 1);

  // create cuda stream for sequential execution of kernels
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // 1. Calculate the distance matrix in the array
  if (camSearch->getDistType() == "euclidean") {
    throw std::runtime_error(
        "NotImplementedError: Euclidean distance is not implemented yet");
  } else if (camSearch->getDistType() == "manhattan") {
    throw std::runtime_error(
        "NotImplementedError: Manhattan distance is not implemented yet");
  } else if (camSearch->getDistType() == "hamming") {
    throw std::runtime_error(
        "NotImplementedError: Hamming distance is not implemented yet");
  } else if (camSearch->getDistType() == "innerproduct") {
    throw std::runtime_error(
        "NotImplementedError: Inner product distance is not implemented yet");
  } else if (camSearch->getDistType() == "range") {
    // throw std::runtime_error(
    //     "NotImplementedError: Range distance is not implemented yet");
    if (camDim.nBoundaries != 2) {
      throw std::runtime_error(
          "Range distance requires ACAM, with 2 boundaries per cell!");
    }
    rangeQueryPairwise<<<grid4Dist, block4Dist, 0, stream>>>(
        rawCamData_d, rawQueryData_d, distanceArray_d, camDim, queryDim);
  } else if (camSearch->getDistType() == "softRange") {
    throw std::runtime_error(
        "NotImplementedError: Soft range distance is not implemented yet");
  } else {
    throw std::runtime_error("NotImplementedError: Unknown distance type");
  }

  // 2. Find the output IDs of the array
  dim3 block4Sensing(SENSING_THREAD_X);
  dim3 grid4Sensing((nVectors - 1) / block4Sensing.x +
                    1);  // we need #nVectors threads
  uint32_t errorCode = 0, *errorCode_d = nullptr;
  CHECK(cudaMalloc((void **)&errorCode_d, sizeof(uint32_t)));
  CHECK(cudaMemcpy(errorCode_d, &errorCode, sizeof(uint32_t),
                   cudaMemcpyHostToDevice));
  if (camSearch->getSensing() == "exact") {
    getArrayExactResults<<<grid4Sensing, block4Sensing, 0, stream>>>(
        distanceArray_d, matchIdx_d, matchIdxDist_d, camDim, queryDim,
        rowCamIdx, colCamIdx, camData->getColCams(), errorCode_d);
  } else if (camSearch->getSensing() == "best") {
    throw std::runtime_error(
        "NotImplementedError: Best sensing is not implemented yet");
  } else if (camSearch->getSensing() == "threshold") {
    throw std::runtime_error(
        "NotImplementedError: Threshold sensing is not implemented yet");
  } else {
    throw std::runtime_error("NotImplementedError: Unknown sensing type");
  }
  // check error code
  CHECK(cudaMemcpy(&errorCode, errorCode_d, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  switch (errorCode) {
    case 1:
      throw std::runtime_error("Error: more than " +
                               std::to_string(MAX_MATCHED_ROWS) +
                               " matched. Please increase MAX_MATCHED_ROWS in "
                               "<CuCAMASim dir>/include/util/consts.h");
  }
  // Synchronize stream
  cudaStreamSynchronize(stream);
  // Destroy stream
  cudaStreamDestroy(stream);

  // for debug
  double *distanceArray_h = new double[nVectors * rowSize];
  CHECK(cudaMemcpy(distanceArray_h, distanceArray_d, nBytes,
                   cudaMemcpyDeviceToHost));

  // export distanceArray_h to csv file
  std::ofstream file("/workspaces/CuCAMASim/distances.csv");
  file << ",";
  for (uint32_t i = 0; i < rowSize; i++) {
    file << i << ",";
  }
  file << std::endl;
  for (uint32_t i = 0; i < nVectors; i++) {
    file << i << ",";
    for (uint32_t j = 0; j < rowSize; j++) {
      file << distanceArray_h[i * rowSize + j] << ",";
    }
    file << std::endl;
  }
  file.close();

  // export rawCamData_h to csv file
  std::ofstream file2("/workspaces/CuCAMASim/rawCamData.csv");
  // print col2featureID as column name
  file2 << ",";
  for (uint32_t i = 0; i < colSize; i++) {
    file2 << "col_" << i << ",";
  }
  file2 << "classID" << std::endl;
  for (uint32_t i = 0; i < rowSize; i++) {
    file2 << "row_" << i << ",";
    for (uint32_t j = 0; j < colSize; j++) {
      uint64_t lowerBdIdx = 0 + 2 * (j + colSize * i);
      uint64_t upperBdIdx = 1 + 2 * (j + colSize * i);
      file2 << rawCamData_h[lowerBdIdx]
            << " < x <= " << rawCamData_h[upperBdIdx] << ",";
    }
    file2 << std::endl;
  }
  file2.close();

  camData->at(rowCamIdx, colCamIdx)
      ->toCSV("/workspaces/CuCAMASim/camArray.csv");

  // export rawQueryData_h to csv file
  std::ofstream file3("/workspaces/CuCAMASim/rawQueryData.csv");
  file3 << ",";
  for (uint32_t i = 0; i < colSize; i++) {
    file3 << i << ",";
  }
  file3 << std::endl;
  for (uint32_t i = 0; i < nVectors; i++) {
    file3 << i << ",";
    for (uint32_t j = 0; j < colSize; j++) {
      file3 << rawQueryData_h[i * colSize + j] << ",";
    }
    file3 << std::endl;
  }
  file3.close();

  // export matchIdx_d to csv file
  uint32_t *matchIdx_h =
      new uint32_t[nVectors * MAX_MATCHED_ROWS * camData->getColCams()];
  CHECK(cudaMemcpy(
      matchIdx_h, matchIdx_d,
      nVectors * MAX_MATCHED_ROWS * camData->getColCams() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  std::ofstream file4("/workspaces/CuCAMASim/matchIdx.csv");
  file4 << ",";
  for (uint32_t i = 0; i < MAX_MATCHED_ROWS * camData->getColCams(); i++) {
    file4 << i << ",";
  }
  file4 << std::endl;
  for (uint32_t i = 0; i < nVectors; i++) {
    file4 << i << ",";
    for (uint32_t j = 0; j < MAX_MATCHED_ROWS * camData->getColCams(); j++) {
      file4 << matchIdx_h[j + MAX_MATCHED_ROWS * camData->getColCams() * i]
            << ",";
    }
    file4 << std::endl;
  }
  file4.close();

  // export matchIdxDist_d to csv file
  double *matchIdxDist_h =
      new double[nVectors * MAX_MATCHED_ROWS * camData->getColCams()];
  CHECK(cudaMemcpy(
      matchIdxDist_h, matchIdxDist_d,
      nVectors * MAX_MATCHED_ROWS * camData->getColCams() * sizeof(double),
      cudaMemcpyDeviceToHost));
  std::ofstream file5("/workspaces/CuCAMASim/matchIdxDist.csv");
  file5 << ",";
  for (uint32_t i = 0; i < MAX_MATCHED_ROWS * camData->getColCams(); i++) {
    file5 << i << ",";
  }
  file5 << std::endl;
  for (uint32_t i = 0; i < nVectors; i++) {
    file5 << i << ",";
    for (uint32_t j = 0; j < MAX_MATCHED_ROWS * camData->getColCams(); j++) {
      file5 << matchIdxDist_h[j + MAX_MATCHED_ROWS * camData->getColCams() * i]
            << ",";
    }
    file5 << std::endl;
  }
  file5.close();

  // std::cerr << "\033[33mWARNING: arraySearch() is still under "
  //              "development\033[0m"
  //           << camSearch << camData << queryData << std::endl;
}