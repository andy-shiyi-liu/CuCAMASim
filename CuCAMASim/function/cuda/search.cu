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
  initDevice(GPU_DEVICE_ID);
  const uint32_t nVectors = queryData->getNVectors();
  const uint32_t rowCams = camData->getRowCams();
  const uint32_t colCams = camData->getColCams();
  const uint32_t rowSize = camData->getRowSize();
  const uint64_t nCamSubarrays = rowCams * colCams;

  assert(colCams == camSearch->getColCams());
  assert(rowCams == camSearch->getRowCams());
  assert(camData->getRowCams() == camSearch->getRowCams());
  assert(camData->getColCams() == camSearch->getColCams());

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
  cudaStream_t streams[nCamSubarrays];
  double *rawCamDataAll_d[nCamSubarrays] = {nullptr};
  double *rawQueryDataAll_d[nCamSubarrays] = {nullptr};
  double *distanceArrayAll_d[nCamSubarrays] = {nullptr};
  uint32_t *errorCodeAll_d[nCamSubarrays] = {nullptr};
  for (uint32_t rowCamIdx = 0; rowCamIdx < rowCams; rowCamIdx++) {
    for (uint32_t colCamIdx = 0; colCamIdx < colCams; colCamIdx++) {
      uint64_t camSubarrayIdx = rowCamIdx * colCams + colCamIdx;
      // for parallel search in multiple subarrays, we need to create all the
      // resources needed here and pass it into the kernel
      CHECK(cudaStreamCreate(&streams[camSubarrayIdx]));
      arraySearch(camSearch, camData, queryData, matchIdx_d, matchIdxDist_d,
                  &rawCamDataAll_d[camSubarrayIdx],
                  &rawQueryDataAll_d[camSubarrayIdx],
                  &distanceArrayAll_d[camSubarrayIdx], rowCamIdx, colCamIdx,
                  streams[camSubarrayIdx], &errorCodeAll_d[camSubarrayIdx]);
    }
  }
  // Synchronize and delete all streams
  for (int i = 0; i < nCamSubarrays; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    uint32_t errorCode = 0;
    // check error code
    CHECK(cudaMemcpy(&errorCode, errorCodeAll_d[i], sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    switch (errorCode) {
      case 1:
        throw std::runtime_error(
            "Error: more than " + std::to_string(MAX_MATCHED_ROWS) +
            " rows matched. Please increase MAX_MATCHED_ROWS in "
            "<CuCAMASim dir>/include/util/consts.h");
        break;
    }
    CHECK(cudaFree(rawCamDataAll_d[i]));
    CHECK(cudaFree(rawQueryDataAll_d[i]));
    CHECK(cudaFree(distanceArrayAll_d[i]));
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

  // sync and free resources
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(matchIdx_d));
  CHECK(cudaFree(matchIdxDist_d));

  // 3. write result from GPU to simResult
  uint32_t *result_h = new uint32_t[nVectors * MAX_MATCHED_ROWS];
  CHECK(cudaMemcpy(result_h, result_d, nBytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(result_d));

  simResult->writeFuncSimResult(result_h, nVectors, MAX_MATCHED_ROWS);

  delete result_h;

  cudaDeviceReset();
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

  // // for debug
  // // export result_d to csv file
  // uint32_t *result_h = new uint32_t[nVectors * MAX_MATCHED_ROWS * 1];
  // CHECK(cudaMemcpy(result_h, result_d,
  //                  nVectors * MAX_MATCHED_ROWS * 1 * sizeof(uint32_t),
  //                  cudaMemcpyDeviceToHost));
  // std::ofstream file5("/workspaces/CuCAMASim/result.csv");
  // file5 << ",";
  // for (uint32_t i = 0; i < MAX_MATCHED_ROWS * 1; i++) {
  //   file5 << i << ",";
  // }
  // file5 << std::endl;
  // for (uint32_t i = 0; i < nVectors; i++) {
  //   file5 << i << ",";
  //   for (uint32_t j = 0; j < MAX_MATCHED_ROWS * 1; j++) {
  //     file5 << result_h[j + MAX_MATCHED_ROWS * 1 * i] << ",";
  //   }
  //   file5 << std::endl;
  // }
  // file5.close();
}

// for each CAM subarray, search and give the matched index and distance
void arraySearch(const CAMSearch *camSearch, const CAMDataBase *camData,
                 const QueryData *queryData, uint32_t *matchIdx_d,
                 double *matchIdxDist_d, double **rawCamData_d,
                 double **rawQueryData_d, double **distanceArray_d,
                 const uint32_t rowCamIdx, const uint32_t colCamIdx,
                 const cudaStream_t &stream, uint32_t **errorCode_d) {
  // get and check data dimensions
  const uint32_t rowSize = camData->getRowSize(),
                 colSize = camData->getColSize(),
                 nVectors = queryData->getNVectors();
  const CAMArrayDim camDim = camData->at(rowCamIdx, colCamIdx)->getDim();
  const InputDataDim queryDim = queryData->at(colCamIdx)->getDim();

  assert(camDim.nCols == queryDim.nFeatures);
  assert(camDim.nRows == rowSize);
  assert(queryDim.nVectors == nVectors);
  assert(camDim.nCols == colSize);

  // init data for cuda kernel
  const double *rawCamData_h =
      camData->at(rowCamIdx, colCamIdx)->getData(FOR_CUDA_MEM_CPY);
  uint64_t nBytes =
      camDim.nRows * camDim.nCols * camDim.nBoundaries * sizeof(double);
  assert(*rawCamData_d == nullptr);
  CHECK(cudaMalloc((void **)rawCamData_d, nBytes));
  CHECK(
      cudaMemcpy(*rawCamData_d, rawCamData_h, nBytes, cudaMemcpyHostToDevice));

  const double *rawQueryData_h =
      queryData->at(colCamIdx)->getData(FOR_CUDA_MEM_CPY);
  nBytes = nVectors * colSize * sizeof(double);
  assert(*rawQueryData_d == nullptr);
  CHECK(cudaMalloc((void **)rawQueryData_d, nBytes));
  CHECK(cudaMemcpy(*rawQueryData_d, rawQueryData_h, nBytes,
                   cudaMemcpyHostToDevice));

  nBytes = nVectors * rowSize * sizeof(double);
  assert(*distanceArray_d == nullptr);
  CHECK(cudaMalloc((void **)distanceArray_d, nBytes));
  CHECK(cudaMemset(*distanceArray_d, 255, nBytes));

  // cuda grid and block size
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE_ID); // assuming device 0, adjust as needed
  if (DIST_FUNC_THREAD_X * DIST_FUNC_THREAD_Y >
      deviceProp.maxThreadsPerBlock) {
    throw std::runtime_error(
        "The number of threads per block exceeds the limit. Conside decrease "
        "DIST_FUNC_THREAD_X and/or DIST_FUNC_THREAD_Y. Current "
        "DIST_FUNC_THREAD_X=" +
        std::to_string(DIST_FUNC_THREAD_X) +
        ", DIST_FUNC_THREAD_Y=" + std::to_string(DIST_FUNC_THREAD_Y) +
        ", current thread per block: " + std::to_string(DIST_FUNC_THREAD_Y) +
        ",max threads per block supported=" +
        std::to_string(deviceProp.maxThreadsPerBlock));
  }
  dim3 block4Dist(DIST_FUNC_THREAD_X, DIST_FUNC_THREAD_Y);
  dim3 grid4Dist((long long int)(rowSize - 1) / block4Dist.x + 1,
                 (long long int)(nVectors - 1) / block4Dist.y + 1);

  // create cuda stream for sequential execution of kernels

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
        *rawCamData_d, *rawQueryData_d, *distanceArray_d, camDim, queryDim);
  } else if (camSearch->getDistType() == "softRange") {
    throw std::runtime_error(
        "NotImplementedError: Soft range distance is not implemented yet");
  } else {
    throw std::runtime_error("NotImplementedError: Unknown distance type");
  }

  // // for debug
  // double *distanceArray_h = new double[nVectors * rowSize];
  // CHECK(cudaMemcpy(distanceArray_h, *distanceArray_d, nBytes,
  //                  cudaMemcpyDeviceToHost));

  // // export distanceArray_h to csv file
  // std::ofstream file("/workspaces/CuCAMASim/distances.csv");
  // file << ",";
  // for (uint32_t i = 0; i < rowSize; i++) {
  //   file << i << ",";
  // }
  // file << std::endl;
  // for (uint32_t i = 0; i < nVectors; i++) {
  //   file << i << ",";
  //   for (uint32_t j = 0; j < rowSize; j++) {
  //     file << distanceArray_h[i * rowSize + j] << ",";
  //   }
  //   file << std::endl;
  // }
  // file.close();

  // 2. Find the output IDs of the array
  dim3 block4Sensing(SENSING_THREAD_X);
  dim3 grid4Sensing((nVectors - 1) / block4Sensing.x +
                    1);  // we need #nVectors threads

  assert(*errorCode_d == nullptr);
  CHECK(cudaMalloc((void **)errorCode_d, sizeof(uint32_t)));
  CHECK(cudaMemset(*errorCode_d, 0, 1 * sizeof(uint32_t)));
  if (camSearch->getSensing() == "exact") {
    getArrayExactResults<<<grid4Sensing, block4Sensing, 0, stream>>>(
        *distanceArray_d, matchIdx_d, matchIdxDist_d, camDim, queryDim,
        rowCamIdx, colCamIdx, camData->getColCams(), *errorCode_d);
  } else if (camSearch->getSensing() == "best") {
    throw std::runtime_error(
        "NotImplementedError: Best sensing is not implemented yet");
  } else if (camSearch->getSensing() == "threshold") {
    throw std::runtime_error(
        "NotImplementedError: Threshold sensing is not implemented yet");
  } else {
    throw std::runtime_error("NotImplementedError: Unknown sensing type");
  }

  // // for debug

  // // export rawCamData_h to csv file
  // std::ofstream file2("/workspaces/CuCAMASim/rawCamData.csv");
  // // print col2featureID as column name
  // file2 << ",";
  // for (uint32_t i = 0; i < colSize; i++) {
  //   file2 << "col_" << i << ",";
  // }
  // file2 << "classID" << std::endl;
  // for (uint32_t i = 0; i < rowSize; i++) {
  //   file2 << "row_" << i << ",";
  //   for (uint32_t j = 0; j < colSize; j++) {
  //     uint64_t lowerBdIdx = 0 + 2 * (j + colSize * i);
  //     uint64_t upperBdIdx = 1 + 2 * (j + colSize * i);
  //     file2 << rawCamData_h[lowerBdIdx]
  //           << " < x <= " << rawCamData_h[upperBdIdx] << ",";
  //   }
  //   file2 << std::endl;
  // }
  // file2.close();

  // camData->at(rowCamIdx, colCamIdx)
  //     ->toCSV("/workspaces/CuCAMASim/camArray.csv");

  // // export rawQueryData_h to csv file
  // std::ofstream file3("/workspaces/CuCAMASim/rawQueryData.csv");
  // file3 << ",";
  // for (uint32_t i = 0; i < colSize; i++) {
  //   file3 << i << ",";
  // }
  // file3 << std::endl;
  // for (uint32_t i = 0; i < nVectors; i++) {
  //   file3 << i << ",";
  //   for (uint32_t j = 0; j < colSize; j++) {
  //     file3 << rawQueryData_h[i * colSize + j] << ",";
  //   }
  //   file3 << std::endl;
  // }
  // file3.close();

  // // export matchIdx_d to csv file
  // uint32_t *matchIdx_h =
  //     new uint32_t[nVectors * MAX_MATCHED_ROWS * camData->getColCams()];
  // CHECK(cudaMemcpy(
  //     matchIdx_h, matchIdx_d,
  //     nVectors * MAX_MATCHED_ROWS * camData->getColCams() * sizeof(uint32_t),
  //     cudaMemcpyDeviceToHost));
  // std::ofstream file4("/workspaces/CuCAMASim/matchIdx.csv");
  // file4 << ",";
  // for (uint32_t i = 0; i < MAX_MATCHED_ROWS * camData->getColCams(); i++) {
  //   file4 << i << ",";
  // }
  // file4 << std::endl;
  // for (uint32_t i = 0; i < nVectors; i++) {
  //   file4 << i << ",";
  //   for (uint32_t j = 0; j < MAX_MATCHED_ROWS * camData->getColCams(); j++) {
  //     file4 << matchIdx_h[j + MAX_MATCHED_ROWS * camData->getColCams() * i]
  //           << ",";
  //   }
  //   file4 << std::endl;
  // }
  // file4.close();

  // // export matchIdxDist_d to csv file
  // double *matchIdxDist_h =
  //     new double[nVectors * MAX_MATCHED_ROWS * camData->getColCams()];
  // CHECK(cudaMemcpy(
  //     matchIdxDist_h, matchIdxDist_d,
  //     nVectors * MAX_MATCHED_ROWS * camData->getColCams() * sizeof(double),
  //     cudaMemcpyDeviceToHost));
  // std::ofstream file5("/workspaces/CuCAMASim/matchIdxDist.csv");
  // file5 << ",";
  // for (uint32_t i = 0; i < MAX_MATCHED_ROWS * camData->getColCams(); i++) {
  //   file5 << i << ",";
  // }
  // file5 << std::endl;
  // for (uint32_t i = 0; i < nVectors; i++) {
  //   file5 << i << ",";
  //   for (uint32_t j = 0; j < MAX_MATCHED_ROWS * camData->getColCams(); j++) {
  //     file5 << matchIdxDist_h[j + MAX_MATCHED_ROWS * camData->getColCams() *
  //     i]
  //           << ",";
  //   }
  //   file5 << std::endl;
  // }
  // file5.close();

  // std::cerr << "\033[33mWARNING: arraySearch() is still under "
  //              "development\033[0m"
  //           << camSearch << camData << queryData << std::endl;
}