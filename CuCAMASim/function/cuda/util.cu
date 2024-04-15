#include <cassert>

#include "function/cuda/util.cuh"
#include "util/data.h"

void initDevice(int devNum) {
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using GPU device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
};

// __device__ inline uint64_t getCamIdx(const uint32_t rowIdx,
//                                      const uint32_t colIdx,
//                                      const uint32_t bdIdx,
//                                      const CAMArrayDim camDim) {
//   assert(rowIdx < camDim.nRows);
//   assert(colIdx < camDim.nCols);
//   assert(bdIdx < camDim.nBoundaries);
//   return bdIdx + camDim.nBoundaries * (colIdx + camDim.nCols * rowIdx);
// }

// __device__ inline uint64_t getQueryIdx(const uint32_t vectorIdx,
//                                        const uint32_t featureIdx,
//                                        const InputDataDim queryDim) {
//   assert(vectorIdx < queryDim.nVectors);
//   assert(featureIdx < queryDim.nFeatures);
//   return featureIdx + queryDim.nFeatures * vectorIdx;
// }