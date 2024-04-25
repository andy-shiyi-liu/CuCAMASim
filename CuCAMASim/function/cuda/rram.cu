#include <curand_kernel.h>
#include <stdio.h>

#include <cstring>
#include <fstream>

#include "function/cuda/rram.cuh"
#include "function/cuda/util.cuh"
#include "util/consts.h"

// use newton's method to solve conductance from Vbd
__device__ inline double solveConductanceFromVbd(double Vbd,
                                                 RRAMCellType type) {
  double x = RRAM_STARTPOINT;
  double f = RRAMConduct2Vbd(x, type) - Vbd;
  double df = d_RRAMConduct2Vbd(x, type);
  for (uint64_t i = 0; fabs(f) > RRAM_TOLERANCE && i < RRAM_MAX_ITER; i++) {
    x = x - f / df;
    f = RRAMConduct2Vbd(x, type) - Vbd;
    df = d_RRAMConduct2Vbd(x, type);
  }
  if (fabs(f) > RRAM_TOLERANCE) {
    printf("\033[0;31mERROR: Newton's method failed to converge!\033[0m\n");
  }
  return x;
};

__global__ void Vbd2conductance(double *array, const CAMArrayDim camDim,
                                const RRAMCellType cellType) {
  getIx;
  getIy;

  uint32_t nx = camDim.nCols;
  uint32_t ny = camDim.nRows;

  outOfRangeReturn2D;
  uint32_t rowIdx = iy;
  uint32_t colIdx = ix;

  double lowerBdVbd = array[getCamIdx(rowIdx, colIdx, 0, camDim)];
  array[getCamIdx(rowIdx, colIdx, 0, camDim)] =
      solveConductanceFromVbd(lowerBdVbd, cellType);
  double upperBdVbd = array[getCamIdx(rowIdx, colIdx, 1, camDim)];
  array[getCamIdx(rowIdx, colIdx, 1, camDim)] =
      solveConductanceFromVbd(upperBdVbd, cellType);
};

__global__ void conductance2Vbd(double *array, const CAMArrayDim camDim,
                                const RRAMCellType cellType) {
  assert(camDim.nBoundaries == 2);

  uint32_t nx = camDim.nCols;
  uint32_t ny = camDim.nRows;
  getIx;
  getIy;
  outOfRangeReturn2D;
  uint32_t rowIdx = iy;
  uint32_t colIdx = ix;

  double lowerBdConductance = array[getCamIdx(rowIdx, colIdx, 0, camDim)];
  array[getCamIdx(rowIdx, colIdx, 0, camDim)] =
      RRAMConduct2Vbd(lowerBdConductance, cellType);
  double upperBdConductance = array[getCamIdx(rowIdx, colIdx, 1, camDim)];
  array[getCamIdx(rowIdx, colIdx, 1, camDim)] =
      RRAMConduct2Vbd(upperBdConductance, cellType);
};

__global__ void addBoundedGaussianVariation(double *array,
                                            const CAMArrayDim camDim,
                                            const double stdDev,
                                            const double bound,
                                            const double minConductance,
                                            const double maxConductance) {
  // printf("in addBoundedGaussianVariation()\n");
  assert(camDim.nBoundaries == 2);
  assert(minConductance <= maxConductance);
  uint32_t nx = camDim.nCols;
  uint32_t ny = camDim.nRows;
  getIx;
  getIy;
  getIdx2D;
  outOfRangeReturn2D;
  uint32_t rowIdx = iy;
  uint32_t colIdx = ix;

  // Initialize the random number generator
  curandState state;
  curand_init(clock64(), idx, 0, &state);

  double lowerBdConductance = array[getCamIdx(rowIdx, colIdx, 0, camDim)];
  double noise = max(min(curand_normal_double(&state) * stdDev, bound), -bound);
  assert(noise >= -bound);
  assert(noise <= bound);
  lowerBdConductance = max(
      min(array[getCamIdx(rowIdx, colIdx, 0, camDim)] + noise, maxConductance),
      minConductance);
  assert(lowerBdConductance >= minConductance);
  assert(lowerBdConductance <= maxConductance);
  array[getCamIdx(rowIdx, colIdx, 0, camDim)] = lowerBdConductance;

  double upperBdConductance = array[getCamIdx(rowIdx, colIdx, 1, camDim)];
  noise = max(min(curand_normal_double(&state) * stdDev, bound), -bound);
  assert(noise >= -bound);
  assert(noise <= bound);
  upperBdConductance = max(
      min(array[getCamIdx(rowIdx, colIdx, 1, camDim)] + noise, maxConductance),
      minConductance);
  assert(upperBdConductance >= minConductance);
  assert(upperBdConductance <= maxConductance);
  array[getCamIdx(rowIdx, colIdx, 1, camDim)] = upperBdConductance;
};

void addRRAMNoise(WriteNoise *writeNoise, ACAMArray *array) {
  assert(writeNoise->getNoiseConfig()->device == "RRAM");
  assert(writeNoise->getHasNoise());

  // get info
  uint32_t nRows = array->getNRows();
  uint32_t nCols = array->getNCols();
  assert(nRows != 0 && nCols != 0);
  uint32_t nBoundaries = array->getDim().nBoundaries;
  assert(nBoundaries == 2);
  std::string cellType = writeNoise->getCellDesign();
  std::map<std::string, std::map<std::string, std::string>> noiseType =
      writeNoise->getNoiseType();

  RRAMCellType cellTypeCUDA = INVALID_RRAM_CELL_TYPE;
  if (cellType == "6T2M") {
    cellTypeCUDA = CELL_6T2M;
  } else if (cellType == "8T2M") {
    cellTypeCUDA = CELL_8T2M;
  } else {
    throw std::runtime_error("Invalid RRAM cell type");
  }

  // copy data
  uint64_t nByte = nRows * nCols * nBoundaries * sizeof(double);
  double *camRawData_h = array->getData(FOR_CUDA_MEM_CPY);
  double *camRawData_d;
  CHECK(cudaMalloc(&camRawData_d, nByte));
  CHECK(cudaMemcpy(camRawData_d, camRawData_h, nByte, cudaMemcpyHostToDevice));

  // grid block size
  const dim3 block(RRAM_NOISE_THREAD_X, RRAM_NOISE_THREAD_Y);
  const dim3 grid((long long int)(nCols - 1) / block.x + 1,
                  (long long int)(nRows - 1) / block.y + 1);
  checkGridBlockSize(grid, block);

  // cuda stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // convert to conductance
  Vbd2conductance<<<grid, block, 0, stream>>>(camRawData_d, array->getDim(),
                                              cellTypeCUDA);

  // iterate through noise types
  for (auto &it : noiseType) {
    std::string noise = it.first;
    if (noise == "variation") {
      std::map<std::string, std::string> params = it.second;

      std::string noiseType = params["type"];

      if (noiseType == "bounded_gaussian") {
        double stdDev = std::stod(params["stdDev"]);
        double bound = std::stod(params["bound"]);
        addBoundedGaussianVariation<<<grid, block, 0, stream>>>(
            camRawData_d, array->getDim(), stdDev, bound,
            writeNoise->getMinConductance(), writeNoise->getMaxConductance());
      } else {
        throw std::runtime_error("Invalid variation type: " + noiseType);
      }
    } else {
      throw std::runtime_error("Invalid noise type: " + noise);
    }
  }

  // convert back to Vbd
  conductance2Vbd<<<grid, block, 0, stream>>>(camRawData_d, array->getDim(),
                                              cellTypeCUDA);

  // post process
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(camRawData_h, camRawData_d, nByte, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(camRawData_d));
  CHECK(cudaStreamDestroy(stream));
}