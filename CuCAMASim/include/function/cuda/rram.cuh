#ifndef RRAM_CUH
#define RRAM_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

enum RRAMCellType {
  CELL_6T2M,
  CELL_8T2M,
};

extern "C" {
__host__ __device__ inline double RRAMConduct2Vbd6T2M(double x) {
  return -0.18858359 * exp(-0.16350861 * (x)) + 0.00518336 * (x) + 0.56900874;
};

__host__ __device__ inline double RRAMConduct2Vbd8T2M(double x) {
  return -2.79080037e-01 * exp(-1.24915981e-01 * (x)) + 6.36010747e-04 * (x) +
         1.00910243;
};

__host__ __device__ inline double RRAMConduct2Vbd(double x, RRAMCellType type) {
  switch (type) {
    case CELL_6T2M:
      return RRAMConduct2Vbd6T2M(x);
    case CELL_8T2M:
      return RRAMConduct2Vbd8T2M(x);
    default:
      printf("\033[0;31mERROR: Invalid RRAM cell type\033[0m\n");
      return 0;
  }
}
}

#endif