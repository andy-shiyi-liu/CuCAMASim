#include "CuCAMASim.h"

#include <iostream>

#include "util/config.h"
#include "function/FunctionSimulator.h"

CuCAMASim::CuCAMASim(CamConfig *camConfig) : config(camConfig) {
  std::cout << "in CuCAMASim()" << std::endl;
  functionSimulator = new FunctionSimulator(camConfig);
  std::cout << "CuCAMASim() done" << std::endl;
}