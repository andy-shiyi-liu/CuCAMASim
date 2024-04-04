#include "CuCAMASim.h"

#include <iostream>

#include "config.h"

CuCAMASim::CuCAMASim(CamConfig *camConfig) : config(camConfig) {
  std::cout << "in CuCAMASim()" << std::endl;
}