#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>

#include "function/FunctionSimulator.h"
#include "util/config.h"

class CuCAMASim {
 private:
  const CamConfig *config;
  const FunctionSimulator *functionSimulator;

 public:
  CuCAMASim(CamConfig *camConfig);
};
#endif