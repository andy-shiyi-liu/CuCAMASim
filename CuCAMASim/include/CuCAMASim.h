#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>

#include "function/FunctionSimulator.h"
#include "util/config.h"
#include "arch/ArchEstimator.h"

class CuCAMASim {
 private:
  const CamConfig *config;
  const FunctionSimulator *functionSimulator;
  const ArchEstimator *archEstimator;

 public:
  CuCAMASim(CamConfig *camConfig) : config(camConfig) {
  std::cout << "in CuCAMASim()" << std::endl;
  functionSimulator = new FunctionSimulator(camConfig);
  archEstimator = new ArchEstimator(camConfig);
  std::cout << "CuCAMASim() done" << std::endl;
};
  ~CuCAMASim(){
    delete functionSimulator;
    delete archEstimator;
  };
};
#endif