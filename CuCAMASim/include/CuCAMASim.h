#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>

#include "arch/ArchEstimator.h"
#include "function/FunctionSimulator.h"
#include "util/config.h"

class SimResult {
 public:
  struct {
    double latency, energy;
    bool valid = false;
  } perf;
  struct {
    bool valid = false;
  } func;
};

class CuCAMASim {
 private:
  const CamConfig *config;
  const FunctionSimulator *functionSimulator;
  const ArchEstimator *archEstimator;
  SimResult simResult;

 public:
  CuCAMASim(CamConfig *camConfig) : config(camConfig) {
    std::cout << "in CuCAMASim()" << std::endl;
    functionSimulator = new FunctionSimulator(camConfig);
    archEstimator = new ArchEstimator(camConfig);
    std::cout << "CuCAMASim() done" << std::endl;
  };
  ~CuCAMASim() {
    delete functionSimulator;
    delete archEstimator;
  };
};
#endif