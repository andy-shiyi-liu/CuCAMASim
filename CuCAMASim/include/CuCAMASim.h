#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>

#include "arch/ArchEstimator.h"
#include "function/FunctionSimulator.h"
#include "performance/PerformanceEvaluator.h"
#include "util/config.h"
#include "util/data.h"

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
  CamConfig *config;
  FunctionSimulator *functionSimulator;
  ArchEstimator *archEstimator;
  PerformanceEvaluator *performanceEvaluator;
  SimResult simResult;

 public:
  CuCAMASim(CamConfig *camConfig) : config(camConfig) {
    std::cout << "in CuCAMASim()" << std::endl;
    functionSimulator = new FunctionSimulator(camConfig);
    archEstimator = new ArchEstimator(camConfig);
    performanceEvaluator = new PerformanceEvaluator();
    std::cout << "CuCAMASim() done" << std::endl;
  };
  void write(CAMData *CAMData);
  ~CuCAMASim() {
    delete functionSimulator;
    functionSimulator = nullptr;
    delete archEstimator;
    archEstimator = nullptr;
    delete performanceEvaluator;
    performanceEvaluator = nullptr;
  };
};
#endif