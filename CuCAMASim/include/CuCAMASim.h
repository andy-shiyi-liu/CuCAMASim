#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>

#include "arch/ArchEstimator.h"
#include "function/FunctionSimulator.h"
#include "performance/PerformanceEvaluator.h"
#include "util/config.h"
#include "util/data.h"

class CuCAMASim {
 private:
  CamConfig *config;
  FunctionSimulator *functionSimulator;
  ArchEstimator *archEstimator;
  PerformanceEvaluator *performanceEvaluator;
  SimResult *simResult;

 public:
  CuCAMASim(CamConfig *camConfig) : config(camConfig) {
    functionSimulator = new FunctionSimulator(camConfig);
    archEstimator = new ArchEstimator(camConfig);
    performanceEvaluator = new PerformanceEvaluator();
    simResult = new SimResult();
  };
  void write(CAMArrayBase *camArray);
  void query(InputData *inputData, SimResult *simResult);

  inline SimResult *getSimResult() const { return simResult; };

  ~CuCAMASim() {
    delete functionSimulator;
    functionSimulator = nullptr;
    delete archEstimator;
    archEstimator = nullptr;
    delete performanceEvaluator;
    performanceEvaluator = nullptr;
    delete simResult;
    simResult = nullptr;
  };
};
#endif