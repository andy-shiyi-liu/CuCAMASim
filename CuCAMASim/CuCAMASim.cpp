#include "CuCAMASim.h"

#include <iostream>

#include "function/FunctionSimulator.h"
#include "util/config.h"

void CuCAMASim::write(CAMArrayBase *camArray) {
  std::cout << "*** Write Data to CAM Arrays ***" << std::endl;

  // Function Simulation
  if (config->queryConfig->funcSim == true) {
    functionSimulator->write(camArray);
  }

  // Performance Evaluation
  if (config->queryConfig->perfEval == true) {
    throw std::runtime_error("Performance Evaluation is not supported yet");
  }
}

void CuCAMASim::query(InputData *inputData, SimResult *simResult) {
  std::cout << "*** Query CAM Arrays ***" << std::endl;

  // Function Simulation
  if (config->queryConfig->funcSim == true) {
    functionSimulator->query(inputData, simResult);
  }

  // Performance Evaluation
  if (config->queryConfig->perfEval == true) {
    throw std::runtime_error("Performance Evaluation is not supported yet");
  }
}