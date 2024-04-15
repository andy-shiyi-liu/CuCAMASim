#include "CuCAMASim.h"

#include <iostream>

#include "function/FunctionSimulator.h"
#include "util/config.h"

void CuCAMASim::write(CAMArrayBase *camArray) {
  std::cout << "in CuCAMASim::write()" << std::endl;
  std::cout << "*** Write Data to CAM Arrays ***" << std::endl;
  std::cout << "CAMArray dims:" << std::endl;
  camArray->printDim();

  // Function Simulation
  if (config->queryConfig->funcSim == true) {
    functionSimulator->write(camArray);
  }

  // Performance Evaluation
  if (config->queryConfig->perfEval == true) {
    throw std::runtime_error("Performance Evaluation is not supported yet");
  }

  std::cout << "CuCAMASim::write() done" << std::endl;
}

void CuCAMASim::query(InputData *inputData,SimResult *simResult) {
  std::cout << "*** Query CAM Arrays ***" << std::endl;

  // Function Simulation
  if (config->queryConfig->funcSim == true) {
    functionSimulator->query(inputData, simResult);
  }
  if (config->queryConfig->perfEval == true) {
    throw std::runtime_error("Performance Evaluation is not supported yet");
  }
}