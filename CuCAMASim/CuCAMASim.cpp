#include "CuCAMASim.h"

#include <iostream>

#include "util/config.h"
#include "function/FunctionSimulator.h"

void CuCAMASim::write(CAMData &CAMData) {
  std::cout << "in CuCAMASim::write()" << std::endl;
  std::cout <<"*** Write Data to CAM Arrays ***" <<std::endl;
  std::cout << "CAMData dims:" <<std::endl;
  CAMData.printDim();

  if (config->queryConfig->funcSim == true){
    functionSimulator->write(CAMData);
  }

  std::cout << "CuCAMASim::write() done" << std::endl;
}