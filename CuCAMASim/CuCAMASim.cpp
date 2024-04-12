#include "CuCAMASim.h"

#include <iostream>

#include "util/config.h"
#include "function/FunctionSimulator.h"

void CuCAMASim::write(CAMData *camData) {
  std::cout << "in CuCAMASim::write()" << std::endl;
  std::cout <<"*** Write Data to CAM Arrays ***" <<std::endl;
  std::cout << "CAMData dims:" <<std::endl;
  camData->printDim();

  if (config->queryConfig->funcSim == true){
    functionSimulator->write(camData);
  }

  std::cout << "CuCAMASim::write() done" << std::endl;
}