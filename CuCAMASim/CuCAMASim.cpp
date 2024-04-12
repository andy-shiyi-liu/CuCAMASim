#include "CuCAMASim.h"

#include <iostream>

#include "util/config.h"
#include "function/FunctionSimulator.h"

void CuCAMASim::write(CAMArrayBase *camArray) {
  std::cout << "in CuCAMASim::write()" << std::endl;
  std::cout <<"*** Write Data to CAM Arrays ***" <<std::endl;
  std::cout << "CAMArray dims:" <<std::endl;
  camArray->printDim();

  if (config->queryConfig->funcSim == true){
    functionSimulator->write(camArray);
  }

  std::cout << "CuCAMASim::write() done" << std::endl;
}