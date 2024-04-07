#ifndef FUNCTION_SIMULATOR_H
#define FUNCTION_SIMULATOR_H

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "util/config.h"
#include "convert.h"
#include "mapping.h"
#include "quantize.h"
#include "search.h"
#include "writeNoise.h"

class FunctionSimulator {
 private:
  CamConfig *camConfig;
  Quantize *quantizer;
  ConvertToPhys *converter;
  Mapping *mapping;
  CAMSearch *search;
  WriteNoise *writeNoise;

 public:
  FunctionSimulator(CamConfig *camConfig) : camConfig(camConfig) {
    std::cout << "in FunctionSimulator()" << std::endl;
    quantizer = new Quantize(camConfig->getQueryConfig());
    converter = new ConvertToPhys(camConfig->getCellConfig());
    mapping = new Mapping(camConfig->getArrayConfig());
    search = new CAMSearch(camConfig->getQueryConfig(), camConfig->getArrayConfig());
    writeNoise = new WriteNoise(camConfig->getNoiseConfig());
    std::cout << "FunctionSimulator() done" << std::endl;
  }
  ~FunctionSimulator() {
    delete quantizer;
    delete converter;
    delete mapping;
    delete search;
    delete writeNoise;
  }
};

#endif  // FUNCTION_SIMULATOR_H