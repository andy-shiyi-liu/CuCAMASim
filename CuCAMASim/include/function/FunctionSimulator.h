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
    quantizer = new Quantize(camConfig->queryConfig);
    converter = new ConvertToPhys(camConfig->cellConfig);
    mapping = new Mapping(camConfig->arrayConfig);
    search = new CAMSearch(camConfig->queryConfig, camConfig->arrayConfig);
    writeNoise = new WriteNoise(camConfig->noiseConfig);
    std::cout << "FunctionSimulator() done" << std::endl;
  }
  void write(CAMData &CAMData);
  ~FunctionSimulator() {
    delete quantizer;
    quantizer = nullptr;
    delete converter;
    converter = nullptr;
    delete mapping;
    mapping = nullptr;
    delete search;
    search = nullptr;
    delete writeNoise;
    writeNoise = nullptr;
  }
};

#endif  // FUNCTION_SIMULATOR_H