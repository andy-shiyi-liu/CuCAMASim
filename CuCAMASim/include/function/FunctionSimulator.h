#ifndef FUNCTION_SIMULATOR_H
#define FUNCTION_SIMULATOR_H

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "config.h"
#include "convert.h"
#include "mapping.h"
#include "quantize.h"
#include "search.h"
#include "writeNoise.h"

class FunctionSimulator {
 private:
  const CamConfig *camConfig;
  const Quantize *quantizer;
  const ConvertToPhys *converter;
  const Mapping *mapping;
  const CAMSearch *search;
  const WriteNoise *writeNoise;

 public:
  FunctionSimulator(const CamConfig *camConfig) : camConfig(camConfig) {
    std::cout << "in FunctionSimulator()" << std::endl;
    quantizer = new Quantize();
    converter = new ConvertToPhys();
    mapping = new Mapping();
    search = new CAMSearch();
    writeNoise = new WriteNoise();
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