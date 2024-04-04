#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <iostream>

#include "config.h"

class Quantize {
 private:
  const uint16_t numBits;
  double minVal, maxVal;

 public:
  Quantize(QueryConfig *queryConfig) : numBits(queryConfig->getBit()) {
    std::cout << "in Quantize()" << std::endl;
    std::cout << "Quantize() done" << std::endl;
  }
};

#endif