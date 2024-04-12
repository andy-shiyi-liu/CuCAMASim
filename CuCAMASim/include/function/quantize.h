#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <iostream>

#include "util/config.h"
#include "util/data.h"

class Quantize {
 private:
  const uint16_t numBits;
  double minVal, maxVal;

 public:
  Quantize(QueryConfig* queryConfig) : numBits(queryConfig->bit) {
    std::cout << "in Quantize()" << std::endl;
    std::cout << "Quantize() done" << std::endl;
  }
  void write(CAMArray* camData);
};

#endif