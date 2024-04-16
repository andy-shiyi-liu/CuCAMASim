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
  }

  void write(CAMArrayBase* camArray);
  void query(InputData* inputData);
};

#endif