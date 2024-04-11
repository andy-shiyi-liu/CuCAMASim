#ifndef MAPPING_H
#define MAPPING_H

#include <iostream>

#include "util/config.h"

class Mapping {
 private:
  const uint32_t rowSize, colSize;
  uint32_t rowCams = 0, colCams = 0;
  double *camData = nullptr, *queryData = nullptr;

 public:
  Mapping(ArrayConfig *arrayConfig)
      : rowSize(arrayConfig->row), colSize(arrayConfig->col) {
    std::cout << "in Mapping()" << std::endl;
    std::cout << "Mapping() done" << std::endl;
  }
  ~Mapping() {
    if (camData != nullptr) {
      delete[] camData;
      camData = nullptr;
    }
    if (queryData != nullptr) {
      delete[] queryData;
      queryData = nullptr;
    }
  }
};

#endif  // MAPPING_H