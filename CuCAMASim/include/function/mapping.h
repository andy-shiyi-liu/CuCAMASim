#ifndef MAPPING_H
#define MAPPING_H

#include <iostream>

#include "config.h"

class Mapping {
 private:
  const uint32_t rowSize, colSize;
  uint32_t rowCams = 0, colCams = 0;
  double *camData = NULL, *queryData = NULL;

 public:
  Mapping(ArrayConfig *arrayConfig)
      : rowSize(arrayConfig->getRow()), colSize(arrayConfig->getCol()) {
    std::cout << "in Mapping()" << std::endl;
    std::cout << "Mapping() done" << std::endl;
  }
  ~Mapping() {
    if (camData != NULL) {
      delete[] camData;
    }
    if (queryData != NULL) {
      delete[] queryData;
    }
  }
};

#endif  // MAPPING_H