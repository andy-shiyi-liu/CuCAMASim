#ifndef MAPPING_H
#define MAPPING_H

#include <iostream>

#include "util/config.h"
#include "util/data.h"


class  Mapping {
 private:
  const uint32_t rowSize, colSize;
  uint32_t rowCams = (uint32_t)-1, colCams = (uint32_t)-1;
  uint64_t camSize = (uint64_t)-1;
  double *camData = nullptr, *queryData = nullptr;
  double checkSize(CAMArray *camData);

 public:
  Mapping(ArrayConfig *arrayConfig)
      : rowSize(arrayConfig->row), colSize(arrayConfig->col) {
    std::cout << "in Mapping()" << std::endl;
    std::cout << "Mapping() done" << std::endl;
  }
  void addNewMapping(CAMArray *camData);
  double write(CAMArray *camData);
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