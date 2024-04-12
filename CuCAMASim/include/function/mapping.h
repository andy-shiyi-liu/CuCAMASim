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
  CAMDataBase *camData = nullptr;
  double *queryData = nullptr;
  double checkSize(CAMArrayBase *camArray);

 public:
  Mapping(ArrayConfig *arrayConfig)
      : rowSize(arrayConfig->row), colSize(arrayConfig->col) {
    std::cout << "in Mapping()" << std::endl;
    std::cout << "Mapping() done" << std::endl;
  }
  void addNewMapping(CAMArrayBase *camArray);
  double write(CAMArrayBase *camArray);
  ~Mapping() {
    if (camData != nullptr) {
      delete camData;
      camData = nullptr;
    }
    if (queryData != nullptr) {
      delete[] queryData;
      queryData = nullptr;
    }
  }
};

#endif  // MAPPING_H