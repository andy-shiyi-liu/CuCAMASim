#ifndef ARCHESTIMATOR_H
#define ARCHESTIMATOR_H

#include <iostream>

#include "util/config.h"

class ArchEstimator {
 private:
  ArchConfig *archConfig;
  uint64_t nCol;
  uint64_t nRow;

 public:
  ArchEstimator(CamConfig *camConfig) {
    std::cout << "in ArchEstimator()" << std::endl;
    archConfig = camConfig->archConfig;
    nCol = camConfig->arrayConfig->col;
    nRow = camConfig->arrayConfig->row;
    std::cout << "ArchEstimator() done" << std::endl;
  };
};

#endif