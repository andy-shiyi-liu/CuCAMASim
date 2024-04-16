#ifndef ARCHESTIMATOR_H
#define ARCHESTIMATOR_H

#include <iostream>

#include "util/config.h"

class ArchEstimator {
 private:
  ArchConfig *archConfig;
  uint32_t nCol;
  uint32_t nRow;

 public:
  ArchEstimator(CamConfig *camConfig) {
    archConfig = camConfig->archConfig;
    nCol = camConfig->arrayConfig->col;
    nRow = camConfig->arrayConfig->row;
  };
};

#endif