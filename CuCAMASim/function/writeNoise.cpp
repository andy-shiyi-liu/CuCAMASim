#include "function/writeNoise.h"

#include <iostream>

#include "util/data.h"

void WriteNoise::addWriteNoise(CAMData *camData) {
  std::cerr << "\033[33mWARNING: WriteNoise::addWriteNoise() is not "
               "implemented yet\033[0m"
            << std::endl;
  camData->at(0, 0, 0);
  return;
}