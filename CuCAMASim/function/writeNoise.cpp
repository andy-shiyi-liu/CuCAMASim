#include "function/writeNoise.h"

#include <iostream>

#include "function/cuda/rram.cuh"
#include "util/data.h"

void WriteNoise::addWriteNoise(CAMArrayBase *camArray) {
  if (!hasNoise) {
    return;
  }

  if (noiseConfig->device == "RRAM") {
    if (camArray->getType() != ACAM_ARRAY_COLD_START &&
        camArray->getType() != ACAM_ARRAY_EXISTING_DATA) {
      throw std::runtime_error("RRAM is only supported for ACAM array");
    }
    addRRAMNoise(this, dynamic_cast<ACAMArray *>(camArray));
  } else {
    throw std::runtime_error("Add write noise for device '" + noiseConfig->device +
                             "' is not supported yet");
  }

  std::cerr << "\033[33mWARNING: WriteNoise::addWriteNoise() is still under "
               "development"
            << std::endl;
  return;
}