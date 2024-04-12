#include "function/FunctionSimulator.h"

#include <iostream>

#include "function/cuda/distance.cuh"
#include "util/data.h"

void FunctionSimulator::write(CAMArrayBase *camArray) {
  // 1. Quantization (optional for ACAM)
  if (camConfig->arrayConfig->cell != "ACAM") {
    quantizer->write(camArray);
  }

  // 2. Conversion to voltage/conductance representation
  converter->write(camArray);

  // 3. add new mapping
  mapping->addNewMapping(camArray);

  // 4. add write noise
  writeNoise->addWriteNoise(camArray);

  // 5. Data mapping to CAM arrays
  mapping->write(camArray);
}

void FunctionSimulator::query(InputData *inputData) {
  inputData->getNFeatures();

  // 1. Quantization (optional for ACAM)
  if (camConfig->arrayConfig->cell != "ACAM") {
    quantizer->query(inputData);
  }

  // 2. Conversion to the same representation
  converter->query(inputData);

  // 3. Data mapping to CAM arrays for queries
  mapping->query(inputData);

  // 4. Searching in each array and merging results
  search->defineSearchArea(mapping->getRowCams(), mapping->getColCams());
  search->search(mapping->getCamData(), mapping->getQueryData());

  std::cerr << "\033[33mWARNING: FunctionSimulator::query() is still under "
               "development\033[0m"
            << std::endl;
}