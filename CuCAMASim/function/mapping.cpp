#include "function/mapping.h"

#include <cmath>
#include <iostream>

#include "util/data.h"

void Mapping::addNewMapping(CAMArray *camData) {
  std::cerr << "\033[33mWARNING: Mapping::addNewMapping() is not implemented "
               "yet\033[0m"
            << std::endl;
  camData->getType();
  return;
}

// Check if the size of the write data is within the maximum size of the CAM.
// Args:
//     camData: Input data to be checked.
// Returns:
//     camUsage: Fraction of CAM array usage.
// Raises:
//     NotImplementedError: If the CAM size is smaller than the dataset size.
double Mapping::checkSize(CAMArray *camData) {
  double camUsage = 0.0;
  uint64_t dataSize = camData->getNRows() * camData->getNCols();

  if (camSize == (uint64_t)-1) {
    camSize = dataSize;
    camUsage = 1;
  } else if (camSize < dataSize) {
    throw std::runtime_error(
        "CAM Size is smaller than dataset size. Not supported now.");
  } else {
    std::cout
        << "CAM size is larger than dataset size. Write all data into CAM."
        << std::endl;
    camUsage = (double)dataSize / camSize;
  }
  return camUsage;
}

double Mapping::write(CAMArray *camData) {
  uint32_t nRows = camData->getNRows();
  uint32_t nCols = camData->getNCols();

  rowCams = std::ceil(nRows / rowSize);
  colCams = std::ceil(nCols / colSize);

  double camUsage = checkSize(camData);
  if (camData->getType() == ACAM_ARRAY){

  }else{
    throw std::runtime_error("Write data other than ACAM is not supported yet");
  }

  std::cerr
      << "\033[33mWARNING: Mapping::write() is still under development!\033[0m"
      << std::endl;
  return camUsage;
}