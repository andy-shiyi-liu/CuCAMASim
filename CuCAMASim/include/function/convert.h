#ifndef CONVERT_H
#define CONVERT_H

#include <iostream>
#include <limits>
#include <string>

#include "util/config.h"
#include "util/data.h"
#include "function/cuda/rram.cuh"

class ConvertToPhys {
 private:
  const std::string physicalRep;
  const std::string cell;
  const std::string device;
  const std::string design;
  CellConfig *cellConfig;
  typedef double (*Conduct2VbdFuncPtr)(double); 
  Conduct2VbdFuncPtr conduct2Vbd;
  double VbdMin = std::numeric_limits<double>::quiet_NaN(),
         VbdMax = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMargin = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMargin = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMin = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMax = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMin = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMax = std::numeric_limits<double>::quiet_NaN();

  void acamN2V(ACAMArray *camArray) const;

 public:
  ConvertToPhys(CellConfig *cellConfig, MappingConfig *mappingConfig);

  void write(CAMArrayBase *camArray);
  void query(InputData *inputData) const ;

  ~ConvertToPhys() {}
};

#endif  // CONVERT_H