#ifndef CONVERT_H
#define CONVERT_H

#include <iostream>
#include <limits>
#include <string>

#include "util/config.h"
#include "util/data.h"

class ConvertToPhys {
 private:
  const std::string physicalRep;
  const std::string cell;
  const std::string device;
  const std::string design;
  CellConfig *cellConfig;
  typedef double (ConvertToPhys::*Conduct2VbdFuncPtr)(double) const;
  Conduct2VbdFuncPtr conduct2Vbd;
  double VbdMin = std::numeric_limits<double>::quiet_NaN(),
         VbdMax = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMargin = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMargin = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMin = std::numeric_limits<double>::quiet_NaN(),
         lineConvertRangeMax = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMin = std::numeric_limits<double>::quiet_NaN(),
         queryClipRangeMax = std::numeric_limits<double>::quiet_NaN();

  double conduct2Vbd6T2M(double x) const {
    return -0.18858359 * std::exp(-0.16350861 * x) + 0.00518336 * x +
           0.56900874;
  };
  double conduct2Vbd8T2M(double x) const {
    return -2.79080037e-01 * std::exp(-1.24915981e-01 * x) +
           6.36010747e-04 * x + 1.00910243;
  };
  void acamN2V(ACAMArray *camArray) const;

 public:
  ConvertToPhys(CellConfig *cellConfig, MappingConfig *mappingConfig);

  void write(CAMArrayBase *camArray);
  void query(InputData *inputData) const ;

  ~ConvertToPhys() {}
};

#endif  // CONVERT_H