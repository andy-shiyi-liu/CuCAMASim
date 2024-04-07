#ifndef CONVERT_H
#define CONVERT_H

#include <string>
#include <iostream>

#include "util/config.h"

class ConvertToPhys {
 private:
  const std::string physicalRep;
  const std::string cell;
  const std::string device;
  const std::string design;
  CellConfig *cellConfig;
  typedef double (ConvertToPhys::*Conduct2VbdFuncPtr)(double);
  Conduct2VbdFuncPtr conduct2Vbd;
  double VbdMin, VbdMax;

  double conduct2Vbd6T2M(double x) {
    return -0.18858359 * std::exp(-0.16350861 * x) + 0.00518336 * x +
           0.56900874;
  };
  double conduct2Vbd8T2M(double x) {
    return -2.79080037e-01 * std::exp(-1.24915981e-01 * x) +
           6.36010747e-04 * x + 1.00910243;
  };

 public:
ConvertToPhys(CellConfig *cellConfig)
        : physicalRep(cellConfig->getPhysicalRep()),
            cell(cellConfig->getType()),
            device(cellConfig->getDevice()),
            design(cellConfig->getDesign()),
            cellConfig(cellConfig) {
    std::cout << "in ConvertToPhys()" << std::endl;
    if (cellConfig->getDesign() == "6T2M") {
        conduct2Vbd = &ConvertToPhys::conduct2Vbd6T2M;
    } else if (cellConfig->getDesign() == "8T2M") {
        conduct2Vbd = &ConvertToPhys::conduct2Vbd8T2M;
    }
    VbdMax = (this->*conduct2Vbd)(cellConfig->getMaxConductance());
    VbdMin = (this->*conduct2Vbd)(cellConfig->getMinConductance());
    std::cout << "ConvertToPhys() done" << std::endl;
}
~ConvertToPhys() {}
};

#endif  // CONVERT_H