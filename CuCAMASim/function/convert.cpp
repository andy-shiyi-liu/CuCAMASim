#include "function/convert.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>

#include "util/data.h"

template <typename T>
T num2Vbd(T num, T numMin, T numMax, T VbdMin, T VbdMax) {
  assert(numMin < numMax && VbdMin < VbdMax && "ERROR: invalid range");
  T percentage = (num - numMin) / (numMax - numMin);
  return VbdMin + (VbdMax - VbdMin) * percentage;
}

ConvertToPhys::ConvertToPhys(CellConfig *cellConfig,
                             MappingConfig *mappingConfig)
    : physicalRep(cellConfig->representation),
      cell(cellConfig->type),
      device(cellConfig->device),
      design(cellConfig->design),
      cellConfig(cellConfig) {
  std::cout << "in ConvertToPhys()" << std::endl;
  if (cellConfig->design == "6T2M") {
    conduct2Vbd = &ConvertToPhys::conduct2Vbd6T2M;
  } else if (cellConfig->design == "8T2M") {
    conduct2Vbd = &ConvertToPhys::conduct2Vbd8T2M;
  }
  // if "N2VConvert", set min/max Vbd according to the config
  if (mappingConfig->strategies.find("N2VConvert") !=
      mappingConfig->strategies.end()) {
    if (this->device != "RRAM") {
      throw std::runtime_error("ERROR: N2VConvert is only for RRAM devices");
    }
    double maxConvertConductance, minConvertConductance;
    std::map<std::string, std::string> N2VCvtCfg =
        mappingConfig->strategies["N2VConvert"];
    if (N2VCvtCfg.find("maxConvertConductance") != N2VCvtCfg.end()) {
      maxConvertConductance = std::stod(N2VCvtCfg["maxConvertConductance"]);
    } else {
      maxConvertConductance = cellConfig->maxConductance;
    }
    if (N2VCvtCfg.find("minConvertConductance") != N2VCvtCfg.end()) {
      minConvertConductance = std::stod(N2VCvtCfg["minConvertConductance"]);
    } else {
      minConvertConductance = cellConfig->minConductance;
    }
    if (maxConvertConductance <= minConvertConductance) {
      throw std::runtime_error(
          "ERROR: maxConvertConductance should be larger than "
          "minConvertConductance");
    }
    if (maxConvertConductance > cellConfig->maxConductance) {
      throw std::runtime_error(
          "ERROR: maxConvertConductance should be smaller or equal to the "
          "maxConductance in the cell config");
    }
    if (minConvertConductance < cellConfig->minConductance) {
      throw std::runtime_error(
          "ERROR: minConvertConductance should be larger or equal to the "
          "minConductance in the cell config");
    }
    VbdMax = (this->*conduct2Vbd)(maxConvertConductance);
    VbdMin = (this->*conduct2Vbd)(minConvertConductance);
  } else {
    VbdMax = (this->*conduct2Vbd)(cellConfig->maxConductance);
    VbdMin = (this->*conduct2Vbd)(cellConfig->minConductance);
    assert(VbdMax > VbdMin && "ERROR: VbdMax should be larger than VbdMin");
  }

  // if N2VConvert, set convert margin and convert ranges.
  if (mappingConfig->strategies.find("N2VConvert") !=
      mappingConfig->strategies.end()) {
    if (this->device != "RRAM") {
      throw std::runtime_error("ERROR: N2VConvert is only for RRAM devices");
    }
    std::map<std::string, std::string> N2VCvtCfg =
        mappingConfig->strategies["N2VConvert"];

    // setup lineConvertRangeMargin and queryClipRangeMargin
    if (N2VCvtCfg.find("lineConvertRangeMargin") != N2VCvtCfg.end()) {
      lineConvertRangeMargin = std::stod(N2VCvtCfg["lineConvertRangeMargin"]);
    } else {
      lineConvertRangeMargin = 0.05;
    }
    if (N2VCvtCfg.find("queryClipRangeMargin") != N2VCvtCfg.end()) {
      queryClipRangeMargin = std::stod(N2VCvtCfg["queryClipRangeMargin"]);
    } else {
      queryClipRangeMargin = 0.03;
    }
    if (queryClipRangeMargin > lineConvertRangeMargin) {
      std::cout << "\033[93mWARNING: queryClipRangeMargin should be smaller or "
                   "equal than lineConvertRangeMargin\033[0m"
                << std::endl;
    }

    // setup lineConvertRangeMin and lineConvertRangeMax
    if (N2VCvtCfg.find("lineConvertRangeMin") != N2VCvtCfg.end() ||
        N2VCvtCfg.find("lineConvertRangeMax") != N2VCvtCfg.end()) {
      if (N2VCvtCfg.find("lineConvertRangeMin") == N2VCvtCfg.end() ||
          N2VCvtCfg.find("lineConvertRangeMax") == N2VCvtCfg.end()) {
        throw std::runtime_error(
            "ERROR: lineConvertRangeMin and lineConvertRangeMax should be both "
            "set");
      }
      if (N2VCvtCfg.find("lineConvertRangeMargin") != N2VCvtCfg.end() &&
          std::stod(N2VCvtCfg["lineConvertRangeMargin"]) != 0.0) {
        throw std::runtime_error(
            "ERROR: lineConvertRangeMargin should be 0 when "
            "lineConvertRangeMin and lineConvertRangeMax are set");
      }

      lineConvertRangeMin = std::stod(N2VCvtCfg["lineConvertRangeMin"]);
      lineConvertRangeMax = std::stod(N2VCvtCfg["lineConvertRangeMax"]);
    }

    // setup queryClipRangeMin and queryClipRangeMax
    if (N2VCvtCfg.find("queryClipRangeMin") != N2VCvtCfg.end() ||
        N2VCvtCfg.find("queryClipRangeMax") != N2VCvtCfg.end()) {
      if (N2VCvtCfg.find("queryClipRangeMin") == N2VCvtCfg.end() ||
          N2VCvtCfg.find("queryClipRangeMax") == N2VCvtCfg.end()) {
        throw std::runtime_error(
            "ERROR: queryClipRangeMin and queryClipRangeMax should be both "
            "set");
      }
      if (N2VCvtCfg.find("queryClipRangeMargin") != N2VCvtCfg.end() &&
          std::stod(N2VCvtCfg["queryClipRangeMargin"]) != 0.0) {
        throw std::runtime_error(
            "ERROR: queryClipRangeMargin should be 0 when "
            "queryClipRangeMin and queryClipRangeMax are set");
      }

      queryClipRangeMin = std::stod(N2VCvtCfg["queryClipRangeMin"]);
      queryClipRangeMax = std::stod(N2VCvtCfg["queryClipRangeMax"]);
    }
  }
  std::cout << "ConvertToPhys() done" << std::endl;
}

// Converts data to a physical representation suitable for write operations.
// Depending on the CAM cell type (e.g., ACAM), it converts data to a physical
// voltage representation.
void ConvertToPhys::write(CAMArrayBase *camArray) {
  if (cell == "ACAM") {
    if (camArray->getType() != ACAM_ARRAY_COLD_START) {
      throw std::runtime_error("ERROR: CAMArray type should be ACAM_ARRAY");
    }
    ACAMArray *acamArray = static_cast<ACAMArray *>(camArray);
    double boundaryMin = acamArray->min();
    double boundaryMax = acamArray->max();

    assert(boundaryMin < boundaryMax &&
           "ERROR: boundaryMin should be smaller than boundaryMax");
    assert(!std::isinf(boundaryMin) &&
           "ERROR: boundaryMin should not be infinity");
    assert(!std::isinf(boundaryMax) &&
           "ERROR: boundaryMax should not be infinity");

    double Nrange = boundaryMax - boundaryMin;

    // setup lineConvertRange
    if (std::isnan(lineConvertRangeMin)) {
      lineConvertRangeMin = boundaryMin - lineConvertRangeMargin * Nrange;
    }
    if (std::isnan(lineConvertRangeMax)) {
      lineConvertRangeMax = boundaryMax + lineConvertRangeMargin * Nrange;
    }

    // setup queryClipRange
    if (std::isnan(queryClipRangeMin)) {
      queryClipRangeMin = boundaryMin - queryClipRangeMargin * Nrange;
    }
    if (std::isnan(queryClipRangeMax)) {
      queryClipRangeMax = boundaryMax + queryClipRangeMargin * Nrange;
    }
    acamN2V(acamArray);
  } else {
    throw std::runtime_error(
        "Cell types other than ACAM are not implemented yet");
  }
}

// convert the data in camArray from numerical to voltage representation.
void ConvertToPhys::acamN2V(ACAMArray *camArray) {
  for (uint32_t i = 0; i < camArray->getNRows(); i++) {
    for (uint32_t j = 0; j < camArray->getNCols(); j++) {
      if (std::isinf(camArray->at(i, j, 0))) {
        camArray->set(i, j, 0, VbdMin);
      } else {
        camArray->set(i, j, 0,
                     num2Vbd<double>(camArray->at(i, j, 0), lineConvertRangeMin,
                                     lineConvertRangeMax, VbdMin, VbdMax));
      }
      if (std::isinf(camArray->at(i, j, 1))) {
        camArray->set(i, j, 1, VbdMax);
      } else {
        camArray->set(i, j, 1,
                     num2Vbd<double>(camArray->at(i, j, 1), lineConvertRangeMin,
                                     lineConvertRangeMax, VbdMin, VbdMax));
      }
    }
  }
  return;
}