#ifndef WRITE_NOISE_H
#define WRITE_NOISE_H

#include "util/config.h"
#include "util/data.h"
#include <iostream>

class WriteNoise {
 private:
  NoiseConfig *noiseConfig;
  const bool hasNoise;
  std::map<std::string, std::map<std::string, std::string>> noiseType;
  std::string cellDesign;
  double minConductance;
  double maxConductance;

 public:
  WriteNoise(NoiseConfig *noiseConfig)
      : noiseConfig(noiseConfig),
        hasNoise(noiseConfig->hasWriteNoise),
        noiseType(noiseConfig->noiseType),
        cellDesign(noiseConfig->cellDesign),
        minConductance(noiseConfig->minConductance),
        maxConductance(noiseConfig->maxConductance) {
  }

  void addWriteNoise(CAMArrayBase *camArray);

  inline bool getHasNoise() const { return hasNoise; }
  inline const std::map<std::string, std::map<std::string, std::string>>& getNoiseType() const { return noiseType; }
  inline std::string getCellDesign() const { return cellDesign; }
  inline double getMinConductance() const { return minConductance; }
  inline double getMaxConductance() const { return maxConductance; }
  inline const NoiseConfig *getNoiseConfig() const { return noiseConfig; }
};

#endif  // WRITE_NOISE_H