#ifndef CONFIG_H
#define CONFIG_H

#include <yaml-cpp/yaml.h>

#include <string>
class Config {};

class ArchConfig : public Config {
 private:
  const uint64_t arrays_per_mat, mats_per_bank, subarrays_per_array;

 public:
  ArchConfig(uint64_t arrays_per_mat, uint64_t mats_per_bank,
             uint64_t subarrays_per_array);
  ArchConfig(YAML::Node archConfig);
  void print();
};

class QueryConfig : public Config {
 private:
  const bool funcSim, perfEval;
  const uint16_t bit;
  const double distanceParameter, parameter;
  const std::string distance, searchScheme;

 public:
  QueryConfig(YAML::Node queryConfig);
  void print();
  uint16_t getBit() { return bit; }
  double getDistanceParameter() { return distanceParameter; }
  double getParameter() { return parameter; }
  std::string getDistance() { return distance; }
  std::string getSearchScheme() { return searchScheme; }
  bool getFuncSim() { return funcSim; }
  bool getPerfEval() { return perfEval; }
};

class ArrayConfig : public Config {
 private:
  const uint16_t bit;
  const std::string cell;
  const uint32_t col;
  const uint32_t row;
  const std::string sensing;
  const double sensingLimit;
  const bool useEVACAMCost;

 public:
  ArrayConfig(YAML::Node arrayConfig);
  uint16_t getBit() { return bit; }
  uint32_t getCol() { return col; }
  uint32_t getRow() { return row; }
  std::string getCell() { return cell; }
  std::string getSensing() { return sensing; }
  double getSensingLimit() { return sensingLimit; }
  bool getUseEVACAMCost() { return useEVACAMCost; }
  void print();
};

class NoiseConfig : public Config {
 private:
  const std::string cellDesign;
  const std::string device;
  const bool hasWriteNoise;
  const double maxConductance;
  const double minConductance;
  std::map<std::string, std::map<std::string, std::string>> noiseType;

 public:
  NoiseConfig(YAML::Node noiseConfig);
  std::string getCellDesign() { return cellDesign; }
  std::string getDevice() { return device; }
  bool getHasWriteNoise() { return hasWriteNoise; }
  double getMaxConductance() { return maxConductance; }
  double getMinConductance() { return minConductance; }
  std::map<std::string, std::map<std::string, std::string>> getNoiseType() {
    return noiseType;
  }
  void print();
};

class CellConfig : public Config {
 private:
  const std::string design;
  const std::string device;
  const double maxConductance;
  const double minConductance;
  const std::string representation;
  const std::string type;
  NoiseConfig *noiseConfig;

 public:
  CellConfig(YAML::Node cellConfig, NoiseConfig *noiseConfig);
  std::string getDesign() { return design; }
  std::string getDevice() { return device; }
  std::string getPhysicalRep() { return representation; }
  std::string getType() { return type; }
  NoiseConfig *getNoiseConfig() { return noiseConfig; }
  double getMaxConductance() { return maxConductance; }
  double getMinConductance() { return minConductance; }
  void print();
};

class MappingConfig : public Config {
 private:
  std::map<std::string, std::map<std::string, std::string>> strategies;

 public:
  MappingConfig(YAML::Node MappingConfig);
  void print();
};

class CamConfig : public Config {
 private:
  ArchConfig *archConfig;
  ArrayConfig *arrayConfig;
  CellConfig *cellConfig;
  QueryConfig *queryConfig;
  NoiseConfig *noiseConfig;
  MappingConfig *mappingConfig;

 public:
  CamConfig(std::string configPath);
  QueryConfig *getQueryConfig() { return queryConfig; }
  CellConfig *getCellConfig() { return cellConfig; }
  ArrayConfig *getArrayConfig() { return arrayConfig; }
  ArchConfig *getArchConfig() { return archConfig; }
  MappingConfig *getMappingConfig() { return mappingConfig; }
  NoiseConfig* getNoiseConfig() { return noiseConfig; }
  void print() {
    archConfig->print();
    arrayConfig->print();
    cellConfig->print();
    queryConfig->print();
    noiseConfig->print();
    mappingConfig->print();
  };

  ~CamConfig() {
    delete archConfig;
    delete arrayConfig;
    delete cellConfig;
    delete queryConfig;
    delete noiseConfig;
    delete mappingConfig;
  };
};

#endif