#ifndef CONFIG_H
#define CONFIG_H

#include <yaml-cpp/yaml.h>

#include <string>
class Config {};

class ArchConfig : public Config {
 private:
  const unsigned arrays_per_mat, mats_per_bank, subarrays_per_array;

 public:
  ArchConfig(unsigned arrays_per_mat, unsigned mats_per_bank,
             unsigned subarrays_per_array);
  ArchConfig(YAML::Node archConfig);
  void print();
};

class QueryConfig : public Config {
 private:
  const bool funcSim, perfEval;
  uint16_t bit;
  double distanceParameter, parameter;
  std::string distance, searchScheme;

 public:
  QueryConfig(YAML::Node queryConfig);
  void print();
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