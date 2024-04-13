#ifndef CONFIG_H
#define CONFIG_H

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

class Config {};

class ArchConfig : public Config {
 public:
  const uint32_t arrays_per_mat, mats_per_bank, subarrays_per_array;
  ArchConfig(uint32_t arrays_per_mat, uint32_t mats_per_bank,
             uint32_t subarrays_per_array);
  ArchConfig(YAML::Node archConfig);
  void print();
};

class QueryConfig : public Config {
 public:
  const bool funcSim, perfEval;
  const uint16_t bit;
  const double distanceParameter, parameter;
  const std::string distance, searchScheme;
  QueryConfig(YAML::Node queryConfig);
  void print();
};

class ArrayConfig : public Config {
 public:
  const uint16_t bit;
  const std::string cell;
  const uint32_t col;
  const uint32_t row;
  const std::string sensing;
  const double sensingLimit;
  const bool useEVACAMCost;
  ArrayConfig(YAML::Node arrayConfig);
  void print();
};

class NoiseConfig : public Config {
 public:
  const std::string cellDesign;
  const std::string device;
  const bool hasWriteNoise;
  const double maxConductance;
  const double minConductance;
  std::map<std::string, std::map<std::string, std::string>> noiseType;
  NoiseConfig(YAML::Node noiseConfig);
  void print();
};

class CellConfig : public Config {
 public:
  const std::string design;
  const std::string device;
  const double maxConductance;
  const double minConductance;
  const std::string representation;
  const std::string type;
  NoiseConfig *noiseConfig;
  CellConfig(YAML::Node cellConfig, NoiseConfig *noiseConfig);
  void print();
};

class MappingConfig : public Config {
 public:
  std::map<std::string, std::map<std::string, std::string>> strategies;
  MappingConfig(YAML::Node MappingConfig);
  void print();
};

class CamConfig : public Config {
 public:
  ArchConfig *archConfig;
  ArrayConfig *arrayConfig;
  CellConfig *cellConfig;
  QueryConfig *queryConfig;
  NoiseConfig *noiseConfig;
  MappingConfig *mappingConfig;
  CamConfig(const std::filesystem::path &configPath);
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
    archConfig = nullptr;
    delete arrayConfig;
    arrayConfig = nullptr;
    delete cellConfig;
    cellConfig = nullptr;
    delete queryConfig;
    queryConfig = nullptr;
    delete noiseConfig;
    noiseConfig = nullptr;
    delete mappingConfig;
    mappingConfig = nullptr;
  };
};

#endif