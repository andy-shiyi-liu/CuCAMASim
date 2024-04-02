#ifndef CONFIG_H
#define CONFIG_H

#include <yaml-cpp/yaml.h>
#include <string>
class Config {};

class ArchConfig : public Config {
 public:
  const unsigned arrays_per_mat, mats_per_bank, subarrays_per_array;
  ArchConfig(unsigned arrays_per_mat, unsigned mats_per_bank,
             unsigned subarrays_per_array);
  ArchConfig(YAML::Node archConfig);
  void print();
};

class QueryConfig : public Config {};

class ArrayConfig : public Config {
  public:
    uint16_t bit;
    std::string cell;
    uint32_t col;
    uint32_t row;
    std::string sensing;
    double sensingLimit;
    bool useEVACAMCost;

    ArrayConfig(YAML::Node arrayConfig);
    void print();
};

class NoiseConfig : public Config {};

class CellConfig : public Config {};

class CamConfig : public Config {
 public:
  ArchConfig *archConfig;
  ArrayConfig *arrayConfig;
  CellConfig *cellConfig;
  QueryConfig *queryConfig;
  NoiseConfig *noiseConfig;

  CamConfig(std::string configPath);
  ~CamConfig() { delete archConfig; };
};

#endif