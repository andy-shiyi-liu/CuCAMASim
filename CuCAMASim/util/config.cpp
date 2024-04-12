#include "util/config.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>

ArchConfig::ArchConfig(uint64_t arrays_per_mat, uint64_t mats_per_bank,
                       uint64_t subarrays_per_array)
    : arrays_per_mat(arrays_per_mat),
      mats_per_bank(mats_per_bank),
      subarrays_per_array(subarrays_per_array){};

ArchConfig::ArchConfig(YAML::Node archConfig)
    : arrays_per_mat(archConfig["ArraysPerMat"].as<uint64_t>()),
      mats_per_bank(archConfig["MatsPerBank"].as<uint64_t>()),
      subarrays_per_array(archConfig["SubarraysPerArray"].as<uint64_t>()){};

void ArchConfig::print() {
  std::cout << "Arch Config: " << std::endl;
  std::cout << "- arrays_per_mat: " << arrays_per_mat << std::endl;
  std::cout << "- mats_per_bank: " << mats_per_bank << std::endl;
  std::cout << "- subarrays_per_array: " << subarrays_per_array << std::endl;
}

ArrayConfig::ArrayConfig(YAML::Node arrayConfig)
    : bit(arrayConfig["bit"].as<uint16_t>()),
      cell(arrayConfig["cell"].as<std::string>()),
      col(arrayConfig["col"].as<uint32_t>()),
      row(arrayConfig["row"].as<uint32_t>()),
      sensing(arrayConfig["sensing"].as<std::string>()),
      sensingLimit(arrayConfig["sensingLimit"].as<double>()),
      useEVACAMCost(arrayConfig["useEVACAMCost"].as<bool>()){};

void ArrayConfig::print() {
  std::cout << "Array Config: " << std::endl;
  std::cout << "- bit: " << bit << std::endl;
  std::cout << "- cell: " << cell << std::endl;
  std::cout << "- col: " << col << std::endl;
  std::cout << "- row: " << row << std::endl;
  std::cout << "- sensing: " << sensing << std::endl;
  std::cout << "- sensingLimit: " << sensingLimit << std::endl;
  std::cout << "- useEVACAMCost: " << useEVACAMCost << std::endl;
}

QueryConfig::QueryConfig(YAML::Node queryConfig)
    : funcSim(queryConfig["FuncSim"].as<bool>()),
      perfEval(queryConfig["PerfEval"].as<bool>()),
      bit(queryConfig["bit"].as<uint16_t>()),
      distanceParameter(queryConfig["distanceParameter"].as<double>()),
      parameter(queryConfig["parameter"].as<double>()),
      distance(queryConfig["distance"].as<std::string>()),
      searchScheme(queryConfig["searchScheme"].as<std::string>()){};

void QueryConfig::print() {
  std::cout << "Query Config: " << std::endl;
  std::cout << "- funcSim: " << funcSim << std::endl;
  std::cout << "- perfEval: " << perfEval << std::endl;
  std::cout << "- bit: " << bit << std::endl;
  std::cout << "- distanceParameter: " << distanceParameter << std::endl;
  std::cout << "- parameter: " << parameter << std::endl;
  std::cout << "- distance: " << distance << std::endl;
  std::cout << "- searchScheme: " << searchScheme << std::endl;
}



NoiseConfig::NoiseConfig(YAML::Node noiseConfig)
    : cellDesign(noiseConfig["cellDesign"].as<std::string>()),
      device(noiseConfig["device"].as<std::string>()),
      hasWriteNoise(noiseConfig["hasWriteNoise"].as<bool>()),
      maxConductance(noiseConfig["maxConductance"].as<double>()),
      minConductance(noiseConfig["minConductance"].as<double>()) {
      YAML::Node noiseTypeNode = noiseConfig["noiseType"];
      for (const auto& it : noiseTypeNode) {
        std::string key = it.first.as<std::string>();
        YAML::Node innerNode = it.second;
        std::map<std::string, std::string> innerMap;
        for (const auto& innerIt : innerNode) {
          std::string innerKey = innerIt.first.as<std::string>();
          std::string innerValue = innerIt.second.as<std::string>();
          innerMap[innerKey] = innerValue;
        }
        noiseType[key] = innerMap;
      }
};

void NoiseConfig::print() {
  std::cout << "Noise Config: " << std::endl;
  std::cout << "- cellDesign: " << cellDesign << std::endl;
  std::cout << "- device: " << device << std::endl;
  std::cout << "- hasWriteNoise: " << hasWriteNoise << std::endl;
  std::cout << "- maxConductance: " << maxConductance << std::endl;
  std::cout << "- minConductance: " << minConductance << std::endl;
  std::cout << "- noiseType: " << std::endl;
  for (const auto& it : noiseType) {
    std::cout << "  - " << it.first << ":" << std::endl;
    for (const auto& innerIt : it.second) {
      std::cout << "    - " << innerIt.first << ": " << innerIt.second << std::endl;
    }
  }
}

CellConfig::CellConfig(YAML::Node cellConfig, NoiseConfig *noiseConfig)
    : design(cellConfig["design"].as<std::string>()),
      device(cellConfig["device"].as<std::string>()),
      maxConductance(cellConfig["maxConductance"].as<double>()),
      minConductance(cellConfig["minConductance"].as<double>()),
      representation(cellConfig["representation"].as<std::string>()),
      type(cellConfig["type"].as<std::string>()),
      noiseConfig(noiseConfig){};

void CellConfig::print() {
  std::cout << "Cell Config: " << std::endl;
  std::cout << "- design: " << design << std::endl;
  std::cout << "- device: " << device << std::endl;
  std::cout << "- maxConductance: " << maxConductance << std::endl;
  std::cout << "- minConductance: " << minConductance << std::endl;
  std::cout << "- representation: " << representation << std::endl;
  std::cout << "- type: " << type << std::endl;
  noiseConfig->print();
}

MappingConfig::MappingConfig(YAML::Node mappingConfig) {
  for (const auto& it : mappingConfig) {
    std::string key = it.first.as<std::string>();
    YAML::Node innerNode = it.second;
    std::map<std::string, std::string> innerMap;
    for (const auto& innerIt : innerNode) {
      std::string innerKey = innerIt.first.as<std::string>();
      std::string innerValue = innerIt.second.as<std::string>();
      innerMap[innerKey] = innerValue;
    }
    strategies[key] = innerMap;
  }
};

void MappingConfig::print() {
  std::cout << "Mapping Config: " << std::endl;
  for (const auto& it : strategies) {
    std::cout << "  - " << it.first << ":" << std::endl;
    for (const auto& innerIt : it.second) {
      std::cout << "    - " << innerIt.first << ": " << innerIt.second << std::endl;
    }
  }
}

CamConfig::CamConfig(const std::filesystem::path& configPath) {
  std::cout << "Using config: " << configPath << std::endl;
  if (!std::filesystem::exists(configPath)) {
    throw std::runtime_error("Config file not found!");
  }
  YAML::Node camConfig = YAML::LoadFile(configPath);
  YAML::Node arch_config_yaml = camConfig["arch"];
  archConfig = new ArchConfig(camConfig["arch"]);
  archConfig->print();
  arrayConfig = new ArrayConfig(camConfig["array"]);
  arrayConfig->print();
  queryConfig = new QueryConfig(camConfig["query"]);
  queryConfig->print();
  noiseConfig = new NoiseConfig(camConfig["cell"]["writeNoise"]);
  cellConfig = new CellConfig(camConfig["cell"], noiseConfig);
  cellConfig->print();
  mappingConfig = new MappingConfig(camConfig["mapping"]);
  mappingConfig->print();
  std::cout << "config read successful!" << std::endl;
}