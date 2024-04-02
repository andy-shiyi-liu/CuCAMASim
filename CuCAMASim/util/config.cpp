#include "config.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>

ArchConfig::ArchConfig(unsigned arrays_per_mat, unsigned mats_per_bank,
                       unsigned subarrays_per_array)
    : arrays_per_mat(arrays_per_mat),
      mats_per_bank(mats_per_bank),
      subarrays_per_array(subarrays_per_array){};

ArchConfig::ArchConfig(YAML::Node archConfig)
    : arrays_per_mat(archConfig["ArraysPerMat"].as<unsigned>()),
      mats_per_bank(archConfig["MatsPerBank"].as<unsigned>()),
      subarrays_per_array(archConfig["SubarraysPerArray"].as<unsigned>()){};

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
  std::cout << "Arch Config: " << std::endl;
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
     searchScheme(queryConfig["searchScheme"].as<std::string>())
    {};

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

CamConfig::CamConfig(std::string configPath) {
  std::cout << "Using config: " << configPath << std::endl;
  YAML::Node camConfig = YAML::LoadFile(configPath);
  YAML::Node arch_config_yaml = camConfig["arch"];
  archConfig = new ArchConfig(camConfig["arch"]);
  archConfig->print();
  arrayConfig = new ArrayConfig(camConfig["array"]);
  arrayConfig->print();
  queryConfig = new QueryConfig(camConfig["query"]);
  queryConfig->print();
  std::cout << "config read successful!" << std::endl;
}