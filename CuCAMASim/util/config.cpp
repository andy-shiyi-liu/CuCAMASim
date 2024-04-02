#include "config.h"

#include <yaml-cpp/yaml.h>

#include <iostream>

camConfig::camConfig(std::string configPath) {
  std::cout << "Using config: " << configPath << std::endl;
  YAML::Node cam_config = YAML::LoadFile(configPath);
}