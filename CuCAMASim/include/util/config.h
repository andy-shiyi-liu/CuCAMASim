#ifndef CONFIG_H
#define CONFIG_H
#include <iostream>

class config {};

class archConfig : config {};

class queryConfig : config {};

class arrayConfig : config {};

class noiseConfig : config {};

class cellConfig : config {};

class camConfig : config {
 public:
 camConfig(std::string configPath);
};

#endif