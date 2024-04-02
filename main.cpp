#include <iostream>
#include <yaml-cpp/yaml.h>

#include "CuCAMASim.h"

using namespace std;

int main() {
  cout << "hello world!" << endl;

  YAML::Node config = YAML::LoadFile("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");

  camConfig cam_config("accuracy_with_hardboundary.yml");
  CuCAMASim camasim;
  return 0;
}