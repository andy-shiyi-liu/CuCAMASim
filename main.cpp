#include <iostream>

#include "CuCAMASim.h"

using namespace std;

int main() {
  cout << "hello world!" << endl;
  CamConfig cam_config("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");
  CuCAMASim camasim;
  return 0;
}