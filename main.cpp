#include <iostream>

#include "CuCAMASim.h"

using namespace std;

int main() {
  cout << "hello world!" << endl;
  CamConfig *camConfig = new CamConfig("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");
  CuCAMASim camasim(camConfig);

  delete camConfig;
  return 0;
}