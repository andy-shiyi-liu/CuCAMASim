#include <iostream>

#include "CuCAMASim.h"

using namespace std;

int main() {
  cout << "hello world!" << endl;
  camConfig cam_config("accuracy_with_hardboundary.yml");
  CuCAMASim camasim;
  return 0;
}