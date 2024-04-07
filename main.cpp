#include <iostream>

#include "CuCAMASim.h"
#include "dt2cam.h"

using namespace std;

int main() {
  cout << "hello world!" << endl;
  DecisionTree dt("/workspaces/CuCAMASim/exampleTreeText.txt");
  dt.print();

  CamConfig camConfig("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");
  CuCAMASim camasim(&camConfig);

  return 0;
}