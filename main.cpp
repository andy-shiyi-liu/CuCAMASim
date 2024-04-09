#include <iostream>

#include "CuCAMASim.h"
#include "dt2cam.h"

int main() {
  std::cout << "hello world!" << std::endl;
  DecisionTree dt("/workspaces/CuCAMASim/exampleTreeText.txt");
  std::cout <<"Original TreeText:"<<std::endl;
  dt.printTreeText();
  std::cout <<"Exported TreeText:"<<std::endl;
  dt.printTree();

  CamConfig camConfig("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");
  CuCAMASim camasim(&camConfig);

  return 0;
}