#include <cassert>
#include <filesystem>
#include <iostream>
#include <map>
#include <cstring>

#include "CuCAMASim.h"
#include "dt2cam.h"
#include "matio.h"

Dataset* loadDataset(std::string datasetName){
  std::cout << "Loading dataset: " << datasetName << std::endl;
  std::map<std::string, std::filesystem::path> datasetPath = {
    {"BTSC_adapted_rand", "/workspaces/CuCAMASim/dataset/BTSC/rand/300train_100validation_-1test.mat"},
    {"test", "/workspaces/CuCAMASim/dataset/test/test.mat"}
  };
  Dataset *dataset = new Dataset(datasetPath[datasetName]);
  std::cout << "Dataset loaded!" << std::endl;
  return dataset;
}

int main() {
  std::cout << "hello world!" << std::endl;
  DecisionTree dt("/workspaces/CuCAMASim/data/treeText/exampleTreeText.txt");
  ACAMArray *camArray = dt.toACAM();
  std::cout << "Original TreeText:" << std::endl;
  dt.printTreeText();
  std::cout << "Exported TreeText:" << std::endl;
  dt.printTree();
  camArray->printDim();

  Dataset *dataset = loadDataset("BTSC_adapted_rand");

  CamConfig camConfig("/workspaces/CuCAMASim/data/config/hard bd.yml");
  CuCAMASim camasim(&camConfig);

  camasim.write(camArray);
  camasim.query(dataset->testInputs, camasim.getSimResult());

  if(dataset!= nullptr){
    delete dataset;
  }

  std::cout << "\033[1;32m" << "main() finished without error" << "\033[0m" << std::endl;
  return 0;
}