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
    {"BTSC_adapted_rand", "/workspaces/CuCAMASim/dataset/BTSC/rand/300train_100validation_-1test.mat"}
  };
  Dataset *dataset = new Dataset(datasetPath[datasetName]);
  std::cout << "Dataset loaded!" << std::endl;
  return dataset;
}

int main() {
  std::cout << "hello world!" << std::endl;
  DecisionTree dt("/workspaces/CuCAMASim/data/treeText/exampleTreeText.txt");
  ACAMData *camData = dt.toACAM();
  std::cout << "Original TreeText:" << std::endl;
  dt.printTreeText();
  std::cout << "Exported TreeText:" << std::endl;
  dt.printTree();
  camData->printDim();
  camData->toCSV("/workspaces/CuCAMASim/camData_beforeConvert.csv");

  Dataset *dataset = loadDataset("BTSC_adapted_rand");
  dataset->testInputs->toCSV("/workspaces/CuCAMASim/testInputs.csv");
  dataset->testLabels->toCSV("/workspaces/CuCAMASim/testLabels.csv");
  dataset->trainInputs->toCSV("/workspaces/CuCAMASim/trainInputs.csv");
  dataset->trainLabels->toCSV("/workspaces/CuCAMASim/trainLabels.csv");

  CamConfig camConfig("/workspaces/CuCAMASim/data/config/hard bd.yml");
  CuCAMASim camasim(&camConfig);

  camasim.write(camData);

  camData->toCSV("/workspaces/CuCAMASim/camData_afterConvert.csv");

  if(dataset!= nullptr){
    delete dataset;
  }

  std::cout << "\033[32m" << "main() finished without error" << "\033[0m" << std::endl;
  return 0;
}