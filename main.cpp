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
  DecisionTree dt("/workspaces/CuCAMASim/exampleTreeText.txt");
  CAMData *camData = dt.toCAM();
  std::cout << "Original TreeText:" << std::endl;
  dt.printTreeText();
  std::cout << "Exported TreeText:" << std::endl;
  dt.printTree();
  camData->printDim();
  camData->toCSV("/workspaces/CuCAMASim/test.csv");

  Dataset *dataset = loadDataset("BTSC_adapted_rand");
  dataset->testInputs->toCSV("/workspaces/CuCAMASim/testInputs.csv");
  dataset->testLabels->toCSV("/workspaces/CuCAMASim/testLabels.csv");
  dataset->trainInputs->toCSV("/workspaces/CuCAMASim/trainInputs.csv");
  dataset->trainLabels->toCSV("/workspaces/CuCAMASim/trainLabels.csv");

  CamConfig camConfig("/workspaces/CuCAMASim/accuracy_with_hardboundary.yml");
  CuCAMASim camasim(&camConfig);

  if(dataset!= nullptr){
    delete dataset;
  }
  return 0;
}