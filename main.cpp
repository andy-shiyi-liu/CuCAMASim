#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>

#include "CuCAMASim.h"
#include "dt2cam.h"
#include "matio.h"

// #define DATASET_NAME "BTSC_adapted_rand"
// #define DATASET_NAME "gas"
#define DATASET_NAME "gas_normalized"

Dataset *loadDataset(std::string datasetName) {
  std::cout << "Loading dataset: " << datasetName << std::endl;
  std::map<std::string, std::filesystem::path> datasetPath = {
      {"BTSC_adapted_rand",
       "/workspaces/CuCAMASim/data/datasets/BelgiumTSC/"
       "300train_100validation_-1test.mat"},
      {"gas_normalized",
       "/workspaces/CuCAMASim/data/datasets/gas_concentrations/"
       "gas_concentrations_normalized.mat"},
      {"gas",
       "/workspaces/CuCAMASim/data/datasets/gas_concentrations/"
       "gas_concentrations.mat"},
      {"test", "/workspaces/CuCAMASim/dataset/test/test.mat"}};
  Dataset *dataset = new Dataset(datasetPath[datasetName]);
  std::cout << "Dataset loaded!" << std::endl;
  return dataset;
}

std::string treeTextPath(std::string datasetName) {
  std::map<std::string, std::string> treeTextPath = {
      {"BTSC_adapted_rand",
       "/workspaces/CuCAMASim/data/treeText/BTSC/"
       "0.0stdDev_100sampleTimes_treeText.txt"},
      {"gas_normalized",
       "/workspaces/CuCAMASim/data/treeText/gas/gas_normalized.txt"},
      {"gas", "/workspaces/CuCAMASim/data/treeText/gas/gas.txt"}};
  return treeTextPath[datasetName];
}

int main() {
  DecisionTree dt(treeTextPath(DATASET_NAME));
  ACAMArray *camArray = dt.toACAM();

  Dataset *dataset = loadDataset(DATASET_NAME);

  std::cout << "Software Accuracy: "
            << dt.score(dataset->testInputs, dataset->testLabels) << std::endl;

  CamConfig camConfig("/workspaces/CuCAMASim/data/config/hard bd.yml");
  CuCAMASim camasim(&camConfig);

  camasim.write(camArray);
  camasim.query(dataset->testInputs, camasim.getSimResult());

  std::cout << "CAM Accuracy: "
            << camasim.getSimResult()->calculateInferenceAccuracy(
                   dataset->testLabels, camArray->getRow2classID())
            << std::endl;

  delete camArray;
  if (dataset != nullptr) {
    delete dataset;
  }

  std::cout << "\033[1;32m"
            << "main() finished without error"
            << "\033[0m" << std::endl;
  return 0;
}