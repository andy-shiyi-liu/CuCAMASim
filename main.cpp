#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>

#include "CuCAMASim.h"
#include "dt2cam.h"
#include "matio.h"
#include "util/CLI11.hpp"

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

int main(int argc, char *argv[]) {
  CLI::App app{"Decision Tree inference on ACAM"};

  // Adding options
  std::string configPath = "/workspaces/CuCAMASim/data/config/hard bd.yml";
  app.add_option("--config", configPath, "CAM config file path");

  std::string datasetName = "BTSC_adapted_rand";
  app.add_option("--dataset", datasetName,
                 "Name of dataset to be used. Available options: "
                 "BTSC_adapted_rand, gas_normalized, gas");

  // Parsing command-line arguments
  CLI11_PARSE(app, argc, argv);

  DecisionTree dt(treeTextPath(datasetName));
  ACAMArray *camArray = dt.toACAM();

  Dataset *dataset = loadDataset(datasetName);

  std::cout << "Software Accuracy: "
            << dt.score(dataset->testInputs, dataset->testLabels) << std::endl;

  CamConfig camConfig(configPath);
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