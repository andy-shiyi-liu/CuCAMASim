#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <chrono>

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

std::filesystem::path getTreeTextPath(std::string datasetName) {
  std::map<std::string, std::filesystem::path> treeTextPath = {
      {"BTSC_adapted_rand",
       "/workspaces/CuCAMASim/data/treeText/BTSC/"
       "0.0stdDev_100sampleTimes_treeText.txt"},
      {"gas_normalized",
       "/workspaces/CuCAMASim/data/treeText/gas/gas_normalized.txt"},
      {"gas", "/workspaces/CuCAMASim/data/treeText/gas/gas.txt"}};
  return treeTextPath[datasetName];
}

double CAMInference(const std::filesystem::path configPath,
                    const std::filesystem::path treeTextPath,
                    const std::string datasetName) {
  std::cout << "Doing CAM inference" << std::endl;
  std::cout << "Using tree text: " << treeTextPath << std::endl;

  DecisionTree dt(treeTextPath);
  ACAMArray *camArray = dt.toACAM();

  Dataset *dataset = loadDataset(datasetName);

  std::cout << "DT Accuracy (original): "
            << dt.score(dataset->testInputs, dataset->testLabels) << std::endl;

  CamConfig camConfig(configPath);
  CuCAMASim camasim(&camConfig);

  camasim.write(camArray);
  camasim.query(dataset->testInputs, camasim.getSimResult());

  double CAMAccuracy = camasim.getSimResult()->calculateInferenceAccuracy(
      dataset->testLabels, camArray->getRow2classID());
  std::cout << "DT Accuracy (CAM): " << CAMAccuracy << std::endl;

  delete camArray;
  if (dataset != nullptr) {
    delete dataset;
  }

  std::cout << "\033[1;32m"
            << "CAMInference() finished without error"
            << "\033[0m" << std::endl;
  return CAMAccuracy;
}

double softwareInference(const std::filesystem::path configPath,
                         const std::filesystem::path treeTextPath,
                         const std::string datasetName) {
  std::cout << "Doing software inference" << std::endl;

  std::cout << "Using software Inference config: " << configPath << std::endl;
  if (!std::filesystem::exists(configPath)) {
    throw std::runtime_error("Config file not found!");
  }
  YAML::Node config = YAML::LoadFile(configPath);

  Dataset *dataset = loadDataset(datasetName);

  double accuracy = 0;
  if (config["weightVar"]["enabled"].as<bool>() == false) {
    DecisionTree dt(treeTextPath);
    accuracy = dt.score(dataset->testInputs, dataset->testLabels);
    std::cout << "Software Inference accuracy: " << accuracy << std::endl;
  } else {
    uint64_t sampleTimes = config["weightVar"]["sampleTimes"].as<uint64_t>();
    for (uint64_t i = 0; i < sampleTimes; i++) {
      DecisionTree dt(treeTextPath);
      dt.parseTreeText();
      dt.addVariation(config["weightVar"]);
      accuracy += dt.score(dataset->testInputs, dataset->testLabels);
    }
    accuracy /= sampleTimes;
    std::cout << "Software Average inference accuracy: " << accuracy
              << std::endl;
  }
  return accuracy;
}

int main(int argc, char *argv[]) {
  CLI::App app{"Decision Tree inference on ACAM"};

  // Adding options

  std::string task = "CAM_inference";
  app.add_option(
      "--task", task,
      "The task which this program is going to perform. Available options: "
      " - CAM Inference"
      " - Software Inference"
      "The default task is " +
          task);

  std::string configPath = "/workspaces/CuCAMASim/data/camConfig/hard bd.yml";
  app.add_option(
      "--config", configPath,
      "Config file path. This can be either the config for CAM inference or "
      "software inference, depending on the --task flag.");

  std::string datasetName = "BTSC_adapted_rand";
  app.add_option("--dataset", datasetName,
                 "Name of dataset to be used. Available options: "
                 "BTSC_adapted_rand, gas_normalized, gas");

  std::string treeTextPath = "default";
  app.add_option("--use_trained_tree", treeTextPath,
                 "Path to the tree text file. If not provided, the default "
                 "path for the dataset will be used.");

  // Parsing command-line arguments
  CLI11_PARSE(app, argc, argv);

  if (treeTextPath == "default") {
    treeTextPath = getTreeTextPath(datasetName);
  }

  auto start = std::chrono::high_resolution_clock::now();

  if (task == "CAM_inference") {
    CAMInference(configPath, treeTextPath, datasetName);
  } else if (task == "software_inference") {
    softwareInference(configPath, treeTextPath, datasetName);
  } else {
    std::cerr << "Invalid task: " << task << std::endl;
    return 1;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;
  return 0;
}