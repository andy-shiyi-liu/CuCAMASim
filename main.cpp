#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>

#include "CuCAMASim.h"
#include "dt2cam.h"
#include "matio.h"
#include "util/CLI11.hpp"

double CAMInference(const std::filesystem::path configPath,
                    const std::filesystem::path treeTextPath,
                    const std::string datasetName) {
  std::cout << "Doing CAM inference" << std::endl;
  std::cout << "Using tree text: " << treeTextPath << std::endl;

  DecisionTree dt(treeTextPath);
  ACAMArray *camArray = dt.toACAM();

  std::cout << "CAM array size after DT mapping: " << camArray->getDim().nCols
            << " Cols, " << camArray->getDim().nRows << " Rows" << std::endl;

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
      " - CAM_inference"
      " - software_inference"
      "The default task is " +
          task);

  std::string configPath = "/workspaces/CuCAMASim/data/camConfig/hard bd.yml";
  app.add_option(
      "--config", configPath,
      "Config file path. This can be either the config for CAM inference or "
      "software inference, depending on the --task flag.");

  std::string datasetName = "BTSC_adapted_rand";
  app.add_option("--dataset", datasetName,
                 "Name of dataset to be used.");

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
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;
  return 0;
}