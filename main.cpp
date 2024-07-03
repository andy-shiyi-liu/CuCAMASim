#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

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
  std::cout << "Dataset Size: " << dataset->testInputs->getNFeatures()
            << " features, "
            << dataset->testInputs->getNVectors() +
                   dataset->trainInputs->getNVectors()
            << " samples." << std::endl;

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

  std::cout << "\033[1;32m" << "CAMInference() finished without error"
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

void errorDistribution(const std::filesystem::path configPath,
                       const std::filesystem::path treeTextPath,
                       const std::filesystem::path outputPath,
                       const std::string datasetName,
                       const uint32_t sampleTimes) {
  std::cout << "Doing statistic for error distribution" << std::endl;

  std::cout << "Using tree text: " << treeTextPath << std::endl;

  struct errDistrResult {
    double softwareIdealAccuracy = 0;
    double camAccuracy = 0;
    uint64_t softwareWrong = 0;
    uint64_t oneMatchCorrect = 0;
    uint64_t oneMatchWrong = 0;
    uint64_t multiMatchCorrect = 0;
    uint64_t multiMatchWrong = 0;
    uint64_t noMatch = 0;
  } result;

  const Dataset *dataset = loadDataset(datasetName);

  DecisionTree dt4idealSwInf(treeTextPath);
  dt4idealSwInf.parseTreeText();
  std::vector<uint32_t> swPred;
  dt4idealSwInf.pred(dataset->testInputs, swPred);
  result.softwareIdealAccuracy =
      dataset->testLabels->calculateInferenceAccuracy(swPred);

  delete dataset;

  for (uint32_t nIter = 0; nIter < sampleTimes; nIter++) {
    const Dataset *dataset = loadDataset(datasetName);
    DecisionTree dt(treeTextPath);
    ACAMArray *camArray = dt.toACAM();

    CamConfig camConfig(configPath);
    CuCAMASim camasim(&camConfig);

    camasim.write(camArray);
    camasim.query(dataset->testInputs, camasim.getSimResult());

    const std::vector<std::vector<uint32_t>> camMatchedIdx =
        camasim.getSimResult()->getMatchedIdx();

    assert(swPred.size() == camMatchedIdx.size() && "Pred length mismatch!");

    for (uint32_t i = 0; i < camMatchedIdx.size(); i++) {
      const LabelData *testLabels = dataset->testLabels;
      if (swPred[i] != testLabels->at(i)) {
        result.softwareWrong += 1;
      } else {
        if (camMatchedIdx[i].size() == 0) {
          result.noMatch += 1;
        } else if (camMatchedIdx[i].size() == 1) {
          if ((*camArray->getRow2classID())[camMatchedIdx[i][0]] ==
              testLabels->at(i)) {
            result.oneMatchCorrect += 1;
          } else {
            result.oneMatchWrong += 1;
          }
        } else {
          std::set<uint32_t> uniquePred(camMatchedIdx[i].begin(),
                                        camMatchedIdx[i].end());
          if (uniquePred.size() == 1 &&
              (*camArray->getRow2classID())[*uniquePred.begin()] ==
                  testLabels->at(i)) {
            result.multiMatchCorrect += 1;
          } else {
            result.multiMatchWrong += 1;
          }
        }
      }
    }

    double CAMAccuracy = camasim.getSimResult()->calculateInferenceAccuracy(
        dataset->testLabels, camArray->getRow2classID());
    result.camAccuracy += CAMAccuracy;

    delete camArray;
    if (dataset != nullptr) {
      delete dataset;
    }
  }

  dataset = loadDataset(datasetName);
  uint64_t cnt = result.softwareWrong + result.oneMatchCorrect +
                 result.oneMatchWrong + result.multiMatchCorrect +
                 result.multiMatchWrong + result.noMatch;
  assert(cnt == sampleTimes * dataset->testLabels->getNVectors());

  if (dataset != nullptr) {
    delete dataset;
  }

  // write results
  std::ofstream outputFile(outputPath);

  if (!outputFile.is_open()) {
    throw std::runtime_error("cannot open output file");
  }

  outputFile << "," << "softwareIdealAccuracy," << "camAccuracy,"
             << "softwareWrong," << "oneMatchCorrect," << "oneMatchWrong,"
             << "multiMatchCorrect," << "multiMatchWrong," << "noMatch"
             << std::endl;
  outputFile << "value," << result.softwareIdealAccuracy << ","
             << (double)result.camAccuracy / sampleTimes << ","
             << (double)result.softwareWrong / (double)cnt << ","
             << (double)result.oneMatchCorrect / (double)cnt << ","
             << (double)result.oneMatchWrong / (double)cnt << ","
             << (double)result.multiMatchCorrect / (double)cnt << ","
             << (double)result.multiMatchWrong / (double)cnt << ","
             << (double)result.noMatch / (double)cnt;

  outputFile.close();

  std::cout << "\033[1;32m" << "errorDistribution() finished without error"
            << "\033[0m" << std::endl;
}

void printInfo(const std::filesystem::path treeTextPath,
               const std::string datasetName) {
  std::cout << "Doing CAM inference" << std::endl;
  std::cout << "Using tree text: " << treeTextPath << std::endl;

  DecisionTree dt(treeTextPath);
  ACAMArray *camArray = dt.toACAM();

  std::cout << "DT depth: " << dt.getTreeDepth() << std::endl;

  std::cout << "CAM array size after DT mapping: " << camArray->getDim().nRows
            << " Rows, " << camArray->getDim().nCols << " Cols" << std::endl;

  Dataset *dataset = loadDataset(datasetName);

  std::vector<uint32_t> uniqueLabels;
  for (uint32_t i = 0; i < dataset->testLabels->getNVectors(); i++) {
    if (std::find(uniqueLabels.begin(), uniqueLabels.end(),
                  dataset->testLabels->at(i)) == uniqueLabels.end()) {
      uniqueLabels.push_back(dataset->testLabels->at(i));
    }
  }

  std::cout << "Dataset Size: " << dataset->testInputs->getNFeatures()
            << " features, "
            << dataset->testInputs->getNVectors() +
                   dataset->trainInputs->getNVectors()
            << " samples, " << uniqueLabels.size() << " classes." << std::endl
            << " - " << dataset->trainInputs->getNVectors() << " train samples,"
            << std::endl
            << " - " << dataset->testInputs->getNVectors() << " test samples."
            << std::endl;
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
      " - errDistr"
      "The default task is " +
          task);

  std::string configPath = "/workspaces/CuCAMASim/data/camConfig/hard bd.yml";
  app.add_option(
      "--config", configPath,
      "Config file path. This can be either the config for CAM inference or "
      "software inference, depending on the --task flag.");

  std::string datasetName = "BTSC_adapted_rand";
  app.add_option("--dataset", datasetName, "Name of dataset to be used.");

  std::string treeTextPath = "default";
  app.add_option("--use_trained_tree", treeTextPath,
                 "Path to the tree text file. If not provided, the default "
                 "path for the dataset will be used.");

  std::string outputPath = "invalid_path";
  app.add_option("--output", outputPath, "Output path for simulation results");

  std::uint32_t sampleTimes = 0;
  app.add_option("--sample_time", sampleTimes,
                 "Do Monte Carlo Method by averaging the result of N samples. This argument is only valid for ``");

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
  } else if (task == "errDistr") {
    if (outputPath == "invalid_path") {
      throw std::runtime_error(
          "Output path is needed for task 'errDistr'. Please specify a result "
          "output path by '--output' argument.");
    }
    if (sampleTimes == 0) {
      throw std::runtime_error(
          "Sample time is needed for task 'errDistr' and cannot be 0. Please "
          "specify by '--sample_time' argument.");
    }
    errorDistribution(configPath, treeTextPath, outputPath, datasetName,
                      sampleTimes);
  } else if (task == "print_info") {
    printInfo(treeTextPath, datasetName);
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