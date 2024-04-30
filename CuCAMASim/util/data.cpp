#include "util/data.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

#include "matio.h"
#include "util/consts.h"

double* loadInputMatrix(mat_t* matfp, const char* variableName, size_t& rows,
                        size_t& cols) {
  matvar_t* matvar = Mat_VarRead(matfp, variableName);
  if (!matvar)
    throw std::runtime_error("Variable " + std::string(variableName) +
                             " not found");
  if (matvar->data_type != MAT_T_DOUBLE)
    throw std::runtime_error("Data type mismatch");

  rows = matvar->dims[1];
  cols = matvar->dims[0];
  size_t nElem = rows * cols;
  double* data = new double[nElem];
  memcpy(data, matvar->data, nElem * sizeof(double));

  Mat_VarFree(matvar);
  return data;
}

uint64_t* loadLabelMatrix(mat_t* matfp, const char* variableName,
                          size_t& nLabels) {
  matvar_t* matvar = Mat_VarRead(matfp, variableName);
  if (!matvar)
    throw std::runtime_error("Variable " + std::string(variableName) +
                             " not found");
  if (matvar->data_type != MAT_T_UINT64)
    throw std::runtime_error("Data type mismatch for " +
                             std::string(variableName));

  size_t nRows = matvar->dims[0];
  size_t nCols = matvar->dims[1];
  nLabels = nRows * nCols;
  uint64_t* data = new uint64_t[nLabels];
  memcpy(data, matvar->data, nLabels * sizeof(uint64_t));

  Mat_VarFree(matvar);
  return data;
}

void Dataset::loadDataset(std::filesystem::path datasetPath) {
  // Check if the dataset exists
  if (!std::filesystem::exists(datasetPath)) {
    throw std::runtime_error("Dataset '" + datasetPath.string() +
                             "' not found");
  }

  mat_t* matfp = Mat_Open(datasetPath.string().c_str(), MAT_ACC_RDONLY);
  size_t nVectors, nFeatures;

  // Load TrainInputs
  double* trainInputsData =
      loadInputMatrix(matfp, "trainInputs", nVectors, nFeatures);
  this->trainInputs = new InputData(nVectors, nFeatures, trainInputsData);

  // Load TrainLabels
  uint64_t* trainLabelsData = loadLabelMatrix(matfp, "trainLabels", nVectors);
  this->trainLabels = new LabelData(nVectors, trainLabelsData);

  // Load TestInputs
  double* testInputsData =
      loadInputMatrix(matfp, "testInputs", nVectors, nFeatures);
  this->testInputs = new InputData(nVectors, nFeatures, testInputsData);

  // Load TestLabels
  uint64_t* testLabelsData = loadLabelMatrix(matfp, "testLabels", nVectors);
  this->testLabels = new LabelData(nVectors, testLabelsData);
}

void InputData::clip(double min, double max) {
  for (uint32_t i = 0; i < dim.nVectors; i++) {
    for (uint32_t j = 0; j < dim.nFeatures; j++) {
      double val = at(i, j);
      if (val < min) {
        set(i, j, min);
      } else if (val > max) {
        set(i, j, max);
      }
    }
  }
}

void CAMArray::initData() {
  type = CAM_ARRAY_COLD_START;
  uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
  this->data = new double[nElem];
  for (uint32_t i = 0; i < dim.nRows; i++) {
    for (uint32_t j = 0; j < dim.nCols; j++) {
      set(i, j, std::numeric_limits<float>::quiet_NaN());
    }
  }
}

void CAMArray::initData(double initVal) {
  type = CAM_ARRAY_COLD_START;
  uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
  this->data = new double[nElem];
  memset(this->data, initVal, nElem * sizeof(double));
}

void ACAMArray::initData() {
  type = ACAM_ARRAY_COLD_START;
  assert(dim.nBoundaries == 2);
  uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
  this->data = new double[nElem];
  for (uint32_t i = 0; i < dim.nRows; i++) {
    for (uint32_t j = 0; j < dim.nCols; j++) {
      set(i, j, 0, -std::numeric_limits<double>::infinity());
      set(i, j, 1, +std::numeric_limits<double>::infinity());
    }
  }
}

void ACAMArray::initData(double initVal) {
  type = ACAM_ARRAY_COLD_START;
  assert(dim.nBoundaries == 2);
  uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
  this->data = new double[nElem];
  memset(this->data, initVal, nElem * sizeof(double));
}

void CAMData::initData(CAMArray* camArray) {
  uint64_t nElem = _rowCams * _colCams;
  this->camArrays = new CAMArray*[nElem];
  for (uint64_t i = 0; i < nElem; i++) {
    this->camArrays[i] = nullptr;
  }
  camArray->getType();
  throw std::runtime_error("Not implemented");
}

void ACAMData::initData(ACAMArray* acamArray) {
  uint64_t nElem = _rowCams * _colCams;
  this->camArrays = new ACAMArray*[nElem];

  if (_colCams > 1 || _rowCams > 1) {
    std::cerr << "\033[33mWARNING: mapping to multiple cam subarrays is not "
                 "tested, the result may be wrong.\033[0m"
              << std::endl;
  }

  // check data size validity
  uint32_t nRows = acamArray->getNRows(), nCols = acamArray->getNCols();
  if (nRows > _rowSize * _rowCams || nCols > _colSize * _colCams) {
    throw std::runtime_error("Data size exceeds the total CAM size");
  }

  // create and init subarrays
  for (uint64_t i = 0; i < nElem; i++) {
    this->camArrays[i] = new ACAMArray(_rowSize, _colSize);
    this->camArrays[i]->initData(0.0);
  }

  // copy data from acamArray to subarrays
  for (uint32_t i = 0; i < nRows; i++) {
    for (uint32_t j = 0; j < nCols; j++) {
      uint32_t camRowIdx = i / _rowSize, camColIdx = j / _colSize;
      uint32_t subArrayRowIdx = i % _rowSize, subArrayColIdx = j % _colSize;
      double lowerBd = acamArray->at(i, j, 0), upperBd = acamArray->at(i, j, 1);
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->set(subArrayRowIdx, subArrayColIdx, 0, lowerBd);
      targetSubarray->set(subArrayRowIdx, subArrayColIdx, 1, upperBd);
    }
  }

  // set row2classID for each subarray
  for (uint32_t i = 0; i < nRows; i++) {
    uint32_t camRowIdx = i / _rowSize;
    for (uint32_t camColIdx = 0; camColIdx < _colCams; camColIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->row2classID.push_back(acamArray->row2classID[i]);
    }
  }
  for (uint32_t i = nRows; i < _rowCams * _rowSize; i++) {
    uint32_t camRowIdx = i / _rowSize;
    for (uint32_t camColIdx = 0; camColIdx < _colCams; camColIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->row2classID.push_back(uint32_t(-1));
    }
  }
  // set col2featureID for each subarray
  for (uint32_t j = 0; j < nCols; j++) {
    uint32_t camColIdx = j / _colSize;
    for (uint32_t camRowIdx = 0; camRowIdx < _rowCams; camRowIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->col2featureID.push_back(acamArray->col2featureID[j]);
    }
  }
  for (uint32_t j = nCols; j < _colCams * _colSize; j++) {
    uint32_t camColIdx = j / _colSize;
    for (uint32_t camRowIdx = 0; camRowIdx < _rowCams; camRowIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->col2featureID.push_back(uint32_t(-1));
    }
  }

  // check if the data is valid
  for (uint64_t i = 0; i < nElem; i++) {
    assert(camArrays[i]->getType() == ACAM_ARRAY_COLD_START);
    assert(camArrays[i]->isDimMatch());
  }
  // check the consistency of row2classID
  for (uint32_t rowCamIdx = 0; rowCamIdx < _rowCams; rowCamIdx++) {
    for (uint32_t colCamIdx = 0; colCamIdx < _colCams; colCamIdx++) {
      ACAMArray* targetSubarray = at(rowCamIdx, colCamIdx);
      assert(std::equal(targetSubarray->row2classID.begin(),
                        targetSubarray->row2classID.end(),
                        at(rowCamIdx, 0)->row2classID.begin()));
    }
  }
  // check the consistency of col2featureID
  for (uint32_t colCamIdx = 0; colCamIdx < _colCams; colCamIdx++) {
    for (uint32_t rowCamIdx = 0; rowCamIdx < _rowCams; rowCamIdx++) {
      ACAMArray* targetSubarray = at(rowCamIdx, colCamIdx);
      assert(std::equal(targetSubarray->col2featureID.begin(),
                        targetSubarray->col2featureID.end(),
                        at(0, colCamIdx)->col2featureID.begin()));
    }
  }

  type = ACAM_DATA_COLD_START;
}

void QueryData::initData(const InputData* inputData,
                         const CAMDataBase* camData) {
  assert(_colCams != uint32_t(-1) && _colSize != uint32_t(-1) &&
         _nVectors != uint32_t(-1));
  assert(_colCams == camData->getColCams());
  assert(_colSize == camData->getColSize());
  assert(_nVectors == inputData->getNVectors());
  if (_colCams > 1) {
    std::cerr << "\033[33mWARNING: mapping to multiple colCams is not tested, "
                 "the result may be wrong.\033[0m"
              << std::endl;
  }

  for (uint32_t colCamIdx = 0; colCamIdx < _colCams; colCamIdx++) {
    for (uint32_t colIdx = 0; colIdx < _colSize; colIdx++) {
      uint32_t featureIdx = camData->at(0, colCamIdx)->col2featureID[colIdx];
      for (uint32_t vectorIdx = 0; vectorIdx < _nVectors; vectorIdx++) {
        if (featureIdx != uint32_t(-1)) {
          at(colCamIdx)->set(vectorIdx, colIdx,
                             inputData->at(vectorIdx, featureIdx));
        } else {
          at(colCamIdx)->set(vectorIdx, colIdx, 0);
        }
      }
    }
  }
  std::cout << "--> Mapping the query..., " << _nVectors << " Query, "
            << _colCams << " COL" << std::endl;
  return;
}

void SimResult::writeFuncSimResult(uint32_t* result, uint32_t nVectors,
                                   uint32_t nMatchedRowsMax) {
  if (func.valid) {
    throw std::runtime_error(
        "Functional simulation result has already been written");
  }
  func.nVectors = nVectors;
  for (uint32_t vectorIdx = 0; vectorIdx < nVectors; vectorIdx++) {
    if (func.matchedIdx.size() <= vectorIdx) {
      func.matchedIdx.push_back(std::vector<uint32_t>());
    }
    for (uint32_t numMatchRow = 0; numMatchRow < nMatchedRowsMax;
         numMatchRow++) {
      uint32_t matchedIdx = result[vectorIdx * nMatchedRowsMax + numMatchRow];
      if (matchedIdx == uint32_t(-1)) {
        continue;
      }
      func.matchedIdx[vectorIdx].push_back(matchedIdx);
    }
  }
  func.valid = true;
  assert(func.matchedIdx.size() == nVectors);
}

void SimResult::printFuncSimResult() const {
  if (!func.valid) {
    throw std::runtime_error("Functional simulation result is not valid");
  }
  std::cout << "Functional Simulation Result:" << std::endl;
  for (uint32_t vectorIdx = 0; vectorIdx < func.nVectors; vectorIdx++) {
    std::cout << "Input vector #" << vectorIdx << ": ";
    for (uint32_t matchedIdx : func.matchedIdx[vectorIdx]) {
      std::cout << matchedIdx << " ";
    }
    std::cout << std::endl;
  }
}

double SimResult::calculateInferenceAccuracy(
    const LabelData* label, const std::vector<uint32_t>* row2classID) const {
  assert(func.valid);
  assert(label->getNVectors() == func.nVectors);

  std::random_device rd;
  std::mt19937 gen(rd());
  uint32_t correctCnt = 0;
  for (uint32_t vectorIdx = 0; vectorIdx < func.nVectors; vectorIdx++) {
    // guess class 0 if none-match
    uint32_t finalPred = 0;
    if (!func.matchedIdx[vectorIdx].empty()) {
      // randomly chose one if multi-match
      std::uniform_int_distribution<> distrib(
          0, func.matchedIdx[vectorIdx].size() - 1);
      uint32_t finalMatchedRowIdx = func.matchedIdx[vectorIdx][distrib(gen)];
      finalPred = (*row2classID)[finalMatchedRowIdx];
    }
    if (finalPred == label->at(vectorIdx)) {
      correctCnt++;
    }
  }
  assert(correctCnt <= func.nVectors);
  return double(correctCnt) / double(func.nVectors);
}

double LabelData::calculateInferenceAccuracy(
    const std::vector<uint32_t>& predLabel) const {
  if (predLabel.size() != getNVectors()) {
    throw std::runtime_error(
        "Size of predLabel does not match the number of vectors in LabelData");
  }
  uint32_t correctCnt = 0;
  for (uint32_t vectorIdx = 0; vectorIdx < getNVectors(); vectorIdx++) {
    if (predLabel[vectorIdx] == at(vectorIdx)) {
      correctCnt++;
    }
  }
  assert(correctCnt <= getNVectors());
  return double(correctCnt) / double(getNVectors());
}

Dataset* loadDataset(std::string datasetName) {
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
      {"iris", "/workspaces/CuCAMASim/data/datasets/iris/iris.mat"},
      {"iris_normalized",
       "/workspaces/CuCAMASim/data/datasets/iris/iris_normalized.mat"},
      {"survival", "/workspaces/CuCAMASim/data/datasets/survival/survival.mat"},
      {"survival_normalized",
       "/workspaces/CuCAMASim/data/datasets/survival/survival_normalized.mat"},
      {"breast_cancer",
       "/workspaces/CuCAMASim/data/datasets/breast_cancer/breast_cancer.mat"},
      {"breast_cancer_normalized",
       "/workspaces/CuCAMASim/data/datasets/breast_cancer/"
       "breast_cancer_normalized.mat"},
      {"MNIST", "/workspaces/CuCAMASim/data/datasets/MNIST/MNIST.mat"},
      {"MNIST_normalized",
       "/workspaces/CuCAMASim/data/datasets/MNIST/MNIST_normalized.mat"},
      {"MNIST_small",
       "/workspaces/CuCAMASim/data/datasets/MNIST/MNIST_small.mat"},
      {"MNIST_small_normalized",
       "/workspaces/CuCAMASim/data/datasets/MNIST/MNIST_small_normalized.mat"},
      {"eye_movements",
       "/workspaces/CuCAMASim/data/datasets/eye_movements/eye_movements.mat"},
      {"eye_movements_normalized",
       "/workspaces/CuCAMASim/data/datasets/eye_movements/"
       "eye_movements_normalized.mat"},
      {"gesture_phase_segmentation",
       "/workspaces/CuCAMASim/data/datasets/gesture_phase_segmentation/"
       "gesture_phase_segmentation.mat"},
      {"gesture_phase_segmentation_normalized",
       "/workspaces/CuCAMASim/data/datasets/gesture_phase_segmentation/"
       "gesture_phase_segmentation_normalized.mat"},
  };
  Dataset* dataset = new Dataset(datasetPath[datasetName]);
  std::cout << "Dataset loaded!" << std::endl;
  return dataset;
}