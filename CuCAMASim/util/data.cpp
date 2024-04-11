#include "util/data.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "matio.h"

double* loadInputMatrix(mat_t* matfp, const char* variableName, size_t& rows,
                         size_t& cols) {
  matvar_t* matvar = Mat_VarRead(matfp, variableName);
  if (!matvar)
    throw std::runtime_error("Variable " + std::string(variableName) +
                             " not found");
  if (matvar->data_type != MAT_T_DOUBLE)
    throw std::runtime_error("Data type mismatch");

  rows = matvar->dims[0];
  cols = matvar->dims[1];
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