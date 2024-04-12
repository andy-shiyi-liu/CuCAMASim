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

  // check data size validity
  uint32_t nRows = acamArray->getNRows(), nCols = acamArray->getNCols();
  if (nRows > _rowSize * _rowCams || nCols > _colSize * _colCams) {
    throw std::runtime_error("Data size exceeds the total CAM size");
  }

  // create and init subarrays
  for (uint64_t i = 0; i < nElem; i++) {
    this->camArrays[i] = new ACAMArray(_rowSize, _colSize);
    this->camArrays[i]->initData();
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
  for (uint64_t i = nRows; i < _rowCams * _rowSize; i++) {
    uint32_t camRowIdx = i / _rowSize;
    for (uint32_t camColIdx = 0; camColIdx < _colCams; camColIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->row2classID.push_back((uint32_t)-1);
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
  for (uint64_t j = nCols; j < _colCams * _colSize; j++) {
    uint32_t camColIdx = j / _colSize;
    for (uint32_t camRowIdx = 0; camRowIdx < _rowCams; camRowIdx++) {
      ACAMArray* targetSubarray = at(camRowIdx, camColIdx);
      targetSubarray->col2featureID.push_back((uint32_t)-1);
    }
  }
  for (uint64_t i = 0; i < nElem; i++) {
    assert(camArrays[i]->getType() == ACAM_ARRAY_COLD_START);
    camArrays[i]->toCSV("/workspaces/CuCAMASim/subarray" + std::to_string(i) +
                        ".csv");
    assert(camArrays[i]->isDimMatch());
  }
  type = ACAM_DATA_COLD_START;
}