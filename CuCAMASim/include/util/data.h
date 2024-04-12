#ifndef DATA_H
#define DATA_H

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

class Data {};

class CAMData : public Data {
 private:
  struct CAMDataDim {
    uint32_t nRows;
    uint32_t nCols;
    uint32_t nBoundaries;
  } dim;
  double _min = +std::numeric_limits<double>::infinity(),
         _max = -std::numeric_limits<double>::infinity();
  double *data = nullptr;

 public:
  std::vector<double> col2featureID;
  std::vector<double> row2classID;

  CAMData(uint32_t nRows, uint32_t nCols) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 2;
    uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
    this->data = new double[nElem];
    for (uint32_t i = 0; i < nRows; i++) {
      for (uint32_t j = 0; j < nCols; j++) {
        set(i, j, 0, -std::numeric_limits<double>::infinity());
        set(i, j, 1, +std::numeric_limits<double>::infinity());
      }
    }
  };
  double at(int rowNum, int colNum, int bdNum) {
    int index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    return data[index];
  }
  void set(int rowNum, int colNum, int bdNum, double val) {
    int index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    if (!std::isinf(val)) {
      _min = std::min(_min, val);
      _max = std::max(_max, val);
    }
  }
  bool checkDim() {
    return col2featureID.size() == dim.nCols && row2classID.size() == dim.nRows;
  }
  void printDim() {
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }
  void toCSV(const std::filesystem::path &outputPath) {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) {
    std::ofstream file(outputPath);
    // print col2featureID as column name
    file << sep;
    for (uint32_t i = 0; i < dim.nCols; i++) {
      file << "feature_" << col2featureID[i] << sep;
    }
    file << "classID" << std::endl;
    for (uint32_t i = 0; i < dim.nRows; i++) {
      file << "row_" << i << sep;
      for (uint32_t j = 0; j < dim.nCols; j++) {
        file << at(i, j, 0) << " < x <= " << at(i, j, 1) << sep;
      }
      file << "class_" << row2classID[i] << std::endl;
    }
    file.close();
  }
  double min() { return _min; }
  double max() { return _max; }

  uint32_t getNRows() { return dim.nRows; }
  uint32_t getNCols() { return dim.nCols; }

  ~CAMData() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

class QueryData : public Data {
 private:
  struct QueryDataDim {
    uint32_t nVectors;
    uint32_t nFeatures;
  } dim;
  double *data = nullptr;

 public:
  QueryData(uint32_t nVectors, uint32_t nFeatures) {
    dim.nVectors = nVectors;
    dim.nFeatures = nFeatures;
    uint64_t nElem = dim.nVectors * dim.nFeatures;
    this->data = new double[nElem];
  };
  double &at(int vecNum, int featureNum) {
    return data[featureNum + dim.nFeatures * vecNum];
  }
  ~QueryData() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

// for test data in dataset
class InputData : public Data {
 private:
  struct InputDataDim {
    uint32_t nVectors;
    uint32_t nFeatures;
  } dim;
  double *data = nullptr;

 public:
  InputData(uint32_t nVectors, uint32_t nFeatures, double *data) {
    dim.nVectors = nVectors;
    dim.nFeatures = nFeatures;
    this->data = data;
  }
  double &at(int vecNum, int featureNum) {
    return data[featureNum + dim.nFeatures * vecNum];
  }
  void toCSV(const std::filesystem::path &outputPath) {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) {
    std::ofstream file(outputPath);
    for (uint32_t i = 0; i < dim.nVectors; i++) {
      for (uint32_t j = 0; j < dim.nFeatures; j++) {
        file << at(i, j) << sep;
      }
      file << std::endl;
    }
    file.close();
  }
  ~InputData() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

// for label data in the dataset
class LabelData : public Data {
 private:
  struct LabelDataDim {
    uint32_t nVectors;
  } dim;
  uint64_t *data = nullptr;

 public:
  LabelData(uint32_t nVectors, uint64_t *data) {
    dim.nVectors = nVectors;
    this->data = data;
  }
  void toCSV(const std::filesystem::path &outputPath) {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) {
    std::ofstream file(outputPath);
    for (uint32_t i = 0; i < dim.nVectors; i++) {
      file << at(i) << sep;
    }
    file.close();
  }
  uint64_t &at(int vecNum) { return data[vecNum]; }
  ~LabelData() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

class Dataset {
 private:
  void loadDataset(std::filesystem::path datasetPath);

 public:
  InputData *trainInputs = nullptr;
  LabelData *trainLabels = nullptr;
  InputData *testInputs = nullptr;
  LabelData *testLabels = nullptr;
  Dataset(std::filesystem::path datasetPath) {
    // load the dataset
    loadDataset(datasetPath);
  };
  ~Dataset() {
    if (trainInputs != nullptr) {
      delete trainInputs;
      trainInputs = nullptr;
    }
    if (trainLabels != nullptr) {
      delete trainLabels;
      trainLabels = nullptr;
    }
    if (testInputs != nullptr) {
      delete testInputs;
      testInputs = nullptr;
    }
    if (testLabels != nullptr) {
      delete testLabels;
      testLabels = nullptr;
    }
  };
};

#endif