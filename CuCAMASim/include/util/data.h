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

enum CAMDataType { CAM_DATA, ACAM_DATA, INVALID_CAMDATA };

class Data {};

class CAMData : public Data {
 private:
  virtual inline void checkDataValid() {
    assert(type == CAM_DATA &&
           "self data type is not CAM_DATA, have you initialized the data by "
           "initData()?");
  }

 protected:
  CAMDataType type = INVALID_CAMDATA;
  struct CAMDataDim {
    uint32_t nRows;
    uint32_t nCols;
    uint32_t nBoundaries;  // for ACAM
  } dim;
  double _min = +std::numeric_limits<double>::infinity(),
         _max = -std::numeric_limits<double>::infinity();
  double *data = nullptr;
  virtual inline void updateMinMax(double val) {
    if (!std::isnan(val)) {
      _min = std::min(_min, val);
      _max = std::max(_max, val);
    }
  }

 public:
  std::vector<double> col2featureID;
  std::vector<double> row2classID;

  CAMData(uint32_t nRows, uint32_t nCols) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 1;
  };
  virtual void initData() {
    type = CAM_DATA;
    uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
    this->data = new double[nElem];
    for (uint32_t i = 0; i < dim.nRows; i++) {
      for (uint32_t j = 0; j < dim.nCols; j++) {
        set(i, j, std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  virtual inline double at(int rowNum, int colNum) {
    checkDataValid();
    int index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
    return data[index];
  }
  virtual inline void set(int rowNum, int colNum, double val) {
    checkDataValid();
    int index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }

  inline bool checkDim() {
    return col2featureID.size() == dim.nCols && row2classID.size() == dim.nRows;
  }
  void printDim() {
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }

  inline double min() { return _min; }
  inline double max() { return _max; }
  inline CAMDataType getType() { return type; }

  inline uint32_t getNRows() { return dim.nRows; }
  inline uint32_t getNCols() { return dim.nCols; }

  virtual ~CAMData() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

class ACAMData : public CAMData {
 private:
 inline void checkDataValid() override {
    assert(type == ACAM_DATA &&
           "self data type is not ACAM_DATA, have you initialized the data by "
           "initData()?");
  }
 public:
  void initData() override {
    type = ACAM_DATA;
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
  ACAMData(uint32_t nRows, uint32_t nCols) : CAMData(nRows, nCols) {
    dim.nBoundaries = 2;
  }
  inline void set(int rowNum, int colNum, double val) override {
    assert(0 && rowNum && colNum && val &&
           "boundary number is needed for ACAMData");
  }
  inline void set(int rowNum, int colNum, int bdNum, double val) {
    checkDataValid();
    int index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }
  inline double at(int rowNum, int colNum) override {
    checkDataValid();
    assert(0 && rowNum && colNum && "boundary number is needed for ACAMData");
  }
  inline double at(int rowNum, int colNum, int bdNum) {
    checkDataValid();
    int index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    return data[index];
  }
  inline void toCSV(const std::filesystem::path &outputPath) {
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
  inline void updateMinMax(double val) override {
    if (!std::isinf(val)) {
      _min = std::min(_min, val);
      _max = std::max(_max, val);
    }
  }
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