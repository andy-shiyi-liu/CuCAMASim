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

enum CAMArrayType {
  CAM_ARRAY_COLD_START,
  CAM_ARRAY_EXISTING_DATA,
  ACAM_ARRAY_COLD_START,
  ACAM_ARRAY_EXISTING_DATA,
  INVALID_CAMARRAY
};

class Data {};

// single, coutinuous CAM array
class CAMArray : public Data {
 private:
  virtual inline void checkDataValid() const {
    assert((type == CAM_ARRAY_COLD_START || type == CAM_ARRAY_EXISTING_DATA) &&
           "self data type is not CAM_ARRAY, have you initialized the data by "
           "initData()?");
  }

 protected:
  CAMArrayType type = INVALID_CAMARRAY;
  struct CAMArrayDim {
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

  CAMArray(uint32_t nRows, uint32_t nCols, double *arrayData,
           std::vector<double> &col2featureID,
           std::vector<double> &row2classID) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 1;
    data = arrayData;
    type = CAM_ARRAY_EXISTING_DATA;
    this->col2featureID = col2featureID;
    this->row2classID = row2classID;
    checkDim();
    checkDataValid();
  };
  CAMArray(uint32_t nRows, uint32_t nCols) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 1;
  };
  virtual void initData();
  virtual inline double at(uint32_t rowNum, uint32_t colNum) const {
    checkDataValid();
    if (rowNum >= dim.nRows || colNum >= dim.nCols) {
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      uint32_t index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
      return data[index];
    }
  }
  virtual inline void set(uint32_t rowNum, uint32_t colNum, double val) {
    assert(rowNum < dim.nRows && colNum < dim.nCols && "Index out of range");
    checkDataValid();
    int index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }
  inline bool checkDim() const {
    return col2featureID.size() == dim.nCols && row2classID.size() == dim.nRows;
  }
  void printDim() const {
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }
  inline double min() const { return _min; }
  inline double max() const { return _max; }
  inline CAMArrayType getType() const { return type; }
  inline uint32_t getNRows() const { return dim.nRows; }
  inline uint32_t getNCols() const { return dim.nCols; }

  virtual ~CAMArray() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  };
};

// single, coutinuous ACAM array
class ACAMArray : public CAMArray {
 private:
  inline void checkDataValid() const override {
    assert(
        (type == ACAM_ARRAY_COLD_START || type == ACAM_ARRAY_EXISTING_DATA) &&
        "self data type is not ACAM_ARRAY, have you initialized the data by "
        "initData()?");
  }

 public:
  void initData() override;
  ACAMArray(uint32_t nRows, uint32_t nCols, double *arrayData,
            std::vector<double> &col2featureID,
            std::vector<double> &row2classID)
      : CAMArray(nRows, nCols, arrayData, col2featureID, row2classID) {
    dim.nBoundaries = 2;
    type = ACAM_ARRAY_EXISTING_DATA;
    checkDim();
    checkDataValid();
  }
  ACAMArray(uint32_t nRows, uint32_t nCols) : CAMArray(nRows, nCols) {
    dim.nBoundaries = 2;
  }
  inline void set(uint32_t rowNum, uint32_t colNum, double val) override {
    assert(0 && rowNum && colNum && val &&
           "boundary number is needed for ACAMArray");
  };
  inline void set(uint32_t rowNum, uint32_t colNum, uint32_t bdNum,
                  double val) {
    checkDataValid();
    uint32_t index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }
  inline double at(uint32_t rowNum, uint32_t colNum) const override {
    assert(0 && rowNum && colNum && "boundary number is needed for ACAMArray");
  }
  inline double at(uint32_t rowNum, uint32_t colNum, uint32_t bdNum) const {
    checkDataValid();
    if (rowNum >= dim.nRows || colNum >= dim.nCols) {
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      uint32_t index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
      return data[index];
    }
  }
  inline void toCSV(const std::filesystem::path &outputPath) const {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) const {
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

enum CAMDataType { CAM_DATA_COLD_START, ACAM_DATA_COLD_START, INVALID_CAMDATA };

// CAM data with multiple sub-arrays
class CAMData : public Data {
 protected:
  uint32_t rowCams = (uint32_t)-1, colCams = (uint32_t)-1;
  CAMArray **camArray = nullptr;
  CAMDataType type = INVALID_CAMDATA;

 public:
  CAMData(uint32_t rowCams, uint32_t colCams) {
    this->rowCams = rowCams;
    this->colCams = colCams;
    uint64_t nElem = rowCams * colCams;
    this->camArray = new CAMArray *[nElem];
  };
  virtual void initData(CAMArray *camArray);
  virtual inline CAMArray *at(uint32_t rowNum, uint32_t colNum) const {
    assert(rowNum < rowCams && colNum < colCams && "Index out of range");
    return camArray[colNum + colCams * rowNum];
  }
  virtual ~CAMData() {
    if (camArray != nullptr) {
      for (uint32_t i = 0; i < rowCams * colCams; i++) {
        if (camArray[i] != nullptr) {
          delete camArray[i];
          camArray[i] = nullptr;
        }
      }
      delete[] camArray;
      camArray = nullptr;
    }
  };
};

// ACAM data with multiple sub-arrays
class ACAMData : public CAMData {
  public:
  ACAMData(uint32_t rowCams, uint32_t colCams) : CAMData(rowCams, colCams) {};
  void initData(CAMArray *camArray)override{
    initData(dynamic_cast<ACAMArray *>(camArray));
  };
  void initData(ACAMArray *acamArray) ;
  inline ACAMArray *at(uint32_t rowNum, uint32_t colNum) const override {
    assert(rowNum < rowCams && colNum < colCams && "Index out of range");
    return dynamic_cast<ACAMArray *>(camArray[colNum + colCams * rowNum]);
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