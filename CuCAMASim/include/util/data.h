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
  CAM_ARRAY_BASE,
  INVALID_CAMARRAY
};

class Data {};

class CAMArrayBase : public Data {
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

  virtual inline void assertDataValid() const = 0;
  virtual void initData() = 0;

 public:
  std::vector<uint32_t> col2featureID;
  std::vector<uint32_t> row2classID;
  CAMArrayBase(uint32_t nRows, uint32_t nCols) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = (uint32_t)-1;
  };
  CAMArrayBase(uint32_t nRows, uint32_t nCols, double *arrayData,
               std::vector<uint32_t> &col2featureID,
               std::vector<uint32_t> &row2classID) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = (uint32_t)-1;
    data = arrayData;
    type = CAM_ARRAY_BASE;
    this->col2featureID = col2featureID;
    this->row2classID = row2classID;
    assert(isDimMatch());
  };

  virtual inline double at(uint32_t rowNum, uint32_t colNum) const = 0;
  virtual inline double at(uint32_t rowNum, uint32_t colNum,
                           uint32_t bdNum) const = 0;
  virtual inline void set(uint32_t rowNum, uint32_t colNum, double val) = 0;
  virtual inline void set(uint32_t rowNum, uint32_t colNum, uint32_t bdNum,
                          double val) = 0;

  inline double min() const { return _min; }
  inline double max() const { return _max; }
  inline CAMArrayType getType() const { return type; }
  inline uint32_t getNRows() const { return dim.nRows; }
  inline uint32_t getNCols() const { return dim.nCols; }
  inline bool isDimMatch() const {
    return col2featureID.size() == dim.nCols && row2classID.size() == dim.nRows;
  }
  void printDim() const {
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }
  virtual inline void updateMinMax(double val) = 0;
  virtual ~CAMArrayBase() {
    if (data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
};

class CAMArray : public CAMArrayBase {
 private:
  inline void assertDataValid() const override {
    assert((type == CAM_ARRAY_COLD_START || type == CAM_ARRAY_EXISTING_DATA) &&
           "self data type is not CAM_ARRAY, have you initialized the data by "
           "initData()?");
  }

 public:
  CAMArray(uint32_t nRows, uint32_t nCols, double *arrayData,
           std::vector<uint32_t> &col2featureID, std::vector<uint32_t> &row2classID)
      : CAMArrayBase(nRows, nCols, arrayData, col2featureID, row2classID) {
    dim.nBoundaries = 1;
    type = CAM_ARRAY_EXISTING_DATA;
    assert(isDimMatch());
    assertDataValid();
    std::cerr
        << "\e[33mThis constructor has not been tested!\n - CAMArray(uint32_t "
           "nRows, uint32_t nCols, double *arrayData, std::vector<double> "
           "&col2featureID, std::vector<double> &row2classID)\e[0m"
        << std::endl;
  };
  CAMArray(uint32_t nRows, uint32_t nCols) : CAMArrayBase(nRows, nCols) {
    dim.nBoundaries = 1;
  };
  void initData() override;
  inline void updateMinMax(double val) override {
    if (!std::isnan(val)) {
      _min = std::min(_min, val);
      _max = std::max(_max, val);
    }
  }
  inline double at(uint32_t rowNum, uint32_t colNum,
                   uint32_t bdNum) const override {
    assert(0 && rowNum && colNum && bdNum &&
           "boundary number is not needed for CAMArray");
  }
  inline double at(uint32_t rowNum, uint32_t colNum) const override {
    assertDataValid();
    if (rowNum >= dim.nRows || colNum >= dim.nCols) {
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      uint32_t index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
      return data[index];
    }
  }
  inline void set(uint32_t rowNum, uint32_t colNum, uint32_t bdNum,
                  double val) override {
    assert(0 && rowNum && colNum && bdNum && val &&
           "boundary number is not needed for CAMArray");
  }
  inline void set(uint32_t rowNum, uint32_t colNum, double val) override {
    assert(rowNum < dim.nRows && colNum < dim.nCols && "Index out of range");
    assertDataValid();
    int index = dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }
};

// single, coutinuous ACAM array
class ACAMArray : public CAMArrayBase {
 private:
  inline void assertDataValid() const override {
    assert(
        (type == ACAM_ARRAY_COLD_START || type == ACAM_ARRAY_EXISTING_DATA) &&
        "self data type is not ACAM_ARRAY, have you initialized the data by "
        "initData()?");
  }

 public:
  void initData() override;
  ACAMArray(uint32_t nRows, uint32_t nCols, double *arrayData,
            std::vector<uint32_t> &col2featureID,
            std::vector<uint32_t> &row2classID)
      : CAMArrayBase(nRows, nCols, arrayData, col2featureID, row2classID) {
    dim.nBoundaries = 2;
    type = ACAM_ARRAY_EXISTING_DATA;
    assert(isDimMatch());
    assertDataValid();
    std::cerr
        << "\e[33mThis constructor has not been tested!\n - ACAMArray(uint32_t "
           "nRows, uint32_t nCols, double *arrayData, std::vector<double> "
           "&col2featureID, std::vector<double> &row2classID)\e[0m"
        << std::endl;
  }
  ACAMArray(uint32_t nRows, uint32_t nCols) : CAMArrayBase(nRows, nCols) {
    dim.nBoundaries = 2;
  }
  inline void set(uint32_t rowNum, uint32_t colNum, double val) override {
    assert(0 && rowNum && colNum && val &&
           "boundary number is needed for ACAMArray");
  };
  inline void set(uint32_t rowNum, uint32_t colNum, uint32_t bdNum,
                  double val) override {
    assertDataValid();
    uint32_t index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    data[index] = val;
    updateMinMax(val);
  }
  inline double at(uint32_t rowNum, uint32_t colNum) const override {
    assert(0 && rowNum && colNum && "boundary number is needed for ACAMArray");
  }
  inline double at(uint32_t rowNum, uint32_t colNum,
                   uint32_t bdNum) const override {
    assertDataValid();
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

class CAMDataBase : public Data {
 protected:
  uint32_t _rowCams = (uint32_t)-1, _colCams = (uint32_t)-1,
           _rowSize = (uint32_t)-1, _colSize = (uint32_t)-1;

  CAMDataType type = INVALID_CAMDATA;

 public:
  CAMDataBase(uint32_t rowCams, uint32_t colCams, uint32_t rowSize,
              uint32_t colSize) {
    this->_rowCams = rowCams;
    this->_colCams = colCams;
    this->_rowSize = rowSize;
    this->_colSize = colSize;
  };
  inline CAMDataType getType() const { return type; }
  virtual void initData(CAMArrayBase *camArray) = 0;
  virtual ~CAMDataBase(){};
};

// CAM data with multiple sub-arrays
class CAMData : public CAMDataBase {
 protected:
  CAMArray **camArrays = nullptr;

 public:
  CAMData(uint32_t rowCams, uint32_t colCams, uint32_t rowSize,
          uint32_t colSize)
      : CAMDataBase(rowCams, colCams, rowSize, colSize){};

  void initData(CAMArrayBase *camArray) override {
    initData(dynamic_cast<CAMArray *>(camArray));
  };
  void initData(CAMArray *camArray);
  inline CAMArrayBase *at(uint32_t rowNum, uint32_t colNum) const {
    assert(rowNum < _rowCams && colNum < _colCams && "Index out of range");
    return camArrays[colNum + _colCams * rowNum];
  }
  ~CAMData() override {
    if (camArrays != nullptr) {
      for (uint32_t i = 0; i < _rowCams * _colCams; i++) {
        if (camArrays[i] != nullptr) {
          delete camArrays[i];
          camArrays[i] = nullptr;
        }
      }
      delete[] camArrays;
      camArrays = nullptr;
    }
  };
};

// ACAM data with multiple sub-arrays
class ACAMData : public CAMDataBase {
 protected:
  ACAMArray **camArrays = nullptr;

 public:
  ACAMData(uint32_t rowCams, uint32_t colCams, uint32_t rowSize,
           uint32_t colSize)
      : CAMDataBase(rowCams, colCams, rowSize, colSize){};
  void initData(CAMArrayBase *camArray) override {
    initData(dynamic_cast<ACAMArray *>(camArray));
  };
  void initData(ACAMArray *acamArray);
  inline ACAMArray *at(uint32_t rowNum, uint32_t colNum) const {
    assert(rowNum < _rowCams && colNum < _colCams && "Index out of range");
    return this->camArrays[colNum + _colCams * rowNum];
  }
  ~ACAMData() override {
    if (camArrays != nullptr) {
      for (uint32_t i = 0; i < _rowCams * _colCams; i++) {
        if (camArrays[i] != nullptr) {
          delete camArrays[i];
          camArrays[i] = nullptr;
        }
      }
      delete[] camArrays;
      camArrays = nullptr;
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
  inline void set(int vecNum, int featureNum, double val) {
    data[featureNum + dim.nFeatures * vecNum] = val;
  };
  inline double at(int vecNum, int featureNum) const {
    return data[featureNum + dim.nFeatures * vecNum];
  }

  void toCSV(const std::filesystem::path &outputPath) const {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) const {
    std::ofstream file(outputPath);
    file << sep;
    for (uint32_t i = 0; i < dim.nFeatures; i++) {
      file << i << sep;
    }
    file << std::endl;
    for (uint32_t i = 0; i < dim.nVectors; i++) {
      file << i << sep;
      for (uint32_t j = 0; j < dim.nFeatures; j++) {
        file << at(i, j) << sep;
      }
      file << std::endl;
    }
    file.close();
  }

  void clip(double min, double max);

  inline uint32_t getNVectors() const { return dim.nVectors; }
  inline uint32_t getNFeatures() const { return dim.nFeatures; }

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
  void toCSV(const std::filesystem::path &outputPath) const {
    toCSV(outputPath, ",");
  }
  void toCSV(const std::filesystem::path &outputPath, std::string sep) const {
    std::ofstream file(outputPath);
    for (uint32_t i = 0; i < dim.nVectors; i++) {
      file << at(i) << sep;
    }
    file.close();
  }
  inline uint64_t at(int vecNum) const { return data[vecNum]; }
  inline uint32_t getNVectors() const { return dim.nVectors; }
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

class QueryData : public Data {
 protected:
  const uint32_t _nVectors = (uint32_t)-1, _colCams = (uint32_t)-1,
                 _colSize = (uint32_t)-1;
  InputData **camQuries = nullptr;

 public:
  QueryData(uint32_t nVectors, uint32_t colCams, uint32_t colSize)
      : _nVectors(nVectors), _colCams(colCams), _colSize(colSize) {
    camQuries = new InputData *[colCams];
    for (uint32_t i = 0; i < colCams; i++) {
      camQuries[i] =
          new InputData(nVectors, colSize, new double[nVectors * colSize]);
    }
  };

  void initData(const InputData *inputData, const CAMData *camData); 

  ~QueryData() {
    if (camQuries != nullptr) {
      for (uint32_t i = 0; i < _colCams; i++) {
        if (camQuries[i] != nullptr) {
          delete camQuries[i];
          camQuries[i] = nullptr;
        }
      }
      delete[] camQuries;
      camQuries = nullptr;
    }
  };
};

#endif