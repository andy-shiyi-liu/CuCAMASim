#ifndef DATA_H
#define DATA_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

class Data {};

class CAMData : public Data {
 private:
  struct CAMDataDim {
    uint32_t nRows;
    uint32_t nCols;
    uint32_t nBoundaries;
  } dim;

 public:
  std::vector<double> col2featureID;
  std::vector<double> row2classID;
  double *data;

  CAMData(uint32_t nRows, uint32_t nCols, double *data) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 2;
    uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
    this->data = new double[nElem];
    std::memcpy(this->data, data, nElem * sizeof(double));
  };
  CAMData(uint32_t nRows, uint32_t nCols) {
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 2;
    uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
    this->data = new double[nElem];
    for (uint32_t i = 0; i < nRows; i++) {
      for (uint32_t j = 0; j < nCols; j++) {
        at(i, j, 0) = -std::numeric_limits<double>::infinity();
        at(i, j, 1) = +std::numeric_limits<double>::infinity();
      }
    }
  };
  double &at(int rowNum, int colNum, int bdNum) {
    int index = bdNum + dim.nBoundaries * (colNum + dim.nCols * rowNum);
    return data[index];
  }
  bool checkDim() {
    return col2featureID.size() == dim.nCols && row2classID.size() == dim.nRows;
  }
  void printDim() {
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }

  void toCSV(std::string outputPath) {
    toCSV(outputPath, ",");
  }
  void toCSV(std::string outputPath, std::string sep) {
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
        file << at(i, j, 0) << " <= x < " << at(i, j, 1) << sep;
      }
      file << "class_" << row2classID[i] << std::endl;
    }
    file.close();
  }
  ~CAMData() { delete[] data; };
};

class QueryData : public Data {
 private:
  struct QueryDataDim {
    uint32_t nVectors;
    uint32_t nFeatures;
  } dim;
  double *data;

 public:
  QueryData(uint32_t nVectors, uint32_t nFeatures, double *data) {
    dim.nVectors = nVectors;
    dim.nFeatures = nFeatures;
    uint64_t nElem = dim.nVectors * dim.nFeatures;
    this->data = new double[nElem];
    for (uint64_t i = 0; i < nElem; i++) {
      this->data[i] = data[i];
    }
  };
  ~QueryData() { delete[] data; };
};

#endif