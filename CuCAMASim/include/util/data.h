#ifndef DATA_H
#define DATA_H

#include <iostream>

class Data {};

class CAMData : public Data {
 private:
    struct CAMDataDim{
        uint32_t nRows;
        uint32_t nCols;
        uint32_t nBoundaries;
    } dim;
    double *data;
 
 public:
  CAMData(uint32_t nRows, uint32_t nCols, double *data){
    dim.nRows = nRows;
    dim.nCols = nCols;
    dim.nBoundaries = 2;
    uint64_t nElem = dim.nRows * dim.nCols * dim.nBoundaries;
    this->data = new double[nElem];
    for (uint64_t i = 0; i < nElem; i++) {
      this->data[i] = data[i];
    }
  };
  void printDim(){
    std::cout << "nRows: " << dim.nRows << std::endl;
    std::cout << "nCols: " << dim.nCols << std::endl;
    std::cout << "nBoundaries: " << dim.nBoundaries << std::endl;
  }
  ~CAMData(){
    delete[] data;
  };
};

class QueryData: public Data{
  private:
    struct QueryDataDim{
        uint32_t nVectors;
        uint32_t nFeatures;
    } dim;
    double *data;
  public:
    QueryData(uint32_t nVectors, uint32_t nFeatures, double *data){
      dim.nVectors = nVectors;
      dim.nFeatures = nFeatures;
      uint64_t nElem = dim.nVectors * dim.nFeatures;
      this->data = new double[nElem];
      for (uint64_t i = 0; i < nElem; i++) {
        this->data[i] = data[i];
      }
    };
    ~QueryData(){
      delete[] data;
    };
};

#endif