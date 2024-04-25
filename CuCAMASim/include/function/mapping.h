#ifndef MAPPING_H
#define MAPPING_H

#include <iostream>

#include "util/config.h"
#include "util/data.h"

class Mapping {
 private:
  const uint32_t rowSize, colSize;
  uint32_t rowCams = uint32_t(-1), colCams = uint32_t(-1);
  uint64_t camSize = uint64_t(-1);
  CAMDataBase *camData = nullptr;
  QueryData *queryData = nullptr;
  double checkSize(CAMArrayBase *camArray);
  MappingConfig *mappingConfig;
  CellConfig *cellConfig;

 public:
  Mapping(ArrayConfig *arrayConfig, MappingConfig *mappingConfig,
          CellConfig *cellConfig)
      : rowSize(arrayConfig->row),
        colSize(arrayConfig->col),
        mappingConfig(mappingConfig),
        cellConfig(cellConfig) {}

  void addNewMapping(CAMArrayBase *camArray);
  double write(CAMArrayBase *camArray);
  void query(InputData *inputData);

  inline uint32_t getRowCams() const { return rowCams; }
  inline uint32_t getColCams() const { return colCams; }
  inline uint32_t getRowSize() const { return rowSize; }
  inline uint32_t getColSize() const { return colSize; }
  inline const CAMDataBase *getCamData() const { return camData; }
  inline const QueryData *getQueryData() const { return queryData; }
  inline const MappingConfig *getMappingConfig() const { return mappingConfig; }
  inline const CellConfig *getCellConfig() const { return cellConfig; }

  ~Mapping() {
    if (camData != nullptr) {
      delete camData;
      camData = nullptr;
    }
    if (queryData != nullptr) {
      delete queryData;
      queryData = nullptr;
    }
  }
};

#endif  // MAPPING_H