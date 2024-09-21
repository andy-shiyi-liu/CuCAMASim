#ifndef SEARCH_H
#define SEARCH_H

#include <iostream>
#include <string>

#include "cuda/distance.cuh"
#include "util/config.h"
#include "util/data.h"

class CAMSearch {
 private:
  QueryConfig *queryConfig;
  ArrayConfig *arrayConfig;
  const std::string searchScheme;
  const double searchParameter;
  const std::string sensing;
  const double sensingLimit;
  std::string distanceType;
  uint32_t _rowCams = uint32_t(-1), _colCams = uint32_t(-1);

 public:
  CAMSearch(QueryConfig *queryConfig, ArrayConfig *arrayConfig)
      : queryConfig(queryConfig),
        arrayConfig(arrayConfig),
        searchScheme(queryConfig->searchScheme),
        searchParameter(queryConfig->parameter),
        sensing(arrayConfig->sensing),
        sensingLimit(arrayConfig->sensingLimit),
        distanceType(queryConfig->distance) {}

  void defineSearchArea(uint32_t rowCams, uint32_t colCams);
  void search(const CAMDataBase *camData, const QueryData *queryData,
              SimResult *simResult);

  inline uint32_t getRowCams() const { return _rowCams; };
  inline uint32_t getColCams() const { return _colCams; };
  inline std::string getDistType() const { return distanceType; };
  inline std::string getSearchScheme() const { return searchScheme; };
  inline double getSearchParameter() const { return searchParameter; };
  inline std::string getSensing() const { return sensing; };
  inline double getSensingLimit() const { return sensingLimit; };
  inline const QueryConfig *getQueryConfig() const { return queryConfig; };

  ~CAMSearch() {}
};
;

#endif  // SEARCH_H