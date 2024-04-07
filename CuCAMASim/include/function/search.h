#ifndef SEARCH_H
#define SEARCH_H

#include <iostream>
#include <string>

#include "config.h"

class CAMSearch {
 private:
  QueryConfig *queryConfig;
  ArrayConfig *arrayConfig;
  const std::string searchScheme;
  const double searchParameter;
  const std::string sensing;
  const double sensingLimit;
  

 public:
    CAMSearch(QueryConfig *queryConfig, ArrayConfig *arrayConfig)
        : queryConfig(queryConfig),
            arrayConfig(arrayConfig),
            searchScheme(queryConfig->getSearchScheme()),
            searchParameter(queryConfig->getParameter()),
            sensing(arrayConfig->getSensing()),
            sensingLimit(arrayConfig->getSensingLimit()) {
        std::cout << "in CAMSearch()" << std::endl;
        std::cout << "CAMSearch() done" << std::endl;
    }
    ~CAMSearch() {}
};

#endif  // SEARCH_H