#ifndef SEARCH_H
#define SEARCH_H

#include <iostream>
#include <string>

#include "util/config.h"
#include "util/data.h"
#include "cuda/distance.cuh"

class CAMSearch {
 private:
    QueryConfig *queryConfig;
    ArrayConfig *arrayConfig;
    const std::string searchScheme;
    const double searchParameter;
    const std::string sensing;
    const double sensingLimit;
    using DistFunc = double *(*)(CAMArrayBase*, QueryData*); // Function pointer declaration
    DistFunc metric;
    uint32_t _rowCams = (uint32_t)-1, _colCams = (uint32_t)-1;

 public:
    CAMSearch(QueryConfig *queryConfig, ArrayConfig *arrayConfig)
            : queryConfig(queryConfig),
                arrayConfig(arrayConfig),
                searchScheme(queryConfig->searchScheme),
                searchParameter(queryConfig->parameter),
                sensing(arrayConfig->sensing),
                sensingLimit(arrayConfig->sensingLimit) {
        std::cout << "in CAMSearch()" << std::endl;
        std::string distanceType = queryConfig->distance;
        if (distanceType == "euclidean") {
            throw std::runtime_error("NotImplementedError: Euclidean distance is not implemented yet");
        } else if (distanceType == "manhattan") {
            throw std::runtime_error("NotImplementedError: Manhattan distance is not implemented yet");
        } else if (distanceType == "hamming") {
            throw std::runtime_error("NotImplementedError: Hamming distance is not implemented yet");
        } else if (distanceType == "innerproduct") {
            throw std::runtime_error("NotImplementedError: Inner product distance is not implemented yet");
        } else if (distanceType == "range") {
            metric = (DistFunc)rangeQueryPairwise;
        } else if (distanceType == "softRange") {
            metric = (DistFunc)softRangePairwise;
        } else {
            throw std::runtime_error("NotImplementedError: Unknown distance type"); // Raise an exception for unknown distance type
        }
        std::cout << "CAMSearch() done" << std::endl;
    }

    void defineSearchArea(uint32_t rowCams, uint32_t colCams);
    void search(const CAMDataBase *camArray, const QueryData *queryData);

    ~CAMSearch() {}
};;

#endif  // SEARCH_H