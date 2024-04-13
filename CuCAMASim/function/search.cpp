#include "function/search.h"

// Define the search area based on the number of row and column CAMs.
//Args:
//  numRowCAMs (uint32_t): Number of row-wise CAM arrays.
//  numColCAMs (uint32_t): Number of column-wise CAM arrays.
//Sets the number of row and column CAMs for the search operation.
void CAMSearch::defineSearchArea(uint32_t rowCams, uint32_t colCams) {
  this->_rowCams = rowCams;
  this->_colCams = colCams;
}

void CAMSearch::search(const CAMDataBase *camArray,
                       const QueryData *queryData) {
  std::cerr << "\033[33mWARNING: CAMSearch::search() is still under "
               "development\033[0m"
            << camArray << queryData << std::endl;
};