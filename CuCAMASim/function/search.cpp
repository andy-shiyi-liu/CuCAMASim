#include "function/search.h"

#include <chrono>
#include <iostream>

#include "function/cuda/search.cuh"
// Define the search area based on the number of row and column CAMs.
// Args:
//  numRowCAMs (uint32_t): Number of row-wise CAM arrays.
//  numColCAMs (uint32_t): Number of column-wise CAM arrays.
// Sets the number of row and column CAMs for the search operation.
void CAMSearch::defineSearchArea(uint32_t rowCams, uint32_t colCams) {
  this->_rowCams = rowCams;
  this->_colCams = colCams;
}

// Perform a search operation in CAM arrays.
// Args:
//     cam_data (array): Data stored in the CAM arrays.
//     query_data (array): Query data for the search.
// Returns:
//     results (list): List of search results.
// Searches in multiple CAM arrays, merges results, and returns a list of search
// results. The operation of this function is designed to be performed in GPU.
void CAMSearch::search(const CAMDataBase *camData, const QueryData *queryData,
                       SimResult *simResult) {
  assert(camData->getTotalNCol() == queryData->getTotalNCol());
  assert(camData->getColSize() == queryData->getColSize());
  assert(camData->getColCams() == queryData->getColCams());
  assert(_colCams == camData->getColCams());
  assert(_rowCams == camData->getRowCams());
  assert(arrayConfig->row == camData->getRowSize());

  auto start = std::chrono::high_resolution_clock::now();

  CAMSearchCUDA(this, camData, queryData, simResult);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Search time: " << diff.count() << " s\n";
};