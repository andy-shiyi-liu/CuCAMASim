#ifndef DT2CAM_H
#define DT2CAM_H

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "util/data.h"

class treeNode {};

class stemNode;

class leafNode : public treeNode {
 private:
  uint64_t classID;
  treeNode *parent;
};

class stemNode : public treeNode {
 private:
  uint64_t featureID;
  double threshold;
  treeNode *leNode;
  treeNode *gtNode;
  treeNode *parent;
};

class DecisionTree {
 private:
  std::list<std::string> treeText;
  CAMData *camData = NULL;
  void parseTreeText();

 public:
  DecisionTree(const std::string &treeTextPath) {
    // Code to read the tree text from the file at treeTextPath
    // and initialize the treeText member variable
    std::ifstream file(treeTextPath);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        if (!line.empty()) {
          treeText.push_back(line);
        }
      }
      file.close();
    } else {
      // Handle error when file cannot be opened
      throw std::runtime_error("Error: file" + treeTextPath +
                               "cannot be opened");
    }
  };
  void print() {
    std::ostringstream oss;  // Create a string stream
    for (const auto &line : treeText) {
      oss << line + "\n";  // Append each line to the string stream
    }
    std::cout << oss.str();  // Print the string stream
  };
  CAMData *toCAM();
};

#endif