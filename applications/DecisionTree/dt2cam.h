#ifndef DT2CAM_H
#define DT2CAM_H

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/data.h"

enum TreeNodeType { LEAF_NODE, STEM_NODE, INVALID };

class TreeNode {
 private:
  TreeNodeType type = INVALID;

 public:
  virtual TreeNodeType getType() { return type; };
  virtual ~TreeNode() {}
};

class StemNode;

class LeafNode : public TreeNode {
 private:
  uint64_t classID = (uint64_t)-1;
  TreeNode *parent = NULL;
  TreeNodeType type = LEAF_NODE;

 public:
  LeafNode(uint64_t classID, TreeNode *parent)
      : classID(classID), parent(parent) {}
  TreeNodeType getType() override { return type; };
  uint64_t getClassID() { return classID; };
  virtual ~LeafNode() {}
};

class StemNode : public TreeNode {
 private:
  uint64_t featureID = (uint64_t)-1;
  double threshold = 0.0;
  TreeNode *leNode = NULL;
  TreeNode *gtNode = NULL;
  TreeNode *parent = NULL;
  TreeNodeType type = STEM_NODE;

 public:
  StemNode(uint64_t featureID, double threshold, TreeNode *parent)
      : featureID(featureID), threshold(threshold), parent(parent){};
  StemNode(){};
  void init(uint64_t featureID, double threshold, TreeNode *parent,
            TreeNode *leNode, TreeNode *gtNode) {
    this->featureID = featureID;
    this->threshold = threshold;
    this->parent = parent;
    this->leNode = leNode;
    this->gtNode = gtNode;
  };
  uint64_t getFeatureID() { return featureID; };
  double getThreshold() { return threshold; };
  TreeNode *getLeNode() { return leNode; };
  TreeNode *getGtNode() { return gtNode; };
  TreeNode *getParent() { return parent; };
  TreeNodeType getType() override { return type; };

  virtual ~StemNode() {
    delete leNode;
    delete gtNode;
  };
};

class DecisionTree {
 private:
  std::vector<std::string> treeText;
  CAMData *camData = NULL;
  std::list<LeafNode *> leafNodes;
  std::list<uint64_t> featureIDs;
  std::list<uint64_t> classIDs;
  std::list<double> thresholds;
  TreeNode *rootNode = NULL;

  void parseTreeText();
  TreeNode *parseSubTree(uint64_t &lineID, TreeNode *parentNode);
  void printSubTree(TreeNode* treeNode, std::string spacing);

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
    parseTreeText();
  };
  void printTree();
  void printTreeText() {
    std::ostringstream oss;  // Create a string stream
    for (const auto &line : treeText) {
      oss << line + "\n";  // Append each line to the string stream
    }
    std::cout << oss.str();  // Print the string stream
  };
  CAMData *toCAM();
};

#endif