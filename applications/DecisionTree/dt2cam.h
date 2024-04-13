#ifndef DT2CAM_H
#define DT2CAM_H

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

#include "util/data.h"

enum TreeNodeType { LEAF_NODE, STEM_NODE, INVALID };

class TreeNode {
 private:
  TreeNode *parent = nullptr;
  TreeNodeType type = INVALID;

 public:
  virtual TreeNode* getParent() { return parent; }
  virtual TreeNodeType getType() { return type; };
  virtual ~TreeNode() {}
};

class StemNode;

class LeafNode : public TreeNode {
 private:
  uint32_t classID = (uint32_t)-1;
  TreeNode *parent = nullptr;
  TreeNodeType type = LEAF_NODE;

 public:
  LeafNode(uint32_t classID, TreeNode *parent)
      : classID(classID), parent(parent) {}
  TreeNodeType getType() override { return type; };
  uint32_t getClassID() { return classID; };
  TreeNode *getParent() override { return parent; };
  virtual ~LeafNode() {}
};

class StemNode : public TreeNode {
 private:
  uint32_t featureID = (uint32_t)-1;
  double threshold = 0.0;
  TreeNode *leNode = nullptr;
  TreeNode *gtNode = nullptr;
  TreeNode *parent = nullptr;
  TreeNodeType type = STEM_NODE;

 public:
  StemNode(uint32_t featureID, double threshold, TreeNode *parent)
      : featureID(featureID), threshold(threshold), parent(parent){};
  StemNode(){};
  void init(uint32_t featureID, double threshold, TreeNode *parent,
            TreeNode *leNode, TreeNode *gtNode) {
    this->featureID = featureID;
    this->threshold = threshold;
    this->parent = parent;
    this->leNode = leNode;
    this->gtNode = gtNode;
  };
  uint32_t getFeatureID() { return featureID; };
  double getThreshold() { return threshold; };
  TreeNode *getLeNode() { return leNode; };
  TreeNode *getGtNode() { return gtNode; };
  TreeNode *getParent() override { return parent; };
  TreeNodeType getType() override { return type; };

  virtual ~StemNode() {
    delete leNode;
    leNode = nullptr;
    delete gtNode;
    gtNode = nullptr;
  };
};

class DecisionTree {
 private:
  std::vector<std::string> treeText;
  ACAMArray *camArray = nullptr;
  std::list<LeafNode *> leafNodes;
  std::vector<uint32_t> featureIDs;
  std::list<uint32_t> classIDs;
  std::list<double> thresholds;
  TreeNode *rootNode = nullptr;

  void parseTreeText();
  TreeNode *parseSubTree(uint64_t &lineID, TreeNode *parentNode);
  void printSubTree(TreeNode* treeNode, std::string spacing);
  ACAMArray* tree2camThresholdArray();

 public:
  DecisionTree(const std::filesystem::path& treeTextPath) {
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
      throw std::runtime_error("Error: file " + std::string(treeTextPath) +
                               " cannot be opened");
    }
  };
  void printTree();
  void printTreeText() {
    std::ostringstream oss;  // Create a string stream
    for (const auto &line : treeText) {
      oss << line + "\n";  // Append each line to the string stream
    }
    std::cout << oss.str();  // Print the string stream
  };
  ACAMArray *toACAM();
};

#endif