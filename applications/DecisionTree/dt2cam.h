#ifndef DT2CAM_H
#define DT2CAM_H

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/data.h"

enum TreeNodeType { LEAF_NODE, STEM_NODE, INVALID_NODE };

class TreeNode {
 private:
  TreeNode *parent = nullptr;
  TreeNodeType type = INVALID_NODE;

 public:
  inline virtual const TreeNode *getParent() const  { return parent; }
  inline virtual TreeNodeType getType() const { return type; };
  virtual ~TreeNode() {}
};

class StemNode;

class LeafNode : public TreeNode {
 private:
  uint32_t classID = uint32_t(-1);
  TreeNode *parent = nullptr;
  TreeNodeType type = LEAF_NODE;

 public:
  LeafNode(uint32_t classID, TreeNode *parent)
      : classID(classID), parent(parent) {}
  inline TreeNodeType getType() const override { return type; };
  inline uint32_t getClassID() const { return classID; };
  inline const TreeNode *getParent() const override { return parent; };
  virtual ~LeafNode() {}
};

class StemNode : public TreeNode {
 private:
  uint32_t featureID = uint32_t(-1);
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
  inline uint32_t getFeatureID() const { return featureID; };
  inline double getThreshold() const { return threshold; };
  inline TreeNode *getLeNode() const { return leNode; };
  inline TreeNode *getGtNode() const { return gtNode; };
  inline const TreeNode *getParent() const override { return parent; };
  inline TreeNodeType getType() const override { return type; };

  inline void setThreshold(double newThreshold) { this->threshold = newThreshold; }

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

  TreeNode *parseSubTree(uint64_t &lineID, TreeNode *parentNode);
  void printSubTree(TreeNode *treeNode, std::string spacing);
  ACAMArray *tree2camThresholdArray();
  void predRow(InputData *input, uint32_t rowIdx, TreeNode *node,
               std::vector<uint32_t> &predLabel);
  void addNodeVariation(TreeNode *node, const YAML::Node &config);

 public:
  DecisionTree(const std::filesystem::path &treeTextPath) {
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
      throw std::runtime_error("Error: tree text file " + std::string(treeTextPath) +
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
  void pred(InputData *input, std::vector<uint32_t> &predLabel);
  double score(InputData *input, LabelData *label);
  void addVariation(const YAML::Node &config);
  ACAMArray *toACAM();
  void parseTreeText();
};

#endif