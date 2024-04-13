#include "dt2cam.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <regex>

#include "util/data.h"

void DecisionTree::parseTreeText() {
  uint64_t lineID = 0;
  rootNode = parseSubTree(lineID, nullptr);
  // assert that all lines are parsed
  if (lineID != treeText.size()) {
    fflush(stdout);
    throw std::runtime_error("Unable to parse all lines of the tree text");
  }
};

TreeNode* DecisionTree::parseSubTree(uint64_t& lineID, TreeNode* parentNode) {
  std::regex classKwRegex("class");
  std::regex leBranchKwRegex("feature_[0-9]+ <=");
  std::smatch match;

  if (std::regex_search(treeText[lineID], classKwRegex)) {
    // do regex matching to extract classID
    std::regex classIDRegex("[0-9]+.?0?");
    if ((std::regex_search(treeText[lineID], match, classIDRegex))) {
      uint32_t classID = std::stoi(match[0].str());
      LeafNode* leafNode = new LeafNode(classID, parentNode);
      leafNodes.push_back(leafNode);
      // if classID not in classIDs, append current classID to classIDs
      if (std::find(classIDs.begin(), classIDs.end(), classID) ==
          classIDs.end()) {
        classIDs.push_back(classID);
      }
      lineID++;
      return leafNode;
    } else {
      throw std::runtime_error("Unable to parse classID from line:\n" +
                               treeText[lineID]);
    }
  } else if (std::regex_search(treeText[lineID], leBranchKwRegex)) {
    // parse featureID and threshold
    StemNode* newNode = new StemNode();
    std::regex featureIDRegex("feature_[0-9]+");
    // assert that the line contains a featureID
    if (!(std::regex_search(treeText[lineID], match, featureIDRegex))) {
      throw std::runtime_error("Unable to parse featureID from line:\n" +
                               treeText[lineID]);
    }
    uint32_t featureID = std::stoi(match[0].str().substr(8));
    // if featureID not in featureIDs, append current featureID to featureIDs
    if (std::find(featureIDs.begin(), featureIDs.end(), featureID) ==
        featureIDs.end()) {
      featureIDs.push_back(featureID);
    }
    std::regex thresholdRegex("<=[ ]+[-]?\\d+(\\.\\d+)?");
    if (!(std::regex_search(treeText[lineID], match, thresholdRegex))) {
      throw std::runtime_error("Unable to parse threshold from line:\n" +
                               treeText[lineID]);
    }
    double threshold = std::stod(match[0].str().substr(3));
    thresholds.push_back(threshold);

    // parse leNode
    TreeNode* leNode = parseSubTree(++lineID, newNode);

    // parse gtNode
    std::regex gtBranchKwRegex =
        std::regex("feature_" + std::to_string(featureID) + " >");
    if (!(std::regex_search(treeText[lineID], gtBranchKwRegex))) {
      throw std::runtime_error("Unable to parse > branch from line:\n" +
                               treeText[lineID]);
    }
    TreeNode* gtNode = parseSubTree(++lineID, newNode);
    newNode->init(featureID, threshold, parentNode, leNode, gtNode);
    return newNode;
  } else {
    std::regex truncatedRegex("truncated");
    if (std::regex_search(treeText[lineID], truncatedRegex)) {
      throw std::runtime_error("Tree is truncated. Unable to parse line:\n" +
                               treeText[lineID]);
    }
    throw std::runtime_error("Unable to parse line:\n" + treeText[lineID]);
  }
};

ACAMArray* DecisionTree::tree2camThresholdArray() {
  ACAMArray* camArray = new ACAMArray(leafNodes.size(), featureIDs.size());
  camArray->initData();
  std::sort(featureIDs.begin(), featureIDs.end());

  for (uint32_t featureID : featureIDs) {
    camArray->col2featureID.push_back(featureID);
  }

  for (LeafNode* leafNode : leafNodes) {
    TreeNode* currentNode = leafNode;
    camArray->row2classID.push_back(leafNode->getClassID());
    while (currentNode->getParent() != nullptr) {
      // assert that parentNode is a StemNode
      assert(currentNode->getParent()->getType() == STEM_NODE);
      StemNode* parentNode = dynamic_cast<StemNode*>(currentNode->getParent());
      uint32_t featureID = parentNode->getFeatureID();
      uint8_t boundaryID;
      if (currentNode == parentNode->getLeNode()) {
        boundaryID = 1;
      } else {
        assert(currentNode == parentNode->getGtNode());
        boundaryID = 0;
      }
      double threshold = parentNode->getThreshold();
      uint32_t rowID = camArray->row2classID.size() - 1;
      auto it = std::find(featureIDs.begin(), featureIDs.end(), featureID);
      assert(it != featureIDs.end() && "featureID not found in featureIDs");
      uint32_t colID = std::distance(featureIDs.begin(), it);
      if (std::isinf(camArray->at(rowID, colID, boundaryID))) {
        camArray->set(rowID, colID, boundaryID, threshold);
      } else {
        if (boundaryID == 0) {
          camArray->set(rowID, colID, boundaryID,
              std::max(camArray->at(rowID, colID, boundaryID), threshold));
        } else {
          assert(boundaryID == 1);
          camArray->set(rowID, colID, boundaryID,
              std::min(camArray->at(rowID, colID, boundaryID), threshold));
        }
      }
      currentNode = parentNode;
    }
  }
  assert(camArray->isDimMatch() && "CAMArray dimensions do not match");
  return camArray;
};

ACAMArray* DecisionTree::toACAM() {
  parseTreeText();
  ACAMArray* camArray = tree2camThresholdArray();
  return camArray;
};

void DecisionTree::printTree() {
  if (rootNode == nullptr) {
    throw std::runtime_error("Tree not initialized");
  }
  printSubTree(rootNode, "");
}

void DecisionTree::printSubTree(TreeNode* treeNode, std::string spacing) {
  // in case we reached a leafNode
  if (treeNode->getType() == LEAF_NODE) {
    LeafNode* leafNode = dynamic_cast<LeafNode*>(treeNode);
    std::cout << spacing + "|--- class: " << leafNode->getClassID()
              << std::endl;
    return;
  }
  StemNode* stemNode = dynamic_cast<StemNode*>(treeNode);
  std::cout << spacing + "|---feature_" << stemNode->getFeatureID()
            << " <= " << std::fixed << std::setprecision(16)
            << stemNode->getThreshold() << std::endl;
  printSubTree(stemNode->getLeNode(), spacing + "|   ");
  std::cout << spacing + "|---feature_" << stemNode->getFeatureID() << " > "
            << std::setprecision(16) << stemNode->getThreshold() << std::endl;
  printSubTree(stemNode->getGtNode(), spacing + "|   ");
  return;
}