#ifndef DECISION_TREE_MODEL_H
#define DECISION_TREE_MODEL_H

#include <stdint.h>
#include "config.h"
#include "ml_inference.h"
#include <vector>

struct Node{
    scent_class_t label;
    int featureIndex;
    float threshold;
    Node* left;
    Node* right;
    Node() : label(SCENT_CLASS_UNKNOWN), featureIndex(-1), threshold(0.0f), left(nullptr), right(nullptr) {}
};

class DecisionTree{
public:
    DecisionTree() : _root(nullptr), _ownedRoot(false), _featureCount(12){}
    ~DecisionTree();

    void train(const ml_training_sample_t* samples, uint16_t sampleCount, uint16_t featureCount=12, uint8_t maxDepth=10, uint8_t minSamples=5);

    ml_metrics_t evaluate(const ml_training_sample_t* samples, uint16_t sampleCount);

    //tree info
    void getStats();
    uint16_t getNodeCount() const {return nodeCount;}
    uint16_t getDepth() const {return depth;}
    uint16_t getLeafCount() const {return leafCount;}
    void printTree(int maxDepth=5) const;

    scent_class_t predict(const float* features) const;
private:
    const Node* _root;
    bool _ownedRoot;
    uint16_t _featureCount;
    uint16_t nodeCount;
    uint16_t depth;
    uint16_t leafCount;

    Node* buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, uint8_t depth, uint8_t maxDepth, uint8_t minSampleSplit);

    void findBestSplit(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, int &bestFeatureIndex, float &bestThreshold, float &bestGini);

    float calculateGini(const std::vector<uint16_t>& sampleIndicies, const ml_training_sample_t* samples);

    scent_class_t getMajorityClass(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples);

    void deleteTree(Node* node);

    void printNode(const Node* node, int depth, int maxDepth) const;
    void countNodes(const Node* node, uint16_t &count, uint16_t depth, uint16_t &maxDepth, uint16_t &leafCount)const;
};

#endif