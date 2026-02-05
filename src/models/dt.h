#ifndef DECISION_TREE_MODEL_H
#define DECISION_TREE_MODEL_H

#include <stdint.h>
#include <vector>
#include <fstream>

//platform specific
#ifdef ARDUINO
    #include "config.h"
    #include "ml_inference.h"
#else
    #include "csv_loader.h"
    typedef csv_training_sample_t ml_training_sample_t;
    typedef csv_metrics_t ml_metrics_t;
#endif


#ifndef DTNode_STRUCT_DEFINED
#define DTNode_STRUCT_DEFINED
struct DTNode{
    scent_class_t label;
    int featureIndex;
    float threshold;
    uint16_t majorityCount;
    uint16_t totalSamples;
    DTNode* left;
    DTNode* right;
    DTNode() : label(SCENT_CLASS_UNKNOWN), featureIndex(-1), threshold(0.0f), majorityCount(0), totalSamples(0), left(nullptr), right(nullptr) {}
};
#endif


class DecisionTree{
public:
    DecisionTree() : _root(nullptr), _ownedRoot(false), _featureCount(12){}
    ~DecisionTree();

    void train(const ml_training_sample_t* samples, uint16_t sampleCount, uint16_t featureCount=12, uint8_t maxDepth=10, uint8_t minSamples=5);

    ml_metrics_t evaluate(const ml_training_sample_t* samples, uint16_t sampleCount);

    //tree info
    void getStats();
    uint16_t getDTNodeCount() const {return DTNodeCount;}
    uint16_t getDepth() const {return depth;}
    uint16_t getLeafCount() const {return leafCount;}
    void printTree(int maxDepth=5) const;

    scent_class_t predict(const float* features) const;
    scent_class_t predictWithConfidence(const float* features, float &confidenceOut) const;

    bool saveModel(const char* file) const;
    bool loadModel(const char* file);
private:
    const DTNode* _root;
    bool _ownedRoot;
    uint16_t _featureCount;
    uint16_t DTNodeCount;
    uint16_t depth;
    uint16_t leafCount;

    DTNode* buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, uint8_t depth, uint8_t maxDepth, uint8_t minSampleSplit);

    void findBestSplit(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, int &bestFeatureIndex, float &bestThreshold, float &bestGini);

    float calculateGini(const std::vector<uint16_t>& sampleIndicies, const ml_training_sample_t* samples);

    scent_class_t getMajorityClass(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples);

    void deleteTree(DTNode* DTNode);

    void printDTNode(const DTNode* DTNode, int depth, int maxDepth) const;
    void countDTNodes(const DTNode* DTNode, uint16_t &count, uint16_t depth, uint16_t &maxDepth, uint16_t &leafCount)const;

    void saveTreeDTNode(std::ofstream &file, const DTNode* DTNode) const;
    DTNode* loadTreeDTNode(std::ifstream &file, uint16_t version);
};

#endif