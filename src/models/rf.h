#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <random>
#include <map>
#include <fstream>


#ifdef ARDUINO
    #include "config.h"
    #include "ml_inference.h"
#else
    #include "csv_loader.h"
    typedef csv_training_sample_t ml_training_sample_t;
    typedef csv_metrics_t ml_metrics_t;
#endif


struct RFNode{
    scent_class_t label;
    int featureIndex;
    float threshold;
    RFNode* left;
    RFNode* right;

    RFNode() : label(SCENT_CLASS_UNKNOWN), featureIndex(-1), threshold(0.0f), left(nullptr), right(nullptr) {}
};

//for embedded use
struct FlatRFNode{
    int8_t featureIndex;
    uint8_t label;
    float threshold;
    int16_t leftChild;
    int16_t rightChild;
};

class RandomForest {
public:
    RandomForest(uint16_t numTrees = 10, uint8_t maxDepth = 10, uint8_t minSamples=5, float subsetRatio=0.7f);
    ~RandomForest();

    void train(const ml_training_sample_t* samples, uint16_t sampleCount, uint16_t featureCount=12);

    //pred
    scent_class_t predict(const float* feats)const;
    scent_class_t predictWithConfidence(const float* feats, float &confidence) const;
    void predictProb(const float* features, float *confidence) const;

    ml_metrics_t evaluate(const ml_training_sample_t* samples, uint16_t sampleCount)const;

    void computeFeatureImportance(float* importance) const;
    void printFeatureImportance() const;

    //stats
    uint16_t getNumTrees() const {return _numTrees;}
    uint16_t getTotalRFNodes() const;
    float getOOBError() const {return _oobError;}
    uint16_t getFeatureCount() const {return _featureCount;}

    //config
    void setNumTrees(uint16_t n) {_numTrees = n;}
    void setMaxDepth(uint8_t d) {_maxDepth = d;}
    void setMinSamples(uint8_t m) {_minSamples = m;}
    void setFeatureSubsetRatio(float r) {_featureSubsetRatio = r;}

    //save
    bool saveModel(const char* filename) const;
    bool loadModel(const char* filename);
    void exportAsHeader(const char* path)const;

private:
    std::vector<RFNode*> _trees;
    uint16_t _numTrees;
    uint8_t _maxDepth;
    uint8_t _minSamples;
    float _featureSubsetRatio;
    uint16_t _featureCount;
    uint16_t _featureSubsetSize;
    float _oobError;

    std::vector<float> _featureImportance;
    mutable std::mt19937 _rng;

    //tree builder
    RFNode* buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t *samples, uint8_t depth);
    
    void findBestSplit(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t *samples, const std::vector<uint16_t> &featureSubset, int &bestFeatureIndex, float &bestThreshold, float &bestGini);

    float calculateGini(const std::vector<uint16_t> &indicies, const ml_training_sample_t *samples);

    scent_class_t getMajorityClass(const std::vector<uint16_t> &indicies, const ml_training_sample_t *samples);

    scent_class_t predictTree(const RFNode* RFNode, const float* features)const;

    void bootstrapSample(uint16_t samples, std::vector<uint16_t> &inBag, std::vector<uint16_t> &outOfBag);

    std::vector<uint16_t> getRandomFeatureSubset();

    void deleteTree(RFNode* RFNode);
    void clearTrees();
    uint16_t countRFNodes(const RFNode* RFNode) const;

    void saveTree(std::ofstream &file, const RFNode* RFNode)const;
    RFNode* loadTree(std::ifstream &file);
    void flattenTree(const RFNode* RFNode, std::vector<FlatRFNode> &flatRFNodes) const;
};


#endif