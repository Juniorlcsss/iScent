#include "dt.h"
#include <iostream>
#include <set>
#include <map>
#include <algorithm>
#include <fstream>
#include <array>

static const uint32_t DT_MAGIC = 0x44544D4C;
static const uint16_t DT_VERSION = 2;

DecisionTree::~DecisionTree(){
    if(_ownedRoot && _root){
        deleteTree(const_cast<DTNode*>(_root));
    }
}

void DecisionTree::deleteTree(DTNode* n){
    if(n){
        deleteTree(n->left);
        deleteTree(n->right);
        delete n;
    }
}

bool DecisionTree::saveModel(const char* filename)const{
    std::ofstream file(filename, std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    file.write(reinterpret_cast<const char*>(&DT_MAGIC), sizeof(DT_MAGIC));
    file.write(reinterpret_cast<const char*>(&DT_VERSION), sizeof(DT_VERSION));
    file.write(reinterpret_cast<const char*>(&_featureCount), sizeof(_featureCount));

    uint8_t maxDepth = depth;
    uint8_t minSamples = 5;

    file.write(reinterpret_cast<const char*>(&maxDepth), sizeof(maxDepth));
    file.write(reinterpret_cast<const char*>(&minSamples), sizeof(minSamples));

    saveTreeDTNode(file, _root);

    file.close();
    std::cout << "Decision Tree model saved to " << filename << std::endl;
    return true;
}

void DecisionTree::saveTreeDTNode(std::ofstream &file, const DTNode* DTNode) const {
    uint8_t exists = (DTNode != nullptr) ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&exists), sizeof(exists));

    if(!DTNode){
        return;
    }

    uint8_t label = static_cast<uint8_t>(DTNode->label);
    file.write(reinterpret_cast<const char*>(&label), sizeof(label));
    file.write(reinterpret_cast<const char*>(&DTNode->featureIndex), sizeof(DTNode->featureIndex));
    file.write(reinterpret_cast<const char*>(&DTNode->threshold), sizeof(DTNode->threshold));
    file.write(reinterpret_cast<const char*>(&DTNode->majorityCount), sizeof(DTNode->majorityCount));
    file.write(reinterpret_cast<const char*>(&DTNode->totalSamples), sizeof(DTNode->totalSamples));

    saveTreeDTNode(file, DTNode->left);
    saveTreeDTNode(file, DTNode->right);
}

bool DecisionTree::loadModel(const char* filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    uint32_t magic = 0;
    uint16_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if(magic != DT_MAGIC || version < 1 || version > DT_VERSION){
        std::cerr << "Invalid Decision Tree model file." << std::endl;
        return false;
    }

    if(_ownedRoot && _root){
        deleteTree(const_cast<DTNode*>(_root));
    }

    file.read(reinterpret_cast<char*>(&_featureCount), sizeof(_featureCount));

    uint8_t maxDepth = 0;
    uint8_t minSamples = 0;

    file.read(reinterpret_cast<char*>(&maxDepth), sizeof(maxDepth));
    file.read(reinterpret_cast<char*>(&minSamples), sizeof(minSamples));

    _root = loadTreeDTNode(file, version);
    _ownedRoot = true;

    file.close();
    std::cout << "Decision Tree model loaded from " << filename << std::endl;
    return true;
}

DTNode* DecisionTree::loadTreeDTNode(std::ifstream &file, uint16_t version){
    uint8_t exists = 0;
    file.read(reinterpret_cast<char*>(&exists), sizeof(exists));
    if(!exists){
        return nullptr;
    }

    DTNode* node = new DTNode();
    uint8_t label = 0;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));
    node->label = static_cast<scent_class_t>(label);
    file.read(reinterpret_cast<char*>(&node->featureIndex), sizeof(node->featureIndex));
    file.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
    if(version >= 2){
        file.read(reinterpret_cast<char*>(&node->majorityCount), sizeof(node->majorityCount));
        file.read(reinterpret_cast<char*>(&node->totalSamples), sizeof(node->totalSamples));
    }
    else{
        node->majorityCount = 1;
        node->totalSamples = 1;
    }

    node->left = loadTreeDTNode(file, version);
    node->right = loadTreeDTNode(file, version);
    return node;
}

void DecisionTree::train(const ml_training_sample_t *samples, uint16_t count, uint16_t featureCount, uint8_t maxDepth, uint8_t minSamples){
    if(_ownedRoot && _root){
        deleteTree(const_cast<DTNode*>(_root));
    }

    _featureCount=  featureCount;
    std::vector<uint16_t> idx(count);
    for(uint16_t i=0; i<count; i++){
        idx[i] = i;
    }
    _root = buildTree(idx, samples,0,maxDepth, minSamples);
    _ownedRoot = true;
}

DTNode* DecisionTree::buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, uint8_t depth, uint8_t maxDepth, uint8_t minSampleSplit){
    DTNode* node = new DTNode();

    node->totalSamples = sampleIndicies.size();

    std::array<uint16_t, SCENT_CLASS_COUNT> classCounts = {0};
    for(auto idx:sampleIndicies){
        classCounts[samples[idx].label]++;
    }

    uint16_t majorityCount = 0;
    scent_class_t majorityLabel = SCENT_CLASS_UNKNOWN;
    for(uint16_t i=0; i<SCENT_CLASS_COUNT; i++){
        if(classCounts[i] > majorityCount){
            majorityCount = classCounts[i];
            majorityLabel = static_cast<scent_class_t>(i);
        }
    }
    node->majorityCount = majorityCount;

    if(sampleIndicies.size()==0){
        node->label = SCENT_CLASS_UNKNOWN;
        return node;
    }

    std::set<scent_class_t> uniqueClasses;
    for(auto idx:sampleIndicies){
        uniqueClasses.insert(samples[idx].label);
    }

    if(uniqueClasses.size()==1 ||depth >= maxDepth || sampleIndicies.size()<minSampleSplit){
        node->label = majorityLabel;
        return node;
    }

    int bestFeatureIndex = -1;
    float bestThreshold = 0.0f;
    float bestGini = 1.0f;

    findBestSplit(sampleIndicies, samples, bestFeatureIndex, bestThreshold, bestGini);

    if(bestFeatureIndex <0){
        node->label = majorityLabel;
        return node;
    }

    node->featureIndex = bestFeatureIndex;
    node->threshold = bestThreshold;
    std::vector<uint16_t> leftIdx, rightIdx;

    for(auto idx : sampleIndicies){
        if(samples[idx].features[bestFeatureIndex] < bestThreshold){
            leftIdx.push_back(idx);
        }
        else{
            rightIdx.push_back(idx);
        }
    }

    if(leftIdx.empty()||rightIdx.empty()){
        node->featureIndex = -1;
        node->label = majorityLabel;
        return node;
    }

    node->left = buildTree(leftIdx, samples, depth+1, maxDepth, minSampleSplit);
    node->right = buildTree(rightIdx, samples, depth+1, maxDepth, minSampleSplit);
    return node;
}

void DecisionTree::findBestSplit(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, int &bestFeatureIndex, float &bestThreshold, float &bestGini){
    bestGini = 1.0f;
    bestFeatureIndex = -1;
    uint16_t n = sampleIndicies.size();

    for(uint16_t i=0; i<_featureCount; i++){
        std::vector<float> featureValues;
        for(uint16_t idx : sampleIndicies){
            featureValues.push_back(samples[idx].features[i]);
        }
        std::sort(featureValues.begin(), featureValues.end());


        for(size_t j=1; j<featureValues.size();j++){
            if(featureValues[j] == featureValues[j-1]){
                continue;
            }

            float threshold = (featureValues[j-1] + featureValues[j]) / 2.0f;

            std::vector<uint16_t> leftIdx, rightIdx;
            for(auto idx : sampleIndicies){
                if(samples[idx].features[i] < threshold){
                    leftIdx.push_back(idx);
                }
                else{
                    rightIdx.push_back(idx);
                }
            }
            if(leftIdx.empty() || rightIdx.empty()){
                continue;
            }

            //calculate gini
            float leftGini = calculateGini(leftIdx, samples);
            float rightGini = calculateGini(rightIdx, samples);

            float weightedGini = (leftIdx.size() * leftGini + rightIdx.size() * rightGini) / n;

            if(weightedGini < bestGini){
                bestGini = weightedGini;
                bestFeatureIndex = i;
                bestThreshold = threshold;
            }
        }
    }
}

float DecisionTree::calculateGini(const std::vector<uint16_t>& sampleIndicies, const ml_training_sample_t* samples){
    
    if(sampleIndicies.empty()){
        return 0.0f;
    }
    std::map<scent_class_t, uint16_t> classCounts;
    for(auto i : sampleIndicies){
        classCounts[samples[i].label]++;
    }

    float gini = 1.0f;
    float n = sampleIndicies.size();
    for(const auto &p:classCounts){
        float prob = p.second / n;
        gini -= prob * prob;
    }
    return gini;
}

scent_class_t DecisionTree::getMajorityClass(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples){
    std::map<scent_class_t, uint16_t> classCounts;

    for(uint16_t i : sampleIndicies){
        classCounts[samples[i].label]++;
    }

    scent_class_t majorityClass = SCENT_CLASS_UNKNOWN;
    uint16_t maxCount = 0;
    for(const auto &j : classCounts){
        if(j.second > maxCount){
            maxCount = j.second;
            majorityClass = j.first;
        }
    }
    return majorityClass;
}

ml_metrics_t DecisionTree::evaluate(const ml_training_sample_t* samples, uint16_t sampleCount){
    ml_metrics_t metrics = {};
    metrics.total = sampleCount;

    for(uint16_t i=0; i<sampleCount; i++){
        scent_class_t pred = predict(samples[i].features);
        scent_class_t ans = samples[i].label;
        if(pred == ans){
            metrics.correct++;
        }
        
        if(ans >= 0 && ans < SCENT_CLASS_UNKNOWN && pred >= 0 && pred < SCENT_CLASS_COUNT){
            metrics.confusionMatrix[ans][pred]++;
        }
    }
    metrics.accuracy = (float)metrics.correct / (float)metrics.total;
    return metrics;
}

void DecisionTree::printTree(int maxDepth) const{
    if(_root){
        printDTNode(_root,0,maxDepth);
    }
}

void DecisionTree::printDTNode(const DTNode* DTNode, int depth, int maxDepth) const{
    if(!DTNode || depth > maxDepth){
        return;
    }
    for(int i=0; i<depth; i++){
        std::cout << "  ";
    }
    if(DTNode->featureIndex < 0 ){
        std::cout << "Leaf: Class=" << (int)DTNode->label << std::endl;
    }
    else{
        std::cout << "DTNode: Feature[" << DTNode->featureIndex << "] < " << DTNode->threshold << std::endl;
        printDTNode(DTNode->left, depth+1, maxDepth);
        printDTNode(DTNode->right, depth+1, maxDepth);
    }

}

void DecisionTree::countDTNodes(const DTNode* DTNode, uint16_t &count, uint16_t depth, uint16_t &maxDepth, uint16_t &leafCount)const{
    if(!DTNode){
        return;
    }
    count++;
    if(depth > maxDepth){
        maxDepth = depth;
    }
    if(DTNode->featureIndex < 0){
        leafCount++;
    }
    else{
        countDTNodes(DTNode->left, count, depth+1, maxDepth, leafCount);
        countDTNodes(DTNode->right, count, depth+1, maxDepth, leafCount);
    }
}

void DecisionTree::getStats(){
    DTNodeCount =0;
    depth =0;
    leafCount =0;
    countDTNodes(_root, DTNodeCount, 0, depth, leafCount);
}

scent_class_t DecisionTree::predict(const float* features) const{
    float output = 0.0f;
    return predictWithConfidence(features, output);
}

scent_class_t DecisionTree::predictWithConfidence(const float* features, float &confidenceOut) const{
    const DTNode* DTNode = _root;

    while(DTNode && DTNode->featureIndex >=0){
        if(features[DTNode->featureIndex] <DTNode->threshold){
            DTNode = DTNode->left;
        }
        else{
            DTNode = DTNode->right;
        }
    }
    if(DTNode){
        confidenceOut = (DTNode->totalSamples > 0) ? static_cast<float>(DTNode->majorityCount) / static_cast<float>(DTNode->totalSamples) : 1.0f;
        return DTNode->label;
    }
    confidenceOut = 0.0f;
    return SCENT_CLASS_UNKNOWN;
}