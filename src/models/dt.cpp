#include "dt.h"
#include <iostream>
#include <set>
#include <map>
#include <algorithm>

DecisionTree::~DecisionTree(){
    if(_ownedRoot && _root){
        deleteTree(const_cast<Node*>(_root));
    }
}

void DecisionTree::deleteTree(Node* n){
    if(n){
        deleteTree(n->left);
        deleteTree(n->right);
        delete n;
    }
}

void DecisionTree::train(const ml_training_sample_t *samples, uint16_t count, uint16_t featureCount, uint8_t maxDepth, uint8_t minSamples){
    if(_ownedRoot && _root){
        deleteTree(const_cast<Node*>(_root));
    }

    _featureCount=  featureCount;
    std::vector<uint16_t> idx(count);
    for(uint16_t i=0; i<count; i++){
        idx[i] = i;
    }
    _root = buildTree(idx, samples,0,maxDepth, minSamples);
    _ownedRoot = true;
}

Node* DecisionTree::buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples, uint8_t depth, uint8_t maxDepth, uint8_t minSampleSplit){
    Node* node = new Node();

    if(sampleIndicies.size()==0){
        node->label = SCENT_CLASS_UNKNOWN;
        return node;
    }

    std::set<scent_class_t> uniqueClasses;
    for(auto idx:sampleIndicies){
        uniqueClasses.insert(samples[idx].label);
    }

    if(uniqueClasses.size()==1 ||depth >= maxDepth || sampleIndicies.size()<minSampleSplit){
        node->label = getMajorityClass(sampleIndicies,samples);
        return node;
    }

    int bestFeatureIndex = -1;
    float bestThreshold = 0.0f;
    float bestGini = 1.0f;

    findBestSplit(sampleIndicies, samples, bestFeatureIndex, bestThreshold, bestGini);

    if(bestFeatureIndex <0){
        node->label = getMajorityClass(sampleIndicies,samples);
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
        node->label = getMajorityClass(sampleIndicies,samples);
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
        printNode(_root,0,maxDepth);
    }
}

void DecisionTree::printNode(const Node* node, int depth, int maxDepth) const{
    if(!node || depth > maxDepth){
        return;
    }
    for(int i=0; i<depth; i++){
        std::cout << "  ";
    }
    if(node->featureIndex < 0 ){
        std::cout << "Leaf: Class=" << (int)node->label << std::endl;
    }
    else{
        std::cout << "Node: Feature[" << node->featureIndex << "] < " << node->threshold << std::endl;
        printNode(node->left, depth+1, maxDepth);
        printNode(node->right, depth+1, maxDepth);
    }

}

void DecisionTree::countNodes(const Node* node, uint16_t &count, uint16_t depth, uint16_t &maxDepth, uint16_t &leafCount)const{
    if(!node){
        return;
    }
    count++;
    if(depth > maxDepth){
        maxDepth = depth;
    }
    if(node->featureIndex < 0){
        leafCount++;
    }
    else{
        countNodes(node->left, count, depth+1, maxDepth, leafCount);
        countNodes(node->right, count, depth+1, maxDepth, leafCount);
    }
}

void DecisionTree::getStats(){
    nodeCount =0;
    depth =0;
    leafCount =0;
    countNodes(_root, nodeCount, 0, depth, leafCount);
}

scent_class_t DecisionTree::predict(const float* features) const{
    const Node* node = _root;

    while(node && node->featureIndex >=0){
        if(features[node->featureIndex] <node->threshold){
            node = node->left;
        }
        else{
            node = node->right;
        }
    }
    return node ? node->label : SCENT_CLASS_UNKNOWN;
}