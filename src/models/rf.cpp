#include "rf.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>
#include <fstream>
#include <cstring>
#include <queue>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

void RandomForest::exportAsHeader(const char* path) const {
    std::string binPath = std::string(path) + ".bin";
    if(saveModel(binPath.c_str())){
        std::cout << "Binary model saved to " << binPath << std::endl;
    }
}

bool RandomForest::saveModel(const char* filename) const{
    std::ofstream file(filename, std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    //magic number
    uint32_t magic = 0x52464D4C;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    //version
    uint16_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    //model parameters
    file.write(reinterpret_cast<const char*>(&_numTrees), sizeof(_numTrees));
    file.write(reinterpret_cast<const char*>(&_maxDepth), sizeof(_maxDepth));
    file.write(reinterpret_cast<const char*>(&_minSamples), sizeof(_minSamples));
    file.write(reinterpret_cast<const char*>(&_featureSubsetRatio), sizeof(_featureSubsetRatio));
    file.write(reinterpret_cast<const char*>(&_featureCount), sizeof(_featureCount));
    file.write(reinterpret_cast<const char*>(&_featureSubsetSize), sizeof(_featureSubsetSize));
    file.write(reinterpret_cast<const char*>(&_oobError), sizeof(_oobError));

    //write feature importance
    for (uint16_t i = 0; i < _featureCount; i++) {
        float importance = (i < _featureImportance.size()) ? _featureImportance[i] : 0.0f;
        file.write(reinterpret_cast<const char*>(&importance), sizeof(importance));
    }

    //tree number
    uint16_t treeCount = _trees.size();
    file.write(reinterpret_cast<const char*>(&treeCount), sizeof(treeCount));

    //write trees
    for(const RFNode* tree : _trees){
        saveTree(file, tree);
    }
    std::cout << "Model saved to " << filename << std::endl;
    return true;
}

void RandomForest::saveTree(std::ofstream &file, const RFNode* RFNode)const{
    uint8_t exists = (RFNode != nullptr) ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&exists), sizeof(exists));
    if (!RFNode) return;

    //write RFNode data
    uint8_t label = static_cast<uint8_t>(RFNode->label);
    file.write(reinterpret_cast<const char*>(&label), sizeof(label));
    file.write(reinterpret_cast<const char*>(&RFNode->featureIndex), sizeof(RFNode->featureIndex));
    file.write(reinterpret_cast<const char*>(&RFNode->threshold), sizeof(RFNode->threshold));

    //save children
    saveTree(file, RFNode->left);
    saveTree(file, RFNode->right);
}

bool RandomForest::loadModel(const char* filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return false;
    }

    //validate
    uint32_t magic = 0;
    uint16_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if(magic != 0x52464D4C || version != 1){
        std::cerr << "Invalid model file format." << std::endl;
        return false;
    }

    clearTrees();

    //read params
    file.read(reinterpret_cast<char*>(&_numTrees), sizeof(_numTrees));
    file.read(reinterpret_cast<char*>(&_maxDepth), sizeof(_maxDepth));
    file.read(reinterpret_cast<char*>(&_minSamples), sizeof(_minSamples));
    file.read(reinterpret_cast<char*>(&_featureSubsetRatio), sizeof(_featureSubsetRatio));
    file.read(reinterpret_cast<char*>(&_featureCount), sizeof(_featureCount));
    file.read(reinterpret_cast<char*>(&_featureSubsetSize), sizeof(_featureSubsetSize));
    file.read(reinterpret_cast<char*>(&_oobError), sizeof(_oobError));

    //read feature importance
    _featureImportance.resize(_featureCount);
    for(uint16_t i=0; i<_featureCount; i++){
        file.read(reinterpret_cast<char*>(&_featureImportance[i]), sizeof(float));
    }
    //read tree count
    uint16_t treeCount = 0;
    file.read(reinterpret_cast<char*>(&treeCount), sizeof(treeCount));

    //read tree
    _trees.reserve(treeCount);
    for(uint16_t i=0; i<treeCount; i++){
        RFNode* tree = loadTree(file);
        _trees.push_back(tree);
    }

    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
    return true;
}

RFNode* RandomForest::loadTree(std::ifstream &f){
    uint8_t exists = 0;
    f.read(reinterpret_cast<char*>(&exists), sizeof(exists));
    if(!exists){
        return nullptr;
    }
    RFNode* node = new RFNode();

    //read RFNode data
    uint8_t label = 0;
    f.read(reinterpret_cast<char*>(&label), sizeof(label));
    node->label = static_cast<scent_class_t>(label);
    f.read(reinterpret_cast<char*>(&node->featureIndex), sizeof(node->featureIndex));
    f.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold)); 

    //load children
    node->left = loadTree(f);
    node->right = loadTree(f);
    return node;
}

void RandomForest::flattenTree(const RFNode* root, std::vector<FlatRFNode> &flatRFNodes)const{
    if(!root)return;

    flatRFNodes.clear();

    //BFS
    std::queue<const RFNode*> RFNodeQueue;
    std::queue<int16_t> indexQueue;

    RFNodeQueue.push(root);
    indexQueue.push(0);

    //assign indices
    std::vector<const RFNode*> RFNodes;
    std::map<const RFNode*,int16_t> RFNodeIdx;

    while(!RFNodeQueue.empty()){
        const RFNode* current = RFNodeQueue.front();
        RFNodeQueue.pop();
        int16_t currentIdx = RFNodes.size();

        RFNodeIdx[current] = currentIdx;
        RFNodes.push_back(current);

        if(current->left){
            RFNodeQueue.push(current->left);
        }
        if(current->right){
            RFNodeQueue.push(current->right);
        }
    }

    //create flat RFNodes
    for(const RFNode* RFNode : RFNodes){
        FlatRFNode flat;
        flat.featureIndex = static_cast<int8_t>(RFNode->featureIndex);
        flat.label = static_cast<uint8_t>(RFNode->label);
        flat.threshold = RFNode->threshold;
        flat.leftChild = (RFNode->left) ? RFNodeIdx[RFNode->left] : -1;
        flat.rightChild = (RFNode->right) ? RFNodeIdx[RFNode->right] : -1;
        flatRFNodes.push_back(flat);
    }
}


RandomForest::RandomForest(uint16_t numTrees, uint8_t maxDepth, uint8_t minSamples, float featureSubsetRatio):
    _numTrees(numTrees), _maxDepth(maxDepth), _minSamples(minSamples),
    _featureSubsetRatio(featureSubsetRatio), _featureCount(12),
    _featureSubsetSize(0), _oobError(0.0f) {}

RandomForest::~RandomForest() {
    clearTrees();
}

void RandomForest::clearTrees() {
    for (auto tree : _trees) {
        deleteTree(tree);
    }
    _trees.clear();
}

void RandomForest::deleteTree(RFNode* RFNode) {
    if (RFNode) {
        deleteTree(RFNode->left);
        deleteTree(RFNode->right);
        delete RFNode;
    }
}

void RandomForest::bootstrapSample(uint16_t totalSamples, std::vector<uint16_t>& inBag, std::vector<uint16_t>& outOfBag, std::mt19937& rng){
    std::vector<bool> selected(totalSamples, false);
    std::uniform_int_distribution<uint16_t> dst(0, totalSamples - 1);

    inBag.clear();
    inBag.reserve(totalSamples);

    for(uint16_t i=0; i<totalSamples; i++){
        uint16_t idx=dst(rng);
        inBag.push_back(idx);
        selected[idx] = true;
    }

    outOfBag.clear();
    for(uint16_t i=0; i<totalSamples; i++){
        if(!selected[i]){
            outOfBag.push_back(i);
        }
    }
}

std::vector<uint16_t> RandomForest::getRandomFeatureSubset(std::mt19937& rng){
    std::vector<uint16_t> allFeatures(_featureCount);
    for(uint16_t i=0; i<_featureCount; i++){
        allFeatures[i] = i;
    }

    std::shuffle(allFeatures.begin(), allFeatures.end(), rng);
    std::vector<uint16_t> subset(allFeatures.begin(), allFeatures.begin() + _featureSubsetSize);
    return subset;
}

void RandomForest::train(const ml_training_sample_t *samples, uint16_t count, uint16_t featureCount){
    clearTrees();

    _featureCount = featureCount;
    _featureSubsetSize = (uint16_t)(_featureCount * _featureSubsetRatio);

    if(_featureSubsetSize < 1){
        _featureSubsetSize = 1;
    }
    if(_featureSubsetSize > _featureCount){
        _featureSubsetSize = _featureCount;
    }

    //init feature importance
    _featureImportance.assign(_featureCount, 0.0f);

    //per-tree storage
    std::vector<RFNode*> builtTrees(_numTrees, nullptr);
    std::vector<std::vector<uint16_t>> outOfBagSets(_numTrees);
    std::vector<uint32_t> seeds(_numTrees);
    std::random_device rd;
    for (uint16_t i = 0; i < _numTrees; ++i) {
        seeds[i] = rd();
    }

    std::cout << "Training Random Forest with " << _numTrees << " trees." << std::endl;
    std::cout << "Feature subset size: " << _featureSubsetSize << "/" << _featureCount << std::endl;

    std::atomic<int> builtCount{0};

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<static_cast<int>(_numTrees);++i) {
        std::mt19937 rng(seeds[i]);
        std::vector<uint16_t>inBag, outOfBag;
        bootstrapSample(count,inBag,outOfBag,rng);

        std::vector<float> localImportance(_featureCount,  0.0f);
        RFNode* tree = buildTree(inBag,samples,0,localImportance,rng);

        builtTrees[i] = tree;
        outOfBagSets[i] = std::move(outOfBag);

        //reduce fi
        #pragma omp critical
        {
            for (uint16_t f=0; f<_featureCount; ++f) {
                _featureImportance[f] += localImportance[f];
            }
        }

        int done = ++builtCount;
        if (done % 10 == 0 || done == _numTrees) {
            #pragma omp critical
            {
                std::cout <<"Trees Built:"<< done<< "/" << _numTrees<< "\r"<< std::flush;
            }
        }
    }
    std::cout << std::endl;

    _trees = std::move(builtTrees);

    //oob predictions
    std::vector<std::vector<scent_class_t>> oobPredictions(count);
    for (uint16_t i=0; i<_numTrees; ++i) {
        const RFNode* tree=_trees[i];
        for (uint16_t idx:outOfBagSets[i]) {
            scent_class_t pred=predictTree(tree,samples[idx].features);
            oobPredictions[idx].push_back(pred);
        }
    }

    //get oob error
    uint16_t correct=0, total=0;

    for(uint16_t i=0; i<count; i++){
        if(oobPredictions[i].empty()){
            continue;
        }

        //majority vote
        std::vector<uint16_t> votes(SCENT_CLASS_COUNT, 0);
        for(scent_class_t pred : oobPredictions[i]){
            if(pred >= 0 && pred < SCENT_CLASS_COUNT){
                votes[pred]++;
            }
        }

        scent_class_t oobPred = SCENT_CLASS_UNKNOWN;
        uint16_t maxVotes = 0;
        for(int j=0; j<SCENT_CLASS_COUNT; j++){
            if(votes[j] > maxVotes){
                maxVotes = votes[j];
                oobPred = (scent_class_t)j;
            }
        }

        if(oobPred == samples[i].label){
            correct++;
        }
        total++;
    }

    _oobError = (total > 0) ? 1.0f - ((float)correct / total) : 0.0f;

    //normalise
    float maxImportance = 0.0f;
    for(float i : _featureImportance){
        if(i > maxImportance){
            maxImportance = i;
        }
    }
    if(maxImportance > 0.0f){
        for(float &i : _featureImportance){
            i /= maxImportance;
        }
    }

    std::cout << "Training complete. OOB Error: " << std::fixed << std::setprecision(2) << (_oobError * 100) << "%" << std::endl;
}

RFNode* RandomForest::buildTree(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t *samples, uint8_t depth, std::vector<float>& featureImportance, std::mt19937& rng){
    RFNode* node = new RFNode();

    if(sampleIndicies.empty()){
        node->label = SCENT_CLASS_UNKNOWN;
        return node;
    }

    //check all same class
    std::set<scent_class_t> uniqueClasses;

    for(uint16_t idx : sampleIndicies){
        uniqueClasses.insert(samples[idx].label);
    }
    if(uniqueClasses.size() == 1 || depth >= _maxDepth || sampleIndicies.size() < _minSamples){
        node->label = getMajorityClass(sampleIndicies, samples);
        return node;
    }

    std::vector<uint16_t> featureSubset = getRandomFeatureSubset(rng);

    //best split
    int bestFeatureIndex = -1;
    float bestThreshold = 0.0f, bestGini = 1.0f;
    findBestSplit(sampleIndicies, samples, featureSubset, bestFeatureIndex, bestThreshold, bestGini);

    if(bestFeatureIndex < 0){
        node->label = getMajorityClass(sampleIndicies, samples);
        return node;
    }

    node->featureIndex = bestFeatureIndex;
    node->threshold = bestThreshold;

    //importance
    float parentGini = calculateGini(sampleIndicies, samples);
    featureImportance[bestFeatureIndex] += (parentGini - bestGini) * sampleIndicies.size();

    //split samples
    std::vector<uint16_t> leftI, rightI;
    for(uint16_t i : sampleIndicies){
        if(samples[i].features[bestFeatureIndex] < bestThreshold){
            leftI.push_back(i);
        }else{
            rightI.push_back(i);
        }
    }

    if(leftI.empty() || rightI.empty()){
        node->featureIndex = -1;
        node->label = getMajorityClass(sampleIndicies, samples);
        return node;
    }

    node->left = buildTree(leftI, samples, depth + 1, featureImportance, rng);
    node->right = buildTree(rightI, samples, depth + 1, featureImportance, rng);
    return node;
}

void RandomForest::findBestSplit(const std::vector<uint16_t>& sampleIndices,const ml_training_sample_t* samples,const std::vector<uint16_t>& featureSubset,int& bestFeature, float& bestThreshold, float& bestGini){
    bestGini = 1.0f;
    bestFeature = -1;
    bestThreshold = 0.0f;

    uint16_t n = sampleIndices.size();
    for(uint16_t f : featureSubset){
        //get feature values
        std::vector<float> featureValues;
        featureValues.reserve(n);
        for(uint16_t idx : sampleIndices){
            featureValues.push_back(samples[idx].features[f]);
        }

        //unique & sort
        std::set<float> uniqueValues(featureValues.begin(), featureValues.end());
        std::vector<float> thresholds(uniqueValues.begin(), uniqueValues.end());
        std::sort(thresholds.begin(), thresholds.end());

        //try splits
        for(size_t t=1; t<thresholds.size(); t++){
            float threshold = (thresholds[t-1] + thresholds[t]) / 2.0f;

            std::vector<uint16_t> left, right;
            for(uint16_t idx : sampleIndices){
                if(samples[idx].features[f] < threshold){
                    left.push_back(idx);
                }else{
                    right.push_back(idx);
                }
            }

            if(left.empty() || right.empty()) continue;

            float leftGini = calculateGini(left, samples);
            float rightGini = calculateGini(right, samples);
            float weightedGini = (left.size() * leftGini + right.size() * rightGini) / n;

            if(weightedGini < bestGini){
                bestGini = weightedGini;
                bestFeature = f;
                bestThreshold = threshold;
            }
        }
    }
}


float RandomForest::calculateGini(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t *samples){
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

scent_class_t RandomForest::getMajorityClass(const std::vector<uint16_t> &sampleIndicies, const ml_training_sample_t* samples){
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

scent_class_t RandomForest::predictTree(const RFNode* RFNode, const float* features) const {
    while(RFNode && RFNode->featureIndex >=0){
        if (features[RFNode->featureIndex] < RFNode->threshold) {
            RFNode = RFNode->left;
        } else {
            RFNode = RFNode->right;
        }
    }
    return RFNode ? RFNode->label : SCENT_CLASS_UNKNOWN;
}

scent_class_t RandomForest::predict(const float* features) const {
    float confidence;
    return predictWithConfidence(features, confidence);
}

scent_class_t RandomForest::predictWithConfidence(const float* features, float& confidence) const {
    if(_trees.empty()){
        confidence = 0.0f;
        return SCENT_CLASS_UNKNOWN;
    }
    std::vector<uint16_t> votes(SCENT_CLASS_COUNT, 0);

    for(const RFNode* tree : _trees){
        scent_class_t pred = predictTree(tree, features);
        if(pred >=0 && pred < SCENT_CLASS_COUNT){
            votes[pred]++;
        }
    }

    scent_class_t bestClass = SCENT_CLASS_UNKNOWN;
    uint16_t maxVotes = 0;
    for(int i = 0; i < SCENT_CLASS_COUNT; i++){
        if(votes[i] > maxVotes){
            maxVotes = votes[i];
            bestClass = static_cast<scent_class_t>(i);
        }
    }
    confidence = (float)maxVotes / _trees.size();
    return bestClass;
}

void RandomForest::predictProb(const float* features, float* probabilities) const {
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        probabilities[i] = 0.0f;
    }
    if(_trees.empty()) return;

    for(const RFNode* tree : _trees){
        scent_class_t pred = predictTree(tree, features);
        if(pred >=0 && pred < SCENT_CLASS_COUNT){
            probabilities[pred] += 1.0f;
        }
    }
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        probabilities[i] /= _trees.size();
    }
}

ml_metrics_t RandomForest::evaluate(const ml_training_sample_t* samples, uint16_t sampleCount) const {
    ml_metrics_t metrics = {};
    metrics.total = sampleCount;

    for(uint16_t i=0; i<sampleCount; i++){
        scent_class_t pred = predict(samples[i].features);
        scent_class_t ans = samples[i].label;
        if(pred == ans){
            metrics.correct++;
        }

        if(ans >=0 && ans < SCENT_CLASS_COUNT && pred >=0 && pred < SCENT_CLASS_COUNT){
            metrics.confusionMatrix[ans][pred]++;
        }
    }
    metrics.accuracy = (float)metrics.correct / (float)metrics.total;
    return metrics;

}

void RandomForest::computeFeatureImportance(float* importance) const {
    for(uint16_t i=0; i<_featureCount; i++){
        importance[i] = _featureImportance[i];
    }
}

void RandomForest::printFeatureImportance() const {
    std::cout << "Feature Importance:" << std::endl;
    for(uint16_t i=0; i<_featureCount; i++){
        std::cout << "Feature " << i << ": " << std::fixed << std::setprecision(4) << _featureImportance[i] << std::endl;
    }
}

uint16_t RandomForest::getTotalRFNodes() const {
    uint16_t total = 0;
    for (const RFNode* tree : _trees) {
        total += countRFNodes(tree);
    }
    return total;
}

uint16_t RandomForest::countRFNodes(const RFNode* RFNode) const {
    if (!RFNode) {
        return 0;
    }
    return 1 + countRFNodes(RFNode->left) + countRFNodes(RFNode->right);
}