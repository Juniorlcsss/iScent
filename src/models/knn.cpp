#include "knn.h"
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

static const uint32_t KNN_MAGIC = 0x4B4E4E4C;
static const uint16_t KNN_VERSION = 1;

bool KNN::saveModel(const char* filename)const{
    if(!_samples || _sample_count ==0){
        std::cerr << "No model data to save." << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    file.write(reinterpret_cast<const char*>(&KNN_MAGIC), sizeof(KNN_MAGIC));
    file.write(reinterpret_cast<const char*>(&KNN_VERSION), sizeof(KNN_VERSION));
    file.write(reinterpret_cast<const char*>(&_k), sizeof(_k));

    uint16_t featureCount = KNN_FEATURE_COUNT;
    file.write(reinterpret_cast<const char*>(&featureCount), sizeof(featureCount));
    file.write(reinterpret_cast<const char*>(&_sample_count), sizeof(_sample_count));

    for(uint16_t i=0; i<_sample_count; i++){
        uint8_t label = static_cast<uint8_t>(_samples[i].label);
        file.write(reinterpret_cast<const char*>(&label), sizeof(label));
        file.write(reinterpret_cast<const char*>(_samples[i].features), featureCount * sizeof(float));
    }
    file.close();
    std::cout << "KNN model saved to " << filename << std::endl;
    return true;
}

bool KNN::loadModel(const char* filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return false;
    }

    uint32_t magic = 0;
    uint16_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if(magic != KNN_MAGIC || version != KNN_VERSION){
        std::cerr << "Invalid KNN model file." << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&_k), sizeof(_k));

    uint16_t featureCount = 0;
    file.read(reinterpret_cast<char*>(&featureCount), sizeof(featureCount));
    file.read(reinterpret_cast<char*>(&_sample_count), sizeof(_sample_count));

    std::cout << "Loading KNN model with " << _sample_count << " samples." << std::endl;
    file.close();
    return true;
}

struct Neighbour{
    float distance;
    scent_class_t label;
};

static float ecludianDistance(const float* a, const float* b, uint16_t len){
    float sum = 0.0f;
    for(uint16_t i=0; i<len; i++){
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

void KNN::train(const ml_training_sample_t* samples, uint16_t count){
    _samples = samples;
    _sample_count = count;
}

scent_class_t KNN::predict(const float* features, uint16_t featureCount) const{
    float confidence;
    return predictWithConfidence(features, featureCount, confidence);
}

scent_class_t KNN::predictWithConfidence(const float* features, uint16_t featureCount, float& confidence) const{
    if(!_samples || _sample_count ==0){
        confidence = 0.0f;
        return SCENT_CLASS_UNKNOWN;
    }

    std::vector<Neighbour> neighbours;
    neighbours.reserve(_sample_count);

    //dist
    for(uint16_t i=0; i<_sample_count; i++){
        float dist = ecludianDistance(features, _samples[i].features, featureCount);
        neighbours.push_back(Neighbour{dist, _samples[i].label});
    }

    //sort
    uint8_t k = (_k < neighbours.size()) ? _k : neighbours.size();
    std::partial_sort(neighbours.begin(), neighbours.begin()+k, neighbours.end(),[](const Neighbour &a, const Neighbour &b){
        return a.distance < b.distance;
    });
    std::vector<uint16_t> classCount(SCENT_CLASS_COUNT, 0);

    //count labels
    for(uint8_t i=0; i<k; i++){
        if(neighbours[i].label >= 0 && neighbours[i].label < SCENT_CLASS_COUNT){
            classCount[neighbours[i].label]++;
        }
    }

    uint16_t maxCount=0;
    scent_class_t predClass = SCENT_CLASS_UNKNOWN;

    for(uint16_t i=0; i<SCENT_CLASS_COUNT; i++){
        if(classCount[i] > maxCount){
            maxCount = classCount[i];
            predClass = (scent_class_t)i;
        }
    }

    //confidence
    confidence = (k>0)?(float)maxCount / k: 0.0f;
    
    return predClass;
}

ml_metrics_t KNN::evaluate(const ml_training_sample_t *samples, uint16_t count) const{
    ml_metrics_t metrics ={};
    metrics.total = count;
    metrics.correct = 0;

    //cm
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        for(int j=0; j<SCENT_CLASS_COUNT; j++){
            metrics.confusionMatrix[i][j] = 0;
        }
    }

    for(uint16_t i=0; i<count; i++){
        scent_class_t pred = predict(samples[i].features, 12);
        scent_class_t actual = samples[i].label;

        if(pred == actual){
            metrics.correct++;
        }

        if(actual >= 0 && actual < SCENT_CLASS_COUNT && pred >= 0 && pred < SCENT_CLASS_COUNT){
            metrics.confusionMatrix[actual][pred]++;
        }
    }

    metrics.accuracy = (metrics.total >0)? (float)metrics.correct / metrics.total : 0.0f;

    return metrics;
}