#include "knn.h"
#include <algorithm>
#include <vector>

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