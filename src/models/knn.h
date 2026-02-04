#ifndef KNN_MODEL_H
#define KNN_MODEL_H

#include <stdint.h>
#include <math.h>
#include <stddef.h>

#ifdef ARDUINO
    #include "config.h"
    #include "ml_inference.h"
    #define KNN_FEATURE_COUNT TOTAL_ML_FEATURES
#else
    #include "csv_loader.h"
    typedef csv_training_sample_t ml_training_sample_t;
    typedef csv_metrics_t ml_metrics_t;
    #define KNN_FEATURE_COUNT CSV_FEATURE_COUNT
#endif

class KNN{
public:
    KNN(uint8_t k = 5) : _samples(nullptr), _sample_count(0), _k(k) {}

    scent_class_t predict(const float *features, uint16_t featureCount) const;

    void train(const ml_training_sample_t* samples, uint16_t count);
    ml_metrics_t evaluate(const ml_training_sample_t* samples, uint16_t sampleCount) const;

    scent_class_t predictWithConfidence(const float* features, uint16_t featureCount, float& confidence)const;
    void setK(uint8_t k){ _k = k; }
    uint8_t getK() const { return _k; }
    
    bool saveModel(const char* filename) const;
    bool loadModel(const char* filename);
private:
    const ml_training_sample_t *_samples;
    uint16_t _sample_count;
    uint8_t _k;
};

#endif