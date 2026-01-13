#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#include <Arduino.h>
#include "config.h"
#include "bme688_handler.h"

//edge impulse library check
#if __has_include("edge-impulse-sdk/classifier/ei_run_classifier.h")
    #include "edge-impulse-sdk/classifier/ei_run_classifier.h"
    #define EI_CLASSIFIER 1
#else
    #define EI_CLASSIFIER 0
    #warning "Edge Impulse SDK not found, ML inference disabled"
#endif


//prediction struct
typedef struct{
    scent_class_t predictedClass;
    float confidence;
    float classConfidences[SCENT_CLASS_COUNT];
    float anomalyScore;
    bool isAnomalous;
    uint32_t inferenceTimeMs;
    uint32_t timestamp;
    bool valid;
} ml_prediction_t;

//feature buffer struct
typedef struct{
    float features[TOTAL_ML_FEATURES];
    uint16_t featureCount;
    bool ready;
}ml_feature_buffer_t;

//training struct
typedef struct{
    scent_class_t label;
    float features[TOTAL_ML_FEATURES];
    uint32_t timestamp;
} ml_training_sample_t;

class MLInference{
public:
    MLInference();
    ~MLInference();

    //===========================================================================================================
    //init
    //===========================================================================================================
    bool begin();//
    bool isReady() const;//

    //===========================================================================================================
    //feature extraction
    //===========================================================================================================
    bool extractFeatures(const dual_sensor_data_t &data);//
    bool addToWindow(const dual_sensor_data_t &data);//
    void clearFeatureBuffer();//
    bool isFeatureBufferReady() const;//

    const char* getClassName(scent_class_t classId) const;//
    scent_class_t getClassFromName(const char* name) const;//
    
    //===========================================================================================================
    //inference
    //===========================================================================================================
    bool runInference(ml_prediction_t &pred);//
    bool runInferenceOnData(const dual_sensor_data_t &data, ml_prediction_t &pred);//

    //===========================================================================================================
    //data collection
    //===========================================================================================================
    void setDataCollectionMode(bool enabled);
    bool isDataCollectionMode() const;

    void setCurrentLabel(scent_class_t label);//
    bool collectSample(const dual_sensor_data_t &data);
    uint32_t getCollectedSampleCount() const;
    void clearCollectedSamples();
    bool exportTrainingData(const char* file);

    //===========================================================================================================
    //thresholds
    //===========================================================================================================
    void setConfidenceThreshold(float threshold);
    void setAnomalyThreshold(float threshold);
    float getConfidenceThreshold() const;
    float getAnomalyThreshold() const;

    //===========================================================================================================
    //stats
    //===========================================================================================================
    uint32_t getTotalInferences() const;
    float getAverageInferenceTimeMs() const;
    void resetStats();

    //===========================================================================================================
    //export
    //===========================================================================================================
    void printModelInfo();//
    void printPrediction(const ml_prediction_t &pred);
    String getPredictionJSON(const ml_prediction_t &pred);

private:

    //===========================================================================================================
    //vals
    //===========================================================================================================
    ml_feature_buffer_t _feature_buffer;

    //window
    uint16_t _window_index;
    uint16_t _window_size;

    float _windowTempP[ML_WINDOW_SIZE];
    float _windowHumP[ML_WINDOW_SIZE];
    float _windowPresP[ML_WINDOW_SIZE];
    float _windowGasP[ML_WINDOW_SIZE][BME688_NUM_HEATER_STEPS];

    float _windowTempS[ML_WINDOW_SIZE];
    float _windowHumS[ML_WINDOW_SIZE];
    float _windowPresS[ML_WINDOW_SIZE];
    float _windowGasS[ML_WINDOW_SIZE][BME688_NUM_HEATER_STEPS];

    //cfg
    float _confidence_threshold;
    float _anomaly_threshold;
    bool _init;
    bool _data_collection_mode;
    scent_class_t _current_label;
    //stats
    uint32_t _total_inferences;
    uint64_t _total_inference_time_ms;

    //data collection
    static const uint16_t MAX_TRAINING_SAMPLES = 1000;
    ml_training_sample_t *_training_samples;
    uint16_t _training_sample_count;

        //===========================================================================================================
    //methods
    //===========================================================================================================
    void normaliseFeatures();
    void computeStats(float *data, const float *window, uint16_t size);
    void extractGasFeatures(float *output, const float window[][BME_688_NUM_HEATER_STEPS], uint16_t size);

    float computeMean(const float* data, uint16_t size);
    float computeSTD(const float* data, uint16_t size, float mean);
    float computeMin(const float* data, uint16_t size);
    float computeMax(const float* data, uint16_t size);
    float computeSlope(const float* data, uint16_t size);
};
#endif