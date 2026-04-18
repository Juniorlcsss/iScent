#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#include <Arduino.h>
#include "config.h"
#include "bme688_handler.h"

#if !defined(EI_CLASSIFIER)
    #if __has_include("iScent_inferencing.h")
        #define EI_CLASSIFIER 1
    #else
        #define EI_CLASSIFIER 0
        #warning "Edge Impulse SDK not found, ML inference disabled"
    #endif
#endif

#if EI_CLASSIFIER && __has_include("model-parameters/model_metadata.h")
#include "model-parameters/model_metadata.h"
static constexpr uint16_t ML_CLASS_COUNT = EI_CLASSIFIER_LABEL_COUNT;
#else
static constexpr uint16_t ML_CLASS_COUNT = SCENT_CLASS_COUNT;
#endif


#define MAX_FULL_FEATURES 256

typedef struct{
    uint16_t correct;
    uint16_t total;
    float accuracy;
    uint16_t confusionMatrix[ML_CLASS_COUNT][ML_CLASS_COUNT];
} ml_metrics_t;

typedef struct{
    scent_class_t predictedClass;
    float confidence;
    float classConfidences[ML_CLASS_COUNT];
    float anomalyScore;
    bool isAnomalous;
    uint32_t inferenceTimeMs;
    uint32_t timestamp;
    bool valid;
} ml_prediction_t;

typedef enum {
    ML_MODEL_EDGE_IMPULSE = 0,
    ML_MODEL_DECISION_TREE,
    ML_MODEL_KNN,
    ML_MODEL_RANDOM_FOREST,
    ML_MODEL_COUNT
} ml_model_source_t;

typedef struct{
    float features[TOTAL_ML_FEATURES];
    uint16_t featureCount;
    bool ready;
} ml_feature_buffer_t;

typedef struct{
    scent_class_t label;
    float features[TOTAL_ML_FEATURES];
    uint32_t timestamp;
} ml_training_sample_t;

typedef struct{
    scent_class_t predictedClass;
    float confidence;
    float classScores[SCENT_CLASS_COUNT];
    scent_class_t dtClass;
    float dtConf;
    scent_class_t knnClass;
    float knnConf;
    scent_class_t rfClass;
    float rfConf;
    uint32_t inferenceTimeMs;
    uint32_t timestamp;
    bool valid;
} ml_ensemble_prediction_t;

typedef struct{
    bool active;
    uint8_t targetSamples;
    uint8_t collectedSamples;
    uint8_t failedAttempts;
    uint8_t maxFailedAttempts;
    uint32_t startTimeMs;
    uint32_t timeoutMs;
    uint32_t lastUpdateTimeMs;
} temporal_state_t;


typedef struct {
    float gas1[BME688_NUM_HEATER_STEPS];
    float gas2[BME688_NUM_HEATER_STEPS];
    float temp1;
    float hum1;
    float pres1;
    float temp2;
    float hum2;
    float pres2;
    bool valid;
} raw_sensor_snapshot_t;


class MLInference{
public:
    MLInference();
    ~MLInference();

    bool begin();
    bool isReady() const;

    bool extractFeatures(const dual_sensor_data_t &data);
    bool addToWindow(const dual_sensor_data_t &data);
    void clearFeatureBuffer();
    bool isFeatureBufferReady() const;
    bool isAccumulating() const;
    const char* getClassName(scent_class_t classId) const;
    scent_class_t getClassFromName(const char* name) const;

    bool runInference(ml_prediction_t &pred);
    bool runInferenceOnData(const dual_sensor_data_t &data, ml_prediction_t &pred);

    void setActiveModel(ml_model_source_t model);
    void nextModel();
    ml_model_source_t getActiveModel() const;
    const char* getActiveModelName() const;
    bool isModelAvailable(ml_model_source_t model) const;

    void setDataCollectionMode(bool enabled);
    bool isDataCollectionMode() const;
    void setCurrentLabel(scent_class_t label);
    bool collectSample(const dual_sensor_data_t &data);
    uint32_t getCollectedSampleCount() const;
    void clearCollectedSamples();
    bool exportTrainingData(const char* file);

    void setConfidenceThreshold(float threshold);
    void setAnomalyThreshold(float threshold);
    float getConfidenceThreshold() const;
    float getAnomalyThreshold() const;
    float computeAmbientScore(const raw_sensor_snapshot_t& raw);

    uint32_t getTotalInferences() const;
    float getAverageInferenceTimeMs() const;
    void resetStats();

    void printModelInfo();
    void printPrediction(const ml_prediction_t &pred);
    String getPredictionJSON(const ml_prediction_t &pred);
    void printFeatureDebug() const;

    bool runEnsembleInference(ml_ensemble_prediction_t &pred);

    void setInferenceMode(inference_mode_t mode);
    inference_mode_t getInferenceMode() const;
    const char* getInferenceModeName() const;
    void cycleInferenceMode();

    bool runActiveInference(ml_prediction_t &pred);
    static void ensembleToPrediction(const ml_ensemble_prediction_t &ens, ml_prediction_t &pred);

    void startTemporalCollection(uint8_t targetSamples = 0);
    bool updateTemporalCollection(ml_prediction_t &pred);
    bool isTemporalCollectionActive() const;
    bool isTemporalCollectionComplete() const;
    float getTemporalCollectionProgress() const;
    void cancelTemporalCollection();
    void finaliseTemporalPrediction(ml_prediction_t &pred);

    void resetTemporalBuffer();
    bool addToTemporalEnsemble(const ml_ensemble_prediction_t &pred);
    scent_class_t getTemporalPrediction(float &confidence) const;
    uint8_t getTemporalCount() const { return _temporal_count; }
    uint8_t getTemporalBufferSize() const { return TEMPORAL_BUFFER_SIZE; }

private:
    ml_feature_buffer_t _feature_buffer;
    inference_mode_t _inference_mode;
    temporal_state_t _temporal_state;

    raw_sensor_snapshot_t _raw_snapshot;

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

    //model selection
    ml_model_source_t _active_model;
    bool _model_available[ML_MODEL_COUNT];
    const char* const _model_names[ML_MODEL_COUNT];

    //data collection
    static const uint16_t MAX_TRAINING_SAMPLES = 1000;
    ml_training_sample_t *_training_samples;
    uint16_t _training_sample_count;

    //temporal buffer
    static const uint8_t TEMPORAL_BUFFER_SIZE = 10;
    float _temporal_scores[SCENT_CLASS_COUNT];
    uint8_t _temporal_count;

    //===========================================================================================================
    //methods
    //===========================================================================================================
    
    void normaliseFeatures(ml_feature_buffer_t& features);
    
    //return number of features written
    int computeBaseFeatures(const raw_sensor_snapshot_t& raw, float* outFeatures);
    
    //returns total features
    int computeEngineeredFeatures(float* features, int baseCount);
    
    //applys robust scaling and feature selection
    void applyScalingAndSelection(const float* allFeatures,int totalCount,ml_feature_buffer_t& output);

    // Stats helpers
    void computeStats(float *data, const float *window, uint16_t size);
    float computeMean(const float* data, uint16_t size);
    float computeSTD(const float* data, uint16_t size, float mean);
    float computeMin(const float* data, uint16_t size);
    float computeMax(const float* data, uint16_t size);
    float computeSlope(const float* data, uint16_t size);
    float computeTrapezoidalAUC(const float* data, uint16_t size);
    int computePeakIndex(const float* data, uint16_t size);
};
#endif