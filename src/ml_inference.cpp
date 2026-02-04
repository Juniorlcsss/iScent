#include "ml_inference.h"
#include <LittleFS.h>

#if EI_CLASSIFIER
#include "iScent_inferencing.h"
#endif

#include "model headers/dt_model_header.h"
#include "model headers/knn_model_header.h"
#include "model headers/rf_model_header.h"

MLInference::MLInference():
    _window_index(0),
    _window_size(0),
    _confidence_threshold(ML_CONFIDENCE_THRESHOLD),
    _anomaly_threshold(ML_ANOMALY_THRESHOLD),
    _init(false),
    _data_collection_mode(false),
    _current_label(SCENT_CLASS_UNKNOWN),
    _total_inferences(0),
    _total_inference_time_ms(0),
    _active_model(ML_MODEL_EDGE_IMPULSE),
    _model_names{"Edge Impulse", "Decision Tree", "KNN", "Random Forest"},
    _training_samples(nullptr),
    _training_sample_count(0)
{
    memset(_model_available, 0, sizeof(_model_available));
    memset(&_feature_buffer,0,sizeof(_feature_buffer));
    memset(_windowTempP,0,sizeof(_windowTempP));
    memset(_windowHumP,0,sizeof(_windowHumP));
    memset(_windowPresP,0,sizeof(_windowPresP));
    memset(_windowGasP,0,sizeof(_windowGasP));
    memset(_windowTempS,0,sizeof(_windowTempS));
    memset(_windowHumS,0,sizeof(_windowHumS));
    memset(_windowPresS,0,sizeof(_windowPresS));
    memset(_windowGasS,0,sizeof(_windowGasS));
}

MLInference::~MLInference(){
    if(_training_samples!=nullptr){
        clearCollectedSamples();
    }
}

//===========================================================================================================
//init
//===========================================================================================================
bool MLInference::begin(){
    DEBUG_PRINTLN(F("[MLInference] Initializing ML Inference module"));

    _model_available[ML_MODEL_EDGE_IMPULSE] = (EI_CLASSIFIER != 0);
    _model_available[ML_MODEL_DECISION_TREE] = true;
    _model_available[ML_MODEL_KNN] = true;
    _model_available[ML_MODEL_RANDOM_FOREST] = true;

#if EI_CLASSIFIER
    //init edge impulse model info
    DEBUG_PRINTF("[MLInference] Edge Impulse model info:\n");
    DEBUG_PRINTF("[ML] Input features: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    DEBUG_PRINTF("[ML] Label count: %d\n", EI_CLASSIFIER_LABEL_COUNT);
    DEBUG_PRINTF("[ML] Anomaly detection: %s\n", 
    EI_CLASSIFIER_HAS_ANOMALY ? "Y" : "N");

    //verify feature count
    if(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE != TOTAL_ML_FEATURES){
        DEBUG_PRINTF("[ML] WARNING: Feature size mismatch! Expected %d, model has %d\n",
        TOTAL_ML_FEATURES, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    }
    _init = true;

#else
    DEBUG_PRINTLN(F("[MLInference] Edge Impulse SDK not found, ML Inference disabled"));
    DEBUG_PRINTLN(F("[MLInference] Data collection mode only"));
    _init = true;
#endif

    setActiveModel(_active_model);

    clearFeatureBuffer();
    return _init;
}

bool MLInference::isReady() const{
    return _init && _feature_buffer.ready;
}

//===========================================================================================================
//feature extraction
//===========================================================================================================

bool MLInference::extractFeatures(const dual_sensor_data_t &sensor_data){
    uint16_t idx = 0;

    //clear
    memset(_feature_buffer.features, 0, sizeof(_feature_buffer.features));

    //*Primary sensor features
    _feature_buffer.features[idx++] = sensor_data.primary.temperatures[0];
    _feature_buffer.features[idx++] = sensor_data.primary.humidities[0];
    _feature_buffer.features[idx++] = sensor_data.primary.pressures[0];
    _feature_buffer.features[idx++] = sensor_data.primary.gas_resistances[0];

    //*Secondary sensor features
    _feature_buffer.features[idx++] = sensor_data.secondary.temperatures[0];
    _feature_buffer.features[idx++] = sensor_data.secondary.humidities[0];
    _feature_buffer.features[idx++] = sensor_data.secondary.pressures[0];
    _feature_buffer.features[idx++] = sensor_data.secondary.gas_resistances[0];

    //delta
    _feature_buffer.features[idx++] = sensor_data.delta_temp;
    _feature_buffer.features[idx++] = sensor_data.delta_hum;
    _feature_buffer.features[idx++] = sensor_data.delta_pres;
    _feature_buffer.features[idx++] = sensor_data.delta_gas_avg;

    _feature_buffer.featureCount = idx;
    _feature_buffer.ready = true;

    DEBUG_PRINTF("[MLInference] Extracted %d features\n", idx);

    return true;
}


bool MLInference::addToWindow(const dual_sensor_data_t &sensor_data){
    if(!sensor_data.valid){
        DEBUG_PRINTLN(F("[MLInference] Invalid sensor data in addToWindow"));
        return false;
    }

    //add primary sensor data
    if(sensor_data.primary.complete){
        _windowTempP[_window_index] = sensor_data.primary.temperatures[0];
        _windowHumP[_window_index] = sensor_data.primary.humidities[0];
        _windowPresP[_window_index] = sensor_data.primary.pressures[0];

        for(uint8_t i =0; i< BME688_NUM_HEATER_STEPS;i++){
            _windowGasP[_window_index][i] = sensor_data.primary.gas_resistances[i];
        }
    }

    //add secondary sensor data
    if(sensor_data.secondary.complete){
        _windowTempS[_window_index] = sensor_data.secondary.temperatures[0];
        _windowHumS[_window_index] = sensor_data.secondary.humidities[0];
        _windowPresS[_window_index] = sensor_data.secondary.pressures[0];

        for(uint8_t i =0; i< BME688_NUM_HEATER_STEPS;i++){
            _windowGasS[_window_index][i] = sensor_data.secondary.gas_resistances[i];
        }
    }

    //window index
    _window_index = (_window_index +1) % ML_WINDOW_SIZE;

    if(_window_size < ML_WINDOW_SIZE){
        _window_size++;
    }

    if(_window_size >= ML_SAMPLES){
        return extractFeatures(sensor_data);
    }

    return false;
}

void MLInference::clearFeatureBuffer(){
    memset(&_feature_buffer,0,sizeof(_feature_buffer));
    _window_index =0;
    _window_size =0;
    memset(_windowTempP,0,sizeof(_windowTempP));
    memset(_windowHumP,0,sizeof(_windowHumP));
    memset(_windowPresP,0,sizeof(_windowPresP));
    memset(_windowGasP,0,sizeof(_windowGasP));
    memset(_windowTempS,0,sizeof(_windowTempS));
    memset(_windowHumS,0,sizeof(_windowHumS));
    memset(_windowPresS,0,sizeof(_windowPresS));
    memset(_windowGasS,0,sizeof(_windowGasS));
}

bool MLInference::isFeatureBufferReady() const{
    return _feature_buffer.ready;
}

const char* MLInference::getClassName(scent_class_t classId) const{
    if(classId < SCENT_CLASS_COUNT){
        return SCENT_CLASS_NAMES[classId];
    }
    return "Unknown";
}

scent_class_t MLInference::getClassFromName(const char* name) const{
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        if(strcmp(name, SCENT_CLASS_NAMES[i])==0){
            return (scent_class_t)i;
        }

    }
    return SCENT_CLASS_UNKNOWN;
}   

void MLInference::setActiveModel(ml_model_source_t model){
    if(model >= ML_MODEL_COUNT){
        model = ML_MODEL_EDGE_IMPULSE;
    }

    if(_model_available[model]){
        _active_model = model;
        return;
    }

    for(uint8_t i=1; i<=ML_MODEL_COUNT; i++){
        ml_model_source_t candidate = (ml_model_source_t)((model + i) % ML_MODEL_COUNT);
        if(_model_available[candidate]){
            _active_model = candidate;
            return;
        }
    }
}

void MLInference::nextModel(){
    setActiveModel((ml_model_source_t)((_active_model + 1) % ML_MODEL_COUNT));
}

ml_model_source_t MLInference::getActiveModel() const{
    return _active_model;
}

const char* MLInference::getActiveModelName() const{
    return _model_names[_active_model];
}

bool MLInference::isModelAvailable(ml_model_source_t model) const{
    if(model >= ML_MODEL_COUNT){
        return false;
    }
    return _model_available[model];
}


void MLInference::extractGasFeatures(float* output, const float window[][BME688_NUM_HEATER_STEPS], uint16_t samples){
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
        float values[ML_WINDOW_SIZE];

        //get vals from heater across window
        for(uint16_t s=0; s<samples && s<ML_WINDOW_SIZE; s++){
            values[s] = window[s][i];
        }

        //stats
        float mean = computeMean(values, samples);
        float std = computeSTD(values, samples, mean);
        float slope = computeSlope(values, samples);

        //normalise
        output[i*3+0] = (mean >0) ? log10f(mean) / 6.0f:0.0f;
        output[i*3+1] = (std>0) ? log10f(std) / 6.0f:0.0f;
        output[i*3+2] = slope / 1000.0f;
    }
}

void MLInference::computeStats(float* stats, const float* window, uint16_t size){
    if(size==0){
        memset(stats,0,5*sizeof(float));
        return;
    } 

    float mean = computeMean(window, size);

    stats[0] = mean; //mean
    stats[1] = computeSTD(window, size, mean); //std dev
    stats[2] = computeMin(window, size); //min
    stats[3] = computeMax(window, size); //max
    stats[4] = computeSlope(window, size); //slope
}

float MLInference::computeMean(const float* data, uint16_t size){
    if(size == 0){
        return 0.0f;
    }
    float sum =0.0f;
    for(uint16_t i=0; i<size; i++){
        sum += data[i];
    }
    return sum / size;
}

float MLInference::computeSTD(const float* data, uint16_t size, float mean){
    if(size <=1){
        return 0.0f;
    }
    float sum = 0.0f;
    for(uint16_t i=0; i<size; i++){
        float diff = data[i] - mean;
        sum += diff * diff;
    }
    return sqrtf(sum/ (size-1));
}

float MLInference::computeMin(const float* data, uint16_t size){
    if(size == 0){
        return 0.0f;
    }
    float min = data[0];
    for(uint16_t i=1; i<size; i++){
        if(data[i] < min){
            min = data[i];
        }
    }
    return min;
}

float MLInference::computeMax(const float* data, uint16_t size){
    if(size==0){
        return 0.0f;
    }
    float max = data[0];
    for(uint16_t i=1; i<size; i++){
        if(data[i] > max){
            max = data[i];
        }
    }
    return max;
}

float MLInference::computeSlope(const float* data, uint16_t size){
    if(size <= 1){
        return 0.0f;
    }

    //linear regression
    float sumX =0.0f, sumY=0.0f, sumXY=0.0f, sumXX=0.0f;

    for(uint16_t i=0; i<size; i++){
        sumX += i;
        sumY += data[i];
        sumXY += i * data[i];
        sumXX += i * i;
    }

    float denominator = size * sumXX - sumX * sumX;
    if(fabsf(denominator) < 1e-6){
        return 0.0f;
    }
    return (size * sumXY - sumX * sumY) / denominator;
}

//===========================================================================================================
//inference
//===========================================================================================================

bool MLInference::runInference(ml_prediction_t &pred){
    memset(&pred, 0, sizeof(pred));
    pred.timestamp = millis();

    if(!_init){
        DEBUG_PRINTLN(F("[MLInference] Module not initialized"));
        return false;
    }
    if(!_feature_buffer.ready){
        DEBUG_PRINTLN(F("[MLInference] Feature buffer not ready"));
        return false;
    }

    if(!_model_available[_active_model]){
        DEBUG_PRINTLN(F("[MLInference] Selected model not available"));
        return false;
    }

    uint32_t start_time = micros();

    switch(_active_model){
        case ML_MODEL_EDGE_IMPULSE: {
#if EI_CLASSIFIER
            signal_t signal;
            signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;

            static float* featurePointer = nullptr;
            featurePointer = _feature_buffer.features;

            signal.get_data = [](size_t offset, size_t length, float* out_ptr) -> 
            int {
                memcpy(out_ptr, featurePointer + offset, length * sizeof(float));
                return 0;
            };

            ei_impulse_result_t ei_result = {0};
            EI_IMPULSE_ERROR res = run_classifier(&signal, &ei_result, false);

            pred.inferenceTimeMs = (micros() - start_time) / 1000;

            if(res != EI_IMPULSE_OK){
                DEBUG_PRINTLN(F("[MLInference] Inference failed"));
                return false;
            }

            float max_confidence = 0.0f;
            uint8_t max_idx = 0;

            for(size_t i=0; i< EI_CLASSIFIER_LABEL_COUNT; i++){
                pred.classConfidences[i] = ei_result.classification[i].value;
                DEBUG_PRINTF("[MLInference] %s: %.2f%%\n",
                    ei_result.classification[i].label,ei_result.classification[i].value * 100.0f);

                if(ei_result.classification[i].value > max_confidence){
                    max_confidence = ei_result.classification[i].value;
                    max_idx = i;
                }
            }

            pred.predictedClass = getClassFromName(ei_result.classification[max_idx].label);
            pred.confidence = max_confidence;

#if EI_CLASSIFIER_HAS_ANOMALY
            pred.anomalyScore = ei_result.anomaly;
            pred.isAnomalous = (ei_result.anomaly > _anomaly_threshold);
#endif
            pred.valid = true;
#else
            DEBUG_PRINTLN(F("[MLInference] Edge Impulse model not available"));
            pred.valid = false;
#endif
            break;
        }

        case ML_MODEL_DECISION_TREE: {
            uint8_t cls = dt_predict(_feature_buffer.features);
            pred.inferenceTimeMs = (micros() - start_time) / 1000;
            pred.predictedClass = (scent_class_t)cls;
            pred.confidence = 1.0f;
            memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
            if(pred.predictedClass < ML_CLASS_COUNT){
                pred.classConfidences[pred.predictedClass] = 1.0f;
            }
            pred.isAnomalous = false;
            pred.anomalyScore = 0.0f;
            pred.valid = true;
            break;
        }

        case ML_MODEL_KNN: {
            float confidence = 0.0f;
            uint8_t cls = knn_predict_with_confidence(_feature_buffer.features, &confidence);
            pred.inferenceTimeMs = (micros() - start_time) / 1000;
            pred.predictedClass = (scent_class_t)cls;
            pred.confidence = confidence;
            memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
            if(pred.predictedClass < ML_CLASS_COUNT){
                pred.classConfidences[pred.predictedClass] = confidence;
            }
            pred.isAnomalous = false;
            pred.anomalyScore = 0.0f;
            pred.valid = true;
            break;
        }

        case ML_MODEL_RANDOM_FOREST: {
            float confidence = 0.0f;
            uint8_t cls = rf_predict_with_confidence(_feature_buffer.features, &confidence);
            pred.inferenceTimeMs = (micros() - start_time) / 1000;
            pred.predictedClass = (scent_class_t)cls;
            pred.confidence = confidence;
            memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
            if(pred.predictedClass < ML_CLASS_COUNT){
                pred.classConfidences[pred.predictedClass] = confidence;
            }
            pred.isAnomalous = false;
            pred.anomalyScore = 0.0f;
            pred.valid = true;
            break;
        }

        default:
            pred.valid = false;
            DEBUG_PRINTLN(F("[MLInference] Unknown model selection"));
            break;
    }

    if(pred.valid){
        _total_inferences++;
        _total_inference_time_ms += pred.inferenceTimeMs;
    }
    return pred.valid;
}

bool MLInference::runInferenceOnData(const dual_sensor_data_t &sensorData, ml_prediction_t &pred){
    if(!addToWindow(sensorData)){
        DEBUG_PRINTLN(F("[MLInference] Feature buffer not ready after adding data"));
        pred.valid = false;
        return false;

    }

    return runInference(pred);
}

//===========================================================================================================
//DATA collection
//===========================================================================================================

void MLInference::setDataCollectionMode(bool enabled){
    _data_collection_mode = enabled;

    if(enabled && _training_samples ==nullptr){
        _training_samples = new ml_training_sample_t[MAX_TRAINING_SAMPLES];
        _training_sample_count = 0;
        DEBUG_PRINTLN(F("[MLInference] Data collection mode enabled"));
    }

    if(!enabled){
        DEBUG_PRINTLN(F("[MLInference] Data collection mode disabled"));
    }
}

bool MLInference::isDataCollectionMode() const{
    return _data_collection_mode;
}

void MLInference::setCurrentLabel(scent_class_t label){
    _current_label = label;
    DEBUG_VERBOSE_PRINTF("[MLInference] Current data collection label set to %d\n", label); 
}

bool MLInference::collectSample(const dual_sensor_data_t &data){
    if(!_data_collection_mode || _training_samples == nullptr){
        DEBUG_PRINTLN(F("[MLInference] Data collection mode not enabled"));
        return false;
    }

    if(_training_sample_count >= MAX_TRAINING_SAMPLES){
        DEBUG_PRINTLN(F("[MLInference] Maximum training samples reached"));
        return false;
    }

    if(!extractFeatures(data)){
        DEBUG_PRINTLN(F("[MLInference] Feature extraction failed, sample not collected"));
        return false;
    }

    //store
    ml_training_sample_t &sample = _training_samples[_training_sample_count];
    memcpy(sample.features, _feature_buffer.features, sizeof(sample.features));
    sample.label = _current_label;
    sample.timestamp = millis();

    _training_sample_count++;
    DEBUG_VERBOSE_PRINTF("[MLInference] Collected training sample %d with label %d\n",_training_sample_count, _current_label);
    return true;
}

uint32_t MLInference::getCollectedSampleCount() const{
    return _training_sample_count;
}

void MLInference::clearCollectedSamples(){
    if(_training_samples != nullptr){
        delete[] _training_samples;
        _training_samples = nullptr;
        _training_sample_count =0;
        DEBUG_PRINTLN(F("[MLInference] Cleared collected training samples"));
    }
}


bool MLInference::exportTrainingData(const char* file){
    if(_training_sample_count ==0){
        DEBUG_PRINTLN(F("[MLInference] No training samples to export"));
        return false;
    }

    if(!LittleFS.begin()){
        DEBUG_PRINTLN(F("[MLInference] LittleFS mount failed"));
        return false;
    }

    File f = LittleFS.open(file, "w");
    if(!f){
        DEBUG_PRINTLN(F("[MLInference] Failed to open training data file for writing"));
        LittleFS.end();
        return false;
    }

    //write header
    f.printf("timestamp,label");
    for(uint16_t i=0; i<TOTAL_ML_FEATURES;i++){
        f.printf(",feature_%d", i);
    }
    f.println();

    //write samples
    for(uint16_t s =0; s< _training_sample_count; s++){
        f.printf("%lu,%d", _training_samples[s].timestamp, _training_samples[s].label);
        for(uint16_t i = 0; i<TOTAL_ML_FEATURES; i++){
            f.printf(",%f", _training_samples[s].features[i]);
        }
        f.println();
    }
    f.close();
    LittleFS.end();
    DEBUG_VERBOSE_PRINTF("[MLInference] Exported %d training samples to %s\n", _training_sample_count, file);
    return true;
}


//===========================================================================================================
//thresholds
//===========================================================================================================

void MLInference::setConfidenceThreshold(float threshold){
    _confidence_threshold = CONSTRAIN_FLOAT(threshold, 0.0f, 1.0f);
}

void MLInference::setAnomalyThreshold(float threshold){
    _anomaly_threshold = CONSTRAIN_FLOAT(threshold, 0.0f, 1.0f);
}

float MLInference::getConfidenceThreshold() const{
    return _confidence_threshold;
}

float MLInference::getAnomalyThreshold() const{
    return _anomaly_threshold;
}



//===========================================================================================================
//util
//===========================================================================================================


void MLInference::printModelInfo(){
    DEBUG_PRINTLN(F("[MLInference] ML Model Information:"));
#if EI_CLASSIFIER
    DEBUG_PRINTF("Model: Edge Impulse\n");
    DEBUG_PRINTF("Input features: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    DEBUG_PRINTF("Output classes: %d\n", EI_CLASSIFIER_LABEL_COUNT);
    DEBUG_PRINTF("Anomaly detection: %s\n", EI_CLASSIFIER_HAS_ANOMALY ? "Yes" : "No");
#else
    DEBUG_PRINTLN(F("Model: None (Edge Impulse SDK not available)"));
#endif
    DEBUG_PRINTF("Active model: %s\n", getActiveModelName());
    DEBUG_PRINTF("Confidence threshold: %.2f\n", _confidence_threshold);
    DEBUG_PRINTF("Anomaly threshold: %.2f\n", _anomaly_threshold);
    DEBUG_PRINTF("Total inferences: %lu\n", _total_inferences);
}

void MLInference::printPrediction(const ml_prediction_t &result){
    DEBUG_PRINTLN(F("=== Prediction Result ==="));
    DEBUG_PRINTF("Model: %s\n", getActiveModelName());
    DEBUG_PRINTF("Class: %s\n", getClassName(result.predictedClass));
    DEBUG_PRINTF("Confidence: %.2f%%\n", result.confidence * 100);
    DEBUG_PRINTF("Valid: %s\n", result.valid ? "Yes" : "No");
    DEBUG_PRINTF("Anomaly: %s (score: %.2f)\n", result.isAnomalous ? "Yes" : "No", result.anomalyScore);
    DEBUG_PRINTF("Inference time: %lu us\n", result.inferenceTimeMs);
    
    DEBUG_PRINTLN(F("All class confidences:"));
    #if EI_CLASSIFIER
    const int labelCount = EI_CLASSIFIER_LABEL_COUNT;
    #else
    const int labelCount = SCENT_CLASS_COUNT;
    #endif
    for (int i = 0; i < labelCount; i++) {
        const char* name = (i < SCENT_CLASS_COUNT) ? SCENT_CLASS_NAMES[i] : "Unknown";
        DEBUG_PRINTF("  %s: %.2f%%\n", name, result.classConfidences[i] * 100);
    }
}

String MLInference::getPredictionJSON(const ml_prediction_t &result){
    String json = "{";
    json += "\"timestamp\":" + String(result.timestamp) + ",";
    json += "\"model\":\"" + String(getActiveModelName()) + "\",";
    json += "\"class\":\"" + String(getClassName(result.predictedClass)) + "\",";
    json += "\"class_id\":" + String(result.predictedClass) + ",";
    json += "\"confidence\":" + String(result.confidence, 4) + ",";
    json += "\"valid\":" + String(result.valid ? "true" : "false") + ",";
    json += "\"anomaly\":" + String(result.isAnomalous ? "true" : "false") + ",";
    json += "\"anomaly_score\":" + String(result.anomalyScore, 4) + ",";
    json += "\"inference_us\":" + String(result.inferenceTimeMs) + ",";
    json += "\"confidences\":[";
    #if EI_CLASSIFIER
    const int labelCount = EI_CLASSIFIER_LABEL_COUNT;
    #else
    const int labelCount = SCENT_CLASS_COUNT;
    #endif
    for (int i = 0; i < labelCount; i++) {
        json += String(result.classConfidences[i], 4);
        if (i < labelCount - 1) json += ",";
    }
    json += "]}";
    return json;
}

//===========================================================================================================
//stats
//===========================================================================================================

uint32_t MLInference::getTotalInferences() const{
    return _total_inferences;
}

float MLInference::getAverageInferenceTimeMs() const{
    if(_total_inferences ==0){
        return 0.0f;
    }
    return (float)_total_inference_time_ms / _total_inferences;
}

void MLInference::resetStats(){
    _total_inferences =0;
    _total_inference_time_ms =0;
}

