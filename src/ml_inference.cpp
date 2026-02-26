#include "ml_inference.h"
#include <LittleFS.h>
#include <ctype.h>
#include "model headers/feature_select.h"

#if EI_CLASSIFIER
#include "iScent_inferencing.h"
#endif

#include "model headers/dt_model_header.h"
#include "model headers/knn_model_header.h"
#include "model headers/rf_model_header.h"
#include "ensemble_weights.h"

#if SELECTED_FEATURE_COUNT != TOTAL_ML_FEATURES
  #error "TOTAL_ML_FEATURES in config.h must match SELECTED_FEATURE_COUNT from feature_select.h"
#endif

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
    _training_sample_count(0),
    _temporal_count(0),
    _inference_mode(INFERENCE_MODE_SINGLE)
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
    memset(_temporal_scores, 0, sizeof(_temporal_scores));
    memset(&_temporal_state,0,sizeof(_temporal_state));
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

bool MLInference::extractFeatures(const dual_sensor_data_t &sensor_data) {
    memset(_feature_buffer.features, 0,sizeof(_feature_buffer.features));

    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS;i++){
        _feature_buffer.features[i] = sensor_data.primary.gas_resistances[i];
    }
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS;i++){
        _feature_buffer.features[BME688_NUM_HEATER_STEPS + i] = sensor_data.secondary.gas_resistances[i];
    }

    _feature_buffer.featureCount=BME688_NUM_HEATER_STEPS * 2;
    normaliseFeatures(_feature_buffer);

    _feature_buffer.ready = true;
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
    if(name == nullptr){
        return SCENT_CLASS_UNKNOWN;
    }

    auto normalizeSimple = [](const char* src, char* dst, size_t dstSize){
        size_t len = strlen(src);
        size_t start = 0;
        while(start < len && isspace((unsigned char)src[start])){
            start++;
        }

        size_t end = len;
        while(end > start && isspace((unsigned char)src[end - 1])){
            end--;
        }

        size_t out = 0;
        for(size_t i = start;i < end&&out +1<dstSize; i++){
            dst[out++] = (char)tolower((unsigned char)src[i]);
        }

        dst[out] = '\0';
    };

    char target[64];
    normalizeSimple(name, target, sizeof(target));

    //strip positional suffix
    {
        size_t tlen = strlen(target);
        //scan backwards for pos
        for(size_t p = 0; p + 3 < tlen; p++){
            if(target[p] == 'p' && target[p+1] == 'o' && target[p+2] == 's'){
                bool allDigits = true;
                bool hasDigits = (p + 3 < tlen);
                for(size_t d = p + 3; d < tlen; d++){
                    if(!isdigit((unsigned char)target[d])){ allDigits = false; break; }
                }
                if(allDigits && hasDigits){
                    //trim back to before pos
                    size_t cut = p;
                    while(cut > 0 && (target[cut-1] == ' ' || target[cut-1] == '_')){
                        cut--;
                    }
                    target[cut] = '\0';
                    break;
                }
            }
        }
    }

    {
        bool hasDecaf = (strstr(target, "decaf") != nullptr);
        bool hasTea = (strstr(target, "tea") != nullptr);
        bool hasCoffee = (strstr(target, "coffee") != nullptr);
        
        if (hasDecaf && hasTea) {
            return SCENT_CLASS_DECAF_TEA;
        }
        if (hasDecaf && hasCoffee) {
            return SCENT_CLASS_DECAF_COFFEE;
        }
        if (hasTea && !hasDecaf) {
            return SCENT_CLASS_TEA;
        }
        if (hasCoffee && !hasDecaf) {
            return SCENT_CLASS_COFFEE;
        }
    }

    for(int i = 0; i < SCENT_CLASS_COUNT; i++){
        char candidate[64];
        normalizeSimple(SCENT_CLASS_NAMES[i], candidate, sizeof(candidate));
        if(strcmp(target, candidate) == 0){
            return static_cast<scent_class_t>(i);
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
            float confidence = 1.0f;
#ifdef DT_HAS_CONFIDENCE
            uint8_t cls = dt_predict_with_confidence(_feature_buffer.features, &confidence);
#else
            uint8_t cls = dt_predict(_feature_buffer.features);
#endif
            pred.inferenceTimeMs = (micros() - start_time) / 1000;
            pred.predictedClass = (scent_class_t)cls;
            pred.confidence = confidence;
            memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
            if(pred.predictedClass < ML_CLASS_COUNT){
                pred.classConfidences[pred.predictedClass] = confidence;
            }
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - pred.confidence;
            } else {
                pred.isAnomalous = false;
                pred.anomalyScore = 0.0f;
            }
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
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - pred.confidence;
            } else {
                pred.isAnomalous = false;
                pred.anomalyScore = 0.0f;
            }
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
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - pred.confidence;
            } else {
                pred.isAnomalous = false;
                pred.anomalyScore = 0.0f;
            }
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

    File f = LittleFS.open(file, "w");
    if(!f){
        DEBUG_PRINTLN(F("[MLInference] Failed to open training data file for writing"));
        return false;
    }

    //write header
    f.printf("timestamp,label");
    for(int i=0; i<TOTAL_ML_FEATURES;i++){
        f.printf(",feature_%d", i);
    }
    f.println();

    //write samples
    for(uint16_t s =0; s< _training_sample_count; s++){
        f.printf("%lu,%d", _training_samples[s].timestamp, _training_samples[s].label);
        for(int i = 0; i<TOTAL_ML_FEATURES; i++){
            f.printf(",%f", _training_samples[s].features[i]);
        }
        f.println();
    }
    f.close();
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


//===========================================================================================================
//inference mode management
//===========================================================================================================
void MLInference::setInferenceMode(inference_mode_t mode){
    if(mode>= INFERENCE_MODE_COUNT){
        mode = INFERENCE_MODE_SINGLE;
    }
    _inference_mode = mode;

    DEBUG_PRINTF("[MLInference] Inference mode set to %d\n", mode);
}

inference_mode_t MLInference::getInferenceMode() const{
    return _inference_mode;
}

const char* MLInference::getInferenceModeName()const{
    return INFERENCE_MODE_NAMES[_inference_mode];
}

void MLInference::cycleInferenceMode(){
    _inference_mode = (inference_mode_t)((_inference_mode + 1) % INFERENCE_MODE_COUNT);
    DEBUG_PRINTF("[MLInference] Inference mode changed to %s\n", getInferenceModeName());
}

//===========================================================================================================
//inferenceing
//===========================================================================================================

bool MLInference::runActiveInference(ml_prediction_t &pred){
    switch(_inference_mode){
        case INFERENCE_MODE_SINGLE:
            return runInference(pred);

        case INFERENCE_MODE_ENSEMBLE:{
            ml_ensemble_prediction_t ensP;
            if(!runEnsembleInference(ensP)){
                pred.valid = false;
                return false;
            }
            ensembleToPrediction(ensP, pred);

            //apply conf
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass=SCENT_CLASS_UNKNOWN;
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - pred.confidence;
            }
            return true;
        }

        case INFERENCE_MODE_TEMPORAL:{
            ml_ensemble_prediction_t ensPred;
            if(!runEnsembleInference(ensPred)){
                pred.valid = false;
                return false;
            }
            ensembleToPrediction(ensPred, pred);
            return true;

        }
            
        default:
            pred.valid = false;
            return false;
    }
}


//===========================================================================================================
//ens to pred
//===========================================================================================================
void MLInference::ensembleToPrediction(const ml_ensemble_prediction_t &ens, ml_prediction_t &pred){
    memset(&pred,0,sizeof(pred));
    pred.timestamp = ens.timestamp;
    pred.inferenceTimeMs = ens.inferenceTimeMs;
    pred.valid = ens.valid;
    pred.confidence = ens.confidence;
    pred.predictedClass = ens.predictedClass;
    pred.isAnomalous = (ens.confidence < ML_CONFIDENCE_THRESHOLD);
    pred.anomalyScore= pred.isAnomalous ?(1.0f-ens.confidence) :0.0f;


    //normalise scores to confidences
    float total = 0.0f;
    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        total+= ens.classScores[i];
    }

    for(int i=0; i<ML_CLASS_COUNT; i++){
        if(i<SCENT_CLASS_COUNT&&total>0.0f){
            pred.classConfidences[i] =ens.classScores[i]/total;
        }
        else{
            pred.classConfidences[i] = 0.0f;
        }

    }
}
//===========================================================================================================
//temporal
//===========================================================================================================
void MLInference::startTemporalCollection(uint8_t samples){
    resetTemporalBuffer();

    _temporal_state.active = true;
    _temporal_state.targetSamples= (samples > 0 )?samples:TEMPORAL_BUFFER_SIZE;
    _temporal_state.collectedSamples = 0;
    _temporal_state.startTimeMs=millis();
    _temporal_state.failedAttempts=0;
    _temporal_state.maxFailedAttempts=_temporal_state.targetSamples*3;
    _temporal_state.lastUpdateTimeMs=0;
    _temporal_state.timeoutMs=TEMPORAL_TIMEOUT_MS;

    DEBUG_PRINTF("[MLInference] Started temporal collection for %d samples\n", _temporal_state.targetSamples);
}

bool MLInference::updateTemporalCollection(ml_prediction_t &pred){
    if(!_temporal_state.active){
        DEBUG_PRINTLN(F("[MLInference] Temporal collection not active"));
        return false;
    }
    uint32_t time=millis();

    if(time- _temporal_state.startTimeMs> _temporal_state.timeoutMs){
        DEBUG_PRINTLN(F("[MLInference] Temporal collection timed out"));
        _temporal_state.active = false;
        finaliseTemporalPrediction(pred);
        return true;
    }

    //check for max faulure
    if(_temporal_state.failedAttempts>= _temporal_state.maxFailedAttempts){
        DEBUG_PRINTLN(F("[MLInference] Temporal collection stopped due to too many failed attempts"));
        _temporal_state.active = false;
        finaliseTemporalPrediction(pred);
        return true;
    }

    //rate limit
    if((time-_temporal_state.lastUpdateTimeMs)<TEMPOERAL_COLLECTION_INTERVAL_MS){
        return false;
    }

    //run inference
    ml_ensemble_prediction_t ePred;
    if(!runEnsembleInference(ePred)){
        _temporal_state.failedAttempts++;
        _temporal_state.lastUpdateTimeMs = time;
        return false;
    }

    //ad dto buffer
    if(addToTemporalEnsemble(ePred)){
        _temporal_state.active = false;
        finaliseTemporalPrediction(pred);

        DEBUG_PRINTLN(F("[MLInference] Temporal collection complete"));
        return true;
    }

    _temporal_state.collectedSamples = _temporal_count;
    _temporal_state.lastUpdateTimeMs = time;

    //update running best pred
    float tempConf=0.0f;
    scent_class_t tempClass = getTemporalPrediction(tempConf);
    pred.predictedClass=tempClass;
    pred.confidence = tempConf;
    pred.valid = (_temporal_count>0);
    pred.timestamp=time;

    return false;//not complete yet
}

bool MLInference::isTemporalCollectionActive()const{
    return _temporal_state.active;
}

bool MLInference::isTemporalCollectionComplete()const{
    return (!_temporal_state.active&&(_temporal_count >0));
}

float MLInference::getTemporalCollectionProgress()const{
    if(!_temporal_state.active &&_temporal_count==0){
        return 0.0f;
    }
    if(_temporal_state.targetSamples==0){
        return 1.0f;
    }

    return (float)_temporal_count / (float)_temporal_state.targetSamples;
}

void MLInference::cancelTemporalCollection(){
    _temporal_state.active=false;
    DEBUG_PRINTLN(F("[MLInference] Temporal collection cancelled"));
    resetTemporalBuffer();
}


void MLInference::finaliseTemporalPrediction(ml_prediction_t &finalPred){
    memset(&finalPred, 0,sizeof(finalPred));

    finalPred.timestamp=millis();

    float conf=0.0f;
    scent_class_t sClass = getTemporalPrediction(conf);
    finalPred.predictedClass=sClass;
    finalPred.confidence=conf;
    finalPred.valid = (_temporal_count >0);
    finalPred.isAnomalous=(sClass==SCENT_CLASS_UNKNOWN);
    finalPred.anomalyScore = finalPred.isAnomalous ? (1.0f - conf) : 0.0f;

    DEBUG_PRINTF("[MLInference] Final temporal prediction: %s (confidence: %.2f%%)\n",getClassName(sClass), conf * 100.0f);

}

//===========================================================================================================
//temporal buffer
//===========================================================================================================
void MLInference::resetTemporalBuffer(){
    memset(_temporal_scores,0,sizeof(_temporal_scores));
    _temporal_count = 0;
}

bool MLInference::addToTemporalEnsemble(const ml_ensemble_prediction_t &pred){
    if(!pred.valid){
        return false;
    }

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        _temporal_scores[i] += pred.classScores[i];
    }
    _temporal_count++;

    DEBUG_PRINTF("[MLInference] Added to temporal ensemble, count: %d\n", _temporal_count);

    return (_temporal_count >= TEMPORAL_BUFFER_SIZE);
}

scent_class_t MLInference::getTemporalPrediction(float &conf)const{
    if(_temporal_count == 0){
        conf=0;
        return SCENT_CLASS_UNKNOWN;
    }

    float bScore=0.0f;
    uint8_t bIdx=0;
    float total=0.0f;

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        total += _temporal_scores[i];
        if(_temporal_scores[i] > bScore){
            bScore = _temporal_scores[i];
            bIdx = i;
        }
    }
    conf = (total>0) ? (bScore / total) : 0.0f;

    //check 2nd
    float second = 0.0f;
    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        if(i != bIdx && _temporal_scores[i] > second){
            second = _temporal_scores[i];
        }
    }

    float margin = bScore - second;
    if(margin < 0.05f){ //margin lower than threshold
        return SCENT_CLASS_UNKNOWN;
    }

    return (scent_class_t)bIdx;
}

//===========================================================================================================
//ensemble inference
//===========================================================================================================
bool MLInference::runEnsembleInference(ml_ensemble_prediction_t &pred){
    memset(&pred,0, sizeof(pred));
    pred.timestamp = millis();
    if(!_init || !_feature_buffer.ready){
        DEBUG_PRINTLN(F("[MLInference] Module not ready"));
        pred.valid=false;
        return false;
    }

    uint32_t start = micros();

    //run all available models
    float dtConf = 1.0f, knnConf = 1.0f, rfConf = 1.0f;
    //dt
#ifdef DT_HAS_CONFIDENCE
    uint8_t dtCls = dt_predict_with_confidence(_feature_buffer.features, &dtConf);
#else
    uint8_t dtCls = dt_predict(_feature_buffer.features);
#endif

    //knn
    uint8_t knnCls = knn_predict_with_confidence(_feature_buffer.features, &knnConf);

    //rf
    uint8_t rfCls = rf_predict_with_confidence(_feature_buffer.features, &rfConf);

    pred.inferenceTimeMs = (micros() - start) / 1000;

    //store results
    pred.dtClass = (scent_class_t)dtCls;
    pred.dtConf = dtConf;
    pred.knnClass = (scent_class_t)knnCls;
    pred.knnConf = knnConf;
    pred.rfClass = (scent_class_t)rfCls;
    pred.rfConf = rfConf;

    //
    memset(pred.classScores, 0 ,sizeof(pred.classScores));


    if(dtCls<SCENT_CLASS_COUNT){
        pred.classScores[dtCls ]+= dtConf * DT_WEIGHT;
    }
    if(knnCls<SCENT_CLASS_COUNT){
        pred.classScores[knnCls] += knnConf * KNN_WEIGHT;
    }
    if(rfCls<SCENT_CLASS_COUNT){
        pred.classScores[rfCls]+= rfConf * RF_WEIGHT;
    }

    //get best
    float bScore = 0.0f;
    uint8_t bClass= 0;
    float total=0.0f;

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        total+= pred.classScores[i];
        if(pred.classScores[i]>bScore){
            bScore= pred.classScores[i];
            bClass=i;

        }
    }
    pred.predictedClass=(scent_class_t)bClass;
    pred.confidence = (total>0) ? (bScore / total) : 0.0f;
    pred.valid=true;

    _total_inferences++;
    _total_inference_time_ms+=pred.inferenceTimeMs;

    return true;
}

//===========================================================================================================
//normalisation
//===========================================================================================================

void MLInference::normaliseFeatures(ml_feature_buffer_t& features) {
    const uint8_t steps = BME688_NUM_HEATER_STEPS;

    //read raw gas resistances
    float _gas1[steps], _gas2[steps];
    for(uint8_t i=0; i<steps;i++){
        _gas1[i] = features.features[i];
        _gas2[i] = features.features[i+steps];
    }

    //self normalise
    float _gas1_mean =computeMean(_gas1,steps);
    float _gas2_mean = computeMean(_gas2, steps);
    if(_gas1_mean <1.0f){
        _gas1_mean = 1.0f;
    }
    if(_gas2_mean < 1.0f){
        _gas2_mean = 1.0f;
    }

    float _gas1_norm[steps],_gas2_norm[steps];
    for(uint8_t i=0;i<steps;i++){
        _gas1_norm[i]=_gas1[i]/_gas1_mean;
        _gas2_norm[i]=_gas2[i]/_gas2_mean;
    }

    //NOW FOR ALL THE FEATURES :PPP

    float featureArray[ORIGINAL_FEATURE_COUNT];
    uint16_t idx=0;

    //0-9
    for(uint8_t i=0; i<steps; i++){
        featureArray[idx++] = _gas1_norm[i];
    }

    //10-19
    for(uint8_t i=0; i<steps; i++){
        featureArray[idx++] = _gas2_norm[i];
    }

    //20-29
    float cross[steps];
    for(uint8_t i=0;i<steps;i++){
        cross[i]= (_gas2_norm[i]>0)?(_gas1_norm[i]/_gas2_norm[i]):1.0f;
        featureArray[idx++] = cross[i];
    }

    //30-39
    for(uint8_t i=0; i<steps;i++){
        featureArray[idx++]= _gas1_norm[i] - _gas2_norm[i];
    }

    //40-48
    for(uint8_t i=0; i<steps-1;i++){
        featureArray[idx++]= _gas1_norm[i+1] - _gas1_norm[i];
    }

    //49-57
    for(uint8_t i=0; i<steps-1;i++){
        featureArray[idx++]= _gas2_norm[i+1] - _gas2_norm[i];
    }

    //58
    featureArray[idx++]=computeSlope(_gas1_norm, steps);
    //59
    featureArray[idx++]=computeSlope(_gas2_norm, steps);

    //60
    {
        uint8_t h=steps/2;
        float steps_low= computeSlope(_gas1_norm, h);
        float steps_high = computeSlope(_gas1_norm + h, steps - h);
        featureArray[idx++] = steps_high - steps_low;
    }

    //61
    {
        uint8_t h=steps/2;
        float steps_low= computeSlope(_gas2_norm, h);
        float steps_high = computeSlope(_gas2_norm + h, steps - h);
        featureArray[idx++] = steps_high - steps_low;
    }

    //62
    {
        float auc=0.0f;
        for(uint8_t i=0; i<steps-1;i++){
            auc+=(_gas1_norm[i]+_gas1_norm[i+1])/2.0f;
        }
        featureArray[idx++] = auc;
    }

    //63
    {
        float auc=0.0f;
        for(uint8_t i=0; i<steps-1;i++){
            auc+=(_gas2_norm[i]+_gas2_norm[i+1])/2.0f;
        }
        featureArray[idx++] = auc;
    }

    //64
    {
        uint8_t peakIdx=0;
        float peakVal=_gas1_norm[0];
        for(uint8_t i=1;i<steps;i++){
            if(_gas1_norm[i]>peakVal){
                peakVal=_gas1_norm[i];
                peakIdx=i;
            }
        }
        featureArray[idx++]=(float)peakIdx;
    }

    //65
    {
        uint8_t peakIdx=0;
        float peakVal=_gas2_norm[0];
        for(uint8_t i=1;i<steps;i++){
            if(_gas2_norm[i]>peakVal){
                peakVal=_gas2_norm[i];
                peakIdx=i;
            }
        }
        featureArray[idx++]=(float)peakIdx;
    }

    //66
    featureArray[idx++]=computeMax(_gas1_norm, steps) - computeMin(_gas1_norm, steps);
    //SIXSEVENNNN
    featureArray[idx++]=computeMax(_gas2_norm, steps) - computeMin(_gas2_norm, steps);

    //68
    {
        uint8_t h=steps/2;
        float early=computeMean(_gas1_norm, h);
        float late=computeMean(_gas1_norm + h, steps - h);
        featureArray[idx++]= (early > 0.01f) ? (late / early) : 1.0f;
    }
    //69
    {
        uint8_t h=steps/2;
        float early=computeMean(_gas2_norm, h);
        float late=computeMean(_gas2_norm + h, steps - h);
        featureArray[idx++]= (early > 0.01f) ? (late / early) : 1.0f;
    }

    //70
    featureArray[idx++]=computeMean(cross,steps);

    //71
    featureArray[idx++]=computeSlope(cross, steps);

    //72
    {
        float crossMean = computeMean(cross, steps);
        float var = 0.0f;
        for(uint8_t i=0; i<steps; i++){
            float diff=cross[i]-crossMean;
            var+= diff*diff;
        }
        featureArray[idx++]=var/steps;
    }

    //check!!
    if(idx!=ORIGINAL_FEATURE_COUNT){
        DEBUG_PRINTF("[MLInference] Feature extraction error: expected %d features but got %d\n", ORIGINAL_FEATURE_COUNT, idx);
    }

    //zscore
    for(int i=0;i<SELECTED_FEATURE_COUNT;i++){
        int idx=SELECTED_INDICES[i];
        float val=featureArray[idx];

        if(SELECTED_STDS[i]>1e-6f){
            val=(val-SELECTED_MEANS[i])/SELECTED_STDS[i];
        }
        features.features[i]=val;
    }
    features.featureCount=SELECTED_FEATURE_COUNT;
    features.ready=true;
}

void MLInference::printFeatureDebug() const {
    DEBUG_PRINTLN(F("\n=== Feature Debug ==="));
    DEBUG_PRINTF("Pipeline: %d raw gas values -> %d full features -> %d selected (z-scored)\n",
                 BME688_NUM_HEATER_STEPS * 2, ORIGINAL_FEATURE_COUNT, SELECTED_FEATURE_COUNT);
    DEBUG_PRINTF("Feature count in buffer: %d (expected: %d)\n",
                 _feature_buffer.featureCount, SELECTED_FEATURE_COUNT);

    // Feature name lookup using selected indices
    static const char* FULL_FEATURE_SECTIONS[] = {
        //0-9
        "g1n", "g1n", "g1n", "g1n", "g1n", "g1n", "g1n", "g1n", "g1n", "g1n",
        //10-19
        "g2n", "g2n", "g2n", "g2n", "g2n", "g2n", "g2n", "g2n", "g2n", "g2n",
        //20-20
        "cross", "cross", "cross", "cross", "cross", "cross", "cross", "cross", "cross", "cross",
        //30-39
        "diff", "diff", "diff", "diff", "diff", "diff", "diff", "diff", "diff", "diff",
        //40-48
        "g1d", "g1d", "g1d", "g1d", "g1d", "g1d", "g1d", "g1d", "g1d",
        //49-57
        "g2d", "g2d", "g2d", "g2d", "g2d", "g2d", "g2d", "g2d", "g2d",
        //58-72
        "slope1", "slope2", "curv1", "curv2","auc1", "auc2", "peak1", "peak2","range1", "range2", "le_r1", "le_r2","cr_mean", "cr_slope", "cr_var"
    };

    for(int i = 0; i < _feature_buffer.featureCount && i < SELECTED_FEATURE_COUNT; i++){
        int oIdx = SELECTED_INDICES[i];
        const char* section = (oIdx < ORIGINAL_FEATURE_COUNT)?FULL_FEATURE_SECTIONS[oIdx] : "???";

        DEBUG_PRINTF("[%2d] orig[%2d] %-8s = %8.4f",i, oIdx,section, _feature_buffer.features[i]);

        if(fabsf(_feature_buffer.features[i]) > 3.0f){
            DEBUG_PRINTF("!! %.1f sigma", _feature_buffer.features[i]);
        }
        DEBUG_PRINTLN();
    }
}