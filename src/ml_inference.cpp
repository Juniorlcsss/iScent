#include "ml_inference.h"
#include <LittleFS.h>
#include <ctype.h>
#include "model headers/feature_select.h"
#include "model headers/feature_stats.h"
#include "model headers/fisher_weights.h"
#include "data_logger.h"

extern DataLogger logger;

#if EI_CLASSIFIER
#include "iScent_inferencing.h"
#endif

//===========================================================================================================
//model enablers for section 6.3.1 
//===========================================================================================================

#ifndef ISC_ENABLE_EI
    #define ISC_ENABLE_EI EI_CLASSIFIER
#endif

#ifndef ISC_ENABLE_DT
    #define ISC_ENABLE_DT 1
#endif

#ifndef ISC_ENABLE_KNN
    #define ISC_ENABLE_KNN 1
#endif

#ifndef ISC_ENABLE_RF
    #define ISC_ENABLE_RF 1
#endif

#if ISC_ENABLE_DT
#include "model headers/dt_model_header.h"
#endif

#if ISC_ENABLE_KNN
#include "model headers/knn_model_header.h"
#endif

#if ISC_ENABLE_RF
#include "model headers/rf_model_header.h"
#endif

#include "ensemble_weights.h"
#include "model headers/anomaly_threshold.h"




static const float EFFECTIVE_ANOMALY_THRESHOLD =0.95f;

extern BME688Handler sensors;

#if SELECTED_FEATURE_COUNT != TOTAL_ML_FEATURES
  #error "TOTAL_ML_FEATURES in config.h must match SELECTED_FEATURE_COUNT from feature_select.h"
#endif

#if FULL_FEATURE_COUNT < (FISHER_BASE_FEATURE_COUNT + FISHER_PAIR_COUNT)
    #error "feature_stats.h FULL_FEATURE_COUNT is inconsistent with fisher_weights.h"
#endif

static inline bool isLikelyFisherIndex(int originalIdx) {
    // Current trainer appends Fisher projections as the final 4 features.
    return (originalIdx >= 153 && originalIdx <= 156);
}

//===========================================================================================================
//constructer
//===========================================================================================================
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
    memset(&_raw_snapshot, 0, sizeof(_raw_snapshot));
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
#if ISC_ENABLE_KNN
    const float buildKnnScale = KNN_DISTANCE_SCALE;
    const int buildKnnK = KNN_K;
#else
    const float buildKnnScale = 0.0f;
    const int buildKnnK = 0;
#endif

    char buildCheck[256];
    snprintf(buildCheck, sizeof(buildCheck),
        "[BUILD] SELECTED=%d ORIGINAL=%d KNN_SCALE=%.4f KNN_K=%d "
        "CLASSES=%d Class4=%s ANOM_THRESH=%.4f",
        SELECTED_FEATURE_COUNT, ORIGINAL_FEATURE_COUNT,
        buildKnnScale, buildKnnK, SCENT_CLASS_COUNT,
        SCENT_CLASS_NAMES[4], CALIBRATED_ANOMALY_THRESHOLD);
    logger.logDebugMsg(String(buildCheck));


    DEBUG_PRINTLN(F("[MLInference] Initializing ML Inference module"));
    DEBUG_PRINTF("[MLInference] Pipeline: %d base -> engineered -> selected %d/%d (robust scaled)\n",
                 FULL_ML_FEATURES, SELECTED_FEATURE_COUNT, ORIGINAL_FEATURE_COUNT);

    _model_available[ML_MODEL_EDGE_IMPULSE] = (ISC_ENABLE_EI != 0);
    _model_available[ML_MODEL_DECISION_TREE] = (ISC_ENABLE_DT != 0);
    _model_available[ML_MODEL_KNN] = (ISC_ENABLE_KNN != 0);
    _model_available[ML_MODEL_RANDOM_FOREST] = (ISC_ENABLE_RF != 0);

#if ISC_ENABLE_EI
    DEBUG_PRINTF("[MLInference] Edge Impulse model info:\n");
    DEBUG_PRINTF("[ML] Input features: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    DEBUG_PRINTF("[ML] Label count: %d\n", EI_CLASSIFIER_LABEL_COUNT);
    DEBUG_PRINTF("[ML] Anomaly detection: %s\n",
                 EI_CLASSIFIER_HAS_ANOMALY ? "Y" : "N");

    if(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE != TOTAL_ML_FEATURES){
        DEBUG_PRINTF("[ML] WARNING: Feature size mismatch! Expected %d, model has %d\n",
                     TOTAL_ML_FEATURES, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    }
    _init = true;
#else
    DEBUG_PRINTLN(F("[MLInference] Edge Impulse SDK not found"));
    DEBUG_PRINTLN(F("[MLInference] Using DT/KNN/RF models"));
    _init = true;
#endif

    setActiveModel(_active_model);
    clearFeatureBuffer();
    return _init;
}

bool MLInference::isReady() const {
    return _init && _feature_buffer.ready;
}

//===========================================================================================================
//feature extraction
//===========================================================================================================

bool MLInference::extractFeatures(const dual_sensor_data_t &sensor_data) {
    memset(&_feature_buffer, 0, sizeof(_feature_buffer));

    _raw_snapshot.valid = false;

    if(sensor_data.primary.complete){
        _raw_snapshot.temp1 = sensor_data.primary.temperatures[0];
        _raw_snapshot.hum1  = sensor_data.primary.humidities[0];
        _raw_snapshot.pres1 = sensor_data.primary.pressures[0];

        for(uint8_t i = 0; i < BME688_NUM_HEATER_STEPS; i++){
            _raw_snapshot.gas1[i] = sensor_data.primary.gas_resistances[i];
        }
    }

    if(sensor_data.secondary.complete){
        _raw_snapshot.temp2 = sensor_data.secondary.temperatures[0];
        _raw_snapshot.hum2  = sensor_data.secondary.humidities[0];
        _raw_snapshot.pres2 = sensor_data.secondary.pressures[0];
        for(uint8_t i = 0; i < BME688_NUM_HEATER_STEPS; i++){
            _raw_snapshot.gas2[i] = sensor_data.secondary.gas_resistances[i];
        }
    }

    _raw_snapshot.valid = (sensor_data.primary.complete && sensor_data.secondary.complete);

    //run pipeline
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
    
    _feature_buffer.ready = false;
    return false;
}

void MLInference::clearFeatureBuffer(){
    memset(&_feature_buffer,0,sizeof(_feature_buffer));
    memset(&_raw_snapshot, 0, sizeof(_raw_snapshot));
    _window_index = 0;
    _window_size = 0;
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

bool MLInference::isAccumulating() const {
    return _window_size < ML_SAMPLES;
}

//===========================================================================================================
//math helpers
//===========================================================================================================

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

float MLInference::computeTrapezoidalAUC(const float* data, uint16_t size){
    float area = 0.0f;
    for(uint16_t i = 1; i < size; i++){
        area += 0.5f * (data[i] + data[i - 1]);
    }
    return area;
}

int MLInference::computePeakIndex(const float* data, uint16_t size){
    int idx = 0;
    float peak = data[0];
    for(uint16_t i = 1; i < size; i++){
        if(data[i] > peak){ peak = data[i]; idx = i; }
    }
    return idx;
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

//===========================================================================================================
//pipeline
//===========================================================================================================

int MLInference::computeBaseFeatures(const raw_sensor_snapshot_t& raw, float* f) {
    const uint8_t steps = BME688_NUM_HEATER_STEPS;
    uint16_t idx = 0;

    //ambient baselines
    sensor_calibration_t cal = sensors.getCalibrationData();

    float gas1_resp[BME688_NUM_HEATER_STEPS];
    float gas2_resp[BME688_NUM_HEATER_STEPS];

    for(uint8_t i = 0; i < steps; i++){
        float b1 = (cal.calibrated && cal.gas_baseline_primary[i] >= 1.0f) ? cal.gas_baseline_primary[i] : 1.0f;
        float b2 = (cal.calibrated && cal.gas_baseline_secondary[i] >= 1.0f) ? cal.gas_baseline_secondary[i] : 1.0f;
        gas1_resp[i] = raw.gas1[i] / b1;
        gas2_resp[i] = raw.gas2[i] / b2;
    }

    // Normalizes to step 0
    float gas1_base=(fabsf(gas1_resp[0])>1e-6f)?gas1_resp[0]:1.0f;
    float gas2_base=(fabsf(gas2_resp[0])>1e-6f)?gas2_resp[0]:1.0f;

    float gas1_norm[BME688_NUM_HEATER_STEPS];
    float gas2_norm[BME688_NUM_HEATER_STEPS];

    for(uint8_t i = 0; i < steps; i++){
        gas1_norm[i] = gas1_resp[i] / gas1_base;
        gas2_norm[i] = gas2_resp[i] / gas2_base;
    }

    //[0-9] gas1_norm
    for(uint8_t i = 0; i < steps; i++){
        f[idx++] = gas1_norm[i];
    }

    //[10-19] gas2_norm
    for(uint8_t i = 0; i < steps; i++){
        f[idx++] = gas2_norm[i];
    }

    //[20-29] cross_ratio
    float cross[BME688_NUM_HEATER_STEPS];
    for(uint8_t i = 0; i < steps; i++){
        cross[i] = (gas2_resp[i] > 0.01f) ? (gas1_resp[i] / gas2_resp[i]) : 1.0f;
        f[idx++] = cross[i];
    }

    //[30-39] diff
    for(uint8_t i = 0; i < steps; i++){
        f[idx++] = gas1_resp[i] - gas2_resp[i];
    }

    //[40-48] gas1_delta (from normalised)
    for(uint8_t i = 0; i < steps - 1; i++){
        f[idx++] = gas1_norm[i + 1] - gas1_norm[i];
    }

    //[49-57] gas2_delta (from normalised)
    for(uint8_t i = 0; i < steps - 1; i++){
        f[idx++] = gas2_norm[i + 1] - gas2_norm[i];
    }

    //[58] slope1_norm
    f[idx++] = computeSlope(gas1_norm, steps);

    //[59] slope2_norm
    f[idx++] = computeSlope(gas2_norm, steps);

    //[60] curvature1_norm: slope(second_half) - slope(first_half)
    {
        uint8_t h = steps / 2;
        f[idx++] = computeSlope(gas1_norm + h, steps - h) - computeSlope(gas1_norm, h);
    }

    //[61] curvature2_norm
    {
        uint8_t h = steps / 2;
        f[idx++] = computeSlope(gas2_norm + h, steps - h) - computeSlope(gas2_norm, h);
    }

    //[62] auc1_norm
    f[idx++] = computeTrapezoidalAUC(gas1_norm, steps);

    //[63] auc2_norm
    f[idx++] = computeTrapezoidalAUC(gas2_norm, steps);

    //[64] peak_idx1 (normalised 0-1)
    f[idx++] = (float)computePeakIndex(gas1_norm, steps) / (steps - 1);

    //[65] peak_idx2 (normalised 0-1)
    f[idx++] = (float)computePeakIndex(gas2_norm, steps) / (steps - 1);

    //[66] range1
    f[idx++] = computeMax(gas1_norm, steps) - computeMin(gas1_norm, steps);

    //[67] range2
    f[idx++] = computeMax(gas2_norm, steps) - computeMin(gas2_norm, steps);

    //[68] late_early_ratio1
    {
        float early = 0, late = 0;
        for(int i = 0; i < 3; i++){
            early += gas1_norm[i];
            late  += gas1_norm[steps - 3 + i];
        }
        early /= 3.0f; late /= 3.0f;
        f[idx++] = (fabsf(early) > 1e-6f) ? (late / early) : 1.0f;
    }

    //[69] late_early_ratio2
    {
        float early = 0, late = 0;
        for(int i = 0; i < 3; i++){
            early += gas2_norm[i];
            late  += gas2_norm[steps - 3 + i];
        }
        early /= 3.0f; late /= 3.0f;
        f[idx++] = (fabsf(early) > 1e-6f) ? (late / early) : 1.0f;
    }

    //[70] cross_ratio_mean
    float cr_sum = 0;
    for(uint8_t i = 0; i < steps; i++) cr_sum += cross[i];
    f[idx++] = cr_sum / steps;

    //[71] cross_ratio_slope
    f[idx++] = computeSlope(cross, steps);

    //[72] cross_ratio_variance
    {
        float cr_mean = cr_sum / steps;
        float cr_var = 0;
        for(uint8_t i = 0; i < steps; i++){
            float d = cross[i] - cr_mean;
            cr_var += d * d;
        }
        f[idx++] = cr_var / steps;
    }

    //[73] delta_temp
    f[idx++] = fabsf(raw.temp1 - raw.temp2);

    //[74] delta_hum
    f[idx++] = fabsf(raw.hum1 - raw.hum2);

    //[75] delta_pres
    f[idx++] = fabsf(raw.pres1 - raw.pres2);

    //[76] temp1
    f[idx++] = raw.temp1;

    //[77] hum1
    f[idx++] = raw.hum1;

    //[78] pres1
    f[idx++] = raw.pres1;

    //[79] temp2
    f[idx++] = raw.temp2;

    //[80] hum2
    f[idx++] = raw.hum2;

    //[81] pres2
    f[idx++] = raw.pres2;

    if(idx != FULL_ML_FEATURES){
        DEBUG_PRINTF("[MLInference] Base feature count mismatch: expected %d got %d\n",FULL_ML_FEATURES, idx);
    }

    return idx;
}

int MLInference::computeEngineeredFeatures(float* f, int baseCount) {
    //!must match addEngineeredFeatures() in the training code
    int fi = baseCount;

    //GROUP 1: gas self-normalised ratios (10 features)
    for(int i = 0; i < 10; i++){
        if(fi >= MAX_FULL_FEATURES) break;
        float g1 = f[i];       // gas1_norm
        float g2 = f[i + 10];  // gas2_norm
        f[fi++] = (fabsf(g2) > 1e-6f) ? g1 / g2 : 0.0f;
    }

    //GROUP 2: log ratios (5 features)
    for(int i = 0; i < 5; i++){
        if(fi >= MAX_FULL_FEATURES) break;
        float g1 = f[i];
        float g2 = f[i + 10];
        f[fi++] = (g1 > 0 && g2 > 0) ? logf(g1 / g2) : 0.0f;
    }

    //GROUP 3: targeted interactions 
    struct Pair { int a, b; };
    Pair interactions[] ={
        {12, 21},  //g2n_s2 * cr_s1
        {12, 71},  //g2n_s2 * cr_slope
        {2, 31},  //g1n_s2 * diff_s1
        {59, 20},  //slope2 * cr_s0
        {51, 22},  //g2d_s2 * cr_s2
        {13, 70},  //g2n_s3 * cr_mean
        {35, 23},  //diff_s5 * cr_s3
        {3, 46},  //g1n_s3 * g1d_s6
        {2, 35},  //g1n_s2 * diff_s5
        {14, 52},  //g2n_s4 * g2d_s3
    };
    for(int p=0; p<10;p++){
        if(fi >= MAX_FULL_FEATURES) break;
        if(interactions[p].a < baseCount && interactions[p].b < baseCount){
            f[fi++] = f[interactions[p].a] * f[interactions[p].b];
        }
    }

    //GROUP 4: cross ratio temporal derivatives (9 features)
    for(int t=0;t<9;t++){
        if(fi >= MAX_FULL_FEATURES) break;
        f[fi++] = f[20 + t + 1] - f[20 + t];
    }

    //GROUP 5: Delta ratios (9 features)
    for(int t=0; t<9;t++){
        if(fi >= MAX_FULL_FEATURES) break;
        float g1d = f[40+t];  //gas1_delta
        float g2d = f[49+t];  //gas2_delta
        f[fi++]=g1d/(fabsf(g2d) + 1e-6f);
    }

    //GROUP 6: 2nd deltas gas2 acceleration (8 features)
    for(int t=0;t<8;t++){
        if(fi >= MAX_FULL_FEATURES) break;
        f[fi++] = f[49 + t + 1] - f[49 + t];
    }

    //GROUP 7: response shape (4 features)
    for(int sensor = 0; sensor < 2; sensor++){

        if(fi >= MAX_FULL_FEATURES) break;
        int base = sensor * 10;
        float minV = 1e30f, maxV = -1e30f;

        for(int t=0; t<10;t++){
            float v = f[base + t];
            if(v < minV) minV = v;
            if(v > maxV) maxV = v;
        }
        float range = maxV - minV;
        f[fi++] = (range > 1e-6f) ? (f[base + 9] - minV) / range : 0.0f;
    }
    //max divergence
    if(fi < MAX_FULL_FEATURES){
        float maxDiv = 0;
        for(int t = 0; t < 10; t++){
            float d = fabsf(f[t] - f[10 + t]);
            if(d > maxDiv) maxDiv = d;
        }
        f[fi++] = maxDiv;
    }
    //crossratio early/late diff
    if(fi < MAX_FULL_FEATURES && baseCount > 29){
        float cr_early = (f[20] + f[21] + f[22]) / 3.0f;
        float cr_late  = (f[27] + f[28] + f[29]) / 3.0f;
        f[fi++] = cr_late - cr_early;
    }

    //GROUP 8: segment ratios(6 features)
    {
        float seg[2][3]; //[sensor][early/mid/late]
        int segs[][2] = {{0,3}, {4,6}, {7,9}};
        for(int sensor = 0; sensor < 2; sensor++){
            int base = sensor * 10;
            for(int sg = 0; sg < 3; sg++){
                float sum = 0;
                int n = 0;
                for(int t = segs[sg][0]; t <= segs[sg][1]; t++){
                    sum += f[base + t];
                    n++;
                }
                seg[sensor][sg] = sum / n;
            }
        }
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[0][0]/(fabsf(seg[0][2])+1e-6f);
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[0][1]/(fabsf(seg[0][2])+1e-6f);
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[1][0] /(fabsf(seg[1][2])+1e-6f);
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[1][1]/(fabsf(seg[1][2])+1e-6f);
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[0][0] /(fabsf(seg[1][0])+1e-6f);
        if(fi < MAX_FULL_FEATURES) f[fi++]=seg[0][2] / (fabsf(seg[1][2])+1e-6f);
    }

    //GROUP 9:diff channel shape (2 features)
    //gas1 response
    if(fi < MAX_FULL_FEATURES && baseCount > 9){
        float g1_start = f[0], g1_mid = f[4], g1_end = f[9];
        if(fabsf(g1_start + g1_end) > 1e-6f){
            f[fi++] = (2.0f * g1_mid) / (g1_start + g1_end);
        }
        else{
            f[fi++] = 0.0f;
        }
    }
    //gas2 response
    if(fi < MAX_FULL_FEATURES && baseCount > 19){
        float g2_start = f[10], g2_mid = f[14], g2_end = f[19];
        if(fabsf(g2_start + g2_end) > 1e-6f){
            f[fi++] = (2.0f * g2_mid) / (g2_start + g2_end);
        }
        else{
            f[fi++] = 0.0f;
        }
    }

    //GROUP 10: humidity gas interactions (8 features)
    if(baseCount >= 77){
        float hum1 = f[77];
        float hum2 = f[80];
        float avg_hum = (hum1 + hum2) / 2.0f;

        if(fabsf(avg_hum) > 1e-6f){
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[12]/avg_hum;  // g2n_s2 / hum
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[2]/avg_hum;  // g1n_s2 / hum
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[20]/avg_hum;  // cr_s0 / hum
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[31]/avg_hum;  // diff_s1 / hum
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[59]/avg_hum;  // slope2 / hum
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[70]/avg_hum;  // cr_mean / hum
        }
        else{
            for(int z = 0; z < 6 &&fi < MAX_FULL_FEATURES; z++){
                f[fi++] = 0.0f;
            }
        }

        float temp1 = f[76];

        if(fabsf(temp1)>1e-6f){
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[12] / temp1;  // g2n_s2 / temp
            if(fi < MAX_FULL_FEATURES) f[fi++] = f[20] / temp1;  // cr_s0 / temp
        }

        else{
            for(int z = 0; z < 2 && fi < MAX_FULL_FEATURES; z++){
                f[fi++] = 0.0f;
            }
        }
    }

    return fi;
}

void MLInference::applyScalingAndSelection(const float* allFeatures, int totalCount,ml_feature_buffer_t& output) {

    //matches training
    static bool warnedMissingSelected = false;
    for(int i=0;i<SELECTED_FEATURE_COUNT; i++){
        int origIdx=SELECTED_INDICES[i];

        float val = 0.0f;
        if(origIdx <totalCount){
            val =allFeatures[origIdx];
        }

        else{
            //use training median as fallback so normalized value is 0
            val =SELECTED_MEDIANS[i];
            if(!warnedMissingSelected){
                const char* kind = isLikelyFisherIndex(origIdx) ? "Fisher" : "non-Fisher";
                DEBUG_PRINTF("[MLInference] WARNING: selected %s feature index %d is beyond computed range %d; using neutral median fallback.\n", kind, origIdx, totalCount);
                warnedMissingSelected = true;
            }
        }

        // (value - median) / IQR
        if(SELECTED_IQRS[i]>1e-6f){
            val= (val-SELECTED_MEDIANS[i])/SELECTED_IQRS[i];
        }
        else{
            val=val - SELECTED_MEDIANS[i];
        }

        output.features[i] = val;
    }
    output.featureCount = SELECTED_FEATURE_COUNT;
    output.ready = true;
}

void MLInference::normaliseFeatures(ml_feature_buffer_t& features) {
    float allFeatures[MAX_FULL_FEATURES];
    memset(allFeatures, 0, sizeof(allFeatures));

    //step 1: compute base features
    int baseCount = computeBaseFeatures(_raw_snapshot, allFeatures);

    //step 2: compute engineered features
    int totalCount = computeEngineeredFeatures(allFeatures, baseCount);

    //step 3: compute fisher projections
    if(totalCount>=FISHER_BASE_FEATURE_COUNT&&(FISHER_BASE_FEATURE_COUNT + FISHER_PAIR_COUNT) <=MAX_FULL_FEATURES){
        float scaledForFisher[FISHER_BASE_FEATURE_COUNT];

        for(int i=0;i<FISHER_BASE_FEATURE_COUNT; i++){
            float denom =FEATURE_IQRS[i];
            if(fabsf(denom)< 1e-6f){
                denom = 1.0f;
            }
            scaledForFisher[i] = (allFeatures[i] - FEATURE_MEDIANS[i])/denom;
        }

        for(int p=0;p<FISHER_PAIR_COUNT; p++){
            float proj =0.0f;

            for(int i=0;i< FISHER_BASE_FEATURE_COUNT;i++){
                proj+=scaledForFisher[i]*FISHER_WEIGHTS[p][i];
            }
            allFeatures[FISHER_BASE_FEATURE_COUNT + p] = proj;
        }

        totalCount = FISHER_BASE_FEATURE_COUNT + FISHER_PAIR_COUNT;
    }

    DEBUG_VERBOSE_PRINTF("[MLInference] Computed %d base + %d engineered = %d total features\n",baseCount, totalCount-baseCount,totalCount);

    if(totalCount<ORIGINAL_FEATURE_COUNT){
        int missing = ORIGINAL_FEATURE_COUNT - totalCount;
        DEBUG_VERBOSE_PRINTF("[MLInference] Runtime pipeline produced %d/%d original features; %d selected slots may use median fallback.\n",totalCount, ORIGINAL_FEATURE_COUNT, missing);
    }

    //step 4: apply robust scaling and feature selection
    applyScalingAndSelection(allFeatures, totalCount, features);
}

//===========================================================================================================
//class name helpers
//===========================================================================================================

const char* MLInference::getClassName(scent_class_t classId) const {
    if(classId < SCENT_CLASS_COUNT){
        return SCENT_CLASS_NAMES[classId];
    }
    return "Unknown";
}

scent_class_t MLInference::getClassFromName(const char* name) const {
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
        bool hasDecaf  = (strstr(target, "decaf") != nullptr);
        bool hasTea    = (strstr(target, "tea") != nullptr);
        bool hasCoffee = (strstr(target, "coffee") != nullptr);

        if(hasDecaf && hasTea)        return SCENT_CLASS_DECAF_TEA;
        if(hasDecaf && hasCoffee)     return SCENT_CLASS_DECAF_COFFEE;
        if(hasTea && !hasDecaf)       return SCENT_CLASS_TEA;
        if(hasCoffee && !hasDecaf)    return SCENT_CLASS_COFFEE;
    }

    for(int i = 0; i < SCENT_CLASS_COUNT; i++){
        char candidate[64];
        normalizeSimple(SCENT_CLASS_NAMES[i], candidate, sizeof(candidate));
        if(strcmp(target, candidate)==0){
            return static_cast<scent_class_t>(i);
        }
    }
    return SCENT_CLASS_UNKNOWN;
}

//===========================================================================================================
//model selection
//===========================================================================================================

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

//===========================================================================================================
//inference
//===========================================================================================================

bool MLInference::runInference(ml_prediction_t &pred){
    memset(&pred, 0, sizeof(pred));
    pred.predictedClass = SCENT_CLASS_UNKNOWN;
    pred.timestamp = millis();

    if(!_init || !_feature_buffer.ready || !_model_available[_active_model]){
        pred.valid = false;
        return false;
    }

    //pre-classification ambient check
    float ambientScore = computeAmbientScore(_raw_snapshot);
    if(ambientScore > 0.85f){
        pred.predictedClass = SCENT_CLASS_AMBIENT;
        pred.confidence = ambientScore;
        pred.valid = true;
        pred.isAnomalous = false;
        pred.anomalyScore = 0.0f;
        pred.inferenceTimeMs = 0;
        
        DEBUG_PRINTF("[MLInference] Ambient detected (score: %.2f), skipping classification\n",ambientScore);
        logger.logDebugMsg(String("[Ambient] Score: ") + String(ambientScore, 2));
        return true;
    }

    uint32_t start_time = micros();

    switch(_active_model){
        case ML_MODEL_EDGE_IMPULSE: {
    #if ISC_ENABLE_EI
            signal_t signal;
            signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
            static float* featurePointer = nullptr;
            featurePointer = _feature_buffer.features;
            signal.get_data = [](size_t offset, size_t length,float* out_ptr) ->int {
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
            for(size_t i=0; i<EI_CLASSIFIER_LABEL_COUNT; i++){
                pred.classConfidences[i] = ei_result.classification[i].value;
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
            if(pred.isAnomalous) pred.predictedClass = SCENT_CLASS_UNKNOWN;
#else
            pred.isAnomalous = false;
            pred.anomalyScore = 0.0f;
#endif
            if(pred.confidence < _confidence_threshold && pred.predictedClass != SCENT_CLASS_UNKNOWN){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
            }
            pred.valid = true;
#else
            DEBUG_PRINTLN(F("[MLInference] Edge Impulse model not available"));
            pred.valid = false;
#endif
            break;
        }

        case ML_MODEL_DECISION_TREE: {
#if ISC_ENABLE_DT
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
            if(pred.predictedClass < ML_CLASS_COUNT)
                pred.classConfidences[pred.predictedClass] = confidence;
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - confidence;
            }
            else{
                pred.isAnomalous = false;
                pred.anomalyScore = 0.0f;
            }
            pred.valid = true;
#else
            pred.valid = false;
#endif
            break;
        }

        case ML_MODEL_KNN: {
#if ISC_ENABLE_KNN
            float confidence = 0.0f;
            uint8_t cls = knn_predict_with_confidence(_feature_buffer.features, &confidence);

            knn_neighbor_t neighbors[KNN_K];
            knn_find_neighbors(_feature_buffer.features, neighbors);
            float avg_distance = 0.0f;
            for(int i = 0; i < KNN_K; i++){
                avg_distance += neighbors[i].distance;
            }

            avg_distance /=KNN_K;
            float calculated_anomaly =fmin(avg_distance / KNN_DISTANCE_SCALE, 1.0f);

            pred.inferenceTimeMs = (micros() - start_time) / 1000;
            pred.predictedClass = (scent_class_t)cls;
            pred.confidence = confidence;

            memset(pred.classConfidences, 0, sizeof(pred.classConfidences));
            if(pred.predictedClass < ML_CLASS_COUNT){
                pred.classConfidences[pred.predictedClass] = confidence;
            }

            pred.anomalyScore = calculated_anomaly;
            pred.isAnomalous = false;
            
            if(pred.confidence < _confidence_threshold || pred.isAnomalous){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                if(!pred.isAnomalous) pred.anomalyScore = 1.0f - pred.confidence;
            }
            pred.valid = true;
#else
            pred.valid = false;
#endif
            break;
        }

        case ML_MODEL_RANDOM_FOREST: {
#if ISC_ENABLE_RF
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
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - confidence;
            }
            else{
                pred.isAnomalous = false;
                pred.anomalyScore = 0.0f;
            }
            pred.valid = true;
#else
            pred.valid = false;
#endif
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
        char debugMsg[160];
        snprintf(debugMsg, sizeof(debugMsg),
                "[Single] Model:%s Class:%d Conf:%.2f Anom:%.2f TimeMs:%lu",
                getActiveModelName(), pred.predictedClass, pred.confidence,
                pred.anomalyScore, pred.inferenceTimeMs);
        logger.logDebugMsg(String(debugMsg));
    }
    return pred.valid;
}

bool MLInference::runInferenceOnData(const dual_sensor_data_t &sensorData, ml_prediction_t &pred){
    if(!addToWindow(sensorData)){
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


bool MLInference::runActiveInference(ml_prediction_t &pred){
    switch(_inference_mode){
        case INFERENCE_MODE_SINGLE:
            return runInference(pred);

        case INFERENCE_MODE_ENSEMBLE: {
            ml_ensemble_prediction_t ensP;
            if(!runEnsembleInference(ensP)){
                pred.valid = false;
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                return false;
            }
            ensembleToPrediction(ensP, pred);
            if(pred.confidence < _confidence_threshold){
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
                pred.isAnomalous = true;
                pred.anomalyScore = 1.0f - pred.confidence;
            }
            return true;
        }

        case INFERENCE_MODE_TEMPORAL: {
            ml_ensemble_prediction_t ensPred;
            if(!runEnsembleInference(ensPred)){
                pred.valid = false;
                pred.predictedClass = SCENT_CLASS_UNKNOWN;
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
//ensemble to pred
//===========================================================================================================

void MLInference::ensembleToPrediction(const ml_ensemble_prediction_t &ens, ml_prediction_t &pred){
    memset(&pred, 0, sizeof(pred));
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

    uint32_t time = millis();

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

    //add to buffer
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

    return false;
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
    memset(&finalPred, 0, sizeof(finalPred));

    finalPred.predictedClass = SCENT_CLASS_UNKNOWN;
    finalPred.timestamp = millis();

    float conf = 0.0f;
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

scent_class_t MLInference::getTemporalPrediction(float &conf) const {
    if(_temporal_count == 0){
        conf=0;
        return SCENT_CLASS_UNKNOWN;
    }

    float bScore=0.0f;
    uint8_t bIdx=SCENT_CLASS_UNKNOWN;
    float total=0.0f;

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        total += _temporal_scores[i];
        if(_temporal_scores[i] > bScore){
            bScore = _temporal_scores[i];
            bIdx = i;
        }
    }
    conf = (total>0) ? (bScore / total) : 0.0f;

    float second = 0.0f;
    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        if(i != bIdx && _temporal_scores[i] > second){
            second = _temporal_scores[i];
        }
    }

    if(bScore - second < 0.05f){
        return SCENT_CLASS_UNKNOWN;
    }

    return (scent_class_t)bIdx;
}

//===========================================================================================================
//ambient score
//===========================================================================================================
float MLInference::computeAmbientScore(const raw_sensor_snapshot_t& raw) {
    sensor_calibration_t cal=sensors.getCalibrationData();
    if(!cal.calibrated){
        return 0.0f;
    }
    
    float total_deviation=0.0f;
    int valid_steps=0;
    
    for(uint8_t i = 0; i < BME688_NUM_HEATER_STEPS; i++){
        float b1=cal.gas_baseline_primary[i];
        float b2=cal.gas_baseline_secondary[i];
        
        if(b1 >= 1.0f){
            float ratio1=raw.gas1[i] / b1;
            total_deviation+=fabsf(ratio1 - 1.0f);
            valid_steps++;
        }
        if(b2 >= 1.0f){
            float ratio2=raw.gas2[i] / b2;
            total_deviation+=fabsf(ratio2 - 1.0f);
            valid_steps++;
        }
    }
    
    if(valid_steps == 0) return 0.0f;
    
    float avg_deviation = total_deviation / valid_steps;
    float ambient_score =1.0f - fminf(avg_deviation / 0.15f, 1.0f);
    
    return ambient_score;
}


//===========================================================================================================
//ensemble inference
//===========================================================================================================

bool MLInference::runEnsembleInference(ml_ensemble_prediction_t &pred){
    memset(&pred,0, sizeof(pred));
    pred.predictedClass = SCENT_CLASS_UNKNOWN;
    pred.timestamp = millis();

    if(!_init || !_feature_buffer.ready){
        DEBUG_PRINTLN(F("[MLInference] Module not ready"));
        pred.valid=false;
        return false;
    }

    //ambient gate
    float ambientScore = computeAmbientScore(_raw_snapshot);
    if(ambientScore>0.85f){
        pred.predictedClass = SCENT_CLASS_AMBIENT;
        pred.confidence = ambientScore;
        pred.valid = true;

        memset(pred.classScores, 0, sizeof(pred.classScores));

        if(SCENT_CLASS_AMBIENT< SCENT_CLASS_COUNT){
            pred.classScores[SCENT_CLASS_AMBIENT] = ambientScore;
        }
        return true;
    }

    uint32_t start=micros();
    uint32_t dt_start=micros();
    // Run all models
    float dtConf = 1.0f, knnConf = 1.0f, rfConf = 1.0f;

#if ISC_ENABLE_DT
#ifdef DT_HAS_CONFIDENCE
    uint8_t dtCls = dt_predict_with_confidence(_feature_buffer.features, &dtConf);
#else
    uint8_t dtCls = dt_predict(_feature_buffer.features);
#endif
    uint32_t dt_time_us = micros() - dt_start;
#else
    uint8_t dtCls = SCENT_CLASS_UNKNOWN;
    dtConf = 0.0f;
    uint32_t dt_time_us = 0;
#endif

#if ISC_ENABLE_KNN
    uint32_t knn_start = micros();
    uint8_t knnCls = knn_predict_with_confidence(_feature_buffer.features, &knnConf);

    //KNN anomaly detection via neighbor distances
    knn_neighbor_t neighbors[KNN_K];
    knn_find_neighbors(_feature_buffer.features, neighbors);
    float avg_distance = 0.0f;
    for(int i = 0; i < KNN_K; i++){
        avg_distance += neighbors[i].distance;
    }
    avg_distance /= KNN_K;
    float calculated_anomaly =fmin(avg_distance/KNN_DISTANCE_SCALE, 1.0f);
    bool knnIsAnomalous = false;
    uint32_t knn_time_us = micros() - knn_start;
#else
    uint8_t knnCls = SCENT_CLASS_UNKNOWN;
    float calculated_anomaly = 0.0f;
    bool knnIsAnomalous = false;
    uint32_t knn_time_us = 0;
    knnConf = 0.0f;
#endif

    // RF
#if ISC_ENABLE_RF
    uint32_t rf_start = micros();
    uint8_t rfCls = rf_predict_with_confidence(_feature_buffer.features, &rfConf);
    uint32_t rf_time_us = micros() - rf_start;
#else
    uint8_t rfCls = SCENT_CLASS_UNKNOWN;
    rfConf = 0.0f;
    uint32_t rf_time_us = 0;
#endif

    if(dtConf < _confidence_threshold){
        dtCls = SCENT_CLASS_UNKNOWN;
        dtConf = 0.0f;
    }
    if(knnConf < _confidence_threshold || knnIsAnomalous){
        knnCls = SCENT_CLASS_UNKNOWN;
        knnConf = 0.0f;
    }
    if(rfConf < _confidence_threshold){
        rfCls = SCENT_CLASS_UNKNOWN;
        rfConf = 0.0f;
    }

    pred.inferenceTimeMs = (micros() - start);

    //results
    pred.dtClass  = (scent_class_t)dtCls;
    pred.dtConf   = dtConf;
    pred.knnClass = (scent_class_t)knnCls;
    pred.knnConf  = knnConf;
    pred.rfClass  = (scent_class_t)rfCls;
    pred.rfConf   = rfConf;

    //weighted ensemble scoring
    memset(pred.classScores, 0, sizeof(pred.classScores));

    //KNN
    if(knnCls < SCENT_CLASS_COUNT){
        pred.classScores[knnCls] += KNN_WEIGHT * knnConf;
        float knnResid = KNN_WEIGHT * (1.0f - knnConf) / (SCENT_CLASS_COUNT - 1);
        for(int c = 0; c < SCENT_CLASS_COUNT; c++){
            if(c != knnCls) pred.classScores[c] += knnResid;
        }
    }

    //RF
    if(rfCls < SCENT_CLASS_COUNT){
        pred.classScores[rfCls] += RF_WEIGHT * rfConf;
        float rfResid = RF_WEIGHT * (1.0f - rfConf) / (SCENT_CLASS_COUNT - 1);
        for(int c = 0; c < SCENT_CLASS_COUNT; c++){
            if(c != rfCls) pred.classScores[c] += rfResid;
        }
    }

    //DT
    if(dtCls < SCENT_CLASS_COUNT){
        pred.classScores[dtCls] += DT_WEIGHT * dtConf;
        float dtResid = DT_WEIGHT * (1.0f - dtConf) / (SCENT_CLASS_COUNT - 1);
        for(int c = 0; c < SCENT_CLASS_COUNT; c++){
            if(c != dtCls) pred.classScores[c] += dtResid;
        }
    }



    float bScore = 0.0f;
    uint8_t bClass = SCENT_CLASS_UNKNOWN;
    float total = 0.0f;

    for(int i = 0; i < SCENT_CLASS_COUNT; i++){
        total += pred.classScores[i];
        if(pred.classScores[i] > bScore){
            bScore = pred.classScores[i];
            bClass = i;
        }
    }

    //agreement check
    bool dt_knn_agree= (dtCls==knnCls)&& (dtCls!= SCENT_CLASS_UNKNOWN);
    bool knn_rf_agree= (knnCls==rfCls)&&(knnCls!= SCENT_CLASS_UNKNOWN);
    bool dt_rf_agree= (dtCls==rfCls)&& (dtCls!= SCENT_CLASS_UNKNOWN);

    char debugMsg[192];
    snprintf(debugMsg, sizeof(debugMsg),
            "[Ensemble] DT:%d(%.2f,%lums) KNN:%d(%.2f,A:%.2f,%lums) RF:%d(%.2f,%lums) Total:%lums",
            dtCls, dtConf, dt_time_us / 1000,knnCls, knnConf, calculated_anomaly, knn_time_us / 1000,
            rfCls, rfConf, rf_time_us / 1000,pred.inferenceTimeMs);
    DEBUG_PRINTLN(debugMsg);
    logger.logDebugMsg(String(debugMsg));

    //if no two models agree, mark as unknown
    if(!dt_knn_agree && !knn_rf_agree && !dt_rf_agree){
        pred.predictedClass = SCENT_CLASS_UNKNOWN;
        pred.confidence = 0.0f;
        DEBUG_PRINTLN("[Ensemble] Full disagreement: forced Unknown");
    }
    else{
        pred.predictedClass = (scent_class_t)bClass;
        float max_possible_score = DT_WEIGHT + KNN_WEIGHT + RF_WEIGHT;
        pred.confidence = (max_possible_score > 0) ? bScore / max_possible_score : 0.0f;
    }

    pred.valid = true;
    _total_inferences++;
    _total_inference_time_ms += pred.inferenceTimeMs;

    return true;
}

//===========================================================================================================
//util
//===========================================================================================================

void MLInference::printModelInfo(){
    DEBUG_PRINTLN(F("[MLInference] ML Model Information:"));
    DEBUG_PRINTF("Pipeline: %d base -> ~%d engineered -> %d selected (robust scaled)\n",FULL_ML_FEATURES, MAX_FULL_FEATURES, SELECTED_FEATURE_COUNT);
#if EI_CLASSIFIER
    DEBUG_PRINTF("Model: Edge Impulse\n");
    DEBUG_PRINTF("Input features: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    DEBUG_PRINTF("Output classes: %d\n", EI_CLASSIFIER_LABEL_COUNT);
    DEBUG_PRINTF("Anomaly detection: %s\n", EI_CLASSIFIER_HAS_ANOMALY ? "Yes" : "No");
#else
    DEBUG_PRINTLN(F("Models: DT + KNN + RF (custom trained)"));
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
    for(int i = 0; i < labelCount; i++){
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
    for(int i = 0; i < labelCount; i++){
        json += String(result.classConfidences[i], 4);
        if(i < labelCount - 1) json += ",";
    }
    json += "]}";
    return json;
}

void MLInference::printFeatureDebug() const {
    DEBUG_PRINTLN(F("\n=== Feature Debug ==="));
    DEBUG_PRINTF("Pipeline: %d base -> engineered -> %d selected (robust scaled)\n",
                FULL_ML_FEATURES, SELECTED_FEATURE_COUNT);
    DEBUG_PRINTF("Feature count in buffer: %d (expected: %d)\n",
                _feature_buffer.featureCount, SELECTED_FEATURE_COUNT);
    DEBUG_PRINTF("Raw snapshot valid: %s\n", _raw_snapshot.valid ? "Yes" : "No");

    if(_raw_snapshot.valid){
        DEBUG_PRINTF("  Temp: %.1f/%.1f C  Hum: %.1f/%.1f %%  Pres: %.0f/%.0f hPa\n",
                    _raw_snapshot.temp1, _raw_snapshot.temp2,
                    _raw_snapshot.hum1, _raw_snapshot.hum2,
                    _raw_snapshot.pres1, _raw_snapshot.pres2);

        DEBUG_PRINT("  Gas1:");
        for(int i = 0; i < BME688_NUM_HEATER_STEPS; i++){
            DEBUG_PRINTF(" %.0f", _raw_snapshot.gas1[i]);
        }

        DEBUG_PRINTLN();

        DEBUG_PRINT("  Gas2:");
        for(int i = 0; i < BME688_NUM_HEATER_STEPS; i++){
            DEBUG_PRINTF(" %.0f", _raw_snapshot.gas2[i]);
        }
        DEBUG_PRINTLN();
    }

    //feature name lookup
    static const char* BASE_FEATURE_NAMES[] = {
        //0-9: gas1_norm
        "g1n_s0", "g1n_s1", "g1n_s2", "g1n_s3", "g1n_s4",
        "g1n_s5", "g1n_s6", "g1n_s7", "g1n_s8", "g1n_s9",
        //10-19: gas2_norm
        "g2n_s0", "g2n_s1", "g2n_s2", "g2n_s3", "g2n_s4",
        "g2n_s5", "g2n_s6", "g2n_s7", "g2n_s8", "g2n_s9",
        //20-29: cross_ratio
        "cr_s0", "cr_s1", "cr_s2", "cr_s3", "cr_s4",
        "cr_s5", "cr_s6", "cr_s7", "cr_s8", "cr_s9",
        //30-39: diff
        "diff_s0", "diff_s1", "diff_s2", "diff_s3", "diff_s4",
        "diff_s5", "diff_s6", "diff_s7", "diff_s8", "diff_s9",
        //40-48: gas1_delta
        "g1d_s0", "g1d_s1", "g1d_s2", "g1d_s3", "g1d_s4",
        "g1d_s5", "g1d_s6", "g1d_s7", "g1d_s8",
        //49-57: gas2_delta
        "g2d_s0", "g2d_s1", "g2d_s2", "g2d_s3", "g2d_s4",
        "g2d_s5", "g2d_s6", "g2d_s7", "g2d_s8",
        //58-72: summary stats
        "slope1", "slope2", "curv1", "curv2",
        "auc1", "auc2", "peak1", "peak2",
        "range1", "range2", "le_r1", "le_r2",
        "cr_mean", "cr_slope", "cr_var",
        //73-81: environment
        "d_temp", "d_hum", "d_pres",
        "temp1", "hum1", "pres1", "temp2", "hum2", "pres2"
    };
    static const int NUM_BASE_NAMES = sizeof(BASE_FEATURE_NAMES) / sizeof(BASE_FEATURE_NAMES[0]);

    for(int i = 0; i < _feature_buffer.featureCount && i < SELECTED_FEATURE_COUNT; i++){
        int oIdx = SELECTED_INDICES[i];
        const char* name;

        if(oIdx < NUM_BASE_NAMES){
            name = BASE_FEATURE_NAMES[oIdx];
        }
        else if(oIdx < FULL_ML_FEATURES + 10){
            name = "eng_ratio";
        }
        else if(oIdx < FULL_ML_FEATURES + 15){
            name = "eng_log";
        }
        else if(oIdx < FULL_ML_FEATURES + 25){
            name = "eng_interact";
        }
        else if(oIdx < FULL_ML_FEATURES + 34){
            name = "eng_cr_deriv";
        }
        else if(oIdx < FULL_ML_FEATURES + 43){
            name = "eng_d_ratio";
        }
        else if(oIdx < FULL_ML_FEATURES + 51){
            name = "eng_accel";
        }
        else if(oIdx < FULL_ML_FEATURES + 55){
            name = "eng_shape";
        }
        else if(oIdx < FULL_ML_FEATURES + 61){
            name = "eng_seg_rat";
        }
        else if(oIdx < FULL_ML_FEATURES + 63){
            name = "eng_diff_sh";
        }
        else if(oIdx < FULL_ML_FEATURES + 71){
            name = "eng_hum_gas";
        }
        else{
            name = "eng_other";
        }

        DEBUG_PRINTF("[%2d] orig[%3d] %-12s = %8.4f", i, oIdx, name, _feature_buffer.features[i]);

        if(fabsf(_feature_buffer.features[i]) > 3.0f){
            DEBUG_PRINTF(" %.1f sigma", _feature_buffer.features[i]);
        }
        DEBUG_PRINTLN();
    }
}