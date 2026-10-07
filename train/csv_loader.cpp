#include "csv_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <array>
#include <set>
#include <cmath>

namespace{
struct Baseline{
    float temp1, temp2;
    float hum1, hum2;
    float pres1, pres2;
    float gas1_steps[10];
    float gas2_steps[10];
};

static const int RAW_COLS_PER_SENSOR = 3 + NUM_HEATER_STEPS;
static const int TOTAL_RAW_COLS = 2 * RAW_COLS_PER_SENSOR;
struct RawSensorData{
    float temp1, hum1, pres1;
    float gas1[10];
    float temp2, hum2, pres2;
    float gas2[10];
};

static float mean(const std::vector<float> &v){
    if(v.empty()){
        return 0.0f;
    }
    double sum = 0.0;
    for(float x : v){
        sum += x;
    }
    return (float)(sum / v.size());
}

static Baseline computeBaseline(const std::vector<RawSensorData> &cal){
    Baseline b{};
    if(cal.empty()){
        return b;
    }

    std::vector<float> temp1v, temp2v, hum1v, hum2v, pres1v, pres2v;
    std::vector<float> gas1v[NUM_HEATER_STEPS], gas2v[NUM_HEATER_STEPS];

    for(const auto&sample : cal){
        temp1v.push_back(sample.temp1);
        hum1v.push_back(sample.hum1);
        pres1v.push_back(sample.pres1);

        temp2v.push_back(sample.temp2);
        hum2v.push_back(sample.hum2);
        pres2v.push_back(sample.pres2);

        for(int i=0; i<NUM_HEATER_STEPS; i++){
            gas1v[i].push_back(sample.gas1[i]);
            gas2v[i].push_back(sample.gas2[i]);
        }
    }

    b.temp1= mean(temp1v);
    b.temp2= mean(temp2v);
    b.hum1= mean(hum1v);
    b.hum2= mean(hum2v);
    b.pres1= mean(pres1v);
    b.pres2= mean(pres2v);
    
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        b.gas1_steps[i]= mean(gas1v[i]);
        b.gas2_steps[i]= mean(gas2v[i]);
        if(b.gas1_steps[i] < 1.0f){
            b.gas1_steps[i] = 1.0f;
        }
        if(b.gas2_steps[i] < 1.0f){
            b.gas2_steps[i] = 1.0f;
        }
    }
    return b;
}

static float localSlope(const float* data,int n){
    if(n<=1){
        return 0.0f;
    }

    float sumX=0.0f, sumY=0.0f, sumXY=0.0f, sumXX=0.0f;
    for(int i=0; i<n;i++){
        sumX+=i;
        sumY+=data[i];
        sumXY+= i*data[i];
        sumXX+= i*i;
    }
    float denom=n*sumXX - sumX*sumX;
    return (fabsf(denom) < 1e-6f) ? 0.0f : (n * sumXY - sumX * sumY) / denom;
}

static float trapezoidalAUC(const float* data, int n){
    float area=0.0f;
    for(int i=1; i<n; i++){
        area+=0.5f*(data[i]+data[i-1]);
    }
    return area;
}

static int peakIndex(const float* data,int n){
    int idx=0;
    float peak=data[0];
    for(int i=1; i<n;i++){
        if(data[i]>peak){
            peak=data[i];
            idx=i;
        }
    }
    return idx;
}

static void applyMultiStepTransform(csv_training_sample_t& sample,const RawSensorData& raw,const Baseline& b) {
    uint16_t idx = 0;
    float gas1_resp[NUM_HEATER_STEPS];
    float gas2_resp[NUM_HEATER_STEPS];
    
    //get baseline-normalised responses
    for(int i = 0; i < NUM_HEATER_STEPS; i++){
        gas1_resp[i] = raw.gas1[i] / b.gas1_steps[i];
        gas2_resp[i] = raw.gas2[i] / b.gas2_steps[i];
    }
    

    //0-9
    float g1_base = (fabsf(gas1_resp[0]) > 1e-6f) ? gas1_resp[0] : 1.0f;
    float g2_base = (fabsf(gas2_resp[0]) > 1e-6f) ? gas2_resp[0] : 1.0f;
    
    float gas1_norm[NUM_HEATER_STEPS];
    float gas2_norm[NUM_HEATER_STEPS];
    
    for(int i = 0; i < NUM_HEATER_STEPS; i++){
        gas1_norm[i] = gas1_resp[i] / g1_base;
        gas2_norm[i] = gas2_resp[i] / g2_base;
        sample.features[idx++] = gas1_norm[i];
    }
    
    //10-19
    for(int i = 0; i < NUM_HEATER_STEPS; i++){
        sample.features[idx++] = gas2_norm[i];
    }
    

    //20-29
    for(int i = 0; i < NUM_HEATER_STEPS; i++){
        sample.features[idx++] = (gas2_resp[i] > 0.01f)?gas1_resp[i] / gas2_resp[i] : 1.0f;
    }
    
    //30-39
    for(int i = 0; i < NUM_HEATER_STEPS; i++){
        sample.features[idx++] = gas1_resp[i] - gas2_resp[i];
    }
    
    //40-48
    for(int i = 0; i < NUM_HEATER_STEPS - 1; i++){
        sample.features[idx++] = gas1_norm[i + 1] - gas1_norm[i];
    }
    
    //49-57
    for(int i = 0; i < NUM_HEATER_STEPS - 1; i++){
        sample.features[idx++] = gas2_norm[i + 1] - gas2_norm[i];
    }
    

    //58-67
    sample.features[idx++]=localSlope(gas1_norm, NUM_HEATER_STEPS);
    sample.features[idx++]=localSlope(gas2_norm, NUM_HEATER_STEPS);
    
    int half = NUM_HEATER_STEPS / 2;
    sample.features[idx++] =localSlope(gas1_norm+half, NUM_HEATER_STEPS-half)- localSlope(gas1_norm, half);
    sample.features[idx++] =localSlope(gas2_norm+half, NUM_HEATER_STEPS-half)- localSlope(gas2_norm, half);
    
    sample.features[idx++] = trapezoidalAUC(gas1_norm, NUM_HEATER_STEPS);
    sample.features[idx++] = trapezoidalAUC(gas2_norm, NUM_HEATER_STEPS);
    
    sample.features[idx++] = (float)peakIndex(gas1_norm, NUM_HEATER_STEPS)/ (NUM_HEATER_STEPS - 1);
    sample.features[idx++] = (float)peakIndex(gas2_norm, NUM_HEATER_STEPS)/ (NUM_HEATER_STEPS - 1);
    
    float g1_min = gas1_norm[0], g1_max = gas1_norm[0];
    float g2_min = gas2_norm[0], g2_max = gas2_norm[0];
    for(int i = 1; i < NUM_HEATER_STEPS; i++){
        if(gas1_norm[i] < g1_min){
            g1_min = gas1_norm[i];
        }

        if(gas1_norm[i] > g1_max){
            g1_max = gas1_norm[i];
        }

        if(gas2_norm[i] < g2_min){
            g2_min = gas2_norm[i];
        }

        if(gas2_norm[i] > g2_max){
            g2_max = gas2_norm[i];
        }
    }
    sample.features[idx++]=g1_max - g1_min;
    sample.features[idx++]=g2_max - g2_min;

    //68-69
    float g1_early = 0,g1_late = 0,g2_early = 0, g2_late = 0;
    for(int i = 0; i < 3; i++){
        g1_early += gas1_norm[i];
        g2_early += gas2_norm[i];
        g1_late += gas1_norm[NUM_HEATER_STEPS - 3 + i];
        g2_late += gas2_norm[NUM_HEATER_STEPS - 3 + i];
    }

    g1_early /= 3.0f; g1_late /= 3.0f;
    g2_early /= 3.0f; g2_late /= 3.0f;
    sample.features[idx++] = (fabsf(g1_early) > 1e-6f)? g1_late / g1_early : 1.0f; 
    sample.features[idx++] = (fabsf(g2_early) > 1e-6f)? g2_late / g2_early : 1.0f;
    
    
    //70-72
    float cr_vals[NUM_HEATER_STEPS];
    float cr_sum = 0;
    for(int i=0;i<NUM_HEATER_STEPS;i++){
        cr_vals[i]=(gas2_resp[i] > 0.01f) ? gas1_resp[i] / gas2_resp[i] : 1.0f;
        cr_sum+=cr_vals[i];
    }
    sample.features[idx++] = cr_sum / NUM_HEATER_STEPS; //mean cross-ratio
    sample.features[idx++] = localSlope(cr_vals, NUM_HEATER_STEPS); //cross-ratio trend
    
    //cross-ratio variance
    float cr_mean = cr_sum / NUM_HEATER_STEPS;
    float cr_var = 0;
    for(int i= 0;i<NUM_HEATER_STEPS;i++){
        float d = cr_vals[i]-cr_mean;
        cr_var += d*d;
    }
    sample.features[idx++] = cr_var/NUM_HEATER_STEPS; //cross-ratio variance
    


    //env
#if USE_ENV_FEATURES
    sample.features[idx++] = fabsf(raw.temp1 - raw.temp2);
    sample.features[idx++] = fabsf(raw.hum1 - raw.hum2);
    sample.features[idx++] = fabsf(raw.pres1 - raw.pres2);
#endif

    //[76-81] raw environmental values
#if USE_ENV_FEATURES
    sample.features[idx++] = raw.temp1;
    sample.features[idx++] = raw.hum1;
    sample.features[idx++] = raw.pres1;
    sample.features[idx++] = raw.temp2;
    sample.features[idx++] = raw.hum2;
    sample.features[idx++] = raw.pres2;
#endif

    if(idx!=BASE_FEATURE_COUNT){
        std::cerr<< "Feature count mismatch: expected "<< BASE_FEATURE_COUNT << " but got " << idx << std::endl;
        return;
    }
}

}//end of ns



CSVLoader::CSVLoader() : _samples(nullptr), _count(0), _capacity(0) {}

CSVLoader::~CSVLoader() {
    if (_samples) {
        delete[] _samples;
        _samples = nullptr;
    }
}

std::string CSVLoader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return str.substr(start, end - start + 1);
}

std::string CSVLoader::toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

const char* CSVLoader::getClassName(scent_class_t classId) {
    if (classId >= 0 && classId < SCENT_CLASS_COUNT) {
        return CSV_SCENT_CLASS_NAMES[classId];
    }
    return "unknown";
}

scent_class_t CSVLoader::getClassFromName(const std::string &name){
    std::string lower = toLower(trim(name));

    //strip suffix
    {
        size_t p = lower.rfind("pos");
        if(p != std::string::npos && p > 0){
            bool allDigits = true;
            for(size_t i = p + 3; i < lower.size(); i++){
                if(!isdigit((unsigned char)lower[i])){ allDigits = false; break; }
            }
            if(allDigits && (p + 3) < lower.size()){
                lower = lower.substr(0, p);
                //trim trailing whitespace
                while(!lower.empty() && (lower.back() == ' ' || lower.back() == '_')){
                    lower.pop_back();
                }
            }
        }
    }

    std::replace(lower.begin(), lower.end(), ' ', '_');

    //remove parentheses
    size_t parentPos = lower.find('(');
    if(parentPos != std::string::npos){
        lower = lower.substr(0, parentPos);
        lower=trim(lower);

        while(!lower.empty() && lower.back() == '_'){
            lower.pop_back();
        }
    }

    //exact match
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        if(lower == CSV_SCENT_CLASS_NAMES[i]){
            return (scent_class_t)i;
        }
    }

    //keyword matching (order matters: check combined terms first)
    bool hasDecaf = (lower.find("decaf") != std::string::npos);
    bool hasTea = (lower.find("tea") != std::string::npos);
    bool hasCoffee = (lower.find("coffee") != std::string::npos);

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
    
    return SCENT_CLASS_UNKNOWN;
}

bool CSVLoader::load(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    if(_samples) {
        delete[] _samples;
    }
    _classCounts.clear();
    _sessionIds.clear();
    _gateDeviations.clear();
    _calibrationDeviations.clear();
    _hasTimestamps = false;
    _capacity = 50000;
    _samples = new csv_training_sample_t[_capacity];
    _count = 0;
    
    std::string line;
    bool isHeader = true;
    int lineNum = 0, skipped = 0;

    //one block per run
    std::vector<std::vector<RawSensorData>> calibrationBlocks;
    bool inCalibration = false;
    std::vector<RawSensorData> ambientRows;

    struct ParsedRow {
        scent_class_t label;
        RawSensorData raw;
        int session;
        int calBlock; //most recent calibration
    };
    std::vector<ParsedRow> rows;
    
    //col
    int labelCol = 1;
    int featureStartCol = 2;
    int numGasSteps = 1;
    int timestampCol = -1;

    //session tracking
    int sessionId = -1;
    long long prevTimestamp = -1;
    std::string prevLabel;
    
    while(std::getline(file, line)){
        lineNum++;
        line = trim(line);
        if(line.empty()) continue;
        
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while(std::getline(ss, token, ',')) tokens.push_back(trim(token));
        
        if(isHeader) {
            isHeader = false;
            
            //detect format
            for(size_t i = 0; i < tokens.size(); i++){
                std::string col = toLower(tokens[i]);
                if(col == "label"){
                    labelCol = i;
                }
                if(col=="temp1"){
                    featureStartCol = i;
                }
                if(col=="timestamp"){
                    timestampCol = i;
                }
            }
            _hasTimestamps = (timestampCol >= 0);
            
            int gasCount = 0;
            for(size_t i = featureStartCol; i < tokens.size(); i++){
                std::string col = toLower(tokens[i]);
                if(col.find("gas1_") != std::string::npos) gasCount++;
            }
            numGasSteps =(gasCount > 1)? gasCount : 1;
            
            std::cout << "Detected format: " << numGasSteps<< " gas steps per sensor" << std::endl;
            continue;
        }
        
        //parse cols
        int colsPerSensor = 3 + numGasSteps;
        int requiredCols = featureStartCol + 2 * colsPerSensor;
        
        if((int)tokens.size()<requiredCols) { skipped++; continue; }
        
        //parse label
        std::string rawLabel = toLower(tokens[labelCol]);

        long long timestamp = -1;
        if(timestampCol >= 0 && timestampCol < (int)tokens.size()){
            try { timestamp = std::stoll(tokens[timestampCol]); }
            catch(...) { timestamp = -1; }
        }
        if(sessionId < 0 || rawLabel != prevLabel || (timestamp >= 0 && timestamp < prevTimestamp)){
            sessionId++;
        }
        prevLabel = rawLabel;
        prevTimestamp = timestamp;
        bool isCalibration = (rawLabel == "calibration" || rawLabel == "baseline");

        if(!isCalibration){
            inCalibration = false;
        }
        
        scent_class_t label = isCalibration ? SCENT_CLASS_UNKNOWN : getClassFromName(tokens[labelCol]);
        
        if(!isCalibration && label==SCENT_CLASS_UNKNOWN) {
            skipped++; continue;
        }
        
        //parse raw
        RawSensorData raw{};
        bool validFeatures = true;
        int col = featureStartCol;
        
        auto tryParse = [&](float& out) -> bool {
            if(col >= (int)tokens.size()) return false;
            try { out = std::stof(tokens[col++]); return true; }
            catch(...) { return false; }
        };
        
        if(!tryParse(raw.temp1)){ 
            skipped++; 
            continue; 
        }
        if(!tryParse(raw.hum1)) { 
            skipped++; 
            continue; 
        }
        if(!tryParse(raw.pres1)){ 
            skipped++; 
            continue; 
        }
        for(int g = 0; g < numGasSteps; g++){
            if(!tryParse(raw.gas1[g])){ 
                validFeatures = false; 
                break; 
            }
        }


        //fill remaining steps with step 0 if old format
        for(int g = numGasSteps; g < NUM_HEATER_STEPS; g++){
            raw.gas1[g] = raw.gas1[0];
        }
        
        if(!tryParse(raw.temp2)){ 
            skipped++; 
            continue; 
        }
        if(!tryParse(raw.hum2)){ 
            skipped++; 
            continue; 
        }
        if(!tryParse(raw.pres2)){ 
            skipped++; 
            continue; 
        }

        for(int g = 0; g < numGasSteps; g++){
            if(!tryParse(raw.gas2[g])){ 
                validFeatures = false; 
                break; 
            }
        }
        for(int g = numGasSteps; g < NUM_HEATER_STEPS; g++) {
            raw.gas2[g] = raw.gas2[0];
        }
        
        if(!validFeatures){ 
            skipped++; 
            continue; 
        }
        
        if(isCalibration){
            if(!inCalibration){
                calibrationBlocks.emplace_back();
                inCalibration = true;
            }
            calibrationBlocks.back().push_back(raw);
            continue;
        }

        if(label == SCENT_CLASS_AMBIENT && (int)ambientRows.size() < BASELINE_SWEEPS){
            ambientRows.push_back(raw);
        }
        rows.push_back({label, raw, sessionId, (int)calibrationBlocks.size() - 1});
    }

    //first ambient sweeps give a single baseline for the whole file
    std::vector<Baseline> baselines;
    Baseline fallback{};
    if(!calibrationBlocks.empty()){
        std::cout << "Baseline: " << calibrationBlocks.size() << " calibration block(s), sweeps per block:";
        for(const auto &block : calibrationBlocks){
            baselines.push_back(computeBaseline(block));
            std::cout << " " << block.size();
        }
        std::cout << std::endl;
        fallback = baselines.front();

        int before = 0;
    
        for(const ParsedRow &row : rows){
            if(row.calBlock < 0) before++;
        }

        if(before > 0){
            std::cout << "Warning: " << before << " rows come before the first calibration block; they use that block's baseline" << std::endl;
        }
    }

    else if(!ambientRows.empty()){
        fallback = computeBaseline(ambientRows);
        std::cout << "Baseline: mean of the first " << ambientRows.size() << " ambient sweeps" << std::endl;
        if((int)ambientRows.size() < BASELINE_SWEEPS){
            std::cout << "Warning: fewer than " << BASELINE_SWEEPS << " ambient sweeps available for the baseline" << std::endl;
        }
    }

    else{
        for(int i=0;i<NUM_HEATER_STEPS;i++){
            fallback.gas1_steps[i]=1.0f;
            fallback.gas2_steps[i]=1.0f;
        }
        std::cout << "Warning: No ambient/calibration rows found, using neutral baseline 1.0.\n";
    }

    auto baselineOf = [&](int block) -> const Baseline& {
        return (block >= 0 && block < (int)baselines.size()) ? baselines[block] : fallback;
    };

    //
    auto gateDeviation = [](const RawSensorData &raw, const Baseline &b) -> float {
        float total = 0.0f;
        int n = 0;
        for(int i=0;i<NUM_HEATER_STEPS;i++){
            if(b.gas1_steps[i] >= 1.0f){ total += fabsf(raw.gas1[i] / b.gas1_steps[i] - 1.0f); n++; }
            if(b.gas2_steps[i] >= 1.0f){ total += fabsf(raw.gas2[i] / b.gas2_steps[i] - 1.0f); n++; }
        }
        return (n > 0) ? total / n : 0.0f;
    };

    //each calibration sweep against the baseline
    _calibrationDeviations.clear();
    if(!calibrationBlocks.empty()){
        for(size_t k = 0; k < calibrationBlocks.size(); k++){
            for(const RawSensorData &raw : calibrationBlocks[k]){
                _calibrationDeviations.push_back(gateDeviation(raw, baselines[k]));
            }
        }
    }
    else{
        for(const RawSensorData &raw : ambientRows){
            _calibrationDeviations.push_back(gateDeviation(raw, fallback));
        }
    }

    for(const ParsedRow &row : rows){
        if(_count >= _capacity){
            break;
        }
        const Baseline &baseline = baselineOf(row.calBlock);
        csv_training_sample_t sample{};
        sample.label = row.label;
        applyMultiStepTransform(sample, row.raw, baseline);
        _samples[_count++] = sample;
        _classCounts[row.label]++;
        _sessionIds.push_back(row.session);
        _gateDeviations.push_back(gateDeviation(row.raw, baseline));
    }

    file.close();
    std::cout << "Loaded " << _count << " samples (" << numGasSteps << " gas steps/sensor, " << skipped << " skipped)" << std::endl;
    if(_hasTimestamps){
        std::cout << "Detected " << (sessionId + 1) << " collection runs (label change or timestamp reset)" << std::endl;
    }
    return _count > 0;
}

void CSVLoader::split(float ratio, uint32_t seed, bool bySession, csv_training_sample_t*& trainSet, uint16_t &trainCount, csv_training_sample_t*& testSet, uint16_t &testCount){
    //group by class
    std::map<scent_class_t, std::vector<uint16_t>> classIndices;
    for(uint16_t i = 0; i < _count; i++){
        classIndices[_samples[i].label].push_back(i);
    }

    std::mt19937 g(seed);

    std::vector<uint16_t> trainIdx, testIdx;

    if(bySession && !_hasTimestamps){
        std::cout << "WARNING: no timestamp column, falling back to random split" << std::endl;
        bySession = false;
    }

    if(bySession){
        std::cout << "Session-wise split (no run is shared between train and test)" << std::endl;
        for(auto &c : classIndices){
            std::vector<uint16_t> &indices = c.second;
            std::vector<int> runs;
            std::map<int, uint16_t> runSize;
            for(uint16_t idx : indices){
                int sid = _sessionIds[idx];
                if(runSize.count(sid) == 0){
                    runs.push_back(sid);
                }
                runSize[sid]++;
            }

            uint16_t target = (uint16_t)(indices.size() * ratio);
            std::set<int> trainRuns;
            uint16_t cum = 0;
            for(size_t r = 0; r < runs.size(); r++){
                //the first run is always used for training; stop at the first run that overshoots
                if(r > 0 && cum + runSize[runs[r]] > target){
                    break;
                }
                trainRuns.insert(runs[r]);
                cum += runSize[runs[r]];
            }
            if(trainRuns.size() == runs.size()){
                std::cout << "WARNING: " << getClassName(c.first) << " has only " << runs.size() << " run(s), none left for testing" << std::endl;
            }

            std::cout << "  " << getClassName(c.first) << " runs:";
            for(int sid : runs){
                std::cout << " " << runSize[sid] << (trainRuns.count(sid) ? "(train)" : "(test)");
            }
            std::cout << std::endl;

            for(uint16_t idx : indices){
                if(trainRuns.count(_sessionIds[idx])){
                    trainIdx.push_back(idx);
                }
                else{
                    testIdx.push_back(idx);
                }
            }
        }
    }
    
    else{
        for(auto &i : classIndices){
            //split each class
            std::vector<uint16_t> &indices = i.second;
            std::shuffle(indices.begin(), indices.end(), g);

            uint16_t classTrain=(uint16_t)(indices.size() * ratio);

            //ensure atleast 1 test per class
            if(classTrain>=indices.size() &&indices.size()>1){
                classTrain=indices.size()-1;
            }

            for(uint16_t i=0; i<classTrain;i++){
                trainIdx.push_back(indices[i]);
            }
            for(uint16_t i=classTrain; i<indices.size();i++){
                testIdx.push_back(indices[i]);
            }
        }
    }

    std::shuffle(trainIdx.begin(), trainIdx.end(), g);
    std::shuffle(testIdx.begin(), testIdx.end(), g);

    //alloc
    trainCount=trainIdx.size();
    testCount=testIdx.size();
    trainSet = new csv_training_sample_t[trainCount];
    testSet = new csv_training_sample_t[testCount];

    //copy
    for (uint16_t i = 0; i < trainCount; i++) {
        trainSet[i] = _samples[trainIdx[i]];
    }
    for (uint16_t i = 0; i < testCount; i++) {
        testSet[i] = _samples[testIdx[i]];
    }

    std::cout << "Split into " << trainCount << " training samples and " << testCount << " testing samples." << std::endl;


    //print per class split
    std::map<scent_class_t, uint16_t> trainCounts,testCounts;

    for(uint16_t i=0; i<trainCount;i++){
        trainCounts[trainSet[i].label]++;
    }
    for(uint16_t i=0;i<testCount;i++){
        testCounts[testSet[i].label]++;
    }
    for(auto &i : trainCounts){
        std::cout<< getClassName(i.first)<< ": "<< i.second<< " train, "<< testCounts[i.first]<< " test" << std::endl;
    }
}


void CSVLoader::printInfo() const {
    std::cout << "\n========== Dataset Info ==========" << std::endl;
    std::cout << "Total samples: " << _count << std::endl;
    std::cout << "Features: " << CSV_FEATURE_COUNT << std::endl;
    std::cout << "\nClass distribution:" << std::endl;
    
    for (const auto& pair : _classCounts) {
        float pct = (100.0f * pair.second) / _count;
        std::cout << "[" << (int)pair.first << "]" << getClassName(pair.first) << ": " 
        << pair.second << " (" << std::fixed << std::setprecision(1)  << pct << "%)" << std::endl;
    }
    std::cout << "==================================" << std::endl;
}

void CSVLoader::printFeatureNames(){
        std::cout << "\nFeatures: " << CSV_FEATURE_COUNT << std::endl;
    int idx = 0;
    for (int i = 0; i < 10; i++)
        std::cout << "  [" << idx++ << "] gas1_norm_step" <<i << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << "  [" << idx++ << "] gas2_norm_step" <<i << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << "  [" << idx++ << "] cross_ratio_step"<< i << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << "  [" << idx++ << "] diff_step" <<i << std::endl;
    for (int i = 0; i < 9; i++)
        std::cout << "  [" << idx++ << "] gas1_delta_step" <<i << std::endl;
    for (int i = 0; i < 9; i++)
        std::cout << "  [" << idx++ << "] gas2_delta_step" <<i << std::endl;
    std::cout <<"  [" << idx++ << "] slope1_norm"<<std::endl;
    std::cout << "  [" << idx++ << "] slope2_norm" << std::endl;
    std::cout << "  [" << idx++ << "] curvature1_norm" << std::endl;
    std::cout << "  [" << idx++ << "] curvature2_norm" << std::endl;
    std::cout << "  [" << idx++ << "] auc1_norm" << std::endl;
    std::cout << "  [" << idx++ << "] auc2_norm" << std::endl;
    std::cout << "  [" << idx++ << "] peak_idx1" << std::endl;
    std::cout << "  [" << idx++ << "] peak_idx2" << std::endl;
    std::cout << "  [" << idx++ << "] range1" << std::endl;
    std::cout << "  [" << idx++ << "] range2" << std::endl;
    std::cout << "  [" << idx++ << "] late_early_ratio1" << std::endl;
    std::cout << "  [" << idx++ << "] late_early_ratio2" << std::endl;
    std::cout << "  [" << idx++ << "] cross_ratio_mean" << std::endl;
    std::cout << "  [" << idx++ << "] cross_ratio_slope" << std::endl;
    std::cout << "  [" << idx++ << "] cross_ratio_var" << std::endl;
#if USE_ENV_FEATURES
    std::cout << "  [" << idx++ << "] delta_temp" << std::endl;
    std::cout << "  [" << idx++ << "] delta_hum" << std::endl;
    std::cout << "  [" << idx++ << "] delta_pres" << std::endl;
    std::cout << "  [" << idx++ << "] temp1" << std::endl;
    std::cout << "  [" << idx++ << "] hum1" << std::endl;
    std::cout << "  [" << idx++ << "] pres1" << std::endl;
    std::cout << "  [" << idx++ << "] temp2" << std::endl;
    std::cout << "  [" << idx++ << "] hum2" << std::endl;
    std::cout << "  [" << idx++ << "] pres2" << std::endl;
#endif
}
