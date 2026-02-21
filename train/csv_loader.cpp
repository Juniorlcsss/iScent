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
#include <cmath>

namespace{
struct Baseline{
    float temp1, temp2;
    float hum1, hum2;
    float pres1, pres2;
    float gas1_steps[10];
    float gas2_steps[10];
};

static const int NUM_HEATER_STEPS = 10;
static const int RAW_COLS_PER_SENSOR = 3 + NUM_HEATER_STEPS;
static const int TOTAL_RAW_COLS = 2 * RAW_COLS_PER_SENSOR;
struct RawSensorData{
    float temp1, hum1, pres1;
    float gas1[10];
    float temp2, hum2, pres2;
    float gas2[10];
};

static float median(std::vector<float> &v){
    if(v.empty()){
        return 0.0f;
    }

    const size_t mid = v.size() /2;
    std::nth_element(v.begin(),v.begin()+mid, v.end());
    float med = v[mid];

    if(v.size()%2==0){
        std::nth_element(v.begin(),v.begin()+mid-1, v.end());
        med= 0.5f* (med+v[mid-1]);
    }
    return med;
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

    b.temp1= median(temp1v);
    b.temp2= median(temp2v);
    b.hum1= median(hum1v);
    b.hum2= median(hum2v);
    b.pres1= median(pres1v);
    b.pres2= median(pres2v);
    
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        b.gas1_steps[i]= median(gas1v[i]);
        b.gas2_steps[i]= median(gas2v[i]);
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


static void applyMultiStepTransform(csv_training_sample_t& sample, const RawSensorData& raw, const Baseline& b){
    uint16_t idx=0;
    float gas1_resp[NUM_HEATER_STEPS];
    float gas2_resp[NUM_HEATER_STEPS];

    for(int i=0; i<NUM_HEATER_STEPS;i++){
        gas1_resp[i]=raw.gas1[i]/b.gas1_steps[i];
        gas2_resp[i]=raw.gas2[i]/b.gas2_steps[i];
        sample.features[idx++]= gas1_resp[i];
    }
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        sample.features[idx++]= gas2_resp[i];
    }

    //per step cross sensor
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        sample.features[idx++] = (gas2_resp[i] > 0.01f) ? gas1_resp[i] / gas2_resp[i] : 1.0f;
    }

    //profile slopes
    sample.features[idx++]= localSlope(gas1_resp, NUM_HEATER_STEPS);
    sample.features[idx++]= localSlope(gas2_resp, NUM_HEATER_STEPS);

    //profile curvature
    int half = NUM_HEATER_STEPS / 2;
    sample.features[idx++] = localSlope(gas1_resp + half, NUM_HEATER_STEPS - half)- localSlope(gas1_resp, half);
    sample.features[idx++] = localSlope(gas2_resp + half, NUM_HEATER_STEPS - half)- localSlope(gas2_resp, half);
    //Environment differentials
    sample.features[idx++] = fabsf(raw.temp1 - raw.temp2);
    sample.features[idx++] = fabsf(raw.hum1  - raw.hum2);
    sample.features[idx++] = fabsf(raw.pres1 - raw.pres2);

    if(idx != CSV_FEATURE_COUNT){
        std::cerr << "Feature count mismatch: expected " << CSV_FEATURE_COUNT << " but got " << idx << std::endl;
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

    //keyword
    if (lower.find("camomile") != std::string::npos || 
        lower.find("chamomile") != std::string::npos) {
        return SCENT_CLASS_PURE_CAMOMILE;
    }
    if (lower.find("mint") != std::string::npos) {
        return SCENT_CLASS_THOROUGHLY_MINTED_INFUSION;
    }
    if (lower.find("berry") != std::string::npos) {
        return SCENT_CLASS_BERRY_BURST;
    }
    if (lower.find("darjeeling") != std::string::npos) {
        return SCENT_CLASS_DARJEELING_BLEND;
    }
    if (lower.find("nutmeg") != std::string::npos || 
        lower.find("vanilla") != std::string::npos ||
        lower.find("decaf") != std::string::npos) {
        return SCENT_CLASS_DECAF_NUTMEG_VANILLA;
    }
    if (lower.find("earl") != std::string::npos) {
        return SCENT_CLASS_EARL_GREY;
    }
    if (lower.find("breakfast") != std::string::npos) {
        return SCENT_CLASS_ENGLISH_BREAKFAST_TEA;
    }
    if (lower.find("orange") != std::string::npos) {
        return SCENT_CLASS_FRESH_ORANGE;
    }
    if (lower.find("lemon") != std::string::npos || 
        lower.find("garden") != std::string::npos) {
        return SCENT_CLASS_GARDEN_SELECTION_LEMON;
    }
    if (lower.find("green") != std::string::npos) {
        return SCENT_CLASS_GREEN_TEA;
    }
    if (lower.find("raspberry") != std::string::npos) {
        return SCENT_CLASS_RASPBERRY;
    }
    if (lower.find("cherry") != std::string::npos) {
        return SCENT_CLASS_SWEET_CHERRY;
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
    _capacity = 50000;
    _samples = new csv_training_sample_t[_capacity];
    _count = 0;
    
    std::string line;
    bool isHeader = true;
    int lineNum = 0, skipped = 0;
    bool baselineReady = false;
    Baseline baseline{};
    std::vector<RawSensorData> calibrationRows;
    
    //col
    int labelCol = 1;
    int featureStartCol = 2;
    int numGasSteps = 1;
    
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
            }
            
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
        bool isCalibration = (rawLabel == "calibration" || rawLabel == "baseline");
        bool isAmbient = (rawLabel == "ambient");
        scent_class_t label = (isCalibration || isAmbient)? SCENT_CLASS_UNKNOWN : getClassFromName(tokens[labelCol]);
        
        if(!isCalibration&& !isAmbient&&label==SCENT_CLASS_UNKNOWN) {
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
        
        if(isCalibration || isAmbient){
            calibrationRows.push_back(raw);
            baselineReady = false;
            continue;
        }
        
        if(!baselineReady){
            if(calibrationRows.empty()){ 
                skipped++; 
                continue; 
            }

            baseline = computeBaseline(calibrationRows);
            calibrationRows.clear();
            baselineReady = true;
        }
        
        csv_training_sample_t sample{};
        sample.label = label;
        applyMultiStepTransform(sample, raw, baseline);
        
        if(_count < _capacity){
            _samples[_count++] = sample;
            _classCounts[label]++;
        } 
        else {
            break;
        }
    }
    
    file.close();
    std::cout << "Loaded " << _count << " samples (" << numGasSteps << " gas steps/sensor, " << skipped << " skipped)" << std::endl;
    return _count > 0;
}

void CSVLoader::split(float ratio, csv_training_sample_t*& trainSet, uint16_t &trainCount, csv_training_sample_t*& testSet, uint16_t &testCount){
    //group by class
    std::map<scent_class_t, std::vector<uint16_t>> classIndices;
    for(uint16_t i = 0; i < _count; i++){
        classIndices[_samples[i].label].push_back(i);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<uint16_t> trainIdx, testIdx;

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