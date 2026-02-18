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
    float temp1;
    float temp2;
    float hum1;
    float hum2;
    float pres1;
    float pres2;
    float gas1;
    float gas2;
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

static Baseline computeBaseline(const std::vector<std::array<float,8>> &cal){
    Baseline b{};
    if(cal.empty()){
        return b;
    }

    std::vector<float> temp1v, temp2v,hum1v,hum2v,pres1v,pres2v,gas1v,gas2v;
    temp1v.reserve(cal.size());
    temp2v.reserve(cal.size());
    hum1v.reserve(cal.size());
    hum2v.reserve(cal.size());
    pres1v.reserve(cal.size());
    pres2v.reserve(cal.size());
    gas1v.reserve(cal.size());
    gas2v.reserve(cal.size());

    for(const auto &sample:cal){
        temp1v.push_back(sample[0]);
        hum1v.push_back(sample[1]);
        pres1v.push_back(sample[2]);
        gas1v.push_back(sample[3]);
        temp2v.push_back(sample[4]);
        hum2v.push_back(sample[5]);
        pres2v.push_back(sample[6]);
        gas2v.push_back(sample[7]);
    }

    b.temp1= median(temp1v);
    b.temp2= median(temp2v);
    b.hum1= median(hum1v);
    b.hum2= median(hum2v);
    b.pres1= median(pres1v);
    b.pres2= median(pres2v);
    b.gas1= median(gas1v);
    b.gas2= median(gas2v);

    if(b.gas1==0.0f){
        b.gas1=1.0f;
    }
    if(b.gas2==0.0f){
        b.gas2=1.0f;
    }

    return b;
    }

static void applyBaselineTransform(csv_training_sample_t& sample, const Baseline& b){
    const float gas1_raw = sample.features[3];
    const float gas2_raw = sample.features[7];

    //gas response ratios
    float gas1_response = (b.gas1 > 0) ? gas1_raw / b.gas1 : 1.0f;
    float gas2_response = (b.gas2 > 0) ? gas2_raw / b.gas2 : 1.0f;
    //cross-sensor gas ratio
    float gas_cross_ratio = (gas2_raw > 0) ? gas1_raw / gas2_raw : 1.0f;

    //ABSOLUTE gas
    float gas_diff_abs = fabsf(gas1_response - gas2_response);

    //ABSOLUTE differentials
    float delta_temp_abs = fabsf(sample.features[0] - sample.features[4]);
    float delta_hum_abs  = fabsf(sample.features[1] - sample.features[5]);
    float delta_pres_abs = fabsf(sample.features[2] - sample.features[6]);

    float log_gas_cross = fabsf(logf(gas_cross_ratio > 0 ? gas_cross_ratio : 1e-6f));

    sample.features[0] = gas1_response;
    sample.features[1] = gas2_response;
    sample.features[2] = gas_cross_ratio;
    sample.features[3] = gas_diff_abs;
    sample.features[4] = delta_temp_abs;
    sample.features[5] = delta_hum_abs;
    sample.features[6] = delta_pres_abs;
    sample.features[7] = log_gas_cross;
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
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    //Clean up
    if (_samples) {
        delete[] _samples;
    }
    _classCounts.clear();
    
    //alloc
    _capacity = 50000;
    _samples = new csv_training_sample_t[_capacity];
    _count = 0;
    
    std::string line;
    bool isHeader = true;
    int lineNum = 0;
    int skipped = 0;
    int calibrationCount = 0;
    int ambientBlocks = 0;
    bool baselineReady = false;
    Baseline baseline{};
    std::vector<std::array<float, 8>> calibrationRows;
    
    int labelCol = 1;
    int featureStartCol = 2;
    
    while (std::getline(file, line)) {
        lineNum++;
        line = trim(line);
        
        if (line.empty()) continue;
        
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(trim(token));
        }
        
        //handle header
        if (isHeader) {
            isHeader = false;
            
            //find label and feature columns
            for (size_t i = 0; i < tokens.size(); i++) {
                std::string col = toLower(tokens[i]);
                if (col == "label") {
                    labelCol = i;
                }
                if (col == "temp1") {
                    featureStartCol = i;
                }
            }
            
            std::cout << "Header: label at col " << labelCol<< ", features start at col " << featureStartCol << std::endl;
            continue;
        }
        
        const int requiredFeatures = featureStartCol + 8;
        if (tokens.size() < (size_t)requiredFeatures) {
            skipped++;
            continue;
        }
        
        //Parse label 
        std::string rawLabel = toLower(tokens[labelCol]);
        bool isCalibration = (rawLabel == "calibration" || rawLabel == "baseline");
        bool isAmbient = (rawLabel == "ambient");
        scent_class_t label = (isCalibration || isAmbient) ? SCENT_CLASS_UNKNOWN : getClassFromName(tokens[labelCol]);
        if (!isCalibration && !isAmbient && label == SCENT_CLASS_UNKNOWN) {
            std::cerr << "Warning: Unknown label '" << tokens[labelCol]<< "' at line " << lineNum << std::endl;
            skipped++;
            continue;
        }
        
        //parse raw features
        float raw[8]{};
        bool validFeatures = true;
        for (int i = 0; i < 8; i++) {
            size_t tokIdx = featureStartCol + i;
            if (tokIdx >= tokens.size()) { validFeatures = false; break; }
            try {
                raw[i] = std::stof(tokens[tokIdx]);
            } catch (...) {
                validFeatures = false;
                break;
            }
        }

        if (!validFeatures) { skipped++;     continue; }

        if (isCalibration || isAmbient) {
            calibrationRows.push_back({raw[0],raw[1],raw[2],raw[3],raw[4],raw[5],raw[6],raw[7]});
            if (isCalibration) {
                calibrationCount++;
            }
            if (isAmbient) {
                ambientBlocks++;
            }
            baselineReady = false;
            continue;
        }

        //ensure we have a baseline
        if (!baselineReady) {
            if (calibrationRows.empty()) {
                std::cerr <<"Warning: No ambient/calibration rows before line "<<lineNum<< "; skipping sample" << std::endl;
                skipped++;
                continue;
            }
            baseline = computeBaseline(calibrationRows);
            calibrationRows.clear();
            baselineReady = true;
        }

        csv_training_sample_t sample{};
        sample.label = label;
        sample.features[0] = raw[0];
        sample.features[1] = raw[1];
        sample.features[2] = raw[2];
        sample.features[3] = raw[3];
        sample.features[4] = raw[4];
        sample.features[5] = raw[5];
        sample.features[6] = raw[6];
        sample.features[7] = raw[7];

        applyBaselineTransform(sample, baseline);

        if (_count < _capacity) {
            _samples[_count++] = sample;
            _classCounts[label]++;
        } else {
            std::cerr << "Warning: Maximum capacity reached" << std::endl;
            break;
        }
    }
    file.close();
    
    std::cout << "Loaded " << _count << " samples";
    if (calibrationCount == 0 && ambientBlocks == 0) {
        std::cout << " (no baseline rows found; data may be invalid)";
    }
    if (ambientBlocks > 0) {
        std::cout << " | ambient blocks: " << ambientBlocks;
    }
    if (calibrationCount > 0) {
        std::cout << " | calibration rows: " << calibrationCount;
    }
    if (skipped > 0) {
        std::cout << " (skipped " << skipped << " invalid rows)";
    }
    std::cout << std::endl;
    
    return _count > 0;
}

void CSVLoader::split(float ratio, csv_training_sample_t*& trainSet, uint16_t &trainCount, csv_training_sample_t*& testSet, uint16_t &testCount){
    //shuffled indicies
    std::vector<uint16_t> indicies(_count);
    for(uint16_t i=0; i<_count; i++){
        indicies[i]=i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indicies.begin(), indicies.end(), g);

    //calc split
    trainCount = (uint16_t)(_count*ratio);
    testCount = _count - trainCount;

    //alloc
    trainSet = new csv_training_sample_t[trainCount];
    testSet = new csv_training_sample_t[testCount];

    //copy
    for (uint16_t i = 0; i < trainCount; i++) {
        trainSet[i] = _samples[indicies[i]];
    }
    for (uint16_t i = 0; i < testCount; i++) {
        testSet[i] = _samples[indicies[trainCount + i]];
    }

    std::cout << "Split into " << trainCount << " training samples and " << testCount << " testing samples." << std::endl;
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