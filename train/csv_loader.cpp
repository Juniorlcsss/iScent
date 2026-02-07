#include "csv_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>
#include <cctype>
#include <iomanip>

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

bool CSVLoader::load(const std::string &filename){
    std::ifstream file(filename);

    if(!file.is_open()){
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    //clean
    if(_samples){
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

    int labelCol = 1;
    int featureStartCol = 2;

    while(std::getline(file,line)){
        lineNum++;
        line=trim(line);

        if(line.empty()){
            continue;
        }

        //parse
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while(std::getline(ss,token,',')){
            tokens.push_back(trim(token));
        }

        //handle head
        if(isHeader){
            isHeader=false;

            //get label and feature cols
            for(size_t i=0; i<tokens.size(); i++){
                std::string col = toLower(tokens[i]);

                if(col == "label"){
                    labelCol=i;
                }
                if(col == "temp1"){
                    featureStartCol=i;
                }
            }

            std::cout << "Detected label column: " << labelCol << std::endl;
            std::cout << "Detected feature start column: " << featureStartCol << std::endl;
            continue;
        }

        //need label and 12f
        if(tokens.size() < (size_t)(featureStartCol + CSV_FEATURE_COUNT)){
            std::cerr << "Skipping line " << lineNum << ": not enough columns" << std::endl;
            skipped++;
            continue;
        }

        //parse label
        scent_class_t label = getClassFromName(tokens[labelCol]);
        if(label == SCENT_CLASS_UNKNOWN){
            std::cerr << "Skipping line " << lineNum << ": unknown label '" << tokens[labelCol] << "'" << std::endl;
            skipped++;
            continue;
        }

        //parse feats
        csv_training_sample_t sample;
        sample.label = label;

        bool valid=true;
        for(int i=0; i<CSV_FEATURE_COUNT;i++){
            try{
                sample.features[i] = std::stof(tokens[featureStartCol + i]);
            }
            catch(...){
                valid=false;
                std::cerr << "Skipping line " << lineNum << ": invalid feature '" << tokens[featureStartCol + i] << "'" << std::endl;
                break;
            }
        }

        if(!valid){
            skipped++;
            continue;
        }

        //add sample
        if(_count < _capacity){
            _samples[_count++] =sample;
            _classCounts[label]++;
        }
        else{
            std::cerr << "Reached max capacity of " << _capacity << " samples. Stopping load." << std::endl;
            break;
        }
    }
    file.close();

    std::cout << "Loaded " << _count << " samples, skipped " << skipped << " lines." << std::endl;
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