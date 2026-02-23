#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <stdint.h>
#include <string>
#include <vector>
#include <map>


#define NUM_HEATER_STEPS 10
#define USE_ENV_FEATURES false
#if USE_ENV_FEATURES
    #define CSV_FEATURE_COUNT 76
#else
    #define CSV_FEATURE_COUNT 73
#endif

//scent classes
typedef enum {
    SCENT_CLASS_DECAF_TEA = 0,
    SCENT_CLASS_DECAF_COFFEE,
    SCENT_CLASS_TEA,
    SCENT_CLASS_COFFEE,
    SCENT_CLASS_COUNT,
    SCENT_CLASS_UNKNOWN = 255
} scent_class_t;

static const char* CSV_SCENT_CLASS_NAMES[SCENT_CLASS_COUNT] = {
    "decaf_tea",
    "decaf_coffee",
    "tea",
    "coffee"
};

//training sample
typedef struct{
    scent_class_t label;
    float features[CSV_FEATURE_COUNT];
} csv_training_sample_t;

//metrics
typedef struct{
    uint16_t correct;
    uint16_t total;
    float accuracy;
    uint16_t confusionMatrix[SCENT_CLASS_COUNT][SCENT_CLASS_COUNT];
}csv_metrics_t;


class CSVLoader{
public:
    CSVLoader();
    ~CSVLoader();

    //load data
    bool load(const std::string& filename);

    //split data
    void split(float ratio, csv_training_sample_t*& trainSet, uint16_t& trainCount,
        csv_training_sample_t*& testSet, uint16_t& testCount);

    uint16_t getSampleCount() const { return _count; }
    csv_training_sample_t* getSamples() const { return _samples; }

    void printInfo() const;
    static void printFeatureNames();

    static const char* getClassName(scent_class_t id);
    static scent_class_t getClassFromName(const std::string& name);

private:
    csv_training_sample_t* _samples;
    uint16_t _count;
    uint16_t _capacity;
    std::map<scent_class_t, uint16_t> _classCounts;
    static std::string trim(const std::string& str);
    static std::string toLower(const std::string& str);
};

#endif