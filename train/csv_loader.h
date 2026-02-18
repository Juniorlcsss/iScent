#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <stdint.h>
#include <string>
#include <vector>
#include <map>

#define CSV_FEATURE_COUNT 8

//scent classes
typedef enum {
    SCENT_CLASS_PURE_CAMOMILE = 0,
    SCENT_CLASS_THOROUGHLY_MINTED_INFUSION,
    SCENT_CLASS_BERRY_BURST,
    SCENT_CLASS_DARJEELING_BLEND,
    SCENT_CLASS_DECAF_NUTMEG_VANILLA,
    SCENT_CLASS_EARL_GREY,
    SCENT_CLASS_ENGLISH_BREAKFAST_TEA,
    SCENT_CLASS_FRESH_ORANGE,
    SCENT_CLASS_GARDEN_SELECTION_LEMON,
    SCENT_CLASS_GREEN_TEA,
    SCENT_CLASS_RASPBERRY,
    SCENT_CLASS_SWEET_CHERRY,
    SCENT_CLASS_COUNT,
    SCENT_CLASS_UNKNOWN = 255
} scent_class_t;
static const char* CSV_SCENT_CLASS_NAMES[SCENT_CLASS_COUNT] = {
    "camomile",
    "thoroughly_minted_infusion",
    "berry_burst",
    "darjeeling_blend",
    "decaf_nutmeg_vanilla",
    "earl_grey",
    "english_breakfast_tea",
    "fresh_orange",
    "garden_selection_lemon",
    "green_tea",
    "raspberry",
    "sweet_cherry"
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