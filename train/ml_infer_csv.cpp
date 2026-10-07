//device header parity check
//runs the converted model headers exactly as src/ml_inference.cpp does (Fisher on robust-scaled
//features, selection + (x - median) / IQR, DT/KNN/RF with confidence, deployed ensemble) over the
//holdout features written by ml_trainer (holdout_check.csv) and compares with the trainer's predictions.
//
//build (headers dir = convert_model.py output, e.g. "include/model headers"):
//  g++ -std=c++17 -O2 -I"include/model headers" train/ml_infer_csv.cpp -o ml_infer_csv.exe
//run:
//  .\ml_infer_csv.exe output\holdout_check.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "feature_select.h"
#include "feature_stats.h"
#include "fisher_weights.h"
#include "dt_model_header.h"
#include "knn_model_header.h"
#include "rf_model_header.h"
#include "ensemble_weights.h"

static const int CLASS_COUNT = 5;
static const uint8_t UNKNOWN = 255;
static const float CONFIDENCE_THRESHOLD = 0.30f;
static const char* CLASS_NAMES[CLASS_COUNT] = {"decaf tea", "tea", "decaf coffee", "coffee", "ambient"};

#if FULL_FEATURE_COUNT < (FISHER_BASE_FEATURE_COUNT + FISHER_PAIR_COUNT)
    #error "feature_stats.h FULL_FEATURE_COUNT is inconsistent with fisher_weights.h"
#endif
#if DT_FEATURE_COUNT > SELECTED_FEATURE_COUNT || KNN_FEATURE_COUNT > SELECTED_FEATURE_COUNT || RF_FEATURE_COUNT > SELECTED_FEATURE_COUNT
    #error "a model header uses more features than SELECTED_FEATURE_COUNT"
#endif

struct Row {
    int label;
    int dtPred, knnPred, rfPred, ensPred;
    float dtConf, knnConf, rfConf, ensConf;
    std::vector<float> features; //unscaled base + engineered features
};


static void devicePipeline(const std::vector<float>& engineered, float* selected){
    float all[FISHER_BASE_FEATURE_COUNT + FISHER_PAIR_COUNT] = {0};
    for(int i = 0; i < FISHER_BASE_FEATURE_COUNT && i < (int)engineered.size(); i++){
        all[i] = engineered[i];
    }

    float scaledForFisher[FISHER_BASE_FEATURE_COUNT];
    for(int i = 0; i < FISHER_BASE_FEATURE_COUNT; i++){
        float denom = FEATURE_IQRS[i];
        if(fabsf(denom) < 1e-6f){
            denom = 1.0f;
        }
        scaledForFisher[i] = (all[i] - FEATURE_MEDIANS[i]) / denom;
    }
    for(int p = 0; p < FISHER_PAIR_COUNT; p++){
        float proj = 0.0f;
        for(int i = 0; i < FISHER_BASE_FEATURE_COUNT; i++){
            proj += scaledForFisher[i] * FISHER_WEIGHTS[p][i];
        }
        all[FISHER_BASE_FEATURE_COUNT + p] = proj;
    }

    for(int i = 0; i < SELECTED_FEATURE_COUNT; i++){
        float val = all[SELECTED_INDICES[i]];
        if(SELECTED_IQRS[i] > 1e-6f){
            val = (val - SELECTED_MEDIANS[i]) / SELECTED_IQRS[i];
        }
        else{
            val = val - SELECTED_MEDIANS[i];
        }
        selected[i] = val;
    }
}

//ensemble
static uint8_t deviceEnsemble(uint8_t dtCls,float dtConf, uint8_t knnCls, float knnConf,uint8_t rfCls, float rfConf, float& confOut){
    if(dtConf < CONFIDENCE_THRESHOLD){
        dtCls =UNKNOWN;
        dtConf = 0.0f;
    }
    if(knnConf < CONFIDENCE_THRESHOLD){
        knnCls =UNKNOWN;
        knnConf = 0.0f;
    }
    if(rfConf < CONFIDENCE_THRESHOLD){
        rfCls =UNKNOWN;
        rfConf = 0.0f;
    }

    float scores[CLASS_COUNT] = {0};
    auto vote = [&](uint8_t cls, float conf, float weight){
        if(cls>=CLASS_COUNT){
            return;
        }
        scores[cls] += weight * conf;
        float resid = weight * (1.0f - conf) / (CLASS_COUNT - 1);

        for(int c = 0; c < CLASS_COUNT; c++){
            if(c != cls){
                scores[c] += resid;
            }
        }
    };
    vote(knnCls, knnConf, KNN_WEIGHT);
    vote(rfCls, rfConf, RF_WEIGHT);
    vote(dtCls, dtConf, DT_WEIGHT);

    float bScore = 0.0f;
    uint8_t bClass = UNKNOWN;
    for(int c = 0; c < CLASS_COUNT; c++){
        if(scores[c] > bScore){
            bScore = scores[c];
            bClass = c;
        }
    }

    bool dtKnn = (dtCls == knnCls) && (dtCls != UNKNOWN);
    bool knnRf = (knnCls == rfCls) && (knnCls != UNKNOWN);
    bool dtRf = (dtCls == rfCls) && (dtCls != UNKNOWN);

    if(!dtKnn && !knnRf && !dtRf){
        confOut = 0.0f;
        return UNKNOWN;
    }
    float maxScore = DT_WEIGHT + KNN_WEIGHT + RF_WEIGHT;
    confOut = (maxScore > 0.0f) ? bScore / maxScore : 0.0f;
    if(confOut < CONFIDENCE_THRESHOLD){
        return UNKNOWN;
    }
    return bClass;
}

static bool loadRows(const char* path, std::vector<Row>& rows){
    std::ifstream f(path);
    if(!f.is_open()){
        std::cerr << "Could not open " << path << std::endl;
        return false;
    }
    std::string line;
    std::getline(f, line);
    while(std::getline(f, line)){
        if(line.empty()){
            continue;
        }


        std::stringstream ss(line);
        std::string tok;
        std::vector<float> v;

        while(std::getline(ss, tok, ',')){
            v.push_back(std::stof(tok));
        }


        if(v.size()<9){
            continue;
        }

        Row r;
        r.label = (int)v[0];
        r.dtPred = (int)v[1]; r.dtConf = v[2];
        r.knnPred = (int)v[3]; r.knnConf = v[4];
        r.rfPred = (int)v[5]; r.rfConf = v[6];
        r.ensPred = (int)v[7]; r.ensConf = v[8];
        r.features.assign(v.begin() + 9, v.end());
        rows.push_back(r);
    }
    return true;
}

int main(int argc, char** argv){
    const char* path = (argc > 1) ? argv[1] : "holdout_check.csv";
    std::vector<Row> rows;
    if(!loadRows(path, rows) || rows.empty()){
        std::cerr << "No rows loaded" << std::endl;
        return 1;
    }
    if((int)rows[0].features.size() != FISHER_BASE_FEATURE_COUNT){
        std::cerr << "holdout_check.csv has " << rows[0].features.size() << " features, headers expect " << FISHER_BASE_FEATURE_COUNT << std::endl;
        return 1;
    }

    std::cout << "Headers: " << SELECTED_FEATURE_COUNT << " selected of " << FULL_FEATURE_COUNT<< "; DT uses " << DT_FEATURE_COUNT << ", KNN " << KNN_FEATURE_COUNT << " (k=" << KNN_K << ", " << KNN_SAMPLE_COUNT << " samples), RF " << RF_FEATURE_COUNT << " (" << RF_NUM_TREES << " trees)" << std::endl;
    std::cout << "Rows: " << rows.size() << std::endl;

    const char* names[4] = {"DT", "KNN", "RF", "Ensemble"};
    int mismatch[4] = {0}, correct[4] = {0}, unknown[4] = {0};
    float maxConfDiff[4] = {0};
    int shown = 0;

    for(const Row& r : rows){
        float x[SELECTED_FEATURE_COUNT];
        devicePipeline(r.features, x);

        float c[4];
        uint8_t p[4];
        p[0] = dt_predict_with_confidence(x, &c[0]);
        p[1] = knn_predict_with_confidence(x, &c[1]);
        p[2] = rf_predict_with_confidence(x, &c[2]);
        p[3] = deviceEnsemble(p[0], c[0], p[1], c[1], p[2], c[2], c[3]);

        const int ref[4] = {r.dtPred, r.knnPred, r.rfPred, r.ensPred};
        const float refConf[4] = {r.dtConf, r.knnConf, r.rfConf, r.ensConf};
        for(int m = 0; m < 4; m++){
            if((int)p[m] != ref[m]){
                mismatch[m]++;
                if(shown < 10){
                    std::cout << "  mismatch " << names[m] << ": device=" << (int)p[m] << " (" << c[m] << ") trainer=" << ref[m] << " (" << refConf[m] << ")" << std::endl;
                    shown++;
                }
            }
            maxConfDiff[m] = std::max(maxConfDiff[m], fabsf(c[m] - refConf[m]));
            if((int)p[m] == r.label){
                correct[m]++;
            }

            if(p[m] >= CLASS_COUNT){
                unknown[m]++;
            }
        }
    }

    bool ok = true;
    std::cout << "\nModel      agree with trainer   max |conf diff|   accuracy   unknown" << std::endl;
    for(int m = 0; m < 4; m++){
        std::cout << "  " << std::left << std::setw(9) << names[m] << std::right
        << std::setw(6) << (rows.size() - mismatch[m]) << "/" << rows.size() << std::setw(18) << std::fixed << std::setprecision(6) << maxConfDiff[m]
        << std::setw(10) << std::setprecision(1) << (100.0f * correct[m] / rows.size()) << "%" << std::setw(9) << unknown[m] << std::endl;
        
        
        if(mismatch[m] > 0){
            ok = false;
        }
    }
    std::cout << (ok ? "\nPASS: device headers reproduce the trainer's holdout predictions" : "\nFAIL: device headers disagree with the trainer (see mismatches above)") << std::endl;
    (void)CLASS_NAMES;
    return ok ? 0 : 2;
}
