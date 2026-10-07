#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <set>
#include <sstream>
#include <filesystem>
#include <cctype>

#include "csv_loader.h"
#include "dt.h"
#include "knn.h"
#include "rf.h"


//===========================================================================================================
//training config
//===========================================================================================================

static int NUM_SELECTED_FEATURES = 0;
static int MAX_SELECTED_FEATURES = 0;
static int MIN_SEARCH_FEATURES = 10;
static int FEATURE_STEP = 5;

static uint32_t RANDOM_SEED = 42;
static float TRAIN_RATIO = 0.8f;
static bool SESSION_SPLIT = false;
static int CV_FOLDS = 5;
static bool QUICK_RF_GRID = false;

static float CONFIDENCE_THRESHOLD = 0.30f;
static bool EXCLUDE_ENV_FROM_FISHER = true;
static bool EXPORT_ONLY = false;
static bool EXCLUDE_ENV_DERIVED = false;

static std::string OUTPUT_DIR = ".";

//C float literal that round-trips exactly (9 significant digits)
static std::string cFloat(float v){
    std::ostringstream o;
    o << std::setprecision(9) << v;
    std::string t = o.str();
    if(t.find_first_of(".eEn") == std::string::npos){
        t += ".0";
    }
    return t + "f";
}

static std::string outPath(const std::string& name){
    return (std::filesystem::path(OUTPUT_DIR) / name).string();
}

//raw environmental readings [76-81]
static const int ENV_RAW[] = {76, 77, 78, 79, 80, 81};
//environment-derived features: |dT|,|dH|,|dP| [73-75] and humidity/temperature compensation [145-152]
static const int ENV_DERIVED[] = {73, 74, 75, 145, 146, 147, 148, 149, 150, 151, 152};


//timer
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    const char* name;
    Timer(const char* n) : name(n){
        start=std::chrono::high_resolution_clock::now();
        std::cout << "\n>> " << name << " ..." << std::flush;
    }
    ~Timer(){
        auto end=std::chrono::high_resolution_clock::now();
        double sec=std::chrono::duration<double>(end - start).count();
        std::cout << "\n>> " << name << " done in "  << std::fixed << std::setprecision(1) << sec << "s" << std::endl;
    }
};

struct QuietCout {
    class NullBuf : public std::streambuf {
    protected:
        int overflow(int c) override { return (c == EOF) ? 0 : c; }
    };
    NullBuf nullBuf;
    std::streambuf* prev;
    QuietCout(){ prev = std::cout.rdbuf(&nullBuf); }
    ~QuietCout(){ std::cout.rdbuf(prev); }
};


void printMetrics(const ml_metrics_t& m, const char* modelName){
    std::cout << "\n--- " << modelName << " Results ---" << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << (m.accuracy * 100) << "% (" << m.correct << "/" << m.total << ")" << std::endl;
}

void printPerClassMetrics(const ml_metrics_t& m){
    std::cout << "\nPer-Class Precision/Recall/F1:" << std::endl;
    for(int i=0; i < SCENT_CLASS_COUNT; i++){
        int recallTotal=0, predictedTotal=0;
        for(int j=0; j < SCENT_CLASS_COUNT; j++){
            recallTotal += m.confusionMatrix[i][j];
            predictedTotal += m.confusionMatrix[j][i];
        }
        int tp=m.confusionMatrix[i][i];
        float recall=(recallTotal > 0) ? ((float)tp / recallTotal) : 0.0f;
        float precision=(predictedTotal > 0) ? ((float)tp / predictedTotal) : 0.0f;
        float f1=(precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0.0f;
        std::cout << "  " << CSVLoader::getClassName((scent_class_t)i) << ": "
                  << "P=" << std::fixed << std::setprecision(1) << (precision * 100) << "% "
                  << "R=" << (recall * 100) << "% "
                  << "F1=" << (f1 * 100) << "%" << std::endl;
    }
}

void printConfusionMatrix(const ml_metrics_t& m){
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "Predicted -> ";
    for(int i=0; i < SCENT_CLASS_COUNT; i++) std::cout << std::setw(4) << i;
    std::cout << std::endl;
    for(int i=0; i < SCENT_CLASS_COUNT; i++){
        std::cout << "Actual " << std::setw(2) << i << ":   ";
        for(int j=0; j < SCENT_CLASS_COUNT; j++){
            std::cout << std::setw(4) << m.confusionMatrix[i][j];
        }
        std::cout << "  | " << CSVLoader::getClassName((scent_class_t)i) << std::endl;
    }
}

static const char* shortFeatureName(int idx){
    static char buf[32];
    if (idx < 10){ snprintf(buf, sizeof(buf), "g1n_s%d", idx); return buf; }
    if (idx < 20){ snprintf(buf, sizeof(buf), "g2n_s%d", idx - 10); return buf; }
    if (idx < 30){ snprintf(buf, sizeof(buf), "cr_s%d", idx - 20); return buf; }
    if (idx < 40){ snprintf(buf, sizeof(buf), "diff_s%d", idx - 30); return buf; }
    if (idx < 49){ snprintf(buf, sizeof(buf), "g1d_s%d", idx - 40); return buf; }
    if (idx < 58){ snprintf(buf, sizeof(buf), "g2d_s%d", idx - 49); return buf; }
    static const char* summaryNames[]={
        "slope1_n", "slope2_n", "curv1_n", "curv2_n",
        "auc1_n", "auc2_n", "peak1", "peak2",
        "range1", "range2", "late_early1", "late_early2",
        "cr_mean", "cr_slope", "cr_var",
        "abs_dtemp", "abs_dhum", "abs_dpres",
        "temp1", "hum1", "pres1", "temp2", "hum2", "pres2"
    };

    if(idx < 82){
        return summaryNames[idx - 58];
    }

    if(idx < 92){
        snprintf(buf, sizeof(buf), "n_ratio_s%d", idx - 82);
        return buf;
    }
    if(idx < 97){
        snprintf(buf, sizeof(buf), "ln_ratio_s%d", idx - 92);
        return buf;
    }
    static const char* interactionNames[]={
        "g2n_s2*cr_s1", "g2n_s2*cr_slope", "g1n_s2*diff_s1", "slope2*cr_s0", "g2d_s2*cr_s2",
        "g2n_s3*cr_mean", "diff_s5*cr_s3", "g1n_s3*g1d_s6", "g1n_s2*diff_s5", "g2n_s4*g2d_s3"
    };

    if(idx<107){
        return interactionNames[idx - 97];
    }

    if(idx < 116){
        snprintf(buf, sizeof(buf), "cr_d_s%d", idx - 107);
        return buf;
    }

    if(idx < 125){
        snprintf(buf, sizeof(buf), "dratio_s%d", idx - 116);
        return buf; 
    }

    if(idx < 133){
        snprintf(buf, sizeof(buf), "g2dd_s%d", idx - 125);
        return buf;
    }

    static const char* laterNames[]={
        "recovery1", "recovery2", "max_div", "cr_late_early",
        "seg1_early/late", "seg1_mid/late", "seg2_early/late", "seg2_mid/late",
        "seg_early_1/2", "seg_late_1/2", "curv_mid1", "curv_mid2",
        "g2n_s2/hum", "g1n_s2/hum", "cr_s0/hum", "diff_s1/hum", "slope2/hum", "cr_mean/hum",
        "g2n_s2/temp1", "cr_s0/temp1",
        "fisher_coff_dtea", "fisher_dtea_tea", "fisher_dcoff_coff", "fisher_dtea_dcoff"
    };
    int offset = idx - 133;
    if (offset >= 0 && offset < (int)(sizeof(laterNames) / sizeof(laterNames[0]))) return laterNames[offset];
    snprintf(buf, sizeof(buf), "f_%d", idx);
    return buf;
}

//===========================================================================================================
//robust scaling
//===========================================================================================================
void robustScale(csv_training_sample_t* samples, uint16_t count,int featureCount, float* medians, float* iqrs) {
    for(int f=0;f<featureCount;f++){
        std::vector<float> vals(count);

        for(int i=0;i<count;i++){
            vals[i] = samples[i].features[f];
        }

        std::sort(vals.begin(), vals.end());

        medians[f] = vals[count / 2];
        float q1 = vals[count / 4];
        float q3 = vals[3 * count / 4];
        iqrs[f] = q3 - q1;

        if(iqrs[f] < 1e-6f){ 
            iqrs[f] = 1.0f;
        }

        for(int i = 0; i < count; i++){
            samples[i].features[f]=(samples[i].features[f] - medians[f]) / iqrs[f];
        }
    }
}

void robustScaleApply(csv_training_sample_t* samples, uint16_t count,int featureCount,const float* medians,const float* iqrs) {
    for(int i = 0; i < count; i++){
        for(int f = 0; f < featureCount; f++){
            samples[i].features[f] =(samples[i].features[f]-medians[f]) / iqrs[f];
        }
    }
}

//===========================================================================================================
//feature engineering
//===========================================================================================================
int addEngineeredFeatures(csv_training_sample_t* samples, uint16_t count,int currentFeatureCount) {
    int idx = currentFeatureCount;

    for(uint16_t s = 0; s < count; s++){
        int fi = currentFeatureCount;

        //group 1
        for(int i = 0; i < 10; i++){
            if(fi >= CSV_FEATURE_COUNT) break;
            float g1 = samples[s].features[i];
            float g2 = samples[s].features[i + 10];
            samples[s].features[fi++] = (fabsf(g2) > 1e-6f)
                                        ? g1 / g2 : 0.0f;
        }

        //group 2
        for(int i = 0; i < 5; i++){
            if(fi >= CSV_FEATURE_COUNT) break;
            float g1 = samples[s].features[i];
            float g2 = samples[s].features[i + 10];
            samples[s].features[fi++] = (g1 > 0 && g2 > 0)
                                        ? logf(g1 / g2) : 0.0f;
        }

        //group 3
        struct Pair { int a, b; };
        Pair interactions[] = {
            {12, 21},
            {12, 71}, 
            { 2, 31},
            {59, 20}, 
            {51, 22},
            {13, 70},
            {35, 23}, 
            { 3, 46},
            { 2, 35},
            {14, 52},
        };
        for(int p = 0; p < 10; p++){
            if(fi >= CSV_FEATURE_COUNT){
                break;
            }

            if(interactions[p].a < currentFeatureCount&&interactions[p].b <currentFeatureCount){
                samples[s].features[fi++] =samples[s].features[interactions[p].a]*samples[s].features[interactions[p].b];
            }
        }

        //group 4
        for(int t = 0; t < 9; t++){
            if(fi >= CSV_FEATURE_COUNT) break;
            samples[s].features[fi++] = samples[s].features[20 + t + 1]- samples[s].features[20 + t];
        }

        //group 5
        for(int t = 0; t < 9; t++){
            if(fi >= CSV_FEATURE_COUNT) break;

            float g1d=samples[s].features[40 + t];
            float g2d=samples[s].features[49 + t];
            samples[s].features[fi++] = g1d / (fabsf(g2d) + 1e-6f);
        }

        //group 6
        for(int t=0; t<8;t++){
            if(fi >= CSV_FEATURE_COUNT) break;

            samples[s].features[fi++] =samples[s].features[49 + t + 1]-samples[s].features[49+t];
        }

        //group 7
        for(int sensor =0;sensor < 2;sensor++){

            if(fi >= CSV_FEATURE_COUNT) break;
            int base = sensor * 10;
            float minV = 1e30f, maxV = -1e30f;

            for(int t=0;t<10; t++){
                float v=samples[s].features[base + t];
                if(v<minV){
                    minV = v;
                }

                if(v>maxV){
                    maxV = v;
                }
            }

            float range = maxV - minV;
            samples[s].features[fi++] = (range > 1e-6f)? (samples[s].features[base + 9] - minV) / range : 0.0f;
        }

        if(fi <CSV_FEATURE_COUNT){
            float maxDiv = 0;
            for(int t=0; t<10;t++){
                float d = fabsf(samples[s].features[t] - samples[s].features[10 + t]);
                if(d>maxDiv){
                    maxDiv = d;
                }
            }
            samples[s].features[fi++] = maxDiv;
        }

        //early late diff 
        if(fi < CSV_FEATURE_COUNT && currentFeatureCount>29){
            float cr_early = (samples[s].features[20] +samples[s].features[21]+samples[s].features[22])/3.0f;
            float cr_late  = (samples[s].features[27] +samples[s].features[28]+samples[s].features[29])/3.0f;
            samples[s].features[fi++] = cr_late - cr_early;
        }

        //group 8
        {
            float seg[2][3];
            int segs[][2] = {{0,3}, {4,6}, {7,9}};

            for(int sensor=0; sensor<2; sensor++){
                int base = sensor*10;
                for(int sg =0; sg <3; sg++){
                    float sum =0;
                    int n = 0;
                    for(int t =segs[sg][0]; t <=segs[sg][1];t++){
                        sum +=samples[s].features[base + t];
                        n++;
                    }
                    seg[sensor][sg] = sum / n;
                }
            }

            //s1
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[0][0] /(fabsf(seg[0][2])+ 1e-6f);
            }
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[0][1] /(fabsf(seg[0][2])+ 1e-6f);
            }

            //s2
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[1][0] /(fabsf(seg[1][2])+ 1e-6f);
            }
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[1][1] /(fabsf(seg[1][2])+ 1e-6f);
            }

            //cs
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[0][0] /(fabsf(seg[1][0])+ 1e-6f);
            }
            if(fi < CSV_FEATURE_COUNT){
                samples[s].features[fi++] = seg[0][2] /(fabsf(seg[1][2])+ 1e-6f);
            }
        }

        //group 9
        if(fi < CSV_FEATURE_COUNT&&currentFeatureCount>9){
            float g1_start=samples[s].features[0];
            float g1_mid=samples[s].features[4];
            float g1_end=samples[s].features[9];
            if(fabsf(g1_start + g1_end) > 1e-6f)
                samples[s].features[fi++] = (2.0f * g1_mid) / (g1_start + g1_end);
            else
                samples[s].features[fi++] = 0.0f;
        }

        //g2 response shape
        if(fi<CSV_FEATURE_COUNT &&currentFeatureCount> 19){
            float g2_start=samples[s].features[10];
            float g2_mid=samples[s].features[14];
            float g2_end=samples[s].features[19];

            if(fabsf(g2_start+g2_end) >1e-6f){
                    samples[s].features[fi++] = (2.0f * g2_mid) / (g2_start + g2_end);
                }

            else{
                samples[s].features[fi++] = 0.0f;
            }
        }
        
        //group 10
        if(currentFeatureCount >= 77) {
            float hum1 =samples[s].features[77];
            float hum2 =samples[s].features[80];
            float avg_hum =(hum1 + hum2)/2.0f;
            
            if(fabsf(avg_hum)>1e-6f){
                //g2n_s2 / humidity 
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[12] / avg_hum;
                }
                //g1n_s2 / humidity
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[2] / avg_hum;
                }
                //cross_ratio_s0 / humidity
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[20] / avg_hum;
                }
                //diff_s1 / humidity
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[31] / avg_hum;
                }
                //slope2 / humidity
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[59] / avg_hum;
                }
                //cross_ratio_mean / humidity
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++] = samples[s].features[70] / avg_hum;
                }
            }
            else{
                //0
                for(int z=0;z <6&&fi<CSV_FEATURE_COUNT;z++){
                    samples[s].features[fi++] = 0.0f;
                }
            }
            
            //temperature gas interactions
            float temp1 = samples[s].features[76];

            if(fabsf(temp1)>1e-6f){
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++]=samples[s].features[12]/temp1;
                }
                if(fi < CSV_FEATURE_COUNT){
                    samples[s].features[fi++]=samples[s].features[20]/temp1;
                }
            }

            else{
                for(int z = 0; z < 2 && fi < CSV_FEATURE_COUNT; z++)
                    samples[s].features[fi++] = 0.0f;
            }
        }

        idx=fi;
    }

    int added = idx - currentFeatureCount;
    std::cout << "Engineered " << added << " new features (total: " << idx << ")" << std::endl;
    return idx;
}



//===========================================================================================================
//feat selection
//===========================================================================================================
struct FeatureSelector {
    std::vector<int> selectedIndices;
    int originalCount = 0;
    int selectedCount = 0;

    void selectFeaturesHybrid(const csv_training_sample_t *samples, uint16_t count,int featureCount, int maxFeatures, const std::set<int>& blacklist, bool verbose, bool applyMinScore = true) {
        originalCount = featureCount;

        struct FeatureScore {
            int index;
            float globalF;
            float teaPairF;      //decaf_tea vs tea
            float coffeePairF;   //decaf_coffee vs coffee
            float crossPairF;    //coffee vs decaf_tea
            float teaCoffeeF;    //tea vs coffee
            float score;
        };

        std::vector<FeatureScore> scores(featureCount);

        auto pairF = [&](int f, scent_class_t classA, scent_class_t classB) -> float {
            float meanA=0.0f, meanB=0.0f;
            float varA=0.0f, varB=0.0f;
            int countA=0, countB=0;

            for(int i=0; i<count;i++){
                if(samples[i].label==classA){
                    meanA+=samples[i].features[f];
                    countA++;
                }
                else if(samples[i].label==classB){
                    meanB+=samples[i].features[f];
                    countB++;
                }
            }
            if(countA==0 ||countB==0){
                return 0.0f;
            }
            meanA/= countA;
            meanB/= countB;

            for(int i=0; i<count;i++){
                if(samples[i].label==classA){
                    float d=samples[i].features[f] - meanA;
                    varA+= d * d;
                }
                else if(samples[i].label==classB){
                    float d=samples[i].features[f] - meanB;
                    varB+= d * d;
                }
            }

            float pooledVar=(varA+varB)/(countA + countB - 2);
            if(pooledVar < 1e-8f){
                return 0.0f;
            }

            float diff=meanA-meanB;
            return (diff * diff) / pooledVar;
        };

        for(int f=0; f<featureCount;f++){
            scores[f] = {f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if(blacklist.count(f)){
                continue;
            }

            {
                float classMeans[SCENT_CLASS_COUNT]={0};
                float classVars[SCENT_CLASS_COUNT]={0};
                int classCounts[SCENT_CLASS_COUNT]={0};

                for(int j=0; j<count;j++){
                    int c=samples[j].label;
                    if(c>= SCENT_CLASS_COUNT){
                        continue;
                    }

                    classMeans[c]+=samples[j].features[f];
                    classCounts[c]++;
                }

                float grandMean=0.0f;
                int totalValid=0;
                for(int j=0; j<SCENT_CLASS_COUNT; j++){
                    if(classCounts[j]>0){
                        classMeans[j] /= classCounts[j];
                    }
                    grandMean+=classMeans[j] * classCounts[j];
                    totalValid += classCounts[j];
                }
                if(totalValid>0){
                    grandMean/= totalValid;
                }

                for(int i=0; i<count;i++){
                    int c=samples[i].label;
                    if(c>= SCENT_CLASS_COUNT){
                        continue;
                    }
                    float d = samples[i].features[f] - classMeans[c];
                    classVars[c] += d * d;
                }

                float betweenVar=0.0f, withinVar=0.0f;
                for(int i=0; i<SCENT_CLASS_COUNT;i++){
                    if(classCounts[i]>0){
                        float d=classMeans[i] - grandMean;
                        betweenVar+= classCounts[i] * d * d;
                        withinVar+= classVars[i];
                    }
                }

                scores[f].globalF = (withinVar > 1e-8f && (totalValid - SCENT_CLASS_COUNT) > 0)
                ? (betweenVar / (SCENT_CLASS_COUNT - 1)) / (withinVar / (totalValid - SCENT_CLASS_COUNT))
                : 0.0f;
            }

            scores[f].teaPairF= pairF(f, SCENT_CLASS_DECAF_TEA, SCENT_CLASS_TEA);
            scores[f].coffeePairF= pairF(f, SCENT_CLASS_DECAF_COFFEE, SCENT_CLASS_COFFEE);
            scores[f].crossPairF= pairF(f, SCENT_CLASS_COFFEE, SCENT_CLASS_DECAF_TEA);
            scores[f].teaCoffeeF= pairF(f, SCENT_CLASS_TEA, SCENT_CLASS_COFFEE);
        }

        float maxG = 0.0f, maxT = 0.0f, maxC = 0.0f, maxX = 0.0f, maxTC = 0.0f;
        for(int f=0;f<featureCount;f++){
            maxG = std::max(maxG, scores[f].globalF);
            maxT = std::max(maxT, scores[f].teaPairF);
            maxC = std::max(maxC, scores[f].coffeePairF);
            maxX = std::max(maxX, scores[f].crossPairF);
            maxTC = std::max(maxTC, scores[f].teaCoffeeF);
        }

        if(maxG<1e-8f) maxG = 1.0f;
        if(maxT < 1e-8f) maxT = 1.0f;
        if(maxC < 1e-8f) maxC = 1.0f;
        if(maxX < 1e-8f) maxX = 1.0f;
        if(maxTC < 1e-8f) maxTC = 1.0f;

        for(int f = 0; f < featureCount; f++){
            float normG = scores[f].globalF / maxG;
            float normT = scores[f].teaPairF / maxT;
            float normC = scores[f].coffeePairF / maxC;
            float normX = scores[f].crossPairF / maxX;
            float normTC=scores[f].teaCoffeeF/maxTC;

            //take max and average across 4 hard pairs
            float pairMax=std::max({normT,normC,normX,normTC});
            float pairAvg = (normT + normC + normX + normTC)/4.0f;

            scores[f].score = 0.20f*normG+0.45f * pairMax+0.35f*pairAvg;
        }

        //stable sort
        std::stable_sort(scores.begin(), scores.end(),
            [](const FeatureScore& a, const FeatureScore& b){
                return a.score > b.score;
        });

        if(verbose){
            std::cout << "\n=== Hybrid Feature Ranking ===" << std::endl;
            std::cout << "  [*]=selected, [ ]=dropped\n" << std::endl;
        }

        selectedIndices.clear();
        const float minScore = 0.05f;

        for(int i = 0; i < featureCount; i++){
            bool selected = ((int)selectedIndices.size() < maxFeatures&& (!applyMinScore || scores[i].score >= minScore) && !blacklist.count(scores[i].index));

            if(selected){
                selectedIndices.push_back(scores[i].index);
            }

            if(verbose&&i < maxFeatures + 10) {
                std::cout << "  " << (selected ? "[*]" : "[ ]")
                        << " S=" << std::setw(6) << std::fixed << std::setprecision(3)
                        << scores[i].score
                        << "(G=" << std::setprecision(2) << scores[i].globalF / maxG
                        << " T=" << scores[i].teaPairF / maxT
                        << " C=" << scores[i].coffeePairF / maxC
                        << " X=" << scores[i].crossPairF / maxX
                        << " TC=" << scores[i].teaCoffeeF / maxTC
                        << ")  [" << std::setw(3) << scores[i].index << "] "
                        << shortFeatureName(scores[i].index) << std::endl;
            }
        }

        //kept in rank order
        selectedCount = selectedIndices.size();
        if(verbose){
            std::cout << "\nSelected " << selectedCount << "/" << featureCount << " features"
                    << " (" << blacklist.size() << " blacklisted)" << std::endl;
            std::cout << "Score maxima: G=" << std::setprecision(1) << maxG << " T=" << maxT << " C=" << maxC
                    << " X=" << maxX << " TC=" << maxTC << std::endl;
        }
    }

    void projectDataset(const csv_training_sample_t* src, uint16_t count,csv_training_sample_t* dst) const {
        for(uint16_t i=0; i < count; i++){
            dst[i].label=src[i].label;
            memset(dst[i].features, 0, sizeof(dst[i].features));
            for(int f=0; f < selectedCount; f++){
                dst[i].features[f]=src[i].features[selectedIndices[f]];
            }
        }
    }

    void saveHeader(const std::string& filename, const float* medians, const float* iqrs,const std::string& note = "")const{
        std::ofstream f(filename);
        if(!note.empty()) f << "// " << note << "\n";
        f << "#ifndef FEATURE_SELECT_H\n#define FEATURE_SELECT_H\n\n";
        f << "#define SELECTED_FEATURE_COUNT " << selectedCount << "\n";
        f << "#define ORIGINAL_FEATURE_COUNT " << originalCount << "\n\n";

        f << "static const int SELECTED_INDICES[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << selectedIndices[i];
            if (i < selectedCount - 1) f << ", ";
        }
        f << "};\n\n";

        f << "static const float SELECTED_MEDIANS[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << std::setprecision(9) << medians[selectedIndices[i]];
            if (i < selectedCount - 1) f << ", ";
        }
        f << "};\n\n";

        f << "static const float SELECTED_IQRS[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << std::setprecision(9) << iqrs[selectedIndices[i]];
            if (i < selectedCount - 1) f << ", ";
        }
        f << "};\n\n#endif\n";
        f.close();
        std::cout << "Feature selection saved to " << filename << std::endl;
    }
};

void augmentTrainingData(csv_training_sample_t*& trainSet, uint16_t& trainCount,int featureCount, std::default_random_engine& rng){
    uint16_t classCounts[SCENT_CLASS_COUNT]={0};
    std::vector<std::vector<uint16_t>> classIndices(SCENT_CLASS_COUNT);

    for(int i=0; i < trainCount; i++){
        if(trainSet[i].label < SCENT_CLASS_COUNT){
            classCounts[trainSet[i].label]++;
            classIndices[trainSet[i].label].push_back(i);
        }
    }

    uint16_t maxClassCount=*std::max_element(classCounts, classCounts + SCENT_CLASS_COUNT);
    uint16_t targetPerClass=maxClassCount * 2;

    std::vector<csv_training_sample_t> augmented;
    augmented.reserve(targetPerClass * SCENT_CLASS_COUNT);

    for(int i=0; i < trainCount; i++)augmented.push_back(trainSet[i]);

    std::normal_distribution<float> noise(0.0f, 0.03f);
    std::uniform_real_distribution<float> mixDist(0.3f, 0.7f);

    for(int c=0; c < SCENT_CLASS_COUNT; c++){
        if(classIndices[c].empty())continue;
        int need=targetPerClass - (int)classCounts[c];
        std::uniform_int_distribution<uint16_t> srcDist(0, classIndices[c].size() - 1);

        for(int j=0; j <need; j++){
            uint16_t idx1=classIndices[c][srcDist(rng)];
            uint16_t idx2=classIndices[c][srcDist(rng)];
            float alpha=mixDist(rng);

            csv_training_sample_t noisy;
            noisy.label=(scent_class_t)c;
            for(int f=0; f <featureCount; f++){
                noisy.features[f]=alpha * trainSet[idx1].features[f]+(1.0f - alpha) *trainSet[idx2].features[f]+ noise(rng);
            }
            for(int f=featureCount; f <CSV_FEATURE_COUNT; f++){
                noisy.features[f]=0.0f;
            }
            augmented.push_back(noisy);
        }
    }

    delete[] trainSet;
    trainCount=augmented.size();
    trainSet=new csv_training_sample_t[trainCount];
    for(uint16_t i=0; i <trainCount; i++) trainSet[i]=augmented[i];
}

//
void applyCostSensitiveWeighting(csv_training_sample_t*& trainSet, uint16_t& trainCount,int featureCount) {
    std::vector<csv_training_sample_t> weighted;
    weighted.reserve(trainCount * 2);
    
    int dupCoffee = 0, dupDtea = 0;
    
    for(uint16_t i = 0; i < trainCount; i++){
        weighted.push_back(trainSet[i]);
        
        //duplicate coffee and decaf_tea samples
        if(trainSet[i].label==SCENT_CLASS_COFFEE||trainSet[i].label==SCENT_CLASS_DECAF_TEA){
            weighted.push_back(trainSet[i]);
            if(trainSet[i].label ==SCENT_CLASS_COFFEE){
                dupCoffee++;
            }
            else{
                dupDtea++;
            }
        }
    }
    
    delete[] trainSet;
    trainCount = weighted.size();
    trainSet = new csv_training_sample_t[trainCount];
    for(uint16_t i = 0; i < trainCount; i++) trainSet[i] = weighted[i];
    
    std::cout << "Cost-sensitive weighting: duplicated " << dupCoffee << " coffee + " << dupDtea << " decaf_tea samples" << std::endl;
    std::cout << "New training size: " << trainCount << std::endl;
}

//===========================================================================================================
//feature pipeline
//===========================================================================================================
struct FisherPair {
    scent_class_t a, b;
    const char* name;
};
static const FisherPair FISHER_PAIRS[] = {
    {SCENT_CLASS_COFFEE, SCENT_CLASS_DECAF_TEA, "coffee_vs_dtea"},
    {SCENT_CLASS_DECAF_TEA, SCENT_CLASS_TEA, "dtea_vs_tea"},
    {SCENT_CLASS_DECAF_COFFEE, SCENT_CLASS_COFFEE, "dcoffee_vs_coffee"},
    {SCENT_CLASS_DECAF_TEA, SCENT_CLASS_DECAF_COFFEE, "dtea_vs_dcoffee"}
};
static const int NUM_FISHER_PAIRS = 4;

struct Pipeline {
    int baseCount = 0;
    int fullCount = 0;
    int numFisher = 0;
    int fisherPairIdx[NUM_FISHER_PAIRS] = {0};
    float medians[CSV_FEATURE_COUNT] = {0};
    float iqrs[CSV_FEATURE_COUNT] = {0};
    float fisherW[NUM_FISHER_PAIRS][CSV_FEATURE_COUNT] = {{0}};
    FeatureSelector selector;
};

static std::set<int> fisherExcluded(){
    std::set<int> ex;
    if(EXCLUDE_ENV_FROM_FISHER){
        ex.insert(std::begin(ENV_RAW), std::end(ENV_RAW));
    }
    if(EXCLUDE_ENV_DERIVED){
        ex.insert(std::begin(ENV_DERIVED), std::end(ENV_DERIVED));
    }
    return ex;
}

static std::set<int> selectionBlacklist(){
    std::set<int> bl(std::begin(ENV_RAW), std::end(ENV_RAW));
    if(EXCLUDE_ENV_DERIVED){
        bl.insert(std::begin(ENV_DERIVED), std::end(ENV_DERIVED));
    }
    return bl;
}

static float fisherProject(const Pipeline& p, int k, const float* scaled){
    float proj = 0.0f;
    for(int f = 0; f < p.baseCount; f++){
        proj += scaled[f] * p.fisherW[k][f];
    }
    return proj;
}

//fits the pipeline on samples
static void fitPipeline(csv_training_sample_t* samples, uint16_t count, int baseCount, int nSelect,Pipeline& p, bool verbose, bool applyMinScore = true){
    p.baseCount = baseCount;

    //robust scaling of base + engineered features
    robustScale(samples, count, baseCount, p.medians, p.iqrs);

    //Fisher discriminant directions on the scaled features
    std::set<int> excluded = fisherExcluded();
    p.numFisher = 0;
    for(int k = 0; k < NUM_FISHER_PAIRS; k++){
        if(baseCount + p.numFisher >= CSV_FEATURE_COUNT){
            break;
        }
        const FisherPair& pr = FISHER_PAIRS[k];

        std::vector<double> meanA(baseCount, 0.0), meanB(baseCount, 0.0);
        int nA = 0, nB = 0;
        for(uint16_t i = 0; i < count; i++){
            if(samples[i].label == pr.a){
                for(int f = 0; f < baseCount; f++) meanA[f] += samples[i].features[f];
                nA++;
            }
            else if(samples[i].label == pr.b){
                for(int f = 0; f < baseCount; f++) meanB[f] += samples[i].features[f];
                nB++;
            }
        }
        if(nA == 0 || nB == 0){
            continue;
        }
        for(int f = 0; f < baseCount; f++){
            meanA[f] /= nA;
            meanB[f] /= nB;
        }

        float* w = p.fisherW[p.numFisher];
        memset(w, 0, sizeof(float) * CSV_FEATURE_COUNT);
        double norm = 0.0;
        for(int f = 0; f < baseCount; f++){
            if(excluded.count(f)){
                continue;
            }
            double varA = 0.0, varB = 0.0;
            for(uint16_t i = 0; i < count; i++){
                if(samples[i].label == pr.a){
                    double d = samples[i].features[f] - meanA[f];
                    varA += d * d;
                }
                else if(samples[i].label == pr.b){
                    double d = samples[i].features[f] - meanB[f];
                    varB += d * d;
                }
            }
            double pooledStd = std::sqrt((varA / nA + varB / nB) / 2.0);
            if(pooledStd > 1e-6){
                w[f] = (float)((meanA[f] - meanB[f]) / pooledStd);
            }
            norm += (double)w[f] * w[f];
        }
        norm = std::sqrt(norm);
        if(norm < 1e-6){
            continue;
        }
        for(int f = 0; f < baseCount; f++){
            w[f] = (float)(w[f] / norm);
        }

        p.fisherPairIdx[p.numFisher] = k;
        int newIdx = baseCount + p.numFisher;

        for(uint16_t i = 0; i < count; i++){
            samples[i].features[newIdx] = fisherProject(p, p.numFisher, samples[i].features);
        }
        p.numFisher++;

        if(verbose){
            std::cout << "  Fisher [" << newIdx << "]: " << pr.name << std::endl;
        }
    }
    p.fullCount = baseCount + p.numFisher;

    //Fisher outputs get their own median/IQR
    for(int f = baseCount; f < p.fullCount; f++){
        std::vector<float> vals(count);
        for(uint16_t i = 0; i < count; i++){
            vals[i] = samples[i].features[f];
        }

        std::sort(vals.begin(), vals.end());
        p.medians[f] = vals[count / 2];
        p.iqrs[f] = vals[3 * count / 4] - vals[count / 4];

        if(p.iqrs[f] < 1e-6f){
            p.iqrs[f] = 1.0f;
        }

        for(uint16_t i = 0; i < count; i++){
            samples[i].features[f] = (samples[i].features[f] - p.medians[f]) / p.iqrs[f];
        }
    }

    if(verbose){
        std::cout << "Total features with Fisher: " << p.fullCount << " (Fisher excludes " << excluded.size() << " environmental features)" << std::endl;
    }

    //hybrid selection
    p.selector.selectFeaturesHybrid(samples, count, p.fullCount, nSelect, selectionBlacklist(), verbose, applyMinScore);
}

//applies a fitted pipeline to unseen samples
static void applyPipeline(csv_training_sample_t* samples, uint16_t count, const Pipeline& p){
    robustScaleApply(samples, count, p.baseCount, p.medians, p.iqrs);
    for(uint16_t i = 0; i < count; i++){
        for(int k = 0; k < p.numFisher; k++){
            int f = p.baseCount + k;
            float proj = fisherProject(p, k, samples[i].features);
            samples[i].features[f] = (proj - p.medians[f]) / p.iqrs[f];
        }
    }
}

//===========================================================================================================
//cross validation
//===========================================================================================================
struct Fold{
    std::vector<csv_training_sample_t> train;
    std::vector<csv_training_sample_t> test;
    uint16_t featureCount = 0;
};

static std::vector<Fold> buildFolds(const csv_training_sample_t* engineered, uint16_t count, int baseCount,int nSelect, int numFolds, uint32_t seed){
    std::vector<std::vector<uint16_t>> classIndices(SCENT_CLASS_COUNT);
    for(uint16_t i = 0; i < count; i++){
        if(engineered[i].label < SCENT_CLASS_COUNT){
            classIndices[engineered[i].label].push_back(i);
        }
    }

    std::mt19937 rng(seed);
    std::vector<int> foldOf(count, -1);

    for(int c = 0; c < SCENT_CLASS_COUNT; c++){
        std::shuffle(classIndices[c].begin(), classIndices[c].end(), rng);

        for(size_t j = 0; j < classIndices[c].size(); j++){
            foldOf[classIndices[c][j]] = j % numFolds;
        }
    }

    std::vector<Fold> folds(numFolds);

    for(int k = 0; k < numFolds; k++){
        std::vector<csv_training_sample_t> tr, te;

        for(uint16_t i = 0; i < count; i++){
            if(foldOf[i] == k) te.push_back(engineered[i]);
            else if(foldOf[i] >= 0) tr.push_back(engineered[i]);
        }

        Pipeline fp;
        fitPipeline(tr.data(), tr.size(), baseCount, nSelect, fp, false, false);
        applyPipeline(te.data(), te.size(), fp);

        folds[k].featureCount = fp.selector.selectedCount;
        folds[k].train.resize(tr.size());
        folds[k].test.resize(te.size());
        fp.selector.projectDataset(tr.data(), tr.size(), folds[k].train.data());
        fp.selector.projectDataset(te.data(), te.size(), folds[k].test.data());
    }
    return folds;
}

struct CVStat {
    float mean = 0.0f;
    float std = 0.0f;
    ml_metrics_t agg;
};

//fn(fold) trains on fold.train and returns metrics on fold.test
template<typename F>
static CVStat runCV(const std::vector<Fold>& folds, F fn){
    CVStat r;
    memset(&r.agg, 0, sizeof(r.agg));
    std::vector<float> accs;
    for(const Fold& fold : folds){
        ml_metrics_t m = fn(fold);
        accs.push_back(m.accuracy);
        r.agg.correct += m.correct;
        r.agg.total += m.total;
        for(int i = 0; i < SCENT_CLASS_COUNT; i++){
            for(int j = 0; j < SCENT_CLASS_COUNT; j++){
                r.agg.confusionMatrix[i][j] += m.confusionMatrix[i][j];
            }
        }
    }

    if(!accs.empty()){
        r.mean = std::accumulate(accs.begin(), accs.end(), 0.0f) / accs.size();
        float var = 0.0f;
        for(float a : accs) var += (a - r.mean) * (a - r.mean);
        r.std = std::sqrt(var / accs.size());
    }
    r.agg.accuracy = (r.agg.total > 0) ? (float)r.agg.correct / r.agg.total : 0.0f;
    return r;
}

static std::string pct(float mean, float sd){
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << (mean * 100) << "% +/- " << (sd * 100) << "%";
    return o.str();
}


//experiment
void twoStageExperiment(const csv_training_sample_t* trainSet, uint16_t trainCount, const csv_training_sample_t* testSet, uint16_t testCount,uint16_t featureCount){
    std::cout <<"\n=== Classification Experiment ===" <<std::endl;

    std::vector<csv_training_sample_t> s1_train, s1_test;
    for(int i=0; i <trainCount; i++){
        if(trainSet[i].label == SCENT_CLASS_AMBIENT) continue;
        csv_training_sample_t s=trainSet[i];
        s.label=(s.label == SCENT_CLASS_DECAF_TEA || s.label == SCENT_CLASS_TEA)? (scent_class_t)0 : (scent_class_t)1;
        s1_train.push_back(s);
    }
    for(int i=0; i <testCount; i++){
        if(testSet[i].label == SCENT_CLASS_AMBIENT) continue;
        csv_training_sample_t s=testSet[i];
        s.label=(s.label == SCENT_CLASS_DECAF_TEA || s.label == SCENT_CLASS_TEA)? (scent_class_t)0 : (scent_class_t)1;
        s1_test.push_back(s);
    }

    RandomForest rf1(100, 12, 3, 0.3f);
    rf1.train(s1_train.data(), s1_train.size(), featureCount);
    int s1_correct=0;
    for(size_t i=0; i <s1_test.size(); i++){
        if (rf1.predict(s1_test[i].features) == s1_test[i].label) s1_correct++;
    }
    std::cout <<"Stage 1 (tea-type vs coffee-type, beverages only): "<<std::fixed <<std::setprecision(1)<<(100.0f * s1_correct / s1_test.size()) <<"%" <<std::endl;

    auto runStage2=[&](scent_class_t classA, scent_class_t classB, const char* name){
        std::vector<csv_training_sample_t> s2_train, s2_test;
        for(int i=0; i <trainCount; i++){
            if(trainSet[i].label == classA || trainSet[i].label == classB){
                csv_training_sample_t s=trainSet[i];
                s.label=(s.label == classA) ? (scent_class_t)0 : (scent_class_t)1;
                s2_train.push_back(s);
            }
        }

        for(int i=0; i <testCount; i++){
            if(testSet[i].label == classA || testSet[i].label == classB){
                csv_training_sample_t s=testSet[i];
                s.label=(s.label == classA) ? (scent_class_t)0 : (scent_class_t)1;
                s2_test.push_back(s);
            }
        }
        if(s2_train.empty() || s2_test.empty()) return;
        RandomForest rf2(100, 12, 3, 0.3f);
        rf2.train(s2_train.data(), s2_train.size(), featureCount);
        int correct=0;
        for(size_t i=0; i <s2_test.size(); i++){
            if (rf2.predict(s2_test[i].features) == s2_test[i].label) correct++;
        }
        std::cout <<name <<": "<<std::fixed <<std::setprecision(1)<<(100.0f * correct / s2_test.size()) <<"%" <<std::endl;
    };

    runStage2(SCENT_CLASS_DECAF_TEA, SCENT_CLASS_TEA, "Stage 2a (decaf_tea vs tea)");
    runStage2(SCENT_CLASS_DECAF_COFFEE, SCENT_CLASS_COFFEE, "Stage 2b (decaf_coffee vs coffee)");
    std::cout <<"============================================\n" <<std::endl;
}

//===========================================================================================================
//hierarchical classifier
//===========================================================================================================
struct HierarchicalClassifier {
    RandomForest stage1;
    RandomForest stage2_tea;
    RandomForest stage2_coffee;
    std::vector<int> s1_features;
    std::vector<int> s2tea_features;
    std::vector<int> s2coffee_features;
    
    uint16_t trainFeatureCount;

    HierarchicalClassifier() :  stage1(300, 8, 5, 0.3f),
                                stage2_tea(200, 6, 3, 0.5f),
                                stage2_coffee(200, 6, 3, 0.5f),
                                trainFeatureCount(0) {}

    void train(csv_training_sample_t* samples, uint16_t count,uint16_t featureCount){
        
        trainFeatureCount = featureCount;

        //filter out ambient samples
        std::vector<csv_training_sample_t> scent_only;
        scent_only.reserve(count);
        for(uint16_t i=0; i<count; i++){
            if(samples[i].label != SCENT_CLASS_AMBIENT){
                scent_only.push_back(samples[i]);
            }
        }
        std::cout << "  Filtered " << (count - scent_only.size()) << " ambient samples from hierarchical training" << std::endl;
        
        //train only on scent data
        std::vector<csv_training_sample_t> s1_data;
        s1_data.reserve(scent_only.size());
        for(auto& s : scent_only){
            csv_training_sample_t s1 = s;
            s1.label = (s.label == SCENT_CLASS_DECAF_TEA||s.label == SCENT_CLASS_TEA)? (scent_class_t)0 : (scent_class_t)1;
            s1_data.push_back(s1);
        }

        stage1.train(s1_data.data(), s1_data.size(), featureCount);
        
        //evaluate
        int s1_correct = 0;
        for(size_t i = 0; i < s1_data.size(); i++){
            if(stage1.predict(s1_data[i].features) == s1_data[i].label) s1_correct++;
        }
        std::cout << "  Stage 1 train acc: " << std::fixed << std::setprecision(1)
                  << (100.0f * s1_correct / s1_data.size()) << "%" << std::endl;

        //decaf_tea vs tea
        std::cout << "  Training Stage 2a: decaf_tea vs tea..." << std::endl;
        std::vector<csv_training_sample_t> tea_data;
        for(uint16_t i = 0; i < count; i++){
            if(samples[i].label == SCENT_CLASS_DECAF_TEA||samples[i].label == SCENT_CLASS_TEA){
                csv_training_sample_t s = samples[i];
                s.label = (s.label == SCENT_CLASS_DECAF_TEA)?(scent_class_t)0:(scent_class_t)1;
                tea_data.push_back(s);
            }
        }

        if(!tea_data.empty()){
            stage2_tea.train(tea_data.data(), tea_data.size(), featureCount);
            int s2t_correct = 0;
            for(size_t i = 0; i < tea_data.size(); i++){
                if(stage2_tea.predict(tea_data[i].features) == tea_data[i].label) s2t_correct++;
            }
            std::cout << "  Stage 2a train acc: " << std::fixed << std::setprecision(1)<< (100.0f * s2t_correct / tea_data.size()) << "%" << std::endl;
        }

        //decaf_coffee vs coffee(
        std::cout << "  Training Stage 2b: decaf_coffee vs coffee..." << std::endl;
        std::vector<csv_training_sample_t> coffee_data;
        for(uint16_t i = 0; i < count; i++){
            if(samples[i].label == SCENT_CLASS_DECAF_COFFEE||samples[i].label == SCENT_CLASS_COFFEE){
                csv_training_sample_t s = samples[i];
                s.label = (s.label == SCENT_CLASS_DECAF_COFFEE)? (scent_class_t)0:(scent_class_t)1;
                coffee_data.push_back(s);
            }
        }

        if(!coffee_data.empty()){
            stage2_coffee.train(coffee_data.data(), coffee_data.size(), featureCount);
            int s2c_correct = 0;

            for(size_t i = 0; i < coffee_data.size(); i++){
                if(stage2_coffee.predict(coffee_data[i].features) == coffee_data[i].label){
                    s2c_correct++;
                }
            }
            std::cout << "  Stage 2b train acc: " << std::fixed << std::setprecision(1)<< (100.0f * s2c_correct / coffee_data.size()) << "%" << std::endl;
        }
    }

    scent_class_t predict(const float* features) {
        scent_class_t group = stage1.predict(features);
        if(group == 0){
            scent_class_t sub = stage2_tea.predict(features);
            return (sub == 0) ? SCENT_CLASS_DECAF_TEA : SCENT_CLASS_TEA;
        }
        else{
            scent_class_t sub = stage2_coffee.predict(features);
            return (sub == 0) ? SCENT_CLASS_DECAF_COFFEE : SCENT_CLASS_COFFEE;
        }
    }

    scent_class_t predictSoft(const float* features){
        float classScores[SCENT_CLASS_COUNT] = {0};
        
        float s1_conf;
        scent_class_t group = stage1.predictWithConfidence(features, s1_conf);
        
        float tea_prob = (group == 0) ? s1_conf : (1.0f - s1_conf);
        float coffee_prob = 1.0f - tea_prob;
        
        float s2tea_conf;
        scent_class_t tea_sub = stage2_tea.predictWithConfidence(features, s2tea_conf);
        
        float s2coffee_conf;
        scent_class_t coffee_sub = stage2_coffee.predictWithConfidence(features, s2coffee_conf);
        
        if(tea_sub == 0){
            classScores[SCENT_CLASS_DECAF_TEA]+=tea_prob * s2tea_conf;
            classScores[SCENT_CLASS_TEA]+=tea_prob * (1.0f - s2tea_conf);
        }
        else{
            classScores[SCENT_CLASS_TEA]+=tea_prob * s2tea_conf;
            classScores[SCENT_CLASS_DECAF_TEA]+=tea_prob * (1.0f - s2tea_conf);
        }
        
        if(coffee_sub == 0){
            classScores[SCENT_CLASS_DECAF_COFFEE] += coffee_prob * s2coffee_conf;
            classScores[SCENT_CLASS_COFFEE]+=coffee_prob * (1.0f - s2coffee_conf);
        }
        else{
            classScores[SCENT_CLASS_COFFEE]+=coffee_prob * s2coffee_conf;
            classScores[SCENT_CLASS_DECAF_COFFEE]+=coffee_prob * (1.0f - s2coffee_conf);
        }
        
        int bestClass = 0;
        for(int i=1;i<SCENT_CLASS_COUNT;i++){
            if(classScores[i] > classScores[bestClass]){ 
                bestClass = i;
            }
        }
        return (scent_class_t)bestClass;
    }

    ml_metrics_t evaluate(const csv_training_sample_t* testSet, uint16_t testCount){
        ml_metrics_t m, mSoft;
        memset(&m, 0, sizeof(m));
        memset(&mSoft, 0, sizeof(mSoft));
        m.total = testCount;
        mSoft.total = testCount;
        
        int s1_correct =0;
        for(uint16_t i=0; i < testCount; i++){
            scent_class_t actual = testSet[i].label;
            
            scent_class_t pred = predict(testSet[i].features);
            if(pred==actual){
                m.correct++;
            }

            if(actual < SCENT_CLASS_COUNT && pred < SCENT_CLASS_COUNT){
                m.confusionMatrix[actual][pred]++;
            }
            
            scent_class_t predSoft = predictSoft(testSet[i].features);
            if(predSoft == actual){
                mSoft.correct++;
            }
            if(actual < SCENT_CLASS_COUNT && predSoft < SCENT_CLASS_COUNT){
                mSoft.confusionMatrix[actual][predSoft]++;
            }
            
            int actualGroup = (actual == SCENT_CLASS_DECAF_TEA || actual == SCENT_CLASS_TEA) ? 0 : 1;
            int predGroup=stage1.predict(testSet[i].features);
            if(actualGroup==predGroup){
                s1_correct++;
            }
        }

        m.accuracy = (m.total > 0) ? (float)m.correct / m.total : 0.0f;
        mSoft.accuracy = (mSoft.total > 0) ? (float)mSoft.correct / mSoft.total : 0.0f;
        
        int bevTotal = 0, bevHard = 0, bevSoft = 0, bevS1 = 0;
        for(uint16_t i=0; i < testCount; i++){
            if(testSet[i].label == SCENT_CLASS_AMBIENT) continue;
            bevTotal++;
            if(predict(testSet[i].features) == testSet[i].label) bevHard++;
            if(predictSoft(testSet[i].features) == testSet[i].label) bevSoft++;
            int actualGroup = (testSet[i].label == SCENT_CLASS_DECAF_TEA || testSet[i].label == SCENT_CLASS_TEA) ? 0 : 1;
            if((int)stage1.predict(testSet[i].features) == actualGroup) bevS1++;
        }
        float bevDen = (bevTotal > 0) ? (float)bevTotal : 1.0f;

        std::cout << "  Stage 1 test acc (beverages): " << std::fixed << std::setprecision(1)<< (100.0f * bevS1 / bevDen) << "%" << std::endl;
        std::cout << "  Hard routing:  " << (m.accuracy * 100) << "% all, " << (100.0f * bevHard / bevDen) << "% beverages only" << std::endl;
        std::cout << "  Soft routing:  " << (mSoft.accuracy * 100) << "% all, " << (100.0f * bevSoft / bevDen) << "% beverages only" << std::endl;
        (void)s1_correct;

        return m;
    }
};

//===========================================================================================================
//weighted ensemble
//===========================================================================================================
struct EnsembleWeights {
    float dt, knn, rf;
};

static scent_class_t deployedEnsemble(const float* features, const DecisionTree& dt, const KNN& knn, const RandomForest& rf, uint16_t featureCount,const EnsembleWeights& w, float threshold, float* confOut=nullptr){
    float dtConf = 0.0f, knnConf = 0.0f, rfConf = 0.0f;
    int dtCls = dt.predictWithConfidence(features, dtConf);
    int knnCls = knn.predictWithConfidence(features, featureCount, knnConf);
    int rfCls = rf.predictWithConfidence(features, rfConf);

    if(dtConf < threshold){
        dtCls = SCENT_CLASS_UNKNOWN;
        dtConf = 0.0f;
    }

    if(knnConf < threshold){
        knnCls = SCENT_CLASS_UNKNOWN;
        knnConf = 0.0f;
    }
    if(rfConf < threshold){
        rfCls = SCENT_CLASS_UNKNOWN;
        rfConf = 0.0f;
    }

    float scores[SCENT_CLASS_COUNT] = {0};
    auto vote = [&](int cls, float conf, float weight){
        if(cls >= SCENT_CLASS_COUNT) return;
        scores[cls] += weight * conf;
        float resid = weight * (1.0f - conf) / (SCENT_CLASS_COUNT - 1);
        for(int c = 0; c < SCENT_CLASS_COUNT; c++){
            if(c != cls){
                scores[c] += resid;
            }
        }
    };
    vote(knnCls, knnConf, w.knn);
    vote(rfCls, rfConf, w.rf);
    vote(dtCls, dtConf, w.dt);

    float bScore = 0.0f;
    int bClass = SCENT_CLASS_UNKNOWN;
    for(int c = 0; c < SCENT_CLASS_COUNT; c++){
        if(scores[c] > bScore){
            bScore = scores[c];
            bClass = c;
        }
    }

    bool dtKnn = (dtCls == knnCls) && (dtCls != SCENT_CLASS_UNKNOWN);
    bool knnRf = (knnCls == rfCls) && (knnCls != SCENT_CLASS_UNKNOWN);
    bool dtRf = (dtCls == rfCls) && (dtCls != SCENT_CLASS_UNKNOWN);
    if(!dtKnn && !knnRf && !dtRf){
        if(confOut) *confOut = 0.0f;
        return SCENT_CLASS_UNKNOWN;
    }
    float maxScore = w.dt + w.knn + w.rf;
    float conf = (maxScore > 0.0f) ? bScore / maxScore : 0.0f;
    if(confOut) *confOut = conf;

    if(conf < threshold){
        return SCENT_CLASS_UNKNOWN;
    }
    return (scent_class_t)bClass;
}

struct RejectMetrics {
    ml_metrics_t m;
    int rejected = 0;
    float coverage() const { return (m.total > 0) ? 1.0f - (float)rejected / m.total : 0.0f; }
    float acceptedAcc() const { return (m.total - rejected > 0) ? (float)m.correct / (m.total - rejected) : 0.0f; }
};

template<typename P>
static RejectMetrics evaluateWithReject(const csv_training_sample_t* set, uint16_t count, P predict){
    RejectMetrics r;
    memset(&r.m, 0, sizeof(r.m));
    r.m.total = count;
    for(uint16_t i = 0; i < count; i++){
        scent_class_t actual = set[i].label;
        scent_class_t pred = predict(set[i].features);
        if(pred >= SCENT_CLASS_COUNT){
            r.rejected++;
            continue;
        }
        if(pred == actual) r.m.correct++;
        if(actual < SCENT_CLASS_COUNT) r.m.confusionMatrix[actual][pred]++;
    }
    r.m.accuracy = (r.m.total > 0) ? (float)r.m.correct / r.m.total : 0.0f;
    return r;
}

static void printReject(const char* name, const RejectMetrics& r){
    std::cout << "  " << std::left << std::setw(18) << name << std::right << std::fixed << std::setprecision(1)
              << "acc " << (r.m.accuracy * 100) << "%  coverage " << (r.coverage() * 100)
              << "%  acc on accepted " << (r.acceptedAcc() * 100) << "%" << std::endl;
}

static void writeFisherHeader(const std::string& filename, const Pipeline& p){
    std::ofstream f(filename);
    f << "// Auto-generated Fisher projection weights\n";
    f << "// Derived from the training partition + robust-scaled pre-Fisher features\n";
    f << "// Feature space: first " << p.baseCount << " scaled features (indices 0.." << (p.baseCount - 1) << ")\n\n";
    f << "#ifndef FISHER_WEIGHTS_H\n#define FISHER_WEIGHTS_H\n\n";
    f << "#define FISHER_PAIR_COUNT " << p.numFisher << "\n";
    f << "#define FISHER_BASE_FEATURE_COUNT " << p.baseCount << "\n\n";
    f << "static const float FISHER_WEIGHTS[FISHER_PAIR_COUNT][FISHER_BASE_FEATURE_COUNT] = {\n";
    for(int k = 0; k < p.numFisher; k++){
        f << "    { // [" << k << "] " << FISHER_PAIRS[p.fisherPairIdx[k]].name << "\n";
        for(int i = 0; i < p.baseCount; i++){
            f << "        " << cFloat(p.fisherW[k][i])<< (i < p.baseCount - 1 ? "," : "") << "\n";
        }
        f << "    }" << (k < p.numFisher - 1 ? "," : "") << "\n";
    }
    f << "};\n\n#endif // FISHER_WEIGHTS_H\n";
    std::cout << "Fisher weights saved to " << filename << std::endl;
}

static void writeStatsHeader(const std::string& filename, const Pipeline& p){
    std::ofstream sf(filename);
    sf << "#ifndef FEATURE_STATS_H\n#define FEATURE_STATS_H\n\n";
    sf << "#define FULL_FEATURE_COUNT " << p.fullCount << "\n\n";
    sf << "static const float FEATURE_MEDIANS[" << p.fullCount << "]={";
    for(int f = 0; f < p.fullCount; f++){
        sf << std::setprecision(9) << p.medians[f] << (f < p.fullCount - 1 ? ", " : "");
    }
    sf << "};\n\nstatic const float FEATURE_IQRS[" << p.fullCount << "]={";
    for(int f = 0; f < p.fullCount; f++){
        sf << std::setprecision(9) << p.iqrs[f] << (f < p.fullCount - 1 ? ", " : "");
    }
    sf << "};\n\n#endif\n";
}

static void writeKnnBinary(const std::string& filename, const csv_training_sample_t* samples, uint16_t count,uint8_t k, uint16_t featureCount){
    std::ofstream file(filename, std::ios::binary);
    const uint32_t magic = 0x4B4E4E4C;
    const uint16_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&k), sizeof(k));
    file.write(reinterpret_cast<const char*>(&featureCount), sizeof(featureCount));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for(uint16_t i = 0; i < count; i++){
        uint8_t label = static_cast<uint8_t>(samples[i].label);
        file.write(reinterpret_cast<const char*>(&label), sizeof(label));
        file.write(reinterpret_cast<const char*>(samples[i].features), featureCount * sizeof(float));
    }
    std::cout << "KNN model saved to " << filename << " (" << featureCount << " features)" << std::endl;
}

//===========================================================================================================
//Edge Impulse export
//===========================================================================================================
static std::string csvFeatureName(int rank, int idx){
    std::string n = shortFeatureName(idx);
    std::string out;
    for(char c : n){
        if(c == '*') out += "_x_";
        else if(c == '/') out += "_per_";
        else if(std::isalnum((unsigned char)c) || c == '_') out += c;
        else out += '_';
    }
    std::ostringstream o;
    o << "f" << std::setw(3) << std::setfill('0') << rank << "_" << out;
    return o.str();
}

static void writeFeatureCsv(const std::string& filename, const csv_training_sample_t* samples, uint16_t count,const FeatureSelector& sel, int nFeatures){
    std::ofstream f(filename);
    f << "label";
    for(int i = 0; i < nFeatures; i++){
        f << "," << csvFeatureName(i, sel.selectedIndices[i]);
    }
    f << "\n" << std::setprecision(9);
    for(uint16_t s = 0; s < count; s++){
        f << CSVLoader::getClassName(samples[s].label);
        for(int i = 0; i < nFeatures; i++){
            f << "," << samples[s].features[i];
        }
        f << "\n";
    }
    std::cout << "Features exported to " << filename << " (" << count << " rows, " << nFeatures << " features)" << std::endl;
}

//===========================================================================================================
//ambient gate
//===========================================================================================================
static const float AMBIENT_SCORE_SCALE = 0.15f;

static void writeAmbientGateHeader(const std::string& filename, const CSVLoader& loader){
    const std::vector<float>& cal = loader.getCalibrationDeviations();
    const std::vector<float>& dev = loader.getGateDeviations();
    std::cout << "\n=== Ambient Gate Calibration ===" << std::endl;
    if(cal.empty()){
        std::cout << "  No calibration sweeps, gate header not written" << std::endl;
        return;
    }

    float calMax = *std::max_element(cal.begin(), cal.end());
    float threshold = 1.5f * calMax;
    bool usable = threshold < 0.9f * AMBIENT_SCORE_SCALE;
    float scoreThreshold = usable ? 1.0f - threshold / AMBIENT_SCORE_SCALE : 1.0f; //1.0 = never fires

    int n[SCENT_CLASS_COUNT] = {0}, gated[SCENT_CLASS_COUNT] = {0};
    const csv_training_sample_t* samples = loader.getSamples();
    for(size_t i = 0; i < dev.size(); i++){
        int c = samples[i].label;
        if(c >= SCENT_CLASS_COUNT){ 
            continue;
        }

        n[c]++;
        if(dev[i] < threshold){
            gated[c]++;
        }
    }

    std::cout << "  Calibration sweeps: " << cal.size() << ", largest mean deviation " << std::fixed<< std::setprecision(1) << calMax * 100 << "%" << std::endl;
    std::cout << "  Gate: mean deviation < " << threshold * 100 << "% (score > " << std::setprecision(3)<< scoreThreshold << ")" << (usable ? "" : "  -- too noisy, gate disabled") << std::endl;
    for(int c = 0; c < SCENT_CLASS_COUNT; c++){
        std::cout << "    " << std::setw(13) << CSVLoader::getClassName((scent_class_t)c) << ": " << gated[c] << "/" << n[c]<< " sweeps would be gated as ambient" << std::endl;
    }

    std::ofstream f(filename);
    f << "#ifndef AMBIENT_GATE_H\n#define AMBIENT_GATE_H\n\n";
    f << "// ambient gate: report ambient without classifying if computeAmbientScore() > AMBIENT_GATE_SCORE\n";
    f << "// score = 1 - min(mean |R/R_base - 1| / " << AMBIENT_SCORE_SCALE << ", 1); threshold = 1.5 x the largest mean\n";
    f << "// deviation of the " << cal.size() << " calibration sweeps (" << std::setprecision(4) << calMax << ")\n";
    
    if(!usable) f << "// calibration sweeps too noisy for a useful gate: 1.0 disables it\n";
    
    f << "#define AMBIENT_GATE_MAX_DEVIATION " << cFloat(threshold) << "\n";
    f << "#define AMBIENT_GATE_SCORE " << cFloat(scoreThreshold) << "\n";
    f << "\n#endif\n";
    std::cout << "Ambient gate saved to " << filename << std::endl;
}

static void printUsage(){
    std::cout << "Usage: ml_trainer [data.csv] [trainRatio] [--features N|auto] [--max-features N] [--feature-step N]\n" << "                  [--seed N] [--split random|session] [--out DIR] [--quick] [--exclude-env-derived] [--export-only]" << std::endl;
}


int main(int argc, char* argv[]){
    //===========================================================================================================
    //arguments
    //===========================================================================================================
    std::string filename="data.csv";
    {
        int positional = 0;
        for(int i = 1; i < argc; i++){
            std::string a = argv[i];
            auto next = [&](const char* flag) -> std::string {
                if(i + 1 >= argc){
                    std::cerr << "Missing value for " << flag << std::endl;
                    exit(1);
                }
                return argv[++i];
            };
            if(a == "--features"){
                std::string v = next("--features");
                NUM_SELECTED_FEATURES = (v == "auto") ? 0 : std::stoi(v);
            }
            else if(a == "--max-features"){
                MAX_SELECTED_FEATURES = std::stoi(next("--max-features"));
            }
            else if(a == "--feature-step"){
                FEATURE_STEP = std::max(1, std::stoi(next("--feature-step")));
            }
            else if(a == "--seed"){
                RANDOM_SEED = (uint32_t)std::stoul(next("--seed"));
            }
            else if(a == "--split"){
                SESSION_SPLIT = (next("--split") == "session");
            }
            else if(a == "--out"){
                OUTPUT_DIR = next("--out");
            }
            else if(a == "--quick"){
                QUICK_RF_GRID = true;
            }
            else if(a == "--no-sweep") {}
            else if(a == "--exclude-env-derived"){
                EXCLUDE_ENV_DERIVED = true;
            }
            else if(a == "--export-only"){
                EXPORT_ONLY = true;
            }
            else if(a == "--help" || a == "-h"){
                printUsage();
                return 0;
            }
            else if(a.rfind("--", 0) == 0){
                std::cerr << "Unknown option " << a << std::endl; printUsage();
                return 1;
            }
            else if(positional == 0){
                filename = a;
                positional++;
            }
            else if(positional == 1){
                TRAIN_RATIO = std::stof(a);
                positional++;
            }
        }
        if(NUM_SELECTED_FEATURES < 0 || NUM_SELECTED_FEATURES > 157 || MAX_SELECTED_FEATURES < 0){
            std::cerr << "--features must be auto or between 1 and 157" << std::endl;
            return 1;
        }
        std::filesystem::create_directories(OUTPUT_DIR);
    }

    std::ofstream logFile(outPath("training_output.txt"));

    class TeeBuf : public std::streambuf {
    public:
        TeeBuf(std::streambuf* sb1, std::streambuf* sb2) : sb1(sb1), sb2(sb2) {}
    protected:
        int overflow(int c) override {
            if(c == EOF) return !EOF;
            if(sb1->sputc(c) == EOF) return EOF;
            if(sb2->sputc(c) == EOF) return EOF;
            return c;
        }
        int sync() override {
            sb1->pubsync();
            sb2->pubsync();
            return 0;
        }
    private:
        std::streambuf* sb1;
        std::streambuf* sb2;
    };

    TeeBuf teeBuf(std::cout.rdbuf(), logFile.rdbuf());
    std::streambuf* originalBuf = std::cout.rdbuf(&teeBuf);

    //capture error
    TeeBuf teeErrBuf(std::cerr.rdbuf(), logFile.rdbuf());
    std::streambuf* originalErrBuf = std::cerr.rdbuf(&teeErrBuf);

    RandomForest::setSeed(RANDOM_SEED);

    std::cout <<"========================================" <<std::endl;
    std::cout <<"    Machine Learning Model Trainer" <<std::endl;
    std::cout <<"========================================" <<std::endl;
    std::cout << "\nLoading: " << filename << std::endl;
    if(NUM_SELECTED_FEATURES > 0){
        std::cout << "Features: " << NUM_SELECTED_FEATURES << " for every model" << std::endl;
    }
    else{
        std::cout << "Features: chosen per model by CV (" << MIN_SEARCH_FEATURES << " to "
                  << (MAX_SELECTED_FEATURES > 0 ? std::to_string(MAX_SELECTED_FEATURES) : std::string("all eligible"))
                  << ", step " << FEATURE_STEP << ")" << std::endl;
    }
    std::cout << "Split: " << (SESSION_SPLIT ? "session-wise" : "stratified random") << " "
              << (TRAIN_RATIO * 100) << "/" << ((1 - TRAIN_RATIO) * 100) << ", seed " << RANDOM_SEED << std::endl;
    std::cout << "RF grid: " << (QUICK_RF_GRID ? "quick" : "full") << ", CV folds: " << CV_FOLDS << std::endl;
    std::cout << "Env features: raw excluded from selection" << (EXCLUDE_ENV_FROM_FISHER ? " and Fisher" : "")
              << (EXCLUDE_ENV_DERIVED ? "; env-derived features excluded" : "; env-derived features allowed") << std::endl;
    std::cout << "Output directory: " << OUTPUT_DIR << std::endl;

    CSVLoader loader;
    if(!loader.load(filename)){
        std::cerr << "Failed to load data!" << std::endl;
        return 1;
    }

    loader.printInfo();
    CSVLoader::printFeatureNames();
    writeAmbientGateHeader(outPath("ambient_gate.h"), loader);

    //split
    csv_training_sample_t *trainSet=nullptr, *testSet=nullptr;
    uint16_t trainCount=0,testCount=0;
    loader.split(TRAIN_RATIO, RANDOM_SEED, SESSION_SPLIT, trainSet, trainCount, testSet, testCount);

    //===========================================================================================================
    //feature engineering
    //===========================================================================================================
    int engineeredFeatureCount=BASE_FEATURE_COUNT;
    {
        Timer t("Feature Engineering");
        engineeredFeatureCount=addEngineeredFeatures(trainSet, trainCount, BASE_FEATURE_COUNT);
        addEngineeredFeatures(testSet,testCount,BASE_FEATURE_COUNT);
    }

    //unscaled copies: training partition for cross-validation, holdout for the device parity check
    std::vector<csv_training_sample_t> trainEngineered(trainSet, trainSet + trainCount);
    std::vector<csv_training_sample_t> testEngineered(testSet, testSet + testCount);

    //===========================================================================================================
    //scaling, Fisher projections and feature selection
    //===========================================================================================================
    Pipeline pipe;
    {
        Timer t("Scaling, Fisher Features and Selection");
        std::cout << std::endl;
        int requested = (NUM_SELECTED_FEATURES > 0) ? NUM_SELECTED_FEATURES: (MAX_SELECTED_FEATURES > 0) ? MAX_SELECTED_FEATURES : CSV_FEATURE_COUNT;
        fitPipeline(trainSet, trainCount, engineeredFeatureCount, requested, pipe, true);
        applyPipeline(testSet, testCount, pipe);

        std::set<int> blacklist = selectionBlacklist();
        int envSelected = 0;
        for(int idx : pipe.selector.selectedIndices){
            if(blacklist.count(idx)) envSelected++;
        }
        std::cout << "  Blacklisted features in final selection: " << envSelected << "/" << pipe.selector.selectedCount << std::endl;
    }

    const int maxFeatures = pipe.selector.selectedCount;
    const int fullFeatureCount = pipe.fullCount;

    //feature counts searched per model
    std::vector<int> featureCounts;
    if(NUM_SELECTED_FEATURES > 0){
        featureCounts.push_back(maxFeatures);
    }
    else{
        for(int n = std::min(MIN_SEARCH_FEATURES, maxFeatures); n < maxFeatures; n += FEATURE_STEP){
            featureCounts.push_back(n);
        }
        featureCounts.push_back(maxFeatures);
    }
    std::cout << "\nFeature ranking (selected features are stored best-first; a model using N features reads the first N):" << std::endl;
    for(int i = 0; i < maxFeatures; i++){
        int idx = pipe.selector.selectedIndices[i];
        std::cout << "  [" << std::setw(3) << i << "] " << std::setw(3) << idx << " " << shortFeatureName(idx) << std::endl;
    }
    std::cout << "Feature counts searched per model:";
    for(int n : featureCounts) std::cout << " " << n;
    std::cout << std::endl;

    if(EXPORT_ONLY){
        //every ranked feature
        std::vector<csv_training_sample_t> tr(trainCount), te(testCount);
        pipe.selector.projectDataset(trainSet, trainCount, tr.data());
        pipe.selector.projectDataset(testSet, testCount, te.data());
        std::cout << std::endl;
        writeFeatureCsv(outPath("features_train.csv"), tr.data(), trainCount, pipe.selector, maxFeatures);
        writeFeatureCsv(outPath("features_test.csv"), te.data(), testCount, pipe.selector, maxFeatures);
        delete[] trainSet;
        delete[] testSet;
        std::cout.rdbuf(originalBuf);
        std::cerr.rdbuf(originalErrBuf);
        logFile.close();
        std::cout << "\nExport complete; output saved to " << outPath("training_output.txt") << std::endl;
        return 0;
    }

    //batch effect
    std::cout << "\nChecking for batch effects..." << std::endl;

    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        std::vector<uint16_t> idx;
        for(uint16_t j=0;j<trainCount;j++){
            if(trainSet[j].label == i){
                idx.push_back(j);
            }
        }
        if(idx.size()<20){
            continue;
        }

        uint16_t mid=idx.size()/2;

        std::cout << "\n  " << CSVLoader::getClassName((scent_class_t)i)<< " (" << idx.size() << " samples):" << std::endl;

        int checkFeat[] ={20, 21, 22, 23, 25, 27, 30, 31, 33, 35, 37, 42, 47, 70, 71};

        for(int f:checkFeat){
            if(f>= fullFeatureCount){
                continue;
            }

            float s1=0.0f, s2=0.0f;

            for(uint16_t j=0; j<mid;j++){
                s1+=trainSet[idx[j]].features[f];
            }
            for(uint16_t j=mid; j<idx.size();j++){
                s2+=trainSet[idx[j]].features[f];
            }

            float m1= s1 / mid;
            float m2= s2 / (idx.size() - mid);
            float shift= fabsf(m1 - m2);

            if(shift>0.3f){
                std::cout << "[!]"<< shortFeatureName(f) << ":first_half=" << std::fixed << std::setprecision(2) << m1<< " second_half=" << m2<< " shift=" << shift << " std" << std::endl;
            }
        }
    }

    std::cout << "\n=== Coffee vs Decaf_Tea Feature Analysis ===" << std::endl;
    int goodFeats = 0;
    for(int f = 0; f < fullFeatureCount; f++){
        float mean_coffee = 0,mean_dtea = 0, var_coffee=0, var_dtea = 0;
        int n_coffee = 0,n_dtea = 0;

        for(int i = 0; i<trainCount;i++){
            if(trainSet[i].label==SCENT_CLASS_COFFEE){
                mean_coffee+=trainSet[i].features[f];
                n_coffee++;
            }
            else if(trainSet[i].label==SCENT_CLASS_DECAF_TEA){
                mean_dtea+=trainSet[i].features[f];
                n_dtea++;
            }
        }
        if(n_coffee==0||n_dtea==0){
            continue;
        }
        mean_coffee/=n_coffee;
        mean_dtea/=n_dtea;

        for(int i=0; i<trainCount;i++){
            if(trainSet[i].label==SCENT_CLASS_COFFEE){
                float d=trainSet[i].features[f]-mean_coffee;
                var_coffee+= d*d;
            }
            else if(trainSet[i].label==SCENT_CLASS_DECAF_TEA){
                float d=trainSet[i].features[f]-mean_dtea;
                var_dtea+=d*d;
            }
        }
        var_coffee/=n_coffee;
        var_dtea/=n_dtea;

        float pooled_std = sqrtf((var_coffee + var_dtea) / 2.0f);
        float separation = (pooled_std > 1e-6f)? fabsf(mean_coffee - mean_dtea)/pooled_std: 0.0f;

        if(separation > 0.3f){
            goodFeats++;
            std::cout << "  [" << std::setw(3) << f << "] "<< std::setw(16) << shortFeatureName(f)<< " sep=" << std::fixed << std::setprecision(2) << separation
            <<" (coffee=" << std::setprecision(3) << mean_coffee<<" dtea=" << mean_dtea << ")"<<std::endl;
        }
    }

    std::cout << "\nFeatures with separation > 0.3: "<< goodFeats<<"/"<<fullFeatureCount<<std::endl;

    //===========================================================================================================
    //project to the ranked selection
    //===========================================================================================================
    csv_training_sample_t* trainReduced=new csv_training_sample_t[trainCount];
    csv_training_sample_t* testReduced=new csv_training_sample_t[testCount];
    pipe.selector.projectDataset(trainSet, trainCount,trainReduced);
    pipe.selector.projectDataset(testSet, testCount,testReduced);
    delete[] trainSet;
    delete[] testSet;
    trainSet=trainReduced;
    testSet=testReduced;

    //CV folds used for search
    std::vector<Fold> folds;
    {
        Timer t("Building CV folds");
        folds = buildFolds(trainEngineered.data(), trainCount, engineeredFeatureCount,maxFeatures, CV_FOLDS, RANDOM_SEED);
        
        int minFold = maxFeatures;
        for(const Fold& f : folds) minFold = std::min<int>(minFold, f.featureCount);
        if(minFold < maxFeatures){
            std::cout << "\nNOTE: one fold selected only " << minFold << " features (score threshold);"<< " larger counts read zeros there" << std::endl;
        }
    }
    if(SESSION_SPLIT){
        std::cout << "NOTE: CV folds are drawn at random within the training runs, so CV accuracy is within-session" << std::endl;
    }

    std::ofstream gridCsv(outPath("search_grid.csv"));
    gridCsv << "model,features,p1,p2,p3,p4,cv_mean,cv_std\n" << std::fixed << std::setprecision(4);
    std::ofstream countCsv(outPath("feature_count_search.csv"));
    countCsv << "model,features,best_cv_mean,best_cv_std,best_params\n" << std::fixed << std::setprecision(4);

    //========================================================
    //Decision Tree
    //========================================================
    int dtFeat = 0;
    uint8_t finalDTDepth=0, finalDTMinSamples=0;
    CVStat dtCV;
    {
        Timer t("Decision Tree CV Search");
        std::cout << "\n\n=== Decision Tree: feature count x hyperparameters ===" << std::endl;
        for(int n : featureCounts){
            CVStat bestN;
            uint8_t bd = 0, bms = 0;
            for(uint8_t d=6; d <= 14; d += 2){
                for(uint8_t ms : {5, 10, 20, 30}){
                    CVStat s = runCV(folds, [&](const Fold& f){
                        QuietCout q;
                        DecisionTree dtTest;
                        dtTest.train(f.train.data(), f.train.size(), n, d, ms);
                        return dtTest.evaluate(f.test.data(), f.test.size());
                    });
                    gridCsv << "dt," << n << "," << (int)d << "," << (int)ms << ",,," << s.mean << "," << s.std << "\n";
                    if(s.mean > bestN.mean){ bestN = s; bd = d; bms = ms; }
                }
            }
            std::cout << "  features=" << std::setw(3) << n << "  best d=" << (int)bd << " ms=" << (int)bms
                      << "  cv=" << pct(bestN.mean, bestN.std) << std::endl;
            countCsv << "dt," << n << "," << bestN.mean << "," << bestN.std << ",d=" << (int)bd << " ms=" << (int)bms << "\n";
            if(bestN.mean > dtCV.mean){
                dtCV = bestN; dtFeat = n; finalDTDepth = bd; finalDTMinSamples = bms;
            }
        }
        std::cout << "\n  Selected: " << dtFeat << " features, d=" << (int)finalDTDepth << " ms=" << (int)finalDTMinSamples
                  << " cv=" << pct(dtCV.mean, dtCV.std) << std::endl;
    }

    DecisionTree dt;
    dt.train(trainSet, trainCount, dtFeat, finalDTDepth, finalDTMinSamples);
    ml_metrics_t dtMetrics=dt.evaluate(testSet, testCount);
    printMetrics(dtMetrics, "Decision Tree (holdout)");
    printPerClassMetrics(dtMetrics);
    printConfusionMatrix(dtMetrics);

    //========================================================
    //KNN
    //========================================================
    int knnFeat = 0;
    uint8_t bestK = 5;
    CVStat knnCV;
    {
        Timer t("KNN CV Search");
        std::cout << "\n\n=== KNN: feature count x k ===" << std::endl;
        for(int n : featureCounts){
            CVStat bestN;
            int bk = 5;
            for(int k = 5; k <= 25; k += 2){
                CVStat s = runCV(folds, [&](const Fold& f){
                    KNN kf((uint8_t)k);
                    kf.train(f.train.data(), f.train.size());
                    return kf.evaluate(f.test.data(), f.test.size(), n);
                });
                gridCsv << "knn," << n << "," << k << ",,,," << s.mean << "," << s.std << "\n";
                if(s.mean > bestN.mean){ bestN = s; bk = k; }
            }
            std::cout << "  features=" << std::setw(3) << n << "  best k=" << bk << "  cv=" << pct(bestN.mean, bestN.std) << std::endl;
            countCsv << "knn," << n << "," << bestN.mean << "," << bestN.std << ",k=" << bk << "\n";
            if(bestN.mean > knnCV.mean){
                knnCV = bestN; knnFeat = n; bestK = bk;
            }
        }
        std::cout << "\n  Selected: " << knnFeat << " features, k=" << (int)bestK << " cv=" << pct(knnCV.mean, knnCV.std) << std::endl;
    }

    KNN knn(bestK);
    knn.train(trainSet, trainCount);
    ml_metrics_t knnMetrics=knn.evaluate(testSet, testCount, knnFeat);
    printMetrics(knnMetrics,"KNN (holdout)");
    printPerClassMetrics(knnMetrics);
    printConfusionMatrix(knnMetrics);

    //========================================================
    //KNN anomaly calibration]
    //========================================================
    {
        std::cout << "\n=== KNN Distance Calibration ===" << std::endl;
        std::vector<float> meanDists(trainCount);
        std::vector<float> d(trainCount);
        for(uint16_t i = 0; i < trainCount; i++){
            int n = 0;
            for(uint16_t j = 0; j < trainCount; j++){
                if(j == i) continue;
                float s = 0.0f;
                for(int f = 0; f < knnFeat; f++){
                    float diff = trainSet[i].features[f] - trainSet[j].features[f];
                    s += diff * diff;
                }
                d[n++] = sqrtf(s);
            }
            int k = std::min<int>(bestK, n);
            std::partial_sort(d.begin(), d.begin() + k, d.begin() + n);
            float sum = 0.0f;
            for(int j = 0; j < k; j++) sum += d[j];
            meanDists[i] = (k > 0) ? sum / k : 0.0f;
        }
        std::vector<float> sorted = meanDists;
        std::sort(sorted.begin(), sorted.end());
        float p50 = sorted[sorted.size() / 2];
        float p95 = sorted[std::min<size_t>(sorted.size() - 1, (size_t)(sorted.size() * 0.95f))];

        //anomaly score on device = min(meanDist / KNN_DISTANCE_SCALE, 1)
        float distance_scale = p95 * 2.5f;
        float calibrated_threshold = (p95 * 1.5f) / distance_scale; //score units (= 0.6)

        std::cout << "  Median LOO mean distance: " << std::setprecision(4) << p50 << std::endl;
        std::cout << "  95th percentile:          " << p95 << std::endl;
        std::cout << "  Distance scale:           " << distance_scale << std::endl;
        std::cout << "  Anomaly threshold (score):" << calibrated_threshold << std::endl;

        std::ofstream athf(outPath("anomaly_threshold.h"));
        athf << "#ifndef ANOMALY_THRESHOLD_H\n#define ANOMALY_THRESHOLD_H\n\n";
        athf << "// Calibrated from leave-one-out " << (int)bestK << "-NN mean distances of " << trainCount << " training samples (" << knnFeat << " features)\n";
        athf << "// anomaly score = min(mean neighbour distance / KNN_DISTANCE_SCALE, 1)\n";
        athf << "// threshold is in score units (1.5 x the 95th percentile distance)\n";
        athf << "static const float CALIBRATED_ANOMALY_THRESHOLD = " << cFloat(calibrated_threshold) << ";\n";
        athf << "static const float KNN_DISTANCE_SCALE = " << cFloat(distance_scale) << ";\n";
        athf << "\n#endif\n";
    }

    //========================================================
    //RF
    //========================================================
    struct RFConfig {
        int features;
        int trees;
        int depth;
        float sr;
        int minSamples;
        CVStat cv;
    };
    RFConfig bestConfig{0, 0, 0, 0.0f, 0, CVStat()};

    std::vector<int> treeGrid = QUICK_RF_GRID ? std::vector<int>{100, 200} : std::vector<int>{50, 100, 200, 500};
    std::vector<int> depthGrid = QUICK_RF_GRID ? std::vector<int>{8, 0} : std::vector<int>{4, 6, 8, 10, 14, 0}; // 0 = unlimited
    std::vector<float> srGrid = QUICK_RF_GRID ? std::vector<float>{0.3f, 0.5f, 0.7f} : std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f};
    std::vector<int> msGrid = QUICK_RF_GRID ? std::vector<int>{3, 5} : std::vector<int>{1, 3, 5, 10};

    {
        Timer t("Random Forest CV Search");
        std::cout << "\n\n=== Random Forest: feature count x hyperparameters ("
                  << treeGrid.size() * depthGrid.size() * srGrid.size() * msGrid.size() << " configs per count) ===" << std::endl;
        auto searchStart = std::chrono::steady_clock::now();

        for(size_t ci = 0; ci < featureCounts.size(); ci++){
            int n = featureCounts[ci];
            RFConfig bestN{n, 0, 0, 0.0f, 0, CVStat()};
            for(int trees : treeGrid){
                for(int depth : depthGrid){
                    for(float sr : srGrid){
                        for(int ms : msGrid){
                            int actualDepth = (depth == 0) ? 255 : depth;
                            CVStat s = runCV(folds, [&](const Fold& f){
                                QuietCout q;
                                RandomForest rf(trees, actualDepth, ms, sr);
                                rf.train(f.train.data(), f.train.size(), n);
                                return rf.evaluate(f.test.data(), f.test.size());
                            });
                            gridCsv << "rf," << n << "," << trees << "," << depth << "," << sr << "," << ms << ","
                                    << s.mean << "," << s.std << "\n";
                            //first maximum wins, so ties go to the smaller (earlier) configuration
                            if(s.mean > bestN.cv.mean){
                                bestN = {n, trees, actualDepth, sr, ms, s};
                            }
                        }
                    }
                }
            }
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
            std::cout << "  features=" << std::setw(3) << n << "  best t=" << bestN.trees << " d=" << bestN.depth
                      << " sr=" << std::setprecision(1) << bestN.sr << " ms=" << bestN.minSamples
                      << "  cv=" << pct(bestN.cv.mean, bestN.cv.std)
                      << "   [" << (ci + 1) << "/" << featureCounts.size() << ", " << std::setprecision(0) << elapsed / 60.0 << " min]" << std::endl;
            countCsv << "rf," << n << "," << bestN.cv.mean << "," << bestN.cv.std << ",t=" << bestN.trees << " d=" << bestN.depth
                     << " sr=" << bestN.sr << " ms=" << bestN.minSamples << "\n";
            gridCsv.flush();
            countCsv.flush();
            if(bestN.cv.mean > bestConfig.cv.mean){
                bestConfig = bestN;
            }
        }
    }
    const int rfFeat = bestConfig.features;
    CVStat rfCV = bestConfig.cv;

    std::cout << "\n  Selected: " << rfFeat << " features, t=" << bestConfig.trees
            << " d=" << bestConfig.depth << " sr=" << bestConfig.sr
            << " ms=" << bestConfig.minSamples
            << " cv=" << pct(rfCV.mean, rfCV.std) << std::endl;

    std::cout << "\nAggregate CV Results (best RF):" << std::endl;
    printConfusionMatrix(rfCV.agg);
    printPerClassMetrics(rfCV.agg);

    std::cout << "\nTraining final model on full training set..." << std::endl;
    RandomForest* bestRf = new RandomForest(bestConfig.trees, bestConfig.depth,
        bestConfig.minSamples, bestConfig.sr);
    bestRf->train(trainSet, trainCount, rfFeat);
    ml_metrics_t rfMetrics = bestRf->evaluate(testSet, testCount);

    printMetrics(rfMetrics, "Random Forest (holdout)");
    printPerClassMetrics(rfMetrics);
    printConfusionMatrix(rfMetrics);
    bestRf->printFeatureImportance();

    //the exported selection only needs to cover the largest model
    const int usedFeatures = std::max({dtFeat, knnFeat, rfFeat});
    pipe.selector.selectedIndices.resize(usedFeatures);
    pipe.selector.selectedCount = usedFeatures;
    {
        std::ostringstream note;
        note << "ranked best-first: DT uses the first " << dtFeat << ", KNN the first " << knnFeat
             << ", RF the first " << rfFeat;
        pipe.selector.saveHeader(outPath("feature_select.h"), pipe.medians, pipe.iqrs, note.str());
    }
    writeStatsHeader(outPath("feature_stats.h"), pipe);
    writeFisherHeader(outPath("fisher_weights.h"), pipe);
    writeFeatureCsv(outPath("features_train.csv"), trainSet, trainCount, pipe.selector, usedFeatures);
    writeFeatureCsv(outPath("features_test.csv"), testSet, testCount, pipe.selector, usedFeatures);

    //save
    dt.getStats(); //fills in the tree depth stored in the binary
    dt.saveModel(outPath("dt_model.bin").c_str());
    bestRf->saveModel(outPath("rf_model.bin").c_str());
    writeKnnBinary(outPath("knn_model.bin"), trainSet, trainCount, bestK, knnFeat);

    //===========================================================================================================
    //experiments on the RF feature set (not deployed)
    //===========================================================================================================
    twoStageExperiment(trainSet, trainCount, testSet, testCount, rfFeat);

    HierarchicalClassifier hier;
    ml_metrics_t hierMetrics;
    {
        Timer t("Hierarchical Classifier");
        std::cout << std::endl;

        hier.train(trainSet,trainCount,rfFeat);
        hierMetrics = hier.evaluate(testSet, testCount);

        printMetrics(hierMetrics,"Hierarchical Classifier (hard routing)");
        printPerClassMetrics(hierMetrics);
        printConfusionMatrix(hierMetrics);
    }

    //===========================================================================================================
    //ensemble (weights = CV acc^4, evaluated exactly as deployed
    //===========================================================================================================
    EnsembleWeights ensW{powf(dtCV.mean, 4), powf(knnCV.mean, 4), powf(rfCV.mean, 4)};
    RejectMetrics ensHoldout;
    CVStat ensCV;
    float ensCVCoverage = 0.0f;
    {
        Timer t("Weighted Ensemble Evaluation");
        std::cout << "\n\n=== Weighted Ensemble Evaluation (as deployed) ===" << std::endl;
        std::cout << "  Weights (CV acc^4): DT=" << std::fixed << std::setprecision(3) << ensW.dt << " KNN=" << ensW.knn << " RF=" << ensW.rf << "  threshold=" << std::setprecision(2) << CONFIDENCE_THRESHOLD << std::endl;

        ensHoldout = evaluateWithReject(testSet, testCount, [&](const float* x){
            return deployedEnsemble(x, dt, knn, *bestRf, knnFeat, ensW, CONFIDENCE_THRESHOLD);
        });
        printMetrics(ensHoldout.m, "Weighted Ensemble (holdout, Unknown counted as wrong)");
        std::cout << "Unknown outputs: " << ensHoldout.rejected << "/" << testCount << std::endl;
        printPerClassMetrics(ensHoldout.m);
        printConfusionMatrix(ensHoldout.m);

        int cvRejected = 0, cvTotal = 0;
        ensCV = runCV(folds, [&](const Fold& f){
            QuietCout q;
            DecisionTree fdt;
            fdt.train(f.train.data(), f.train.size(), dtFeat, finalDTDepth, finalDTMinSamples);
            KNN fknn(bestK);
            fknn.train(f.train.data(), f.train.size());
            RandomForest frf(bestConfig.trees, bestConfig.depth, bestConfig.minSamples, bestConfig.sr);
            frf.train(f.train.data(), f.train.size(), rfFeat);
            RejectMetrics r = evaluateWithReject(f.test.data(), f.test.size(), [&](const float* x){
                return deployedEnsemble(x, fdt, fknn, frf, knnFeat, ensW, CONFIDENCE_THRESHOLD);
            });
            cvRejected += r.rejected;
            cvTotal += r.m.total;
            return r.m;
        });

        ensCVCoverage = (cvTotal > 0) ? 1.0f - (float)cvRejected / cvTotal : 0.0f;
        std::cout << "\n  Ensemble CV: " << pct(ensCV.mean, ensCV.std)<< " (coverage " << std::setprecision(1) << (ensCVCoverage * 100) << "%)" << std::endl;

        std::cout << "\n=== Holdout with " << CONFIDENCE_THRESHOLD << " confidence threshold (single-model mode) ===" << std::endl;
        printReject("Decision Tree", evaluateWithReject(testSet, testCount, [&](const float* x){
            float c; scent_class_t p = dt.predictWithConfidence(x, c);
            return (c < CONFIDENCE_THRESHOLD) ? SCENT_CLASS_UNKNOWN : p;
        }));

        printReject("KNN", evaluateWithReject(testSet, testCount, [&](const float* x){
            float c; scent_class_t p = knn.predictWithConfidence(x, knnFeat, c);
            return (c < CONFIDENCE_THRESHOLD) ? SCENT_CLASS_UNKNOWN : p;
        }));

        printReject("Random Forest", evaluateWithReject(testSet, testCount, [&](const float* x){
            float c; scent_class_t p = bestRf->predictWithConfidence(x, c);
            return (c < CONFIDENCE_THRESHOLD) ? SCENT_CLASS_UNKNOWN : p;
        }));
        printReject("Ensemble", ensHoldout);

        std::ofstream ef(outPath("ensemble_weights.h"));
        ef << "#ifndef ENSEMBLE_WEIGHTS_H\n#define ENSEMBLE_WEIGHTS_H\n\n";
        ef << "static const float DT_WEIGHT=" << cFloat(ensW.dt) << ";\n";
        ef << "static const float KNN_WEIGHT=" << cFloat(ensW.knn) << ";\n";
        ef << "static const float RF_WEIGHT=" << cFloat(ensW.rf) << ";\n";
        ef << "\n#endif\n";
    }

    //===========================================================================================================
    //summary
    //===========================================================================================================
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "  Exported features: " << usedFeatures << " (ranked) of " << fullFeatureCount << " candidates" << std::endl;
    std::cout << "  " << std::left << std::setw(34) << "Model" << std::setw(10) << "Features" << std::setw(20) << "CV (train partition)" << "  Holdout" << std::right << std::endl;
    auto row = [](const std::string& name, int feats, const CVStat& cv, float holdout){
        std::cout << "  " << std::left << std::setw(34) << name << std::setw(10) << (feats > 0 ? std::to_string(feats) : std::string("-"))
                  << std::setw(20) << pct(cv.mean, cv.std)
                  << std::right << "  " << std::fixed << std::setprecision(1) << (holdout * 100) << "%" << std::endl;
    };
    row("Decision Tree (d=" + std::to_string(finalDTDepth) + ",ms=" + std::to_string(finalDTMinSamples) + ")", dtFeat, dtCV, dtMetrics.accuracy);
    row("KNN (k=" + std::to_string(bestK) + ")", knnFeat, knnCV, knnMetrics.accuracy);
    row("Random Forest (t=" + std::to_string(bestConfig.trees) + ",d=" + std::to_string(bestConfig.depth) + ")", rfFeat, rfCV, rfMetrics.accuracy);
    row("Ensemble (deployed)", 0, ensCV, ensHoldout.m.accuracy);
    std::cout << "  Hierarchical (hard, holdout): " << std::fixed << std::setprecision(1) << (hierMetrics.accuracy * 100) << "%" << std::endl;
    if(NUM_SELECTED_FEATURES == 0){
        std::cout << "  NOTE: CV figures are the best of the whole search, so slightly optimistic; the holdout is untouched." << std::endl;
    }

    {
        std::ofstream rf(outPath("results.csv"));
        rf << "features_requested,features_selected,split,seed,rf_grid,"
           << "dt_features,dt_depth,dt_min_split,dt_cv_mean,dt_cv_std,dt_holdout,"
           << "knn_features,knn_k,knn_cv_mean,knn_cv_std,knn_holdout,"
           << "rf_features,rf_trees,rf_depth,rf_subset,rf_min_split,rf_cv_mean,rf_cv_std,rf_holdout,"
           << "ens_cv_mean,ens_cv_std,ens_cv_coverage,ens_holdout,ens_holdout_coverage,hier_hard_holdout\n";
        rf << std::fixed << std::setprecision(4)
           << (NUM_SELECTED_FEATURES > 0 ? std::to_string(NUM_SELECTED_FEATURES) : std::string("auto")) << ","
           << usedFeatures << "," << (SESSION_SPLIT ? "session" : "random") << ","
           << RANDOM_SEED << "," << (QUICK_RF_GRID ? "quick" : "full") << ","
           << dtFeat << "," << (int)finalDTDepth << "," << (int)finalDTMinSamples << "," << dtCV.mean << "," << dtCV.std << "," << dtMetrics.accuracy << ","
           << knnFeat << "," << (int)bestK << "," << knnCV.mean << "," << knnCV.std << "," << knnMetrics.accuracy << ","
           << rfFeat << "," << bestConfig.trees << "," << bestConfig.depth << "," << bestConfig.sr << "," << bestConfig.minSamples << ","
           << rfCV.mean << "," << rfCV.std << "," << rfMetrics.accuracy << ","
           << ensCV.mean << "," << ensCV.std << "," << ensCVCoverage << ","
           << ensHoldout.m.accuracy << "," << ensHoldout.coverage() << "," << hierMetrics.accuracy << "\n";
    }

    //holdout features before scaling + trainer predictions
    {
        std::ofstream hc(outPath("holdout_check.csv"));
        hc << "label,dt_pred,dt_conf,knn_pred,knn_conf,rf_pred,rf_conf,ens_pred,ens_conf";
        for(int f = 0; f < engineeredFeatureCount; f++) hc << ",f" << f;
        hc << "\n" << std::setprecision(9);
        for(uint16_t i = 0; i < testCount; i++){
            float dc, kc, rc, ec;
            int dp = dt.predictWithConfidence(testSet[i].features, dc);
            int kp = knn.predictWithConfidence(testSet[i].features, knnFeat, kc);
            int rp = bestRf->predictWithConfidence(testSet[i].features, rc);
            int ep = deployedEnsemble(testSet[i].features, dt, knn, *bestRf, knnFeat, ensW, CONFIDENCE_THRESHOLD, &ec);
            hc << (int)testSet[i].label << "," << dp << "," << dc << "," << kp << "," << kc << ","
               << rp << "," << rc << "," << ep << "," << ec;
            for(int f = 0; f < engineeredFeatureCount; f++) hc << "," << testEngineered[i].features[f];
            hc << "\n";
        }
    }

    // Sample predictions
    std::cout << "\nSample Predictions:" << std::endl;
    int sampleIdx=std::min((int)testCount, 5);

    for(int i=0; i < sampleIdx; i++){
        scent_class_t actual=testSet[i].label;
        float dtConf, knnConf, rfConf;
        scent_class_t dtPred=dt.predictWithConfidence(testSet[i].features, dtConf);
        scent_class_t knnPred=knn.predictWithConfidence(testSet[i].features, knnFeat, knnConf);
        scent_class_t rfPred=bestRf->predictWithConfidence(testSet[i].features, rfConf);
        scent_class_t hierPred = hier.predict(testSet[i].features);
        scent_class_t ensPred = deployedEnsemble(testSet[i].features, dt, knn, *bestRf, knnFeat, ensW, CONFIDENCE_THRESHOLD);

        std::cout
                << "\n  Sample " << i << ": actual=" << CSVLoader::getClassName(actual)
                << "\n    DT:       " << CSVLoader::getClassName(dtPred) << " ("
                << std::fixed << std::setprecision(0) << (dtConf * 100) << "%)"
                << (dtPred == actual ? " [OK]" : " [X]")
                << "\n    KNN:      " << CSVLoader::getClassName(knnPred) << " ("
                << (knnConf * 100) << "%)"
                << (knnPred == actual ? " [OK]" : " [X]")
                << "\n    RF:       " << CSVLoader::getClassName(rfPred) << " ("
                << (rfConf * 100) << "%)" << (rfPred == actual ? " [OK]" : " [X]")
                << "\n    Hier:     " << CSVLoader::getClassName(hierPred)
                << (hierPred == actual ? " [OK]" : " [X]")
                << "\n    Ensemble: " << CSVLoader::getClassName(ensPred)
                << (ensPred == actual ? " [OK]" : " [X]")
                << std::endl;
    }

    //cleanup
    delete[] trainSet;
    delete[] testSet;
    delete bestRf;

    std::cout << "\nTraining complete." << std::endl;

    std::cout.rdbuf(originalBuf);
    std::cerr.rdbuf(originalErrBuf);
    logFile.close();

    std::cout << "\nOutput saved to " << outPath("training_output.txt") << std::endl;
    return 0;
}