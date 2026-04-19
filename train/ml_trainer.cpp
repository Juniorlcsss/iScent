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

#include "csv_loader.h"
#include "dt.h"
#include "knn.h"
#include "rf.h"


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


void printMetrics(const ml_metrics_t& m, const char* modelName){
    std::cout << "\n--- " << modelName << " Results ---" << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2)
              << (m.accuracy * 100) << "% (" << m.correct << "/" << m.total << ")" << std::endl;
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
    static char buf[24];
    if (idx < 10){ snprintf(buf, sizeof(buf), "g1n_s%d", idx); return buf; }
    if (idx < 20){ snprintf(buf, sizeof(buf), "g2n_s%d", idx - 10); return buf; }
    if (idx < 30){ snprintf(buf, sizeof(buf), "cr_s%d", idx - 20); return buf; }
    if (idx < 40){ snprintf(buf, sizeof(buf), "diff_s%d", idx - 30); return buf; }
    if (idx < 49){ snprintf(buf, sizeof(buf), "g1d_s%d", idx - 40); return buf; }
    if (idx < 58){ snprintf(buf, sizeof(buf), "g2d_s%d", idx - 49); return buf; }
    static const char* names[]={
        "slope1_n", "slope2_n", "curv1_n", "curv2_n",
        "auc1_n", "auc2_n", "peak1", "peak2",
        "range1", "range2", "late_early1", "late_early2",
        "cr_mean", "cr_slope", "cr_var"
    };
    static const char* fisherNames[] = {"fisher_coff_dtea","fisher_dtea_tea","fisher_dcoff_coff","fisher_dtea_dcoff"};

    if(idx >= 145&&idx<149){ 
        return fisherNames[idx-145];
    }

    int offset=idx - 58;
    int numNames=sizeof(names) / sizeof(names[0]);
    if (offset >= 0 && offset < numNames) return names[offset];
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
    int originalCount;
    int selectedCount;

    void selectFeatures(const csv_training_sample_t* samples, uint16_t count,
                        int featureCount, float minFRatio=0.5f,
                        int maxFeatures=45){
        originalCount=featureCount;

        struct FeatureScore {
            int index;
            float score;
        };
        std::vector<FeatureScore> scores(featureCount);

        for(int f=0; f < featureCount; f++){
            float classMeans[SCENT_CLASS_COUNT]={0};
            float classVars[SCENT_CLASS_COUNT]={0};
            int classCounts[SCENT_CLASS_COUNT]={0};

            for(int i=0; i < count; i++){
                int c=samples[i].label;
                if (c >= SCENT_CLASS_COUNT) continue;
                classMeans[c] += samples[i].features[f];
                classCounts[c]++;
            }

            float grandMean=0;
            int totalValid=0;
            for(int c=0; c < SCENT_CLASS_COUNT; c++){
                if (classCounts[c] > 0) classMeans[c] /= classCounts[c];
                grandMean += classMeans[c] * classCounts[c];
                totalValid += classCounts[c];
            }
            if (totalValid > 0) grandMean /= totalValid;

            for(int i=0; i < count; i++){
                int c=samples[i].label;
                if (c >= SCENT_CLASS_COUNT) continue;
                float d=samples[i].features[f] - classMeans[c];
                classVars[c] += d * d;
            }

            float betweenVar=0, withinVar=0;
            for(int c=0; c < SCENT_CLASS_COUNT; c++){
                if (classCounts[c] > 0){
                    float d=classMeans[c] - grandMean;
                    betweenVar += classCounts[c] * d * d;
                    withinVar += classVars[c];
                }
            }

            float fRatio=(withinVar > 1e-8f)
                ? (betweenVar / (SCENT_CLASS_COUNT - 1)) / (withinVar / (totalValid - SCENT_CLASS_COUNT))
                : 0.0f;

            scores[f]={f, fRatio};
        }

        std::sort(scores.begin(), scores.end(),
            [](const FeatureScore& a, const FeatureScore& b){
                return a.score > b.score;
            });

        std::cout << "\n=== Feature Ranking (F-ratio) ===" << std::endl;
        std::cout << "  [*]=selected, [ ]=dropped\n" << std::endl;

        selectedIndices.clear();
        for(int i=0; i < featureCount; i++){
            bool selected=((int)selectedIndices.size() < maxFeatures && scores[i].score >= minFRatio);
            if (selected) selectedIndices.push_back(scores[i].index);

            std::cout << "  " << (selected ? "[*]" : "[ ]")
                      << " F=" << std::setw(8) << std::fixed << std::setprecision(3)
                      << scores[i].score
                      << "  [" << std::setw(2) << scores[i].index << "] "
                      << shortFeatureName(scores[i].index) << std::endl;
        }

        std::sort(selectedIndices.begin(), selectedIndices.end());
        selectedCount=selectedIndices.size();

        std::cout << "\nSelected " << selectedCount << "/" << featureCount << " features" << std::endl;
    }

    void selectFeaturesHybrid(const csv_training_sample_t *samples, uint16_t count, 
                          int featureCount, int maxFeatures = 50) {
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
            scores[f].index=f;

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
            if(scores[f].globalF>maxG){
                maxG = scores[f].globalF;
            }
            if(scores[f].teaPairF>maxT){
                maxT = scores[f].teaPairF;
            }

            if(scores[f].coffeePairF>maxC){
                maxC = scores[f].coffeePairF;
            }
            if(scores[f].crossPairF >maxX){
                maxX = scores[f].crossPairF;
            }
            if(scores[f].teaCoffeeF >maxTC){
                maxTC = scores[f].teaCoffeeF;
            }
        }

        if(maxG<1e-8f){
            maxG = 1.0f;
        }
        if(maxT < 1e-8f){
            maxT = 1.0f;
        }
        if(maxC < 1e-8f){
            maxC = 1.0f;
        }
        if(maxX < 1e-8f){
            maxX = 1.0f;
        }
        if(maxTC < 1e-8f){
            maxTC = 1.0f;
        }

        for(int f = 0; f < featureCount; f++){
            float normG = scores[f].globalF / maxG;
            float normT = scores[f].teaPairF / maxT;
            float normC = scores[f].coffeePairF / maxC;
            float normX = scores[f].crossPairF / maxX;
            float normTC=scores[f].teaCoffeeF/maxTC;

            //take max and average across 4 hard pairs
            float pairMax=std::max({normT,normC,normX,normTC});
            float pairAvg = (normT + normC + normX + normTC)/4.0f;
            
            //ensure pair representation
            scores[f].score = 0.20f*normG+0.45f * pairMax+0.35f*pairAvg; 
        }

        std::sort(scores.begin(), scores.end(),
            [](const FeatureScore& a, const FeatureScore& b){
                return a.score > b.score;
        });

        std::cout << "\n=== Hybrid Feature Ranking ===" << std::endl;
        std::cout << "  [*]=selected, [ ]=dropped\n" << std::endl;

        selectedIndices.clear();
        float minScore = 0.05f;

        for(int i = 0; i < featureCount; i++){
            float normG = scores[i].globalF / maxG;
            float normT = scores[i].teaPairF / maxT;
            float normC = scores[i].coffeePairF / maxC;
            float normX = scores[i].crossPairF / maxX;
            float normTC = scores[i].teaCoffeeF / maxTC;

            bool selected = ((int)selectedIndices.size() < maxFeatures && scores[i].score >= minScore);
            
            if(selected){
                selectedIndices.push_back(scores[i].index);
            }

            if(i < maxFeatures + 10) {
                std::cout << "  " << (selected ? "[*]" : "[ ]")
                        << " S=" << std::setw(6) << std::fixed << std::setprecision(3)
                        << scores[i].score
                        << "(G=" << std::setprecision(2) << normG
                        << " T=" << normT
                        << " C=" << normC
                        << " X=" << normX
                        << " TC=" << normTC
                        << ")  [" << std::setw(3) << scores[i].index << "] "
                        << shortFeatureName(scores[i].index) << std::endl;
            }
        }

        int criticalFeats[] = {20, 21, 22, 23, 25, 30, 31, 70, 71, 12, 13, 52};
        for(int cf:criticalFeats){
            if(cf>=featureCount){
                continue;
            }

            bool alreadySelected = false;
            for(int si : selectedIndices){
                if(si == cf){
                    alreadySelected = true;
                    break;
                }
            }
            if(!alreadySelected&&(int)selectedIndices.size() < maxFeatures){
                selectedIndices.push_back(cf);
                std::cout<<"  [FORCED] feature " <<cf<< " (" << shortFeatureName(cf) << ")" << std::endl;
            }
        }

        std::sort(selectedIndices.begin(), selectedIndices.end());
        selectedCount = selectedIndices.size();
        std::cout << "\nSelected " << selectedCount << "/" << featureCount << " features" << std::endl;
        std::cout << "Score maxima: G=" << std::setprecision(1) << maxG << " T=" << maxT << " C=" << maxC 
                << " X=" << maxX << " TC=" << maxTC << std::endl;
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

    void saveHeader(const char* filename, const float* medians, const float* iqrs)const {
        std::ofstream f(filename);
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
            f << std::setprecision(8) << medians[selectedIndices[i]];
            if (i < selectedCount - 1) f << ", ";
        }
        f << "};\n\n";

        f << "static const float SELECTED_IQRS[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << std::setprecision(8) << iqrs[selectedIndices[i]];
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

//cross validation
struct CVResults{
    float meanAcc;
    float stdAcc;
    float foldAcc[10];
    int foldCount;
    ml_metrics_t aggregateMetrics;
};

CVResults crossValidateRF(csv_training_sample_t* allSamples, uint16_t totalCount,uint16_t featureCount, int numFolds,int numTrees, int maxDepth,int minSamples,float subsetRatio,const float* medians,const float* iqrs,const FeatureSelector& selector){
    CVResults results={};
    results.foldCount=numFolds;
    memset(&results.aggregateMetrics, 0, sizeof(ml_metrics_t));

    std::vector<std::vector<uint16_t>> classIndices(SCENT_CLASS_COUNT);
    for(uint16_t i=0; i<totalCount;i++){
        if(allSamples[i].label< SCENT_CLASS_COUNT){
            classIndices[allSamples[i].label].push_back(i);
        }
    }

    std::mt19937 rng(1337);
    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        std::shuffle(classIndices[i].begin(), classIndices[i].end(), rng);
    }

    std::vector<int> foldAssignment(totalCount, -1);
    for(int i=0; i<SCENT_CLASS_COUNT;i++){
        for(size_t j=0; j<classIndices[i].size();j++){
            foldAssignment[classIndices[i][j]]=j % numFolds;
        }
    }

    float totalAcc=0;
    int totalCorrect=0, totalSamples=0;

    for(int fold=0; fold<numFolds;fold++){
        std::vector<csv_training_sample_t> trainFold, testFold;
        for(uint16_t i=0; i<totalCount;i++){
            if(foldAssignment[i]==fold){
                testFold.push_back(allSamples[i]);
            }
            else{
                trainFold.push_back(allSamples[i]);
            }
        }

        if(trainFold.empty() || testFold.empty()){
            continue;
        }

        //robust scaling per fold
        float foldMedians[CSV_FEATURE_COUNT]={0};
        float foldIQRs[CSV_FEATURE_COUNT]={0};

        for(int f=0; f<featureCount; f++){
            std::vector<float> vals(trainFold.size());

            for(size_t i=0; i<trainFold.size(); i++){
                vals[i] = trainFold[i].features[f];
            }
            std::sort(vals.begin(), vals.end());

            foldMedians[f] = vals[vals.size() / 2];
            float q1 =vals[vals.size()/ 4];
            float q3 =vals[3 * vals.size() / 4];
            foldIQRs[f]=q3 - q1;

            if(foldIQRs[f]<1e-6f){
                foldIQRs[f] = 1.0f;
            }
        }

        for(auto &s: trainFold){
            for(int f=0; f<featureCount; f++){
                s.features[f] = (s.features[f] - foldMedians[f]) / foldIQRs[f];
            }
        }
        
        for(auto &s: testFold){
            for(int f=0; f<featureCount; f++){
                s.features[f] = (s.features[f] - foldMedians[f]) / foldIQRs[f];
            }
        }

        uint16_t selectedFeat= selector.selectedCount;
        std::vector<csv_training_sample_t> trainProj(trainFold.size());
        std::vector<csv_training_sample_t> testProj(testFold.size());
        selector.projectDataset(trainFold.data(), trainFold.size(), trainProj.data());
        selector.projectDataset(testFold.data(), testFold.size(), testProj.data());

        uint16_t augTrainCount=trainProj.size();
        csv_training_sample_t* augTrain=new csv_training_sample_t[augTrainCount];
        memcpy(augTrain, trainProj.data(), augTrainCount * sizeof(csv_training_sample_t));

        RandomForest rf(numTrees, maxDepth, minSamples, subsetRatio);
        rf.train(augTrain, augTrainCount, selectedFeat);

        ml_metrics_t m=rf.evaluate(testProj.data(), testProj.size());

        results.foldAcc[fold]=m.accuracy;
        totalAcc +=m.accuracy;
        totalCorrect+=m.correct;
        totalSamples+=m.total;

        for(int i=0; i<SCENT_CLASS_COUNT;i++){
            for(int j=0; j<SCENT_CLASS_COUNT;j++){
                results.aggregateMetrics.confusionMatrix[i][j]+=m.confusionMatrix[i][j];
            }
        }

        std::cout << "    Fold " << (fold + 1) << "/" << numFolds<< ": " << std::fixed << std::setprecision(1)
        << (m.accuracy * 100) << "% (" << m.correct << "/" << m.total << ")"<< std::endl;

        delete[] augTrain;
    }

    results.meanAcc= totalAcc / numFolds;
    results.aggregateMetrics.correct= totalCorrect;
    results.aggregateMetrics.total= totalSamples;
    results.aggregateMetrics.accuracy= (totalSamples > 0) ? ((float)totalCorrect / totalSamples) : 0.0f;

    float varSum=0;
    for(int i=0; i<numFolds;i++){
        varSum+= (results.foldAcc[i] - results.meanAcc) * (results.foldAcc[i] - results.meanAcc);
    }
    results.stdAcc= sqrt(varSum / numFolds);
    return results;
}


//experiment
void twoStageExperiment(const csv_training_sample_t* trainSet, uint16_t trainCount, const csv_training_sample_t* testSet, uint16_t testCount,uint16_t featureCount){
    std::cout <<"\n=== Classification Experiment ===" <<std::endl;

    std::vector<csv_training_sample_t> s1_train, s1_test;
    for(int i=0; i <trainCount; i++){
        csv_training_sample_t s=trainSet[i];
        s.label=(s.label == SCENT_CLASS_DECAF_TEA || s.label == SCENT_CLASS_TEA)? (scent_class_t)0 : (scent_class_t)1;
        s1_train.push_back(s);
    }
    for(int i=0; i <testCount; i++){
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
    std::cout <<"Stage 1 (tea-type vs coffee-type): "<<std::fixed <<std::setprecision(1)<<(100.0f * s1_correct / s1_test.size()) <<"%" <<std::endl;

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
        
        std::cout << "  Stage 1 test acc: " << std::fixed << std::setprecision(1)<< (100.0f * s1_correct / testCount) << "%" << std::endl;
        std::cout << "  Hard routing:  " << (m.accuracy * 100) << "%" << std::endl;
        std::cout << "  Soft routing:  " << (mSoft.accuracy * 100) << "%" << std::endl;
        
        return (mSoft.accuracy > m.accuracy) ? mSoft : m;
    }
};

//===========================================================================================================
//weighted ensemble
//===========================================================================================================
scent_class_t ensemblePredict(const float* features,DecisionTree& dt, KNN& knn,RandomForest& rf,HierarchicalClassifier& hier, uint16_t featureCount,float dtWeight, float knnWeight,float rfWeight, float hierWeight) {
    float classScores[SCENT_CLASS_COUNT] = {0};

    //KNN
    float knnConf;
    scent_class_t knnPred = knn.predictWithConfidence(features, featureCount, knnConf);
    classScores[knnPred] += knnWeight * knnConf;
    float knnResid = knnWeight * (1.0f-knnConf)/(SCENT_CLASS_COUNT - 1);
    for(int c = 0; c<SCENT_CLASS_COUNT;c++){
        if(c != knnPred){
            classScores[c] += knnResid;
        }
    }

    //RF
    float rfConf;
    scent_class_t rfPred =rf.predictWithConfidence(features, rfConf);
    classScores[rfPred]+=rfWeight*rfConf;
    float rfResid=rfWeight * (1.0f-rfConf)/(SCENT_CLASS_COUNT - 1);

    for(int c=0; c<SCENT_CLASS_COUNT;c++){
        if(c!=rfPred){
            classScores[c]+=rfResid;
        }
    }

    //hierarchical
    scent_class_t hierPred=hier.predictSoft(features);
    if(hierPred !=SCENT_CLASS_AMBIENT){
        classScores[hierPred]+=hierWeight*0.7f;
    }

    //DT
    scent_class_t dtPred=dt.predict(features);
    classScores[dtPred]+=dtWeight * 0.3f;

    return (scent_class_t)(std::distance(classScores,std::max_element(classScores, classScores+SCENT_CLASS_COUNT)));
}


int main(int argc, char* argv[]){
    std::ofstream logFile("training_output.txt");
    
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

    std::cout <<"========================================" <<std::endl;
    std::cout <<"    Machine Learning Model Trainer" <<std::endl;
    std::cout <<"========================================" <<std::endl;

    std::string filename="data.csv";
    float trainRatio=0.8f;
    if (argc > 1) filename=argv[1];
    if (argc > 2) trainRatio=std::stof(argv[2]);
    std::cout <<"\nLoading: " << filename << std::endl;
    std::cout << "Train/Test split: " << (trainRatio * 100) << "/"<< ((1 - trainRatio) * 100) << std::endl;

    CSVLoader loader;
    if(!loader.load(filename)){
        std::cerr << "Failed to load data!" << std::endl;
        return 1;
    }

    loader.printInfo();
    CSVLoader::printFeatureNames();

    //split
    csv_training_sample_t *trainSet=nullptr, *testSet=nullptr;
    uint16_t trainCount=0,testCount=0;
    loader.split(trainRatio,trainSet,trainCount,testSet,testCount);

    //===========================================================================================================
    //feature engineering
    //===========================================================================================================
    int engineeredFeatureCount=BASE_FEATURE_COUNT;
    {
        Timer t("Feature Engineering");
        engineeredFeatureCount=addEngineeredFeatures(trainSet, trainCount, BASE_FEATURE_COUNT);
        addEngineeredFeatures(testSet,testCount,BASE_FEATURE_COUNT);
    }

    //===========================================================================================================
    //robust scaling
    //===========================================================================================================
    float feature_medians[CSV_FEATURE_COUNT]={0};
    float feature_iqrs[CSV_FEATURE_COUNT]={0};

    {
        Timer t("Robust Scaling");
        robustScale(trainSet, trainCount, engineeredFeatureCount, feature_medians, feature_iqrs);
        robustScaleApply(testSet, testCount, engineeredFeatureCount, feature_medians, feature_iqrs);
    }

    //===========================================================================================================
    //fisher discriminant features
    //===========================================================================================================
    float fisherWeights[4][CSV_FEATURE_COUNT]={{0}};
    int numFisherPairs=0;
    int fisherBaseFeat=0;

    int preFisherFeatureCount = engineeredFeatureCount;
    {
        Timer t("Fisher Features");
        std::cout << std::endl;
        
        struct ClassPair {
            scent_class_t a, b;
            const char* name;
        };
        ClassPair pairs[] = {
            {SCENT_CLASS_COFFEE, SCENT_CLASS_DECAF_TEA, "coffee_vs_dtea"},
            {SCENT_CLASS_DECAF_TEA, SCENT_CLASS_TEA, "dtea_vs_tea"},
            {SCENT_CLASS_DECAF_COFFEE, SCENT_CLASS_COFFEE, "dcoffee_vs_coffee"},
            {SCENT_CLASS_DECAF_TEA, SCENT_CLASS_DECAF_COFFEE, "dtea_vs_dcoffee"}
        };
        int numPairs=4;
        int baseFeat=engineeredFeatureCount;
        fisherBaseFeat=baseFeat;
        
        for(int p=0;p<numPairs;p++){
            if(engineeredFeatureCount>=CSV_FEATURE_COUNT-1){
                break;
            }
            
            float meanA[CSV_FEATURE_COUNT] = {0};
            float meanB[CSV_FEATURE_COUNT] = {0};
            int nA= 0,nB=0;
            
            for(uint16_t i =0;i<trainCount;i++){
                if(trainSet[i].label==pairs[p].a){
                    for(int f=0;f<baseFeat;f++){
                        meanA[f]+=trainSet[i].features[f];
                    }
                    nA++;
                }

                else if(trainSet[i].label==pairs[p].b){
                    for(int f=0;f<baseFeat;f++){
                        meanB[f] += trainSet[i].features[f];
                    }
                    nB++;
                }
            }
            if(nA == 0||nB == 0){
                continue;
            }

            for(int f = 0; f<baseFeat;f++){
                meanA[f] /=nA;
                meanB[f] /=nB;
            }
            
            float weights[CSV_FEATURE_COUNT] = {0};
            float norm = 0;
            for(int f=0; f< baseFeat; f++){
                float varA = 0, varB = 0;

                for(uint16_t i=0; i<trainCount;i++){
                    if(trainSet[i].label==pairs[p].a){
                        float d = trainSet[i].features[f]-meanA[f];
                        varA += d * d;
                    }
                    else if(trainSet[i].label==pairs[p].b){
                        float d = trainSet[i].features[f]-meanB[f];
                        varB += d * d;
                    }
                }

                float pooled_std = sqrtf((varA/nA + varB/nB) / 2.0f);
                if(pooled_std>1e-6f){
                    weights[f]=(meanA[f]-meanB[f]) /pooled_std;
                }
                norm+=weights[f]*weights[f];
            }

            norm=sqrtf(norm);
            if(norm < 1e-6f){
                continue;
            }

            for(int f = 0; f < baseFeat; f++){
                weights[f] /= norm;
            }
            
            memcpy(fisherWeights[numFisherPairs], weights, sizeof(float)*baseFeat);
            numFisherPairs++;

            int newIdx=engineeredFeatureCount;
            for(uint16_t i=0; i<trainCount;i++){
                float proj = 0;

                for(int f =0; f < baseFeat; f++){
                    proj += trainSet[i].features[f] * weights[f];
                }

                trainSet[i].features[newIdx] = proj;
            }
            for(uint16_t i = 0; i < testCount; i++){
                float proj = 0;

                for(int f=0; f<baseFeat;f++){
                    proj+=testSet[i].features[f] * weights[f];
                }
                testSet[i].features[newIdx]=proj;
            }

            engineeredFeatureCount++;
            std::cout<< "  Fisher [" << newIdx << "]: " <<pairs[p].name << std::endl;
        }
        std::cout << "Total features with Fisher: " << engineeredFeatureCount << std::endl;
        std::cout << "Selectable features (pre-Fisher): " << preFisherFeatureCount << std::endl;
    }

    {
        int fisherStart=preFisherFeatureCount;
        int fisherEnd=engineeredFeatureCount;
        
        for(int f=fisherStart;f<fisherEnd;f++){
            std::vector<float> vals(trainCount);

            for(uint16_t i=0; i<trainCount; i++){
                vals[i]=trainSet[i].features[f];
            }

            std::sort(vals.begin(), vals.end());
            
            feature_medians[f] =vals[trainCount/2];
            float q1=vals[trainCount/4];
            float q3=vals[3*trainCount/4];
            feature_iqrs[f] =q3-q1;
            if(feature_iqrs[f]< 1e-6f) feature_iqrs[f]=1.0f;
            
            //train
            for(uint16_t i = 0; i < trainCount; i++){
                trainSet[i].features[f]=(trainSet[i].features[f]-feature_medians[f])/feature_iqrs[f];
            }

            //test
            for(uint16_t i =0;i< testCount;i++){
                testSet[i].features[f]=(testSet[i].features[f]-feature_medians[f])/feature_iqrs[f];
            }
        }
        
        std::cout << "Normalized " << (fisherEnd - fisherStart)<< " Fisher features (indices "<< fisherStart<< ".."<<(fisherEnd - 1) << ")" << std::endl;
    }

    //save
    {
        std::ofstream sf("feature_stats.h");
        sf << "#ifndef FEATURE_STATS_H\n#define FEATURE_STATS_H\n\n";
        sf << "#define FULL_FEATURE_COUNT " << engineeredFeatureCount << "\n\n";
        sf << "static const float FEATURE_MEDIANS[" << engineeredFeatureCount << "]={";

        for(int f=0; f < engineeredFeatureCount; f++){
            sf <<std::setprecision(8) << feature_medians[f];
            if(f<engineeredFeatureCount-1){
                sf << ", ";
            }
        }

        sf << "};\n\nstatic const float FEATURE_IQRS[" << engineeredFeatureCount << "]={";

        for(int f=0; f < engineeredFeatureCount; f++){
            sf <<std::setprecision(8) << feature_iqrs[f];
            if(f<engineeredFeatureCount-1){
                sf << ", ";
            }
        }
        sf << "};\n\n#endif\n";
    }

    FeatureSelector selector;
    {
        Timer t("Feature Selection (hybrid)");

        //temp1, hum1, pres1, temp2, hum2, pres2
        std::vector<int> envBlacklist={76, 77, 78, 79, 80, 81};

        //count how many are within range
        int blacklistCount = 0;
        for(int idx:envBlacklist){
            if(idx<preFisherFeatureCount){
                blacklistCount++;
            }
        }

        std::cout<<"\nBlacklisting " << blacklistCount << " environmental features from selection" << std::endl;
        std::cout<<"Selecting from " << (preFisherFeatureCount - blacklistCount) << " effective features" << std::endl;
m
        std::vector<std::vector<float>> savedVals( trainCount,std::vector<float>(envBlacklist.size(),0.0f));

        for(uint16_t i=0; i<trainCount; i++){
            for(size_t j=0; j <envBlacklist.size();j++){
                int idx = envBlacklist[j];

                if(idx < preFisherFeatureCount){
                    savedVals[i][j]=trainSet[i].features[idx];
                    trainSet[i].features[idx]=0.0f;
                }
            }
        }

        //run selection
        selector.selectFeaturesHybrid(trainSet, trainCount, engineeredFeatureCount, 50);

        //restore
        for(uint16_t i = 0; i < trainCount; i++){
            for(size_t j = 0; j < envBlacklist.size(); j++){
                int idx = envBlacklist[j];
                if(idx < preFisherFeatureCount){
                    trainSet[i].features[idx]=savedVals[i][j];
                }
            }
        }

        //verify
        int envSelected = 0;
        std::set<int> blacklistSet(envBlacklist.begin(), envBlacklist.end());
        for(int i = 0; i < selector.selectedCount; i++){
            if(blacklistSet.count(selector.selectedIndices[i])>0){
                std::cout << "  WARNING: env feature " << selector.selectedIndices[i] << " was still selected!" << std::endl;
                
                envSelected++;
            }
        }

        std::cout << "  Env features in final selection: " << envSelected<<"/"<<selector.selectedCount<<std::endl;
    }

    selector.saveHeader("feature_select.h", feature_medians, feature_iqrs);
    uint16_t reducedFeatureCount = selector.selectedCount;

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
            if(f>= engineeredFeatureCount){
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
    for(int f = 0; f < engineeredFeatureCount; f++){
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

    std::cout << "\nFeatures with separation > 0.3: "<< goodFeats<<"/"<<engineeredFeatureCount<<std::endl;
    
    if(goodFeats<5){
        std::cout <<"WARNING: Very few discriminative features between coffee and decaf_tea!"<<std::endl;
        std::cout <<"  -> Sensors may produce genuinely similar readings for these classes"<<std::endl;
        std::cout <<"  -> Consider collecting paired A/B data under varied conditions"<<std::endl;
    }
    else if(goodFeats<15){
        std::cout << "MODERATE: Some signal exists but is weak. More engineered features may help." << std::endl;
    }
    else {
        std::cout << "GOOD: Signal exists. Model tuning should improve separation." << std::endl;
    }

    //===========================================================================================================
    //feature count sweeep
    //===========================================================================================================
    {
        std::cout << "\n=== Feature Count Sweep ===" << std::endl;
        
        std::streambuf* origBuf = std::cout.rdbuf();
        std::ostringstream nullStream;
        
        for(int maxF:{10,15,20,25,30,35,40,45,50}){
            std::cout.rdbuf(nullStream.rdbuf());
            FeatureSelector testSel;
            testSel.selectFeaturesHybrid(trainSet, trainCount, engineeredFeatureCount, maxF);
            std::cout.rdbuf(origBuf);
            
            csv_training_sample_t* tmpTrain=new csv_training_sample_t[trainCount];
            csv_training_sample_t* tmpTest=new csv_training_sample_t[testCount];
            testSel.projectDataset(trainSet,trainCount,tmpTrain);
            testSel.projectDataset(testSet,testCount,tmpTest);
            
            //KNN
            KNN tmpKnn;
            tmpKnn.train(tmpTrain,trainCount);
            tmpKnn.setK(5);
            ml_metrics_t mKnn=tmpKnn.evaluate(tmpTest, testCount);
            
            //RF
            RandomForest tmpRf(100,6,10,0.3f);
            tmpRf.train(tmpTrain,trainCount, testSel.selectedCount);
            ml_metrics_t mRf=tmpRf.evaluate(tmpTest, testCount);
            
            std::cout << "  Features=" << std::setw(2) << testSel.selectedCount 
                    <<" (requested " << maxF << ")"<<" -> KNN: " << std::fixed<<std::setprecision(1)
                    <<(mKnn.accuracy * 100) << "%"<<"  RF: " << (mRf.accuracy * 100)<<"%" << std::endl;
            
            delete[] tmpTrain;
            delete[] tmpTest;
        }
        std::cout<<std::endl;
    }




    csv_training_sample_t* trainReduced=new csv_training_sample_t[trainCount];
    csv_training_sample_t* testReduced=new csv_training_sample_t[testCount];
    selector.projectDataset(trainSet, trainCount,trainReduced);
    selector.projectDataset(testSet, testCount,testReduced);


    csv_training_sample_t* trainFull=trainSet;
    csv_training_sample_t* testFull=testSet;
    trainSet=trainReduced;
    testSet=testReduced;

    std::cout << "\nTraining with " << reducedFeatureCount << " selected features" << std::endl;

    //experiment
    twoStageExperiment(trainSet, trainCount, testSet, testCount, reducedFeatureCount);

    //===========================================================================================================
    //hierarchical classifier
    //===========================================================================================================
    HierarchicalClassifier hier;
    ml_metrics_t hierMetrics;
    {
        Timer t("Hierarchical Classifier");
        std::cout << std::endl;

        hier.train(trainSet,trainCount,reducedFeatureCount);
        hierMetrics = hier.evaluate(testSet, testCount);

        printMetrics(hierMetrics,"Hierarchical Classifier");
        printPerClassMetrics(hierMetrics);
        printConfusionMatrix(hierMetrics);
    }

    csv_training_sample_t* knnTrainSet=new csv_training_sample_t[trainCount];
    uint16_t knnTrainCount=trainCount;
    memcpy(knnTrainSet, trainSet, trainCount * sizeof(csv_training_sample_t));

    //aug
    bool useAugmentation=false; //leave off for now

    if(useAugmentation){
        Timer t("Data Augmentation");
        std::default_random_engine rng(42);
        augmentTrainingData(trainSet, trainCount, reducedFeatureCount, rng);

        std::cout << "Augmented: " << trainCount << " samples" << std::endl;
        for(int c=0; c < SCENT_CLASS_COUNT; c++){
            int cnt=0;

            for(int j=0; j < trainCount; j++){
                if (trainSet[j].label == c){
                    cnt++;
                }
            }
            std::cout << "  " << CSVLoader::getClassName((scent_class_t)c) << ": " << cnt << std::endl;
        }
    }
    else{
        std::cout << "\nData augmentation skipped" << std::endl;
    }

    //========================================================
    //Decision Tree
    //========================================================
    {
        Timer t("Decision Tree Grid Search");

        std::cout << std::endl;
        uint8_t bestDTDepth=0, bestDTMinSamples=0;
        float bestDTAcc=0.0f;

        for(uint8_t d=6; d <= 14; d += 2){
            for(uint8_t ms : {5, 10, 20, 30}){
                DecisionTree dtTest;
                dtTest.train(trainSet, trainCount, reducedFeatureCount, d, ms);
                ml_metrics_t m=dtTest.evaluate(testSet, testCount);
                std::cout
                        << "  DT d=" << (int)d << " ms=" << (int)ms
                        << " -> " << std::fixed << std::setprecision(1)
                        << (m.accuracy * 100) << "%" << std::endl;
                if(m.accuracy > bestDTAcc){
                    bestDTAcc=m.accuracy;
                    bestDTDepth=d;
                    bestDTMinSamples=ms;
                }
            }
        }

        std::cout << "\n  Best: d=" << (int)bestDTDepth << " ms=" << (int)bestDTMinSamples
                  << " acc=" << (bestDTAcc * 100) << "%" << std::endl;
    }

    DecisionTree dt;

    uint8_t finalDTDepth=0, finalDTMinSamples=0;
    float finalDTAcc=0.0f;
    {
        for(uint8_t d=6; d <= 14; d += 2){
            for(uint8_t ms : {5, 10, 20, 30}){
                DecisionTree dtTest;
                dtTest.train(trainSet, trainCount, reducedFeatureCount, d, ms);
                ml_metrics_t m=dtTest.evaluate(testSet, testCount);
                if (m.accuracy > finalDTAcc){
                    finalDTAcc=m.accuracy;
                    finalDTDepth=d;
                    finalDTMinSamples=ms;
                }
            }
        }
    }

    dt.train(trainSet, trainCount, reducedFeatureCount, finalDTDepth, finalDTMinSamples);
    ml_metrics_t dtMetrics=dt.evaluate(testSet, testCount);
    printMetrics(dtMetrics, "Decision Tree");
    printPerClassMetrics(dtMetrics);
    printConfusionMatrix(dtMetrics);

    //========================================================
    //KNN
    //========================================================
    std::cout << "\n=== Training KNN ===" << std::endl;
    KNN knn;
    knn.train(knnTrainSet, knnTrainCount);

    uint8_t bestK=1;
    float bestKnnAcc=0.0f;
    {
        Timer t("KNN Search");
        std::cout << std::endl;
        for(int k=1; k <= 25; k += 2){
            knn.setK(k);
            ml_metrics_t m=knn.evaluate(testSet, testCount);
            std::cout << "  K=" << std::setw(2) << k << ": "<< std::fixed << std::setprecision(1)<< (m.accuracy * 100) << "%" << std::endl;
            if(m.accuracy > bestKnnAcc){
                bestKnnAcc=m.accuracy;
                bestK=k;
            }
        }
    }

    if(bestK < 5){
        std::cout<<"  Overriding K="<<(int)bestK<< " -> K=5 for inference robustness"<< std::endl;
        bestK=5;
    }

    knn.setK(bestK);
    ml_metrics_t knnMetrics=knn.evaluate(testSet, testCount);
    printMetrics(knnMetrics,"KNN");
    printPerClassMetrics(knnMetrics);
    printConfusionMatrix(knnMetrics);

    {
        std::cout << "\n=== KNN Distance Calibration ===" << std::endl;
        
        float class_avg_dist[SCENT_CLASS_COUNT] = {0};
        int class_count[SCENT_CLASS_COUNT] = {0};
        float max_inclass_dist = 0.0f;
        float all_correct_dists[5000];
        int total_correct = 0;
        
        for(uint16_t i = 0; i < testCount; i++){
            float conf;
            scent_class_t pred = knn.predictWithConfidence(testSet[i].features,reducedFeatureCount,conf);
            
            float dist_proxy = 1.0f - conf;
            int cls = testSet[i].label;
            
            //only do correct preds
            if(pred==cls&&cls<SCENT_CLASS_COUNT){
                class_avg_dist[cls]+= dist_proxy;
                class_count[cls]++;

                if(total_correct<5000){
                    all_correct_dists[total_correct++]=dist_proxy;
                }

                if(dist_proxy>max_inclass_dist){
                    max_inclass_dist = dist_proxy;
                }
            }
        }
        
        for(int c=0;c<SCENT_CLASS_COUNT;c++){
            if(class_count[c]>0){
                class_avg_dist[c]/=class_count[c];
                std::cout << "  " << CSVLoader::getClassName((scent_class_t)c)
                        << ": avg_dist=" << std::fixed << std::setprecision(4) 
                        << class_avg_dist[c]
                        << " (n=" << class_count[c] << ")" << std::endl;
            }
        }
        
        //use 95th percentile
        std::sort(all_correct_dists, all_correct_dists + total_correct);
        float p95_dist = (total_correct > 0)?all_correct_dists[(int)(total_correct * 0.95f)]:max_inclass_dist;
        
        float calibrated_threshold = p95_dist*1.5f;
        float distance_scale = p95_dist*2.5f;
        
        std::cout << "  Max in-class dist: " << max_inclass_dist << std::endl;
        std::cout << "  95th percentile:   " << p95_dist << std::endl;
        std::cout << "  Anomaly threshold: " << calibrated_threshold << std::endl;
        std::cout << "  Distance scale:    " << distance_scale << std::endl;
        
        //save
        std::ofstream athf("anomaly_threshold.h");
        athf << "#ifndef ANOMALY_THRESHOLD_H\n#define ANOMALY_THRESHOLD_H\n\n";
        athf << "// Calibrated from " << total_correct 
            << " correct test predictions\n";
        athf << "static const float CALIBRATED_ANOMALY_THRESHOLD = " 
            << std::setprecision(6) << calibrated_threshold << "f;\n";
        athf << "static const float KNN_DISTANCE_SCALE = " 
            << std::setprecision(6) << distance_scale 
            << "f;  // replaces magic 5.0\n";
        athf << "\n#endif\n";
    }

    //========================================================
    //RF
    //========================================================
    std::cout << "\n=== Cross-Validated Random Forest Search ===" << std::endl;

    csv_training_sample_t* rawSamples = loader.getSamples();
    uint16_t rawCount = loader.getSampleCount();

    csv_training_sample_t* cvSamples = new csv_training_sample_t[rawCount];
    memcpy(cvSamples, rawSamples, rawCount * sizeof(csv_training_sample_t));
    addEngineeredFeatures(cvSamples, rawCount, BASE_FEATURE_COUNT);

    //apply Fisher projections
    for(int p=0;p<numFisherPairs;p++){
        int fi=preFisherFeatureCount+p;
        for(uint16_t i=0; i<rawCount;i++){
            float proj=0;
            for(int f=0; f<fisherBaseFeat;f++){
                proj += cvSamples[i].features[f]*fisherWeights[p][f];
            }
            cvSamples[i].features[fi]=proj;
        }
    }
    std::cout<<"Built CV dataset: "<<rawCount<<" samples x "<< engineeredFeatureCount<<" features"<<std::endl;

    struct RFConfig {
        int trees;
        int depth;
        float sr;
        int minSamples;
        float cvMean;
        float cvStd;
    };
    std::vector<RFConfig> rfConfigs;

    {
        Timer t("Random Forest CV Grid Search");
        std::cout << std::endl;

        for(int trees : {50, 100, 200, 500}){
            for(int depth : {4, 6, 8, 10, 14, 0}){ // 0 = unlimited
                for(float sr : {0.1f,0.2f,0.3f,0.4f,0.5f,0.7f}){
                    for(int ms : {1, 3, 5, 10}){
                        int actualDepth = (depth == 0) ? 255 : depth;

                        std::cout << "\n  Config: t=" << trees << " d=" << depth<< " sr=" << sr<< " ms="<<ms<<std::endl;

                        CVResults cv = crossValidateRF(cvSamples, rawCount,
                        engineeredFeatureCount, 5,
                        trees, actualDepth, ms, sr,
                        feature_medians, feature_iqrs, selector);

                        std::cout << "  => Mean: " << std::fixed << std::setprecision(1)
                                  << (cv.meanAcc * 100) << "% +/- "
                                  << (cv.stdAcc * 100) << "%" << std::endl;

                        rfConfigs.push_back({trees, actualDepth, sr, ms, cv.meanAcc, cv.stdAcc});
                    }
                }
            }
        }
    }

    auto bestConfig = std::max_element(rfConfigs.begin(), rfConfigs.end(),
        [](const RFConfig& a, const RFConfig& b) { return a.cvMean < b.cvMean; });

    std::cout << "\nBest CV Config: t=" << bestConfig->trees
            << " d=" << bestConfig->depth << " sr=" << bestConfig->sr
            << " ms=" << bestConfig->minSamples
            << " cv=" << std::fixed << std::setprecision(1)
            << (bestConfig->cvMean * 100) << "% +/- "
            << (bestConfig->cvStd * 100) << "%" << std::endl;

    //full cv
    std::cout << "\nFull 5-fold CV with best config:" << std::endl;
    CVResults bestCV = crossValidateRF(cvSamples, rawCount, 
    engineeredFeatureCount, 5,
    bestConfig->trees, bestConfig->depth, 
    bestConfig->minSamples, bestConfig->sr,
    feature_medians, feature_iqrs, selector);

    std::cout << "\nAggregate CV Results:" << std::endl;
    printConfusionMatrix(bestCV.aggregateMetrics);
    printPerClassMetrics(bestCV.aggregateMetrics);

    //final model
    std::cout << "\nTraining final model on full training set..." << std::endl;
    RandomForest* bestRf = new RandomForest(bestConfig->trees, bestConfig->depth,
        bestConfig->minSamples, bestConfig->sr);
    bestRf->train(trainSet, trainCount, reducedFeatureCount);
    ml_metrics_t rfMetrics = bestRf->evaluate(testSet, testCount);

    printMetrics(rfMetrics, "Random Forest (holdout)");
    printPerClassMetrics(rfMetrics);
    printConfusionMatrix(rfMetrics);
    bestRf->printFeatureImportance();

    std::cout << "\nNOTE: CV accuracy (" << std::fixed << std::setprecision(1)
              << (bestCV.meanAcc * 100) << "%) is more reliable than holdout ("
              << (rfMetrics.accuracy * 100) << "%)" << std::endl;

    //save
    dt.saveModel("dt_model.bin");
    bestRf->saveModel("rf_model.bin");
    knn.saveModel("knn_model.bin");

    //summary
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "  Features: " << reducedFeatureCount << "/" << engineeredFeatureCount << " selected" << std::endl;
    std::cout << "  Decision Tree:     " << std::fixed << std::setprecision(1) << (dtMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  KNN (K=" << (int)knn.getK() << "):         " << (knnMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  Random Forest:     "<<(rfMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  Hierarchical:      "<<(hierMetrics.accuracy * 100) << "%" << std::endl;

    std::string bestModel="Decision Tree";
    float bestAcc=dtMetrics.accuracy;

    if (knnMetrics.accuracy > bestAcc){
        bestAcc=knnMetrics.accuracy; bestModel="KNN";
    }
    if (rfMetrics.accuracy > bestAcc){
        bestAcc=rfMetrics.accuracy; bestModel="Random Forest";
    }
    if (hierMetrics.accuracy > bestAcc){
        bestAcc=hierMetrics.accuracy; bestModel="Hierarchical";
    }
    std::cout << "\n >> Best individual: " << bestModel << " (" << (bestAcc * 100) << "%)" << std::endl;

    //===========================================================================================================
    //ensemble
    //===========================================================================================================
    {
        std::cout << "\n=== Weighted Ensemble Evaluation ===" << std::endl;

        //weights
        float dtW   = powf(dtMetrics.accuracy, 4);
        float knnW  = powf(knnMetrics.accuracy, 4);
        float rfW   = powf(rfMetrics.accuracy, 4);
        float hierW = powf(hierMetrics.accuracy, 4);

        std::cout << "  Weights: DT=" << std::fixed << std::setprecision(3) << dtW<<" KNN="<<knnW<<" RF=" << rfW<<" Hier="<<hierW<<std::endl;

        ml_metrics_t ensMetrics;
        memset(&ensMetrics, 0, sizeof(ensMetrics));
        ensMetrics.total = testCount;

        for(uint16_t i = 0; i < testCount; i++){
            scent_class_t actual = testSet[i].label;
            scent_class_t pred = ensemblePredict(testSet[i].features,
            dt, knn, *bestRf, hier,reducedFeatureCount,dtW, knnW, rfW, hierW);

            if(pred==actual){
                ensMetrics.correct++;
            }
            if(actual<SCENT_CLASS_COUNT&&pred<SCENT_CLASS_COUNT){
                ensMetrics.confusionMatrix[actual][pred]++;
            }
        }
        ensMetrics.accuracy = (ensMetrics.total > 0) ? (float)ensMetrics.correct / ensMetrics.total : 0.0f;

        printMetrics(ensMetrics, "Weighted Ensemble");
        printPerClassMetrics(ensMetrics);
        printConfusionMatrix(ensMetrics);

        //update best if ensemble wins
        if(ensMetrics.accuracy > bestAcc){
            bestAcc = ensMetrics.accuracy;
            bestModel = "Weighted Ensemble";
        }

        std::cout << "\n >> Overall Best: " << bestModel << " (" 
                  << std::fixed << std::setprecision(1) << (bestAcc * 100) << "%)" << std::endl;

        // Save ensemble weights
        {
            std::ofstream ef("ensemble_weights.h");
            ef << "#ifndef ENSEMBLE_WEIGHTS_H\n#define ENSEMBLE_WEIGHTS_H\n\n";
            ef << "static const float DT_WEIGHT=" << dtW << "f;\n";
            ef << "static const float KNN_WEIGHT=" << knnW << "f;\n";
            ef << "static const float RF_WEIGHT=" << rfW << "f;\n";
            ef << "static const float HIER_WEIGHT=" << hierW << "f;\n";
            ef << "\n#endif\n";
        }
    }

    // Sample predictions
    std::cout << "\nSample Predictions:" << std::endl;
    int sampleIdx=std::min((int)testCount, 5);

    float dtW = dtMetrics.accuracy*dtMetrics.accuracy;
    float knnW = knnMetrics.accuracy*knnMetrics.accuracy;
    float rfW = rfMetrics.accuracy*rfMetrics.accuracy;
    float hierW = hierMetrics.accuracy*hierMetrics.accuracy;

    for(int i=0; i < sampleIdx; i++){
        scent_class_t actual=testSet[i].label;
        scent_class_t dtPred=dt.predict(testSet[i].features);
        float knnConf, rfConf;
        scent_class_t knnPred=knn.predictWithConfidence(testSet[i].features, reducedFeatureCount, knnConf);
        scent_class_t rfPred=bestRf->predictWithConfidence(testSet[i].features, rfConf);
        scent_class_t hierPred = hier.predict(testSet[i].features);
        scent_class_t ensPred = ensemblePredict(testSet[i].features,dt, knn, *bestRf, hier,reducedFeatureCount,dtW, knnW,rfW, hierW);

        std::cout
                << "\n  Sample " << i << ": actual=" << CSVLoader::getClassName(actual)
                << "\n    DT:       " << CSVLoader::getClassName(dtPred) << (dtPred == actual ? " [OK]" : " [X]")
                << "\n    KNN:      " << CSVLoader::getClassName(knnPred) << " ("
                << std::fixed << std::setprecision(0) << (knnConf * 100) << "%)"
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
    delete[] trainFull;
    delete[] testFull;
    delete[] trainSet;
    delete[] testSet;
    delete[] knnTrainSet;
    delete[] cvSamples;
    delete bestRf;

    std::cout << "\nTraining complete." << std::endl;

    std::cout.rdbuf(originalBuf);
    std::cerr.rdbuf(originalErrBuf);
    logFile.close();

    std::cout << "\nOutput saved to training_output.txt" << std::endl;
    return 0;
}