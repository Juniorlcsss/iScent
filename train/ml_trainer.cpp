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
    int offset=idx - 58;
    int numNames=sizeof(names) / sizeof(names[0]);
    if (offset >= 0 && offset < numNames) return names[offset];
    snprintf(buf, sizeof(buf), "f_%d", idx);
    return buf;
}

// ============================================================
// Feature Selection (ANOVA F-ratio)
// ============================================================
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

    void projectDataset(const csv_training_sample_t* src, uint16_t count,
                        csv_training_sample_t* dst) const {
        for(uint16_t i=0; i < count; i++){
            dst[i].label=src[i].label;
            memset(dst[i].features, 0, sizeof(dst[i].features));
            for(int f=0; f < selectedCount; f++){
                dst[i].features[f]=src[i].features[selectedIndices[f]];
            }
        }
    }

    void saveHeader(const char* filename, const float* means, const float* stds) const {
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

        f << "static const float SELECTED_MEANS[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << std::setprecision(8) << means[selectedIndices[i]];
            if (i < selectedCount - 1) f << ", ";
        }
        f << "};\n\n";

        f << "static const float SELECTED_STDS[" << selectedCount << "]={";
        for(int i=0; i < selectedCount; i++){
            f << std::setprecision(8) << stds[selectedIndices[i]];
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
            //remove unused
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


//experiment
void twoStageExperiment(const csv_training_sample_t* trainSet, uint16_t trainCount, const csv_training_sample_t* testSet, uint16_t testCount,uint16_t featureCount){
    std::cout <<"\n=== Classification Experiment ===" <<std::endl;

    //stage 1: tea-type (0,2) vs coffee-type (1,3)
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

    //stage 2a: decaf_tea vs tea
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


int main(int argc, char* argv[]){
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
    uint16_t trainCount=0, testCount=0;
    loader.split(trainRatio, trainSet, trainCount, testSet, testCount);

    //zscore
    float feature_means[CSV_FEATURE_COUNT]={0};
    float feature_stds[CSV_FEATURE_COUNT]={0};

    for(int f=0; f < CSV_FEATURE_COUNT; f++){
        double sum=0;
        for(int i=0; i < trainCount; i++) sum += trainSet[i].features[f];
        feature_means[f]=sum / trainCount;

        double varSum=0;
        for(int i=0; i < trainCount; i++){
            double diff=trainSet[i].features[f] - feature_means[f];
            varSum += diff * diff;
        }
        feature_stds[f]=sqrt(varSum / trainCount);
        if (feature_stds[f] < 1e-6f) feature_stds[f]=1.0f;
    }

    for(int i=0; i < trainCount; i++)
        for(int f=0; f < CSV_FEATURE_COUNT; f++)
            trainSet[i].features[f]=(trainSet[i].features[f] - feature_means[f]) / feature_stds[f];
    for(int i=0; i < testCount; i++)
        for(int f=0; f < CSV_FEATURE_COUNT; f++)
            testSet[i].features[f]=(testSet[i].features[f] - feature_means[f]) / feature_stds[f];

    //save feats
    {
        std::ofstream sf("feature_stats.h");
        sf << "#ifndef FEATURE_STATS_H\n#define FEATURE_STATS_H\n\n";
        sf << "#define FULL_FEATURE_COUNT " << CSV_FEATURE_COUNT << "\n\n";
        sf << "static const float FEATURE_MEANS[" << CSV_FEATURE_COUNT << "]={";
        for(int f=0; f < CSV_FEATURE_COUNT; f++){
            sf << std::setprecision(8) << feature_means[f];
            if (f < CSV_FEATURE_COUNT - 1) sf << ", ";
        }
        sf << "};\n\nstatic const float FEATURE_STDS[" << CSV_FEATURE_COUNT << "]={";
        for(int f=0; f < CSV_FEATURE_COUNT; f++){
            sf << std::setprecision(8) << feature_stds[f];
            if (f < CSV_FEATURE_COUNT - 1) sf << ", ";
        }
        sf << "};\n\n#endif\n";
    }

    //feature selection
    FeatureSelector selector;
    {
        Timer t("Feature Selection");
        selector.selectFeatures(trainSet, trainCount, CSV_FEATURE_COUNT, 0.5f, 45);
    }
    selector.saveHeader("feature_select.h", feature_means, feature_stds);

    uint16_t reducedFeatureCount=selector.selectedCount;

    csv_training_sample_t* trainReduced=new csv_training_sample_t[trainCount];
    csv_training_sample_t* testReduced=new csv_training_sample_t[testCount];
    selector.projectDataset(trainSet, trainCount, trainReduced);
    selector.projectDataset(testSet, testCount, testReduced);


    csv_training_sample_t* trainFull=trainSet;
    csv_training_sample_t* testFull=testSet;
    trainSet=trainReduced;
    testSet=testReduced;

    std::cout << "\nTraining with " << reducedFeatureCount << " selected features" << std::endl;

    //experiment
    twoStageExperiment(trainSet, trainCount, testSet, testCount, reducedFeatureCount);

    csv_training_sample_t* knnTrainSet=new csv_training_sample_t[trainCount];
    uint16_t knnTrainCount=trainCount;
    memcpy(knnTrainSet, trainSet, trainCount * sizeof(csv_training_sample_t));

    //aug
    {
        Timer t("Data Augmentation");
        std::default_random_engine rng(42);
        augmentTrainingData(trainSet, trainCount, reducedFeatureCount, rng);
    }

    std::cout << "Augmented: " << trainCount << " samples" << std::endl;
    for(int c=0; c < SCENT_CLASS_COUNT; c++){
        int cnt=0;
        for(int j=0; j < trainCount; j++)
            if (trainSet[j].label == c) cnt++;
        std::cout << "  " << CSVLoader::getClassName((scent_class_t)c) << ": " << cnt << std::endl;
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
    uint8_t bestDTDepth=10, bestDTMinSamples=20;

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

    knn.setK(bestK);
    ml_metrics_t knnMetrics=knn.evaluate(testSet, testCount);
    printMetrics(knnMetrics, "KNN");
    printPerClassMetrics(knnMetrics);
    printConfusionMatrix(knnMetrics);

    //========================================================
    //RF
    //========================================================
    std::cout << "\n=== Training Random Forest ===" << std::endl;
    RandomForest* bestRf=nullptr;
    float bestRfAcc=0.0f;
    uint16_t bestNumTrees=0;

    {
        Timer t("Random Forest Grid Search");
        std::cout << std::endl;

        for(int trees : {50, 100, 200}){
            for(int depth : {10, 14}){
                for(float sr : {0.3f, 0.5f}){
                    std::cout << "  RF t="<< trees << " d=" << depth<< " sr=" << sr << " ..." << std::flush;

                    RandomForest* rfTest=new RandomForest(trees, depth, 3, sr);
                    rfTest->train(trainSet, trainCount, reducedFeatureCount);

                    ml_metrics_t m=rfTest->evaluate(testSet, testCount);

                    std::cout << " acc=" << std::fixed << std::setprecision(1)<< (m.accuracy * 100) << "% oob="<< (rfTest->getOOBError() * 100) << "%" << std::endl;

                    if(m.accuracy > bestRfAcc){
                        bestRfAcc=m.accuracy;
                        bestNumTrees=trees;
                        delete bestRf;
                        bestRf=rfTest;
                    } 
                    else{
                        delete rfTest;
                    }
                }
            }
        }
    }

    ml_metrics_t rfMetrics=bestRf->evaluate(testSet, testCount);
    printMetrics(rfMetrics, "Random Forest");
    printPerClassMetrics(rfMetrics);
    printConfusionMatrix(rfMetrics);
    bestRf->printFeatureImportance();

    //save
    dt.saveModel("dt_model.bin");
    bestRf->saveModel("rf_model.bin");
    knn.saveModel("knn_model.bin");

    //summary
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "  Features: " << reducedFeatureCount << "/" << CSV_FEATURE_COUNT << " selected" << std::endl;
    std::cout << "  Decision Tree:  " << std::fixed << std::setprecision(1) << (dtMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  KNN (K=" << (int)knn.getK() << "):"<< (knnMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  Random Forest:  " << (rfMetrics.accuracy * 100) << "%" << std::endl;

    std::string bestModel="Decision Tree";
    float bestAcc=dtMetrics.accuracy;
    if (knnMetrics.accuracy > bestAcc){ bestAcc=knnMetrics.accuracy; bestModel="KNN"; }
    if (rfMetrics.accuracy > bestAcc){ bestAcc=rfMetrics.accuracy; bestModel="Random Forest"; }
    std::cout << "\n >> Best: " << bestModel << " (" << (bestAcc * 100) << "%)" << std::endl;

    // Sample predictions
    std::cout << "\nSample Predictions:" << std::endl;
    int sampleIdx=std::min((int)testCount, 5);
    for(int i=0; i < sampleIdx; i++){
        scent_class_t actual=testSet[i].label;
        scent_class_t dtPred=dt.predict(testSet[i].features);
        float knnConf, rfConf;
        scent_class_t knnPred=knn.predictWithConfidence(testSet[i].features, reducedFeatureCount, knnConf);
        scent_class_t rfPred=bestRf->predictWithConfidence(testSet[i].features, rfConf);
        std::cout 
                << "\n  Sample " << i << ": actual=" << CSVLoader::getClassName(actual)
                << "\n    DT:  " << CSVLoader::getClassName(dtPred) << (dtPred == actual ? " [OK]" : " [X]")
                << "\n    KNN: " << CSVLoader::getClassName(knnPred) << " ("
                << std::fixed << std::setprecision(0) << (knnConf * 100) << "%)"
                << (knnPred == actual ? " [OK]" : " [X]")
                << "\n    RF:  " << CSVLoader::getClassName(rfPred) << " ("
                << (rfConf * 100) << "%)" << (rfPred == actual ? " [OK]" : " [X]")
                << std::endl;
    }

    //ensamble weights
    {
        std::ofstream ef("ensemble_weights.h");
        ef << "#ifndef ENSEMBLE_WEIGHTS_H\n#define ENSEMBLE_WEIGHTS_H\n\n";
        ef << "static const float DT_WEIGHT=" << (dtMetrics.accuracy * dtMetrics.accuracy) << "f;\n";
        ef << "static const float KNN_WEIGHT=" << (knnMetrics.accuracy * knnMetrics.accuracy) << "f;\n";
        ef << "static const float RF_WEIGHT=" << (rfMetrics.accuracy * rfMetrics.accuracy) << "f;\n";
        ef << "\n#endif\n";
    }

    //cleanup
    delete[] trainFull;
    delete[] testFull;
    delete[] trainSet;
    delete[] testSet;
    delete[] knnTrainSet;
    delete bestRf;

    std::cout << "\nTraining complete." << std::endl;
    return 0;
}