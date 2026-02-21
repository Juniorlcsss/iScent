#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>


#include "csv_loader.h"
#include "dt.h"
#include "knn.h"
#include "rf.h"

void printMetrics(const ml_metrics_t& m, const char* modelName){
    std::cout << "\n--- " << modelName << " Results ---" << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << (m.accuracy*100)<<"% (" << m.correct << "/" << m.total << ")" << std::endl;
}

void printPerClassMetrics(const ml_metrics_t& m){
    std::cout<<"\nPer-Class Precision/Recall/F1:"<<std::endl;

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        //recall
        int recallTotal=0;
        for(int j=0; j<SCENT_CLASS_COUNT; j++){
            recallTotal+= m.confusionMatrix[i][j];
        }

        //precision
        int predictedTotal=0;
        for(int j=0; j<SCENT_CLASS_COUNT; j++){
            predictedTotal+= m.confusionMatrix[j][i];
        }

        //true positives
        int tp = m.confusionMatrix[i][i];
        float recall = (recallTotal > 0) ? ((float)tp / recallTotal) : 0.0f;
        float precision = (predictedTotal > 0) ? ((float)tp / predictedTotal) : 0.0f;
        float f1 = (precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0.0f;

        std::cout<<"P="<< std::fixed << std::setprecision(2) << (precision*100)<<"%";
        std::cout<<" R="<< std::fixed << std::setprecision(2)<< (recall*100)<<"%";
        std::cout<<" F1="<< std::fixed << std::setprecision(2) << (f1*100)<<"%"<<std::endl;
    }
}


void printConfusionMatrix(const ml_metrics_t &m){
    std::cout << "\nConfusion Matrix:"<<std::endl;
    std::cout << "Predicted -> ";
    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        std::cout << std::setw(4) << i;
    }
    std::cout << std::endl;

    for(int i=0; i<SCENT_CLASS_COUNT; i++){
        std::cout << "Actual " << std::setw(2) << i << ":   ";
        for (int j = 0; j < SCENT_CLASS_COUNT; j++) {
            std::cout << std::setw(4) << m.confusionMatrix[i][j];
        }
        std::cout << "  | " << CSVLoader::getClassName((scent_class_t)i) << std::endl;
    }
}

void printFeatureNames() {
    std::cout<<"\nFeatures:"<< CSV_FEATURE_COUNT <<std::endl;
    for(int i=0; i<10;i++){
        std::cout<<"  ["<< i <<"] gas1_resp_step"<< i <<std::endl;
    }
    for(int i=0; i<10;i++){
        std::cout<<"  ["<< (10+i) <<"] gas2_resp_step"<< i <<std::endl;
    }
    for(int i = 0; i < 10; i++){
        std::cout << "[" << (20+i) << "] cross_ratio_step" << i << std::endl;
    }
    std::cout << "[30] slope1\n[31] slope2\n"<<"[32] curvature1\n[33] curvature2\n"<<"[34] delta_temp\n[35] delta_hum\n[36] delta_pres"<<std::endl;
}


static const char* shortFeatureName(int idx){
    static char buf[16];
    if(idx<10){
        snprintf(buf, sizeof(buf),"g1_s%d",idx);
        return buf;
    }
    if(idx>=10 && idx<20){
        snprintf(buf, sizeof(buf),"g2_s%d",idx-10);
        return buf;
    }
    if(idx>=20 && idx<30){
        snprintf(buf, sizeof(buf),"cr_s%d",idx-20);
        return buf;
    }
    if(idx==30){
        return "slope1";
    }
    if(idx==31){
        return "slope2";
    }
    if(idx==32){
        return "curvature1";
    }
    if(idx==33){
        return "curvature2";
    }
    if(idx==34){
        return "delta_temp";
    }
    if(idx==35){
        return "delta_hum";
    }
    if(idx==36){
        return "delta_pres";
    }

    snprintf(buf, sizeof(buf), "f_%d", idx);
    return buf;

}

int main(int argc, char* argv[]){
    std::cout <<"========================================" << std::endl;
    std::cout <<"    Machine Learning Model Trainer" << std::endl;
    std::cout <<"========================================" << std::endl;

    //get file
    std::string filename = "data.csv";
    float trainRatio = 0.8f;
    
    if (argc > 1) filename = argv[1];
    if (argc > 2) trainRatio = std::stof(argv[2]);
    
    std::cout << "\nLoading: " << filename << std::endl;
    std::cout << "Train/Test split: " << (trainRatio * 100) << "/" << ((1 - trainRatio) * 100) << std::endl;


    //load data
    CSVLoader loader;
    if(!loader.load(filename)){
        std::cerr << "Failed to load data!" << std::endl;
        return 1;
    }

    loader.printInfo();
    printFeatureNames();

    //split
    csv_training_sample_t *trainSet = nullptr;
    csv_training_sample_t *testSet = nullptr;
    uint16_t trainCount=0, testCount=0;
    loader.split(trainRatio, trainSet, trainCount, testSet, testCount);
    std::cout << "\n=== Training Feature Statistics ===" << std::endl;

    
    //per class stats
    for (int c = 0; c < SCENT_CLASS_COUNT; c++) {
        std::vector<std::vector<float>> classFeatures(CSV_FEATURE_COUNT);
        int classCount = 0;
        
        for (int i = 0; i < trainCount; i++) {
            if (trainSet[i].label == c) {
                classCount++;
                for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
                    classFeatures[f].push_back(trainSet[i].features[f]);
                }
            }
        }
        
        if (classCount == 0) continue;
        
        std::cout << "\nClass: " << CSVLoader::getClassName((scent_class_t)c) 
                << " (" << classCount << " samples)" << std::endl;
        for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
            float sum = 0;
            for (float v : classFeatures[f]) sum += v;
            float mean = sum / classFeatures[f].size();
            
            float varSum = 0;
            for (float v : classFeatures[f]) varSum += (v - mean) * (v - mean);
            float std = sqrt(varSum / classFeatures[f].size());
            
            float mn = *std::min_element(classFeatures[f].begin(), classFeatures[f].end());
            float mx = *std::max_element(classFeatures[f].begin(), classFeatures[f].end());
            
            std::cout << "  " << std::setw(8) << shortFeatureName(f)
                    << ": mean=" << std::setw(10) << std::fixed << std::setprecision(4) << mean
                    << " std=" << std::setw(10) << std << " min=" << std::setw(10) << mn 
                    << " max=" << std::setw(10) << mx << std::endl;
        }
    }

    //get global training stats
    float feature_means[CSV_FEATURE_COUNT] = {0};
    float feature_stds[CSV_FEATURE_COUNT] = {0};

    for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
        double sum = 0;
        for (int i = 0; i < trainCount; i++) {
            sum += trainSet[i].features[f];
        }
        feature_means[f] = sum / trainCount;
        
        double varSum = 0;
        for (int i = 0; i < trainCount; i++) {
            double diff = trainSet[i].features[f] - feature_means[f];
            varSum += diff * diff;
        }
        feature_stds[f] = sqrt(varSum / trainCount);
        if (feature_stds[f] < 1e-6f) feature_stds[f] = 1.0f; // prevent division by zero
    }

    //apply z-score
    for (int i = 0; i < trainCount; i++) {
        for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
            trainSet[i].features[f] = (trainSet[i].features[f] - feature_means[f]) / feature_stds[f];
        }
    }
    for (int i = 0; i < testCount; i++) {
        for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
            testSet[i].features[f] = (testSet[i].features[f] - feature_means[f]) / feature_stds[f];
        }
    }

    //save to feature_stats
    std::ofstream statsFile("feature_stats.h");
    statsFile << "#ifndef FEATURE_STATS_H\n#define FEATURE_STATS_H\n\n";
    statsFile << "static const float FEATURE_MEANS[" << CSV_FEATURE_COUNT << "] = {";
    for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
        statsFile << std::setprecision(8) << feature_means[f];
        if (f < CSV_FEATURE_COUNT - 1) statsFile << ", ";
    }
    statsFile << "};\n\n";
    statsFile << "static const float FEATURE_STDS[" << CSV_FEATURE_COUNT << "] = {";
    for (int f = 0; f < CSV_FEATURE_COUNT; f++) {
        statsFile << std::setprecision(8) << feature_stds[f];
        if (f < CSV_FEATURE_COUNT - 1) statsFile << ", ";
    }
    statsFile << "};\n\n#endif\n";
    statsFile.close();

    //save unaugmented for knn (sensitive to duplicates)
    csv_training_sample_t* knnTrainSet = new csv_training_sample_t[trainCount];
    uint16_t knnTrainCount = trainCount;
    memcpy(knnTrainSet, trainSet, trainCount * sizeof(csv_training_sample_t));

    //data augmentation
    std::default_random_engine rng(42);

    //augment
    std::normal_distribution<float> gasNoise(0.0f, 0.05f);    //gas features: +-% noise in z-space
    std::normal_distribution<float> ratioNoise(0.0f, 0.04f);
    std::normal_distribution<float> shapeNoise(0.0f, 0.06f);
    std::normal_distribution<float> envNoise(0.0f, 0.08f);

    uint16_t classCounts[SCENT_CLASS_COUNT] = {0};
    std::vector<std::vector<uint16_t>> classIndices(SCENT_CLASS_COUNT);

    for(int i=0;i<trainCount; i++){
        if(trainSet[i].label <SCENT_CLASS_COUNT){
            classCounts[trainSet[i].label]++;
            classIndices[trainSet[i].label].push_back(i);
        }
    }

    uint16_t maxClassCount = *std::max_element(classCounts,classCounts+SCENT_CLASS_COUNT);
    uint16_t targetPerClass = maxClassCount * 3;

    std::vector<csv_training_sample_t> augmented;
    augmented.reserve(targetPerClass * SCENT_CLASS_COUNT);

    //keep og
    for(int i=0; i<trainCount;i++){
        augmented.push_back(trainSet[i]);
    }

    std::uniform_int_distribution<uint16_t> tempDist(0,1);
    for(int c=0; c<SCENT_CLASS_COUNT;c++){
        if(classIndices[c].empty()){
            continue;
        }

        int need= targetPerClass - (int)classCounts[c];
        std::uniform_int_distribution<uint16_t> srcDist(0, classIndices[c].size()-1);
        for(int j=0;j<need;j++){
            uint16_t srcIdx = classIndices[c][srcDist(rng)];
            csv_training_sample_t noisy = trainSet[srcIdx];

            for(int i=0; i<10;i++){
                noisy.features[i] += gasNoise(rng); //gas1
            }

            for(int i=10; i<20;i++){
                noisy.features[i] += gasNoise(rng); //gas2
            }

            for(int i=20; i<30;i++){
                noisy.features[i] += ratioNoise(rng); //cross ratios
            }

            for(int i=30; i<34;i++){
                noisy.features[i] += shapeNoise(rng); //slopes/curvature
            }
            for(int i=34; i<37;i++){
                noisy.features[i] += envNoise(rng); //env diffs
            }

            augmented.push_back(noisy);
        }
    }

    // Replace training set
    delete[] trainSet;
    trainCount = augmented.size();
    trainSet = new csv_training_sample_t[trainCount];
    for (uint16_t i = 0; i < trainCount; i++) {
        trainSet[i] = augmented[i];
    }
    std::cout << "Augmented training set: " << trainCount << " samples" << std::endl;    
    for(int c=0; c<SCENT_CLASS_COUNT;c++){
        int count=0;
        for(int j=0; j<trainCount; j++){
            if(trainSet[j].label == c){
                count++;
            }
        }
        std::cout << "Class " << CSVLoader::getClassName((scent_class_t)c) << ": " << count << " samples" << std::endl;
    }
    



    //===========================================================================================================
    //DT
    //===========================================================================================================
    std::cout << "\nTraining Decision Tree..." << std::endl;

    uint8_t bestDTDepth=0;
    uint8_t bestDTMinSamples=0;
    float bestDTAcc=0.0f;

    for(uint8_t d=5; d<=20; d++){
        for(uint8_t ms:{1,3,5,8,10,15,20}){
            DecisionTree dtTest;
            dtTest.train(trainSet, trainCount, CSV_FEATURE_COUNT, d, ms);
            ml_metrics_t m = dtTest.evaluate(testSet, testCount);

            std::cout << "DT depth=" << (int)d << " minSamples=" << (int)ms<< " -> " << std::fixed << std::setprecision(2) << (m.accuracy * 100) << "%" << std::endl;
            
            if(m.accuracy > bestDTAcc){
                bestDTAcc = m.accuracy;
                bestDTDepth = d;
                bestDTMinSamples = ms;
            }
        }
    }

    std::cout << "\nBest DT Depth: " << (int)bestDTDepth << " Min Samples: " << (int)bestDTMinSamples << " Accuracy: " << std::fixed << std::setprecision(2) << (bestDTAcc * 100) << "%" << std::endl;



    DecisionTree dt;
    dt.train(trainSet, trainCount, CSV_FEATURE_COUNT, bestDTDepth, bestDTMinSamples);
    dt.getStats();

    std::cout << "Tree Stats:"<<std::endl;
    std::cout << "Depth: " << dt.getDepth() << std::endl;
    std::cout << "Nodes: " << dt.getDTNodeCount() << std::endl;
    std::cout << "Leaf Nodes: " << dt.getLeafCount() << std::endl;

    //print tree
    std::cout << "\nDecision Tree Structure (max depth 3):" << std::endl;
    dt.printTree(3);

    ml_metrics_t dtMetrics = dt.evaluate(testSet, testCount);
    printMetrics(dtMetrics, "Decision Tree");
    printPerClassMetrics(dtMetrics);
    printConfusionMatrix(dtMetrics);



    //===========================================================================================================
    //KNN
    //===========================================================================================================
    std::cout << "\nTraining K-Nearest Neighbors..." << std::endl;
    KNN knn;

    knn.train(knnTrainSet, knnTrainCount); //unaugmented

    uint8_t k=1;
    float bestKnnAcc = 0.0f;

    for(int i=1; i<=20; i++){
        knn.setK(i);
        ml_metrics_t knnM = knn.evaluate(testSet, testCount);
        std::cout << "K=" << std::setw(2) << i << ": " 
        << std::fixed << std::setprecision(2) << (knnM.accuracy * 100) << "%" << std::endl;

        if(knnM.accuracy > bestKnnAcc){
            bestKnnAcc = knnM.accuracy;
            k = i;
        }
    }

    std::cout << "\nBest K: " << (int)k << std::endl;
    std::cout << "Accuracy: "<< std::fixed << std::setprecision(2) << (bestKnnAcc * 100) << "%" << std::endl;

    knn.setK(k);
    ml_metrics_t kNNMetrics = knn.evaluate(testSet, testCount);
    printMetrics(kNNMetrics, "K-Nearest Neighbors");
    printPerClassMetrics(kNNMetrics);
    printConfusionMatrix(kNNMetrics);

    //===========================================================================================================
    //Random Forest
    //===========================================================================================================
    std::cout << "\nTraining Random Forest..." << std::endl;

    RandomForest* bestRf = nullptr;
    float bestOOBError = 1.0f;
    uint16_t bestNumTrees=0;
    float bestRfAcc = 0.0f;

    for (int trees:{10, 25, 50, 100, 125, 150, 175, 200, 225, 250, 255}){
        RandomForest* rfTest= new RandomForest(trees, 15, 3, 0.25f);
        rfTest->train(trainSet, trainCount, CSV_FEATURE_COUNT);

        ml_metrics_t m = rfTest->evaluate(testSet, testCount);
        float oobErr= rfTest->getOOBError();

        std::cout<< "Trees: " << std::setw(3) << trees << std::endl;
        std::cout << "Accuracy: "<< std::fixed << std::setprecision(2) << (m.accuracy * 100) <<"%" << "OOB Error: " <<  (oobErr*100) << "%" << std::endl;
        
        if(oobErr < bestOOBError){
            bestOOBError = oobErr;
            bestRfAcc = m.accuracy;
            bestNumTrees=trees;
            delete bestRf;
            bestRf=rfTest;
        }
        else{
            delete rfTest;
        }
    }
    std::cout << "\nBest Number of Trees: " << bestNumTrees << std::endl;
    std::cout << "Accuracy: "<< std::fixed << std::setprecision(2) << (bestRfAcc * 100) << "%" << std::endl;
    std::cout << "OOB Error: "<< std::fixed << std::setprecision(2) << (bestOOBError * 100) << "%" << std::endl;

    ml_metrics_t rfM = bestRf->evaluate(testSet, testCount);
    printMetrics(rfM, "Random Forest");
    printConfusionMatrix(rfM);
    printPerClassMetrics(rfM);
    bestRf->printFeatureImportance();
    
    
    //===========================================================================================================

    //save models
    dt.saveModel("dt_model.bin");
    bestRf->saveModel("rf_model.bin");
    knn.saveModel("knn_model.bin");

    std::cout << "\nModels saved to disk." << std::endl;

    //sample prediciotns
    std::cout << "\nSample Predictions:" << std::endl;
    int sampleIdx = (testCount <5) ? testCount : 5;
    for(int i=0; i<sampleIdx; i++){
        scent_class_t actual = testSet[i].label;
        scent_class_t dtPred = dt.predict(testSet[i].features);

        float knnConf;
        scent_class_t knnPred = knn.predictWithConfidence(testSet[i].features, CSV_FEATURE_COUNT, knnConf);

        float rfConf;
        scent_class_t rfPred = bestRf->predictWithConfidence(testSet[i].features, rfConf);

        std::cout << "\nSample " << i << ":" << std::endl;

        std::cout << "  Actual: " << CSVLoader::getClassName(actual) << std::endl;
        std::cout << "  DT:     " << CSVLoader::getClassName(dtPred) 
        << (dtPred == actual ? " [OK]" : " [X]") << std::endl;

        std::cout << "  KNN:    " << CSVLoader::getClassName(knnPred) 
        << " (conf: " << std::fixed << std::setprecision(0) << (knnConf * 100) << "%)"
        << (knnPred == actual ? " [OK]" : " [X]") << std::endl;

        std::cout << "  RF:     " << CSVLoader::getClassName(rfPred) 
        << " (conf: " << std::fixed << std::setprecision(0) << (rfConf * 100) << "%)"
        << (rfPred == actual ? " [OK]" : " [X]") << std::endl;
    }

    //model summary
    std::cout << "  Decision Tree:  " << (dtMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  KNN (K=" << (int)knn.getK() << "):       " << (kNNMetrics.accuracy * 100) << "%" << std::endl;
    std::cout << "  Random Forest:  " << (rfM.accuracy * 100) << "%" << std::endl;


    //best model
    std::string bestModel = "Decision Tree";
    float bestAcc = dtMetrics.accuracy;
    if (kNNMetrics.accuracy > bestAcc) { bestAcc = kNNMetrics.accuracy; bestModel = "KNN"; }
    if (rfM.accuracy > bestAcc) { bestAcc = rfM.accuracy; bestModel = "Random Forest"; }
    
    std::cout << "\n  >> Best Model: " << bestModel << " (" << (bestAcc * 100) << "%)" << std::endl;


    //save ensemble weights
    std::ofstream ensembleFile("ensemble_weights.h");
    ensembleFile << "#ifndef ENSEMBLE_WEIGHTS_H\n#define ENSEMBLE_WEIGHTS_H\n\n";
    ensembleFile << "static const float DT_WEIGHT = " << (dtMetrics.accuracy * dtMetrics.accuracy) << "f;\n";
    ensembleFile << "static const float KNN_WEIGHT = " << (kNNMetrics.accuracy * kNNMetrics.accuracy) << "f;\n";
    ensembleFile << "static const float RF_WEIGHT = " << (rfM.accuracy * rfM.accuracy) << "f;\n";
    ensembleFile << "\n#endif\n";
    ensembleFile.close();

    //cleanup
    delete[] trainSet;
    delete[] testSet;
    delete[] knnTrainSet;
    delete bestRf;

    std::cout << "\nTraining complete. Exiting." << std::endl;

    return 0;
}