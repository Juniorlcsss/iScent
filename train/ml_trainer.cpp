#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include "csv_loader.h"
#include "dt.h"
#include "knn.h"
#include "rf.h"

void printMetrics(const ml_metrics_t& m, const char* modelName){
    std::cout << "\n--- " << modelName << " Results ---" << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << (m.accuracy*100)<<"% (" << m.correct << "/" << m.total << ")" << std::endl;
}

void printConfusionMatrix(const ml_metrics_t &m){
    std::cout << "\nConfusion Matrix:"<<std::endl;
    std::cout << "Predicted ->  ";
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

void printFeatureNames(){
    const char* names[]={
        "temp1", "hum1", "pres1", "gas1_0",
        "temp2", "hum2", "pres2", "gas2_0",
        "delta_temp", "delta_hum", "delta_pres", "delta_gas"
    };
    std::cout << "\nFeatures (" << CSV_FEATURE_COUNT << "):" << std::endl;
    for(int i=0; i< CSV_FEATURE_COUNT; i++){
        std::cout<< "[" << i << "] " << names[i] << std::endl;
    }
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

    //===========================================================================================================
    //DT
    //===========================================================================================================
    std::cout << "\nTraining Decision Tree..." << std::endl;
    DecisionTree dt;
    dt.train(trainSet, trainCount, CSV_FEATURE_COUNT, 12, 5);
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
    printConfusionMatrix(dtMetrics);





    //===========================================================================================================
    //KNN
    //===========================================================================================================
    std::cout << "\nTraining K-Nearest Neighbors..." << std::endl;
    KNN knn;

    knn.train(trainSet, trainCount);

    uint8_t k=1;
    float bestKnnAcc = 0.0f;

    for(int i=0; i<=15; i+=2){
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
    printConfusionMatrix(kNNMetrics);

    //===========================================================================================================
    //Random Forest
    //===========================================================================================================
    std::cout << "\nTraining Random Forest..." << std::endl;

    uint16_t bestNumTrees=10;
    float bestRfAcc = 0.0f;

    for(int trees : {10,25,50,100}){
        RandomForest rfTest(trees, 10, 5, 0.7f);
        rfTest.train(trainSet, trainCount, CSV_FEATURE_COUNT);

        ml_metrics_t m = rfTest.evaluate(testSet, testCount);
        std::cout<< "Trees: " << std::setw(3) << trees << std::endl;
        std::cout << "Accuracy: "<< std::fixed << std::setprecision(2) << (m.accuracy * 100) <<"%" << "OOB Error: " <<  (rfTest.getOOBError()*100) << "%" << std::endl;
 
        if(m.accuracy > bestRfAcc){
            bestRfAcc = m.accuracy;
            bestNumTrees = trees;
        }
    }

    std::cout << "\nBest Number of Trees: " << bestNumTrees << std::endl;
    std::cout << "Accuracy: "<< std::fixed << std::setprecision(2) << (bestRfAcc * 100) << "%" << std::endl;

    //train final model
    RandomForest rf(bestNumTrees, 10, 5, 0.7f);
    rf.train(trainSet, trainCount, CSV_FEATURE_COUNT);

    ml_metrics_t rfM = rf.evaluate(testSet, testCount);
    printMetrics(rfM, "Random Forest");
    printConfusionMatrix(rfM);
    rf.printFeatureImportance();
    
    
    //===========================================================================================================

    //save models
    dt.saveModel("dt_model.bin");
    rf.saveModel("rf_model.bin");
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
        scent_class_t rfPred = rf.predictWithConfidence(testSet[i].features, rfConf);

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

    //cleanup
    delete[] trainSet;
    delete[] testSet;

    std::cout << "\nTraining complete. Exiting." << std::endl;

    return 0;
}