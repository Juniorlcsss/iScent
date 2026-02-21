#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "../include/model headers/dt_model_header.h"
#include "../include/model headers/knn_model_header.h"
#include "../include/model headers/rf_model_header.h"
#include "../include/feature_stats.h"

static const int FEATURE_COUNT = 37;
static const int NUM_HEATER_STEPS = 10;

struct RawRow {
    std::string rawLabel;
    float temp1, hum1, pres1;
    float gas1[NUM_HEATER_STEPS];
    float temp2, hum2, pres2;
    float gas2[NUM_HEATER_STEPS];
};

struct Row {
    std::string rawLabel;
    float features[FEATURE_COUNT];
};

static const char* CLASS_NAMES[12] = {
    "camomile",
    "thoroughly minted infusion",
    "berry burst",
    "darjeeling blend",
    "decaf nutmeg and vanilla",
    "earl grey",
    "english breakfast tea",
    "fresh orange",
    "garden selection (lemon)",
    "green tea",
    "raspberry",
    "sweet cherry"
};

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

struct Baseline {
    float temp1, temp2;
    float hum1, hum2;
    float pres1, pres2;
    float gas1_steps[NUM_HEATER_STEPS];
    float gas2_steps[NUM_HEATER_STEPS];
};

static float medianVec(std::vector<float>& v) {
    if(v.empty()){
        return 0.0f;
    }


    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];

    if(v.size() % 2 == 0){
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        m = 0.5f * (m + v[mid - 1]);
    }
    return m;
}

static float localSlope(const float* data, int n){
    if(n<=1){
        return 0.0f;
    }
    float sumX=0.0f, sumY=0.0f, sumXY=0.0f, sumXX=0.0f;
    for(int i=0; i<n; i++){
        sumX+=i;
        sumY+=data[i];
        sumXY+= i*data[i];
        sumXX+= i*i;
    }
    float denom=n*sumXX - sumX*sumX;
    return (fabsf(denom) < 1e-6f) ? 0.0f : (n * sumXY - sumX * sumY) / denom;
}

static Baseline computeBaseline(const std::vector<RawRow>& rows) {
    Baseline b{};
    if(rows.empty()){ 
        return b;
    }

    std::vector<float> temp1v, temp2v, hum1v, hum2v, pres1v, pres2v;
    std::vector<float> gas1v[NUM_HEATER_STEPS], gas2v[NUM_HEATER_STEPS];

    for(const auto& r : rows){
        temp1v.push_back(r.temp1);
        hum1v.push_back(r.hum1);
        pres1v.push_back(r.pres1);
        temp2v.push_back(r.temp2);
        hum2v.push_back(r.hum2);
        pres2v.push_back(r.pres2);
        for(int i=0; i<NUM_HEATER_STEPS;i++){
            gas1v[i].push_back(r.gas1[i]);
            gas2v[i].push_back(r.gas2[i]);
        }
    }

    b.temp1 = medianVec(temp1v);
    b.temp2 = medianVec(temp2v);
    b.hum1 = medianVec(hum1v);
    b.hum2 = medianVec(hum2v);
    b.pres1 = medianVec(pres1v);
    b.pres2 = medianVec(pres2v);
    
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        b.gas1_steps[i] = medianVec(gas1v[i]);
        b.gas2_steps[i] = medianVec(gas2v[i]);
        if(b.gas1_steps[i] < 1.0f){
            b.gas1_steps[i] = 1.0f;
        }
        if(b.gas2_steps[i] < 1.0f){
            b.gas2_steps[i] = 1.0f;
        }

    }
    return b;
}

static Row toEnvironmentInvariantRow(const RawRow& raw, const Baseline& b) {
    Row out{};
    out.rawLabel = raw.rawLabel;
    memset(out.features, 0, sizeof(out.features));

    uint16_t idx=0;

    float gas1_resp[NUM_HEATER_STEPS];
    float gas2_resp[NUM_HEATER_STEPS];

    for(int i=0; i<NUM_HEATER_STEPS;i++){
        gas1_resp[i] = raw.gas1[i]/b.gas1_steps[i];
        gas2_resp[i] = raw.gas2[i]/b.gas2_steps[i];
        out.features[idx++] = gas1_resp[i];
    }
    for(int i=0; i<NUM_HEATER_STEPS;i++){
        out.features[idx++] = gas2_resp[i];
    }

    for(int i=0; i<NUM_HEATER_STEPS;i++){
        out.features[idx++] = (gas2_resp[i] > 0.01f)? gas1_resp[i] / gas2_resp[i] : 1.0f;
    }

    out.features[idx++] = localSlope(gas1_resp, NUM_HEATER_STEPS);
    out.features[idx++] = localSlope(gas2_resp, NUM_HEATER_STEPS);

    int half= NUM_HEATER_STEPS / 2;
    out.features[idx++] = localSlope(gas1_resp + half, NUM_HEATER_STEPS-half)-localSlope(gas1_resp, half);
    out.features[idx++] = localSlope(gas2_resp + half, NUM_HEATER_STEPS - half)- localSlope(gas2_resp, half);

    out.features[idx++] = fabsf(raw.temp1 - raw.temp2);
    out.features[idx++] = fabsf(raw.hum1  - raw.hum2);
    out.features[idx++] = fabsf(raw.pres1 - raw.pres2);

    for(int i = 0; i < FEATURE_COUNT; i++){
        if(FEATURE_STDS[i] > 1e-6f){
            out.features[i] = (out.features[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];
        }
    }
    return out;
}

static std::string toLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return out;
}

static std::string trim(const std::string& s) {
    size_t b = 0;
    while(b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while(e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) e--;
    return s.substr(b, e - b);
}

static std::string normaliseLabel(const std::string& raw) {
    std::string norm = toLower(trim(raw));
    size_t p = norm.find('(');
    if (p != std::string::npos) {
        norm = trim(norm.substr(0, p));
    }
    std::replace(norm.begin(), norm.end(), '_', ' ');
    norm.erase(std::unique(norm.begin(), norm.end(),
        [](char a, char b){
            return std::isspace(static_cast<unsigned char>(a)) &&std::isspace(static_cast<unsigned char>(b));}), norm.end());
    return norm;
}

static int labelIndex(const std::string& raw) {
    const std::string norm = normaliseLabel(raw);
    for (int i = 0; i < 12; ++i) {
        if (norm == CLASS_NAMES[i]) return i;
    }
    if(norm.find("camomile") != std::string::npos || norm.find("chamomile") != std::string::npos) return 0;
    if(norm.find("mint") != std::string::npos) return 1;
    if(norm.find("berry") != std::string::npos) return 2;
    if(norm.find("darjeeling") != std::string::npos) return 3;
    if(norm.find("nutmeg") != std::string::npos || norm.find("vanilla") != std::string::npos || norm.find("decaf") != std::string::npos) return 4;
    if(norm.find("earl") != std::string::npos) return 5;
    if(norm.find("breakfast") != std::string::npos) return 6;
    if(norm.find("orange") != std::string::npos) return 7;
    if(norm.find("lemon") != std::string::npos || norm.find("garden") != std::string::npos) return 8;
    if(norm.find("green") != std::string::npos) return 9;
    if(norm.find("raspberry") != std::string::npos) return 10;
    if(norm.find("cherry") != std::string::npos) return 11;
    return -1;
}

static const char* nameForClass(uint8_t cls) {
    if (cls<12) return CLASS_NAMES[cls];
    return "unknown";
}

static bool parseCsv(const std::string& path,std::vector<RawRow>& dataRows,std::vector<RawRow>& calibrationRows, int& detectedGasSteps) {
    std::ifstream f(path);
    if(!f.is_open()){
        std::cerr<< "Failed to open "<<path << "\n";
        return false;
    }

    std::string line;
    bool headerParsed = false;
    int labelCol = 1;
    int featureStartCol = 2;
    int numGasSteps = 1;

    while(std::getline(f, line)){
        if(line.empty()) continue;

        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')){
            fields.push_back(trim(cell));
        }

        if(!headerParsed) {
            bool hasAlpha = false;
            for(char c : line){
                if(std::isalpha(static_cast<unsigned char>(c))){
                    hasAlpha = true;
                    break;
                }
            }

            if(hasAlpha) {
                for(size_t i = 0; i < fields.size(); i++){
                    std::string col = toLower(fields[i]);
                    if(col == "label") labelCol = i;
                    if(col == "temp1") featureStartCol = i;
                }

                int gasCount = 0;
                for(size_t i = 0; i < fields.size(); i++){
                    std::string col = toLower(fields[i]);
                    if(col.find("gas1_") != std::string::npos) gasCount++;
                }
                numGasSteps =(gasCount>1)?gasCount:1;

                std::cout << "Header: label col=" << labelCol << ", features col=" << featureStartCol<< ", gas steps=" << numGasSteps << "\n";

                headerParsed = true;
                continue;
            }
            headerParsed = true;
            //don't continue
        }

        int colsPerSensor = 3 + numGasSteps;
        int requiredCols = featureStartCol + 2 * colsPerSensor;
        if((int)fields.size() < requiredCols) continue;

        RawRow row{};
        row.rawLabel =(labelCol<(int)fields.size())?fields[labelCol]:"";

        int col = featureStartCol;
        auto toF = [&](int c) -> float {
            return(c < (int)fields.size())? static_cast<float>(std::atof(fields[c].c_str())): 0.0f;
        };

        row.temp1 = toF(col++);
        row.hum1= toF(col++);
        row.pres1=toF(col++);

        for(int g = 0; g < numGasSteps; g++){
            row.gas1[g] = toF(col++);
        }
        for(int g = numGasSteps; g < NUM_HEATER_STEPS; g++){
            row.gas1[g] = row.gas1[0];
        }

        row.temp2 = toF(col++);
        row.hum2  = toF(col++);
        row.pres2 = toF(col++);
        for(int g = 0; g < numGasSteps; g++) {
            row.gas2[g] = toF(col++);
        }
        for(int g = numGasSteps; g < NUM_HEATER_STEPS; g++){
            row.gas2[g] = row.gas2[0];
        }

        const std::string normLabel = normaliseLabel(row.rawLabel);
        if(normLabel == "calibration" || normLabel == "baseline" || normLabel == "ambient"){
            calibrationRows.push_back(row);
        }
        else {
            dataRows.push_back(row);
        }
    }

    detectedGasSteps = numGasSteps;
    return !(dataRows.empty() && calibrationRows.empty());
}

int main(int argc, char** argv) {
    if(argc < 2){
        std::cerr <<"Usage: ml_infer_csv <csv_file>\n";
        return 1;
    }

    std::vector<RawRow> dataRows;
    std::vector<RawRow> calibrationRows;
    int detectedGasSteps = 1;

    if(!parseCsv(argv[1], dataRows, calibrationRows, detectedGasSteps)){
        std::cerr << "No data parsed from CSV.\n";
        return 1;
    }

    if(calibrationRows.empty()){
        std::cerr << "Error: CSV must contain an ambient/baseline block.\n";
        return 1;
    }

    Baseline base = computeBaseline(calibrationRows);

    std::cout << "Calibration from " << calibrationRows.size() << " rows:\n";
    for(int s = 0; s < NUM_HEATER_STEPS; s++) {
        std::cout << "  Step " << s << ": gas1=" << std::fixed << std::setprecision(0)<< base.gas1_steps[s] << ", gas2=" << base.gas2_steps[s] << "\n";
    }

    if(detectedGasSteps == 1) {
        std::cout << "\nWARNING: Old single-step CSV detected. "<< "All gas steps filled from step 0.\n"<< "Multi-step discrimination will be limited.\n\n";
    }

    //transform data rows
    std::vector<Row> rows;
    rows.reserve(dataRows.size());
    for (const auto& raw : dataRows) {
        rows.push_back(toEnvironmentInvariantRow(raw, base));
    }

    //print zscores
    if(!rows.empty()){
        float featSum[FEATURE_COUNT]{};
        for(const auto& r : rows){
            for(int i = 0; i < FEATURE_COUNT; i++){
                featSum[i] += r.features[i];
            }
        }
        float invN = 1.0f / static_cast<float>(rows.size());
        std::cout << "Mean z-scored features:\n";
        for(int i = 0; i < FEATURE_COUNT; i++){
            float mean = featSum[i] * invN;
            std::cout << "  [" << std::setw(2) << i << "] "<< std::setw(12) << shortFeatureName(i)<< " = " << std::fixed << std::setprecision(4) << mean;

            if(fabsf(mean) > 3.0f) std::cout << "  *** OOD WARNING ***";
            std::cout << "\n";
        }
    }

    std::cout<< "\nLoaded " << rows.size() <<" rows. Running DT/KNN/RF...\n";
    std::cout<<"idx,label,dt_cls,dt_conf,knn_cls,knn_conf,rf_cls,rf_conf\n";

    size_t totalLabeled = 0;
    size_t correctDt = 0, correctKnn = 0, correctRf = 0;

    //ensemble
    float ensembleScores[12]{};     //confidence-weighted scores
    int dtVotes[12]{};              //majority vote counts
    int knnVotes[12]{};
    int rfVotes[12]{};
    int groundTruthIdx = -1;

    for(size_t i = 0; i < rows.size(); ++i){
        const Row& r = rows[i];
        const int truthIdx = labelIndex(r.rawLabel);

        //store ground truth from first labeled row
        if(truthIdx >= 0 && groundTruthIdx < 0){
            groundTruthIdx = truthIdx;
        }

        float dtConf = 1.0f;
        uint8_t dtCls = dt_predict_with_confidence(r.features, &dtConf);

        float knnConf = 0.0f;
        uint8_t knnCls = knn_predict_with_confidence(r.features, &knnConf);

        float rfConf = 0.0f;
        uint8_t rfCls = rf_predict_with_confidence(r.features, &rfConf);


        if(dtCls < 12){
            dtVotes[dtCls]++;
            ensembleScores[dtCls] += dtConf;
        }
        if(knnCls < 12){
            knnVotes[knnCls]++;
            ensembleScores[knnCls] += knnConf;
        }
        if(rfCls < 12){
            rfVotes[rfCls]++;
            ensembleScores[rfCls] += rfConf;
        }

        //per-row output
        struct Pred {
            uint8_t cls; 
            float conf; 
            const char* source; 
        };

        std::vector<Pred> preds={
            {dtCls, dtConf, "DT"},
            {knnCls, knnConf, "KNN"},
            {rfCls, rfConf, "RF"}
        };

        std::sort(preds.begin(), preds.end(),
            [](const Pred& a, const Pred& b){ return a.conf > b.conf; });

        std::vector<Pred> uniq;
        std::vector<uint8_t> seen;
        for(const auto& p:preds){
            if(std::find(seen.begin(), seen.end(), p.cls) != seen.end()) continue;
            
            uniq.push_back(p);
            seen.push_back(p.cls);
            
            if(uniq.size() == 3) break;
        }

        std::ostringstream top3;
        top3 << "[";
        for (size_t k = 0; k < uniq.size(); ++k) {
            if (k > 0) top3 << ", ";
            top3 << uniq[k].source << ":(" << nameForClass(uniq[k].cls)<< "," << std::fixed << std::setprecision(3) << uniq[k].conf << ")";
        }
        top3 << "]";

        if (truthIdx >= 0) {
            totalLabeled++;
            if (dtCls == static_cast<uint8_t>(truthIdx)) correctDt++;
            if (knnCls == static_cast<uint8_t>(truthIdx)) correctKnn++;
            if (rfCls == static_cast<uint8_t>(truthIdx)) correctRf++;
        }

        std::cout << i << "," << r.rawLabel << ","<< nameForClass(dtCls) << "," << std::fixed << std::setprecision(3) << dtConf << ","
        << nameForClass(knnCls)<<"," << knnConf << ","<< nameForClass(rfCls) << "," << rfConf<< ",top3=" << top3.str() << "\n";
    }

    //pra
    if(totalLabeled > 0){
        auto pct = [](size_t good, size_t total){
            return 100.0 * static_cast<double>(good) / static_cast<double>(total);
        };

        std::cout << "\nPer-row accuracy over " << totalLabeled << " labeled rows:\n";
        std::cout << "  DT : " << std::fixed << std::setprecision(1) << pct(correctDt, totalLabeled) << "%\n";
        std::cout << "  KNN: " << pct(correctKnn, totalLabeled) << "%\n";
        std::cout << "  RF : " << pct(correctRf, totalLabeled) << "%\n";
    }

    //majority vote
    auto bestVote = [](int votes[12]) -> int {
        int best = 0;
        for(int i = 1; i < 12; i++) {
            if(votes[i] > votes[best]){
                best = i;
            }
        }
        return best;
    };

    int dtMaj = bestVote(dtVotes);
    int knnMaj = bestVote(knnVotes);
    int rfMaj = bestVote(rfVotes);

    //combined majority
    int combinedVotes[12]{};
    for(int i = 0; i < 12; i++){
        combinedVotes[i] = dtVotes[i] + knnVotes[i] + rfVotes[i];
    }
    int combinedMaj = bestVote(combinedVotes);

    std::cout << "\n=== Majority Vote (across " << rows.size() << " readings) ===\n";

    std::cout << "  DT  majority:  " << nameForClass(dtMaj)
            << " (" << dtVotes[dtMaj] << "/" << rows.size() << " votes)\n";
    std::cout << "  KNN majority:  " << nameForClass(knnMaj)
            << " (" << knnVotes[knnMaj] << "/" << rows.size() << " votes)\n";
    std::cout << "  RF  majority:  " << nameForClass(rfMaj)
            << " (" << rfVotes[rfMaj] << "/" << rows.size() << " votes)\n";
    std::cout << "  Combined:      " << nameForClass(combinedMaj)
            << " (" << combinedVotes[combinedMaj] << "/" << rows.size() * 3 << " votes)\n";

    if (groundTruthIdx >= 0) {
        std::cout << "  Ground truth:  " << nameForClass(groundTruthIdx) << "\n";
        std::cout << "  DT  majority " << (dtMaj == groundTruthIdx ? "CORRECT ✓" : "WRONG ✗") << "\n";
        std::cout << "  KNN majority " << (knnMaj == groundTruthIdx ? "CORRECT ✓" : "WRONG ✗") << "\n";
        std::cout << "  RF  majority " << (rfMaj == groundTruthIdx ? "CORRECT ✓" : "WRONG ✗") << "\n";
        std::cout << "  Combined     " << (combinedMaj == groundTruthIdx ? "CORRECT ✓" : "WRONG ✗") << "\n";
    }

    //confidence ensemble
    int bestEnsemble = 0;
    for (int i = 1; i < 12; i++) {
        if (ensembleScores[i] > ensembleScores[bestEnsemble]) bestEnsemble = i;
    }

    //compute total
    float totalScore = 0;
    for (int i = 0; i < 12; i++) totalScore += ensembleScores[i];

    std::cout << "\n=== Confidence-Weighted Ensemble ===\n";
    std::cout << "Final prediction: " << nameForClass(bestEnsemble) << "\n";

    if (groundTruthIdx >= 0) {
        std::cout << "Ground truth:" << nameForClass(groundTruthIdx) << "\n";
        std::cout << "Result: "<< (bestEnsemble == groundTruthIdx ? "CORRECT ✓" : "WRONG ✗") << "\n";
    }

    //show top 5 candidates
    std::vector<std::pair<float, int>> ranked;
    for (int i = 0; i < 12; i++) {
        if (ensembleScores[i] > 0) ranked.push_back({ensembleScores[i], i});
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<float,int>>());

    std::cout << "\n  Top candidates:\n";
    for (size_t k = 0; k < std::min(ranked.size(), size_t(5)); k++) {
        float pct = (totalScore > 0) ? 100.0f * ranked[k].first / totalScore : 0;
        std::cout << "    " << (k + 1) << ". " << std::setw(30) << std::left<< nameForClass(ranked[k].second) << std::right
                  << "  score=" << std::fixed << std::setprecision(3) << ranked[k].first<<"  (" << std::setprecision(1) << pct << "%)\n";
    }

    //
    std::cout << "\n  Per-model contribution to top prediction ("<< nameForClass(bestEnsemble) << "):\n";
    
    float dtContrib = 0, knnContrib = 0, rfContrib = 0;
    for(size_t i = 0; i < rows.size(); ++i) {
        float dtConf; uint8_t dtC = dt_predict_with_confidence(rows[i].features, &dtConf);
        float knnConf; uint8_t knnC = knn_predict_with_confidence(rows[i].features, &knnConf);
        float rfConf; uint8_t rfC = rf_predict_with_confidence(rows[i].features, &rfConf);
        if (dtC == bestEnsemble)  dtContrib += dtConf;
        if (knnC == bestEnsemble) knnContrib += knnConf;
        if (rfC == bestEnsemble)  rfContrib += rfConf;
    }
    std::cout << "DT: " << std::fixed << std::setprecision(3) << dtContrib << "\n";
    std::cout << "KNN: " << knnContrib << "\n";
    std::cout << "RF: " << rfContrib << "\n";


    //per step variance
    if(detectedGasSteps>1){
        std::cout<<"\n===Per-step Gas Response Variance===\n";
        for(int i=0; i<NUM_HEATER_STEPS;i++){
            float s1=0.0f,sq1=0.0f;
            float s2=0.0f,sq2=0.0f;

            for(const auto &r : rows){
                s1 += r.features[i];
                sq1 += r.features[i]*r.features[i];
                s2 += r.features[10+i];
                sq2 += r.features[i+10]*r.features[i+10];
            }
            float n=static_cast<float>(rows.size());
            float m1=s1/n, v1=(sq1/n) - (m1*m1);
            float m2=s2/n, v2=(sq2/n) - (m2*m2);
            std::cout << "  Step " << i
                    << "  gas1: mean_z=" << std::fixed << std::setprecision(3) << m1
                    << " var=" << std::setprecision(4) << v1
                    << "  gas2: mean_z=" << m2
                    << " var=" << v2 << "\n";
        }
    }

    return 0;
}