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

static const int FEATURE_COUNT = 8;

struct RawRow {
    std::string rawLabel;
    float sensors[8]; //temp1,hum1,pres1,gas1_0,temp2,hum2,pres2,gas2_0
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

static const char* FEATURE_NAMES[FEATURE_COUNT] = {
    "gas1_resp", "gas2_resp", "gas_cross", "gas_diff","d_temp", "d_hum", "d_pres", "log_gas_cross"
};

struct Baseline {
    float temp1, temp2;
    float hum1, hum2;
    float pres1, pres2;
    float gas1, gas2;
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

static Baseline computeBaseline(const std::vector<RawRow>& rows) {
    std::vector<float> temp1v, temp2v, hum1v, hum2v, pres1v, pres2v, gas1v, gas2v;
    for(const auto& r : rows){
        temp1v.push_back(r.sensors[0]);
        hum1v.push_back(r.sensors[1]);
        pres1v.push_back(r.sensors[2]);
        gas1v.push_back(r.sensors[3]);
        temp2v.push_back(r.sensors[4]);
        hum2v.push_back(r.sensors[5]);
        pres2v.push_back(r.sensors[6]);
        gas2v.push_back(r.sensors[7]);
    }
    Baseline b{};
    b.temp1 = medianVec(temp1v);
    b.temp2 = medianVec(temp2v);
    b.hum1 = medianVec(hum1v);
    b.hum2 = medianVec(hum2v);
    b.pres1 = medianVec(pres1v);
    b.pres2 = medianVec(pres2v);
    b.gas1 = medianVec(gas1v);
    b.gas2 = medianVec(gas2v);
    if(b.gas1 == 0.0f) b.gas1 = 1.0f;
    if(b.gas2 == 0.0f) b.gas2 = 1.0f;
    return b;
}

static Row toEnvironmentInvariantRow(const RawRow& raw, const Baseline& b) {
    Row out{};
    out.rawLabel = raw.rawLabel;

    float gas1_raw = raw.sensors[3];
    float gas2_raw = raw.sensors[7];

    float gas1_response = (b.gas1 > 0) ? gas1_raw / b.gas1 : 1.0f;
    float gas2_response = (b.gas2 > 0) ? gas2_raw / b.gas2 : 1.0f;
    float gas_cross_ratio = (gas2_raw > 0) ? gas1_raw / gas2_raw : 1.0f;

    out.features[0] = gas1_response;
    out.features[1] = gas2_response;
    out.features[2] = gas_cross_ratio;
    out.features[3] = fabsf(gas1_response - gas2_response);
    out.features[4] = fabsf(raw.sensors[0] - raw.sensors[4]);
    out.features[5] = fabsf(raw.sensors[1] - raw.sensors[5]);
    out.features[6] = fabsf(raw.sensors[2] - raw.sensors[6]);
    out.features[7] = fabsf(logf(gas_cross_ratio > 0 ? gas_cross_ratio : 1e-6f));

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

static bool parseCsv(const std::string& path,std::vector<RawRow>& dataRows,std::vector<RawRow>& calibrationRows) {
    std::ifstream f(path);
    if(!f.is_open()){
        std::cerr<< "Failed to open "<<path << "\n";
        return false;
    }

    std::string line;
    if(!std::getline(f,line)) return false;

    bool hasAlpha = false;
    for(char c : line){
        if (std::isalpha(static_cast<unsigned char>(c))) { 
            hasAlpha = true; 
            break; 
        }
    }

    if(!hasAlpha){
        f.clear();
        f.seekg(0);
    }

    while(std::getline(f, line)){
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            fields.push_back(cell);
        }
        if (fields.size() < 10) continue;

        RawRow row{};
        row.rawLabel = (fields.size() > 1) ? fields[1] : "";
        auto toFloat = [](const std::string& s) {
            return static_cast<float>(std::atof(s.c_str()));
        };
        row.sensors[0] = toFloat(fields[2]);  // temp1
        row.sensors[1] = toFloat(fields[3]);  // hum1
        row.sensors[2] = toFloat(fields[4]);  // pres1
        row.sensors[3] = toFloat(fields[5]);  // gas1_0
        row.sensors[4] = toFloat(fields[6]);  // temp2
        row.sensors[5] = toFloat(fields[7]);  // hum2
        row.sensors[6] = toFloat(fields[8]);  // pres2
        row.sensors[7] = toFloat(fields[9]);  // gas2_0

        const std::string normLabel = normaliseLabel(row.rawLabel);
        if(normLabel == "calibration" || normLabel == "baseline" || normLabel == "ambient"){
            calibrationRows.push_back(row);
        }
        else{
            dataRows.push_back(row);
        }
    }
    return !(dataRows.empty() && calibrationRows.empty());
}

int main(int argc, char** argv) {
    if(argc < 2){
        std::cerr <<"Usage: ml_infer_csv <csv_file>\n";
        return 1;
    }

    std::vector<RawRow> dataRows;
    std::vector<RawRow> calibrationRows;
    if(!parseCsv(argv[1], dataRows, calibrationRows)){
        std::cerr << "No data parsed from CSV.\n";
        return 1;
    }

    if(calibrationRows.empty()){
        std::cerr << "Error: CSV must contain an ambient/baseline block.\n";
        return 1;
    }

    Baseline base = computeBaseline(calibrationRows);

    std::cout << "Calibration medians -> gas1: " << base.gas1<< ", gas2: " << base.gas2<< " (from " << calibrationRows.size() << " calibration rows)\n";

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
            std::cout << "  [" << i << "] " << std::setw(15) << FEATURE_NAMES[i]
                      << " = " << std::fixed << std::setprecision(4) << mean;
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

    return 0;
}