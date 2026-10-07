// Auto-generated C++ header file
// Model Type: Decision Tree
// Generated on: 2026-09-23 00:48:49
// Feature count: 80 (selected from 157)
// Normalisation: robust (median/IQR)
// Classes: 5 (decaf_tea, tea, decaf_coffee, coffee, ambient)
// Info: Nodes: 151

#ifndef DT_MODEL_DATA_H
#define DT_MODEL_DATA_H

#include <stdint.h>
#include <string.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #ifndef PROGMEM
    #define PROGMEM
  #endif
  #ifndef pgm_read_byte
    #define pgm_read_byte(addr) (*(const uint8_t *)(addr))
  #endif
  #ifndef pgm_read_word
    #define pgm_read_word(addr) (*(const uint16_t *)(addr))
  #endif
  #ifndef pgm_read_dword
    #define pgm_read_dword(addr) (*(const uint32_t *)(addr))
  #endif
  #ifndef pgm_read_float
    #define pgm_read_float(addr) (*(const float *)(addr))
  #endif
  #ifndef memcpy_P
    #define memcpy_P(dest, src, n) memcpy((dest), (src), (n))
  #endif
#endif

#if defined(SELECTED_FEATURE_COUNT) && SELECTED_FEATURE_COUNT < 80
  #error "DT model uses more features than SELECTED_FEATURE_COUNT"
#endif

// Model Parameters
#define DT_FEATURE_COUNT 80
#define DT_TOTAL_NODES 151
#define DT_NUM_CLASSES 5
#define DT_MAX_DEPTH 13
#define DT_HAS_CONFIDENCE 1

// Class Names
static const char* DT_CLASS_NAMES[5]={
    "decaf_tea",
    "tea",
    "decaf_coffee",
    "coffee",
    "ambient"
};

// Node Structure
typedef struct {
    int16_t featureIndex;
    uint8_t label;
    float threshold;
    uint16_t majorityCount;
    uint16_t totalSamples;
    int32_t leftChild;
    int32_t rightChild;
} dt_node_t;

static const dt_node_t DT_NODES[151] PROGMEM={
    {53, 255, 0.979152083f, 121, 583, 1, 2},  // g1_early_late_ratio < 0.9792?
    {31, 255, 0.260412753f, 121, 483, 3, 4},  // fisher_dcoffee_vs_coffee < 0.2604?
    {-1, 4, 0.0f, 100, 100, -1, -1},  // LEAF: ambient (100/100)
    {6, 255, -0.382617295f, 101, 319, 5, 6},  // interact_g2n2*cr1 < -0.3826?
    {10, 255, 0.0992792845f, 79, 164, 7, 8},  // diff_step0 < 0.0993?
    {45, 255, -0.518358946f, 53, 138, 9, 10},  // g2n2_div_temp < -0.5184?
    {7, 255, -0.74938339f, 81, 181, 11, 12},  // cr0_div_hum < -0.7494?
    {1, 255, -0.928945065f, 35, 111, 13, 14},  // abs_delta_pres < -0.9289?
    {45, 255, 0.75951612f, 48, 53, 15, 16},  // g2n2_div_temp < 0.7595?
    {63, 255, 0.874016821f, 20, 52, 17, 18},  // gas2_norm_step1 < 0.8740?
    {30, 255, -0.0768277645f, 43, 86, 19, 20},  // fisher_dtea_vs_tea < -0.0768?
    {46, 255, 0.594403148f, 13, 24, 21, 22},  // gas1_norm_step1 < 0.5944?
    {45, 255, -1.29576671f, 79, 157, 23, 24},  // g2n2_div_temp < -1.2958?
    {-1, 2, 0.0f, 19, 19, -1, -1},  // LEAF: decaf_coffee (19/19)
    {45, 255, 3.1710043f, 35, 92, 25, 26},  // g2n2_div_temp < 3.1710?
    {33, 255, -0.0678253546f, 45, 46, 27, 28},  // g1g2_ratio_s2 < -0.0678?
    {32, 255, -0.884489f, 3, 7, 29, 30},  // cross_ratio_slope < -0.8845?
    {1, 255, 0.856582403f, 19, 39, 31, 32},  // abs_delta_pres < 0.8566?
    {0, 255, 0.0681076273f, 12, 13, 33, 34},  // interact_g2n3*crMean < 0.0681?
    {45, 255, 1.76877379f, 17, 24, 35, 36},  // g2n2_div_temp < 1.7688?
    {1, 255, 0.642109871f, 41, 62, 37, 38},  // abs_delta_pres < 0.6421?
    {19, 255, 1.32601917f, 5, 7, 39, 40},  // gas2_norm_step4 < 1.3260?
    {31, 255, -1.05775058f, 13, 17, 41, 42},  // fisher_dcoffee_vs_coffee < -1.0578?
    {-1, 3, 0.0f, 19, 19, -1, -1},  // LEAF: coffee (19/19)
    {1, 255, 0.285527468f, 60, 138, 43, 44},  // abs_delta_pres < 0.2855?
    {68, 255, -0.35581547f, 35, 79, 45, 46},  // range2 < -0.3558?
    {40, 255, -0.237707168f, 12, 13, 47, 48},  // g1g2_ratio_s7 < -0.2377?
    {-1, 2, 0.0f, 43, 43, -1, -1},  // LEAF: decaf_coffee (43/43)
    {-1, 2, 0.0f, 2, 3, -1, -1},  // LEAF: decaf_coffee (2/3)
    {-1, 2, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_coffee (3/3)
    {-1, 1, 0.0f, 2, 4, -1, -1},  // LEAF: tea (2/4)
    {6, 255, -0.437720597f, 19, 28, 49, 50},  // interact_g2n2*cr1 < -0.4377?
    {46, 255, 0.314183772f, 9, 11, 51, 52},  // gas1_norm_step1 < 0.3142?
    {-1, 1, 0.0f, 12, 12, -1, -1},  // LEAF: tea (12/12)
    {-1, 3, 0.0f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {46, 255, -0.289779425f, 17, 20, 53, 54},  // gas1_norm_step1 < -0.2898?
    {-1, 4, 0.0f, 3, 4, -1, -1},  // LEAF: ambient (3/4)
    {60, 255, -0.0884855837f, 14, 21, 55, 56},  // cr0_div_temp < -0.0885?
    {39, 255, 0.0399202518f, 34, 41, 57, 58},  // g2_early_late_ratio < 0.0399?
    {-1, 1, 0.0f, 5, 5, -1, -1},  // LEAF: tea (5/5)
    {-1, 3, 0.0f, 2, 2, -1, -1},  // LEAF: coffee (2/2)
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {4, 255, -1.10176516f, 13, 14, 59, 60},  // g2n2_div_hum < -1.1018?
    {7, 255, -0.43606928f, 31, 78, 61, 62},  // cr0_div_hum < -0.4361?
    {60, 255, -0.252735555f, 39, 60, 63, 64},  // cr0_div_temp < -0.2527?
    {78, 255, -0.398437381f, 28, 36, 65, 66},  // gas2_delta_step5 < -0.3984?
    {8, 255, -0.0634575337f, 18, 43, 67, 68},  // gas2_norm_step2 < -0.0635?
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 4, 0.0f, 12, 12, -1, -1},  // LEAF: ambient (12/12)
    {65, 255, -1.04023004f, 19, 25, 69, 70},  // cross_ratio_step7 < -1.0402?
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {-1, 0, 0.0f, 1, 2, -1, -1},  // LEAF: decaf_tea (1/2)
    {-1, 2, 0.0f, 9, 9, -1, -1},  // LEAF: decaf_coffee (9/9)
    {-1, 2, 0.0f, 2, 2, -1, -1},  // LEAF: decaf_coffee (2/2)
    {7, 255, -0.497456819f, 17, 18, 71, 72},  // cr0_div_hum < -0.4975?
    {39, 255, -0.776048064f, 14, 16, 73, 74},  // g2_early_late_ratio < -0.7760?
    {-1, 0, 0.0f, 5, 5, -1, -1},  // LEAF: decaf_tea (5/5)
    {24, 255, 1.22492194f, 34, 38, 75, 76},  // slope2_div_hum < 1.2249?
    {-1, 1, 0.0f, 3, 3, -1, -1},  // LEAF: tea (3/3)
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 2, 0.0f, 13, 13, -1, -1},  // LEAF: decaf_coffee (13/13)
    {42, 255, 0.340790063f, 15, 35, 77, 78},  // g1g2_ratio_s3 < 0.3408?
    {8, 255, 0.581519425f, 23, 43, 79, 80},  // gas2_norm_step2 < 0.5815?
    {48, 255, 0.727674246f, 11, 28, 81, 82},  // gas2_norm_step8 < 0.7277?
    {23, 255, -0.393044472f, 28, 32, 83, 84},  // diff_step5 < -0.3930?
    {76, 255, -0.564202964f, 27, 30, 85, 86},  // diff_step9 < -0.5642?
    {1, 255, -0.143417612f, 3, 6, 87, 88},  // abs_delta_pres < -0.1434?
    {30, 255, -0.543502688f, 16, 18, 89, 90},  // fisher_dtea_vs_tea < -0.5435?
    {1, 255, 0.214036614f, 9, 25, 91, 92},  // abs_delta_pres < 0.2140?
    {-1, 2, 0.0f, 2, 3, -1, -1},  // LEAF: decaf_coffee (2/3)
    {24, 255, 0.594847083f, 19, 22, 93, 94},  // slope2_div_hum < 0.5948?
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 1, 0.0f, 17, 17, -1, -1},  // LEAF: tea (17/17)
    {-1, 0, 0.0f, 2, 3, -1, -1},  // LEAF: decaf_tea (2/3)
    {-1, 1, 0.0f, 13, 13, -1, -1},  // LEAF: tea (13/13)
    {32, 255, -0.410264492f, 34, 36, 95, 96},  // cross_ratio_slope < -0.4103?
    {-1, 1, 0.0f, 2, 2, -1, -1},  // LEAF: tea (2/2)
    {45, 255, -0.618492246f, 11, 26, 97, 98},  // g2n2_div_temp < -0.6185?
    {-1, 3, 0.0f, 9, 9, -1, -1},  // LEAF: coffee (9/9)
    {76, 255, -0.158443734f, 20, 27, 99, 100},  // diff_step9 < -0.1584?
    {10, 255, -0.132465228f, 10, 16, 101, 102},  // diff_step0 < -0.1325?
    {11, 255, -0.630742908f, 11, 22, 103, 104},  // crMean_div_hum < -0.6307?
    {3, 255, 0.270614862f, 3, 6, 105, 106},  // diff_step1 < 0.2706?
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {6, 255, 0.931825757f, 28, 29, 107, 108},  // interact_g2n2*cr1 < 0.9318?
    {4, 255, 0.382151842f, 3, 6, 109, 110},  // g2n2_div_hum < 0.3822?
    {-1, 1, 0.0f, 24, 24, -1, -1},  // LEAF: tea (24/24)
    {-1, 2, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_coffee (3/3)
    {-1, 3, 0.0f, 2, 3, -1, -1},  // LEAF: coffee (2/3)
    {-1, 1, 0.0f, 1, 2, -1, -1},  // LEAF: tea (1/2)
    {-1, 0, 0.0f, 16, 16, -1, -1},  // LEAF: decaf_tea (16/16)
    {14, 255, -0.40456149f, 7, 15, 111, 112},  // fisher_dtea_vs_dcoffee < -0.4046?
    {18, 255, -0.0887020975f, 8, 10, 113, 114},  // cross_ratio_step2 < -0.0887?
    {45, 255, -1.29149103f, 19, 20, 115, 116},  // g2n2_div_temp < -1.2915?
    {-1, 0, 0.0f, 1, 2, -1, -1},  // LEAF: decaf_tea (1/2)
    {-1, 1, 0.0f, 1, 1, -1, -1},  // LEAF: tea (1/1)
    {69, 255, -0.375f, 34, 35, 117, 118},  // peak_idx1 < -0.3750?
    {5, 255, 0.0994103402f, 5, 6, 119, 120},  // interact_g1n2*diff1 < 0.0994?
    {76, 255, 0.352406263f, 11, 20, 121, 122},  // diff_step9 < 0.3524?
    {34, 255, -0.174015343f, 4, 10, 123, 124},  // interact_g2d2*cr2 < -0.1740?
    {4, 255, 0.204496935f, 16, 17, 125, 126},  // g2n2_div_hum < 0.2045?
    {-1, 3, 0.0f, 3, 3, -1, -1},  // LEAF: coffee (3/3)
    {45, 255, -0.841750622f, 10, 13, 127, 128},  // g2n2_div_temp < -0.8418?
    {0, 255, 0.548579335f, 6, 7, 129, 130},  // interact_g2n3*crMean < 0.5486?
    {67, 255, -0.205607057f, 10, 15, 131, 132},  // gas1_delta_step6 < -0.2056?
    {-1, 2, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_coffee (3/3)
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {-1, 3, 0.0f, 28, 28, -1, -1},  // LEAF: coffee (28/28)
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 3, 0.0f, 2, 3, -1, -1},  // LEAF: coffee (2/3)
    {-1, 1, 0.0f, 3, 3, -1, -1},  // LEAF: tea (3/3)
    {-1, 1, 0.0f, 5, 5, -1, -1},  // LEAF: tea (5/5)
    {7, 255, 0.530648947f, 7, 10, 133, 134},  // cr0_div_hum < 0.5306?
    {-1, 0, 0.0f, 1, 2, -1, -1},  // LEAF: decaf_tea (1/2)
    {-1, 3, 0.0f, 8, 8, -1, -1},  // LEAF: coffee (8/8)
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 3, 0.0f, 19, 19, -1, -1},  // LEAF: coffee (19/19)
    {-1, 1, 0.0f, 1, 1, -1, -1},  // LEAF: tea (1/1)
    {-1, 0, 0.0f, 34, 34, -1, -1},  // LEAF: decaf_tea (34/34)
    {-1, 3, 0.0f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {-1, 0, 0.0f, 5, 5, -1, -1},  // LEAF: decaf_tea (5/5)
    {32, 255, -0.523477197f, 10, 15, 135, 136},  // cross_ratio_slope < -0.5235?
    {0, 255, 0.95193994f, 4, 5, 137, 138},  // interact_g2n3*crMean < 0.9519?
    {-1, 0, 0.0f, 4, 4, -1, -1},  // LEAF: decaf_tea (4/4)
    {1, 255, -0.0719267577f, 2, 6, 139, 140},  // abs_delta_pres < -0.0719?
    {-1, 0, 0.0f, 16, 16, -1, -1},  // LEAF: decaf_tea (16/16)
    {-1, 3, 0.0f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {-1, 2, 0.0f, 10, 10, -1, -1},  // LEAF: decaf_coffee (10/10)
    {-1, 1, 0.0f, 6, 6, -1, -1},  // LEAF: tea (6/6)
    {-1, 3, 0.0f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {-1, 0, 0.0f, 2, 2, -1, -1},  // LEAF: decaf_tea (2/2)
    {24, 255, -0.211568117f, 10, 13, 141, 142},  // slope2_div_hum < -0.2116?
    {24, 255, -0.26024124f, 7, 8, 143, 144},  // slope2_div_hum < -0.2602?
    {-1, 1, 0.0f, 1, 2, -1, -1},  // LEAF: tea (1/2)
    {0, 255, 0.521774232f, 3, 7, 145, 146},  // interact_g2n3*crMean < 0.5218?
    {-1, 1, 0.0f, 8, 8, -1, -1},  // LEAF: tea (8/8)
    {-1, 3, 0.0f, 4, 4, -1, -1},  // LEAF: coffee (4/4)
    {-1, 1, 0.0f, 1, 1, -1, -1},  // LEAF: tea (1/1)
    {-1, 2, 0.0f, 2, 4, -1, -1},  // LEAF: decaf_coffee (2/4)
    {-1, 1, 0.0f, 2, 2, -1, -1},  // LEAF: tea (2/2)
    {-1, 1, 0.0f, 2, 3, -1, -1},  // LEAF: tea (2/3)
    {7, 255, -0.605560303f, 9, 10, 147, 148},  // cr0_div_hum < -0.6056?
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 2, 0.0f, 7, 7, -1, -1},  // LEAF: decaf_coffee (7/7)
    {32, 255, -0.598347187f, 3, 5, 149, 150},  // cross_ratio_slope < -0.5983?
    {-1, 1, 0.0f, 2, 2, -1, -1},  // LEAF: tea (2/2)
    {-1, 0, 0.0f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 3, 0.0f, 9, 9, -1, -1},  // LEAF: coffee (9/9)
    {-1, 0, 0.0f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {-1, 2, 0.0f, 1, 2, -1, -1}  // LEAF: decaf_coffee (1/2)
};

// ===========================================================================
// DT Inference Functions
// ===========================================================================

#define DT_READ_NODE(idx) ({ \
    dt_node_t _n; \
    memcpy_P(&_n, &DT_NODES[idx], sizeof(dt_node_t)); \
    _n; })

static inline uint8_t dt_predict(const float* features) {
    uint16_t nodeIdx=0;
    dt_node_t node=DT_READ_NODE(nodeIdx);
    
    while (node.featureIndex >= 0) {
        if (features[node.featureIndex] < node.threshold) {
            nodeIdx=node.leftChild;
        } else {
            nodeIdx=node.rightChild;
        }
        node=DT_READ_NODE(nodeIdx);
    }
    return node.label;
}

static inline uint8_t dt_predict_with_confidence(const float* features, float* confidence_out) {
    uint16_t nodeIdx=0;
    dt_node_t node=DT_READ_NODE(nodeIdx);
    
    while (node.featureIndex >= 0) {
        if (features[node.featureIndex] < node.threshold) {
            nodeIdx=node.leftChild;
        } else {
            nodeIdx=node.rightChild;
        }
        node=DT_READ_NODE(nodeIdx);
    }
    
    if (confidence_out) {
        const float total=(float)node.totalSamples;
        const float majority=(float)node.majorityCount;
        *confidence_out=(total > 0.0f) ? (majority / total) : 1.0f;
    }
    return node.label;
}

static inline const char* dt_get_class_name(uint8_t idx) {
    return (idx < 5) ? DT_CLASS_NAMES[idx] : "unknown";
}

#endif // DT_MODEL_DATA_H
