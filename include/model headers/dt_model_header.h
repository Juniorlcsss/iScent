// Auto-generated C++ header file
// Model Type: Decision Tree
// Generated on: 2026-03-24 19:15:10
// Feature count: 50 (selected from 157)
// Normalization: robust (median/IQR)
// Classes: 5 (decaf_tea, tea, decaf_coffee, coffee, ambient)
// Info: Nodes: 111

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

// Compile-time feature count validation
#if defined(SELECTED_FEATURE_COUNT) && SELECTED_FEATURE_COUNT != 50
  #error "DT model was trained with 50 features but SELECTED_FEATURE_COUNT is different"
#endif

// Model Parameters
#define DT_FEATURE_COUNT 50
#define DT_TOTAL_NODES 111
#define DT_NUM_CLASSES 5
#define DT_MAX_DEPTH 0
#define DT_HAS_CONFIDENCE 1

// Class Names
static const char* DT_CLASS_NAMES[5] = {
    "decaf_tea",
    "tea",
    "decaf_coffee",
    "coffee",
    "ambient"
};

// Node Structure
typedef struct {
    int8_t featureIndex;
    uint8_t label;
    float threshold;
    uint16_t majorityCount;
    uint16_t totalSamples;
    int32_t leftChild;
    int32_t rightChild;
} dt_node_t;

static const dt_node_t DT_NODES[111] PROGMEM = {
    {36, 255, 1.042571f, 121, 576, 1, 2},  // g1_early_late_ratio < 1.0426?
    {49, 255, -0.370642f, 121, 486, 3, 4},  // fisher_dtea_vs_dcoffee < -0.3706?
    {-1, 4, 0.000000f, 90, 90, -1, -1},  // LEAF: ambient (90/90)
    {48, 255, 0.215978f, 73, 161, 5, 6},  // fisher_dcoffee_vs_coffee < 0.2160?
    {45, 255, 3.185264f, 107, 325, 7, 8},  // g2n2_div_temp < 3.1853?
    {45, 255, -1.495916f, 42, 79, 9, 10},  // g2n2_div_temp < -1.4959?
    {24, 255, 0.665142f, 61, 82, 11, 12},  // env_gas1_baseline < 0.6651?
    {24, 255, -1.166158f, 107, 305, 13, 14},  // env_gas1_baseline < -1.1662?
    {4, 255, -0.220653f, 18, 20, 15, 16},  // gas2_norm_step3 < -0.2207?
    {-1, 3, 0.000000f, 16, 16, -1, -1},  // LEAF: coffee (16/16)
    {0, 255, 0.621554f, 26, 63, 17, 18},  // gas1_norm_step1 < 0.6216?
    {46, 255, 0.341207f, 58, 67, 19, 20},  // fisher_coffee_vs_dtea < 0.3412?
    {8, 255, 0.031659f, 12, 15, 21, 22},  // cross_ratio_step0 < 0.0317?
    {9, 255, -0.482227f, 15, 16, 23, 24},  // cross_ratio_step1 < -0.4822?
    {45, 255, 1.759067f, 107, 289, 25, 26},  // g2n2_div_temp < 1.7591?
    {25, 255, -0.163339f, 18, 19, 27, 28},  // g1g2_ratio_s7 < -0.1633?
    {-1, 0, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {24, 255, 0.166159f, 26, 54, 29, 30},  // env_gas1_baseline < 0.1662?
    {-1, 2, 0.000000f, 8, 9, -1, -1},  // LEAF: decaf_coffee (8/9)
    {11, 255, 0.047643f, 5, 11, 31, 32},  // cross_ratio_step3 < 0.0476?
    {25, 255, -0.201776f, 53, 56, 33, 34},  // g1g2_ratio_s7 < -0.2018?
    {-1, 2, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf_coffee (2/2)
    {0, 255, -0.507491f, 12, 13, 35, 36},  // gas1_norm_step1 < -0.5075?
    {-1, 3, 0.000000f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {-1, 2, 0.000000f, 15, 15, -1, -1},  // LEAF: decaf_coffee (15/15)
    {42, 255, -0.534825f, 105, 267, 37, 38},  // diff1_div_hum < -0.5348?
    {0, 255, -0.499703f, 20, 22, 39, 40},  // gas1_norm_step1 < -0.4997?
    {-1, 0, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 4, 0.000000f, 18, 18, -1, -1},  // LEAF: ambient (18/18)
    {41, 255, -0.546967f, 15, 31, 41, 42},  // cr0_div_hum < -0.5470?
    {39, 255, -0.013680f, 18, 23, 43, 44},  // g2n2_div_hum < -0.0137?
    {-1, 2, 0.000000f, 4, 4, -1, -1},  // LEAF: decaf_coffee (4/4)
    {-1, 3, 0.000000f, 5, 7, -1, -1},  // LEAF: coffee (5/7)
    {-1, 2, 0.000000f, 50, 50, -1, -1},  // LEAF: decaf_coffee (50/50)
    {-1, 2, 0.000000f, 3, 6, -1, -1},  // LEAF: decaf_coffee (3/6)
    {-1, 2, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf_coffee (1/1)
    {-1, 3, 0.000000f, 12, 12, -1, -1},  // LEAF: coffee (12/12)
    {45, 255, -0.191188f, 55, 89, 45, 46},  // g2n2_div_temp < -0.1912?
    {39, 255, -0.418852f, 54, 178, 47, 48},  // g2n2_div_hum < -0.4189?
    {-1, 1, 0.000000f, 2, 2, -1, -1},  // LEAF: tea (2/2)
    {-1, 0, 0.000000f, 20, 20, -1, -1},  // LEAF: decaf_tea (20/20)
    {27, 255, 0.138792f, 6, 12, 49, 50},  // interact_g2n2×cr1 < 0.1388?
    {1, 255, 0.006461f, 15, 19, 51, 52},  // gas1_norm_step2 < 0.0065?
    {45, 255, -0.738500f, 18, 21, 53, 54},  // g2n2_div_temp < -0.7385?
    {-1, 2, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf_coffee (2/2)
    {14, 255, -0.823575f, 8, 23, 55, 56},  // diff_step1 < -0.8236?
    {45, 255, 1.466334f, 51, 66, 57, 58},  // g2n2_div_temp < 1.4663?
    {24, 255, 1.165142f, 44, 111, 59, 60},  // env_gas1_baseline < 1.1651?
    {7, 255, -0.187808f, 26, 67, 61, 62},  // gas2_norm_step8 < -0.1878?
    {-1, 3, 0.000000f, 6, 6, -1, -1},  // LEAF: coffee (6/6)
    {-1, 1, 0.000000f, 5, 6, -1, -1},  // LEAF: tea (5/6)
    {-1, 3, 0.000000f, 2, 4, -1, -1},  // LEAF: coffee (2/4)
    {-1, 0, 0.000000f, 15, 15, -1, -1},  // LEAF: decaf_tea (15/15)
    {-1, 0, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf_tea (2/2)
    {0, 255, -0.029332f, 18, 19, 63, 64},  // gas1_norm_step1 < -0.0293?
    {27, 255, -0.983442f, 7, 14, 65, 66},  // interact_g2n2×cr1 < -0.9834?
    {-1, 0, 0.000000f, 7, 9, -1, -1},  // LEAF: decaf_tea (7/9)
    {24, 255, -0.583333f, 50, 60, 67, 68},  // env_gas1_baseline < -0.5833?
    {-1, 0, 0.000000f, 5, 6, -1, -1},  // LEAF: decaf_tea (5/6)
    {45, 255, -0.740751f, 42, 84, 69, 70},  // g2n2_div_temp < -0.7408?
    {45, 255, -0.405564f, 15, 27, 71, 72},  // g2n2_div_temp < -0.4056?
    {45, 255, 0.588286f, 12, 22, 73, 74},  // g2n2_div_temp < 0.5883?
    {29, 255, 0.129325f, 24, 45, 75, 76},  // interact_g1n2×diff1 < 0.1293?
    {-1, 0, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf_tea (1/1)
    {-1, 3, 0.000000f, 18, 18, -1, -1},  // LEAF: coffee (18/18)
    {-1, 1, 0.000000f, 4, 7, -1, -1},  // LEAF: tea (4/7)
    {-1, 3, 0.000000f, 7, 7, -1, -1},  // LEAF: coffee (7/7)
    {-1, 2, 0.000000f, 4, 8, -1, -1},  // LEAF: decaf_coffee (4/8)
    {27, 255, -0.991185f, 47, 52, 77, 78},  // interact_g2n2×cr1 < -0.9912?
    {32, 255, 0.057886f, 17, 33, 79, 80},  // interact_g2n3×crMean < 0.0579?
    {1, 255, 1.104329f, 34, 51, 81, 82},  // gas1_norm_step2 < 1.1043?
    {0, 255, 0.474696f, 7, 10, 83, 84},  // gas1_norm_step1 < 0.4747?
    {22, 255, -0.392103f, 13, 17, 85, 86},  // cross_ratio_mean < -0.3921?
    {39, 255, 0.305654f, 12, 17, 87, 88},  // g2n2_div_hum < 0.3057?
    {-1, 1, 0.000000f, 4, 5, -1, -1},  // LEAF: tea (4/5)
    {17, 255, -0.468882f, 23, 38, 89, 90},  // diff_step5 < -0.4689?
    {-1, 0, 0.000000f, 6, 7, -1, -1},  // LEAF: decaf_tea (6/7)
    {-1, 0, 0.000000f, 3, 4, -1, -1},  // LEAF: decaf_tea (3/4)
    {11, 255, 0.161102f, 46, 48, 91, 92},  // cross_ratio_step3 < 0.1611?
    {1, 255, -0.215636f, 17, 27, 93, 94},  // gas1_norm_step2 < -0.2156?
    {-1, 0, 0.000000f, 6, 6, -1, -1},  // LEAF: decaf_tea (6/6)
    {27, 255, 0.240417f, 34, 47, 95, 96},  // interact_g2n2×cr1 < 0.2404?
    {-1, 2, 0.000000f, 4, 4, -1, -1},  // LEAF: decaf_coffee (4/4)
    {-1, 0, 0.000000f, 1, 2, -1, -1},  // LEAF: decaf_tea (1/2)
    {-1, 2, 0.000000f, 7, 8, -1, -1},  // LEAF: decaf_coffee (7/8)
    {-1, 0, 0.000000f, 13, 13, -1, -1},  // LEAF: decaf_tea (13/13)
    {-1, 3, 0.000000f, 3, 4, -1, -1},  // LEAF: coffee (3/4)
    {0, 255, 0.003037f, 12, 13, 97, 98},  // gas1_norm_step1 < 0.0030?
    {-1, 2, 0.000000f, 3, 4, -1, -1},  // LEAF: decaf_coffee (3/4)
    {24, 255, 0.748476f, 5, 12, 99, 100},  // env_gas1_baseline < 0.7485?
    {26, 255, -0.473637f, 20, 26, 101, 102},  // g1g2_ratio_s8 < -0.4736?
    {19, 255, -0.856589f, 46, 47, 103, 104},  // gas1_delta_step2 < -0.8566?
    {-1, 3, 0.000000f, 1, 1, -1, -1},  // LEAF: coffee (1/1)
    {-1, 1, 0.000000f, 6, 6, -1, -1},  // LEAF: tea (6/6)
    {0, 255, 0.700710f, 17, 21, 105, 106},  // gas1_norm_step1 < 0.7007?
    {40, 255, -0.995044f, 34, 42, 107, 108},  // g1n2_div_hum < -0.9950?
    {-1, 3, 0.000000f, 4, 5, -1, -1},  // LEAF: coffee (4/5)
    {-1, 0, 0.000000f, 12, 12, -1, -1},  // LEAF: decaf_tea (12/12)
    {-1, 1, 0.000000f, 1, 1, -1, -1},  // LEAF: tea (1/1)
    {-1, 2, 0.000000f, 5, 9, -1, -1},  // LEAF: decaf_coffee (5/9)
    {-1, 0, 0.000000f, 3, 3, -1, -1},  // LEAF: decaf_tea (3/3)
    {-1, 2, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf_coffee (2/2)
    {27, 255, -0.378887f, 20, 24, 109, 110},  // interact_g2n2×cr1 < -0.3789?
    {-1, 2, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf_coffee (1/1)
    {-1, 1, 0.000000f, 46, 46, -1, -1},  // LEAF: tea (46/46)
    {-1, 3, 0.000000f, 17, 18, -1, -1},  // LEAF: coffee (17/18)
    {-1, 1, 0.000000f, 2, 3, -1, -1},  // LEAF: tea (2/3)
    {-1, 0, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf_tea (2/2)
    {-1, 1, 0.000000f, 34, 40, -1, -1},  // LEAF: tea (34/40)
    {-1, 0, 0.000000f, 2, 3, -1, -1},  // LEAF: decaf_tea (2/3)
    {-1, 3, 0.000000f, 19, 21, -1, -1}  // LEAF: coffee (19/21)
};

// ===========================================================================
// DT Inference Functions
// ===========================================================================

#define DT_READ_NODE(idx) ({ \
    dt_node_t _n; \
    memcpy_P(&_n, &DT_NODES[idx], sizeof(dt_node_t)); \
    _n; })

static inline uint8_t dt_predict(const float* features) {
    uint16_t nodeIdx = 0;
    dt_node_t node = DT_READ_NODE(nodeIdx);
    
    while (node.featureIndex >= 0) {
        if (features[node.featureIndex] < node.threshold) {
            nodeIdx = node.leftChild;
        } else {
            nodeIdx = node.rightChild;
        }
        node = DT_READ_NODE(nodeIdx);
    }
    return node.label;
}

static inline uint8_t dt_predict_with_confidence(const float* features, float* confidence_out) {
    uint16_t nodeIdx = 0;
    dt_node_t node = DT_READ_NODE(nodeIdx);
    
    while (node.featureIndex >= 0) {
        if (features[node.featureIndex] < node.threshold) {
            nodeIdx = node.leftChild;
        } else {
            nodeIdx = node.rightChild;
        }
        node = DT_READ_NODE(nodeIdx);
    }
    
    if (confidence_out) {
        const float total = (float)node.totalSamples;
        const float majority = (float)node.majorityCount;
        *confidence_out = (total > 0.0f) ? (majority / total) : 1.0f;
    }
    return node.label;
}

static inline const char* dt_get_class_name(uint8_t idx) {
    return (idx < 5) ? DT_CLASS_NAMES[idx] : "unknown";
}

#endif // DT_MODEL_DATA_H
