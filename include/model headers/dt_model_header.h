// Auto-generated C++ header file
// Model Type: Decision Tree
// Generated on: 2026-02-05 17:54:15
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
  #ifndef pgm_read_float
    #define pgm_read_float(addr) (*(const float *)(addr))
  #endif
  #ifndef memcpy_P
    #define memcpy_P(dest, src, n) memcpy((dest), (src), (n))
  #endif
#endif

// Model Parameters
#define DT_FEATURE_COUNT 12
#define DT_TOTAL_NODES 111
#define DT_NUM_CLASSES 12
#define DT_MAX_DEPTH 12
#define DT_HAS_CONFIDENCE 1

// Class Names
static const char* DT_CLASS_NAMES[12] = {
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

typedef struct {
    int8_t featureIndex;
    uint8_t label;
    float threshold;
    uint16_t majorityCount;
    uint16_t totalSamples;
    int16_t leftChild;
    int16_t rightChild;
} dt_node_t;

static const dt_node_t DT_NODES[111] PROGMEM = {
    {7, 255, 1659140.500000f, 69, 570, 1, 2},
    {4, 255, 29.990000f, 61, 233, 3, 4},
    {4, 255, 28.935001f, 64, 337, 5, 6},
    {7, 255, 1491907.000000f, 61, 119, 7, 8},
    {2, 255, 976.270020f, 41, 114, 9, 10},
    {2, 255, 976.905029f, 45, 267, 11, 12},
    {5, 255, 65.845001f, 57, 70, 13, 14},
    {0, 255, 29.270000f, 57, 60, 15, 16},
    {0, 255, 24.885000f, 54, 59, 17, 18},
    {-1, 6, 0.000000f, 36, 36, -1, -1},
    {3, 255, 687671.500000f, 41, 78, 19, 20},
    {4, 255, 25.735001f, 45, 239, 21, 22},
    {-1, 10, 0.000000f, 28, 28, -1, -1},
    {11, 255, -448127.500000f, 57, 64, 23, 24},
    {1, 255, 84.315002f, 3, 6, 25, 26},
    {-1, 2, 0.000000f, 57, 57, -1, -1},
    {-1, 1, 0.000000f, 3, 3, -1, -1},
    {-1, 2, 0.000000f, 4, 4, -1, -1},
    {0, 255, 30.994999f, 54, 55, 27, 28},
    {4, 255, 32.599998f, 27, 36, 29, 30},
    {2, 255, 976.674988f, 41, 42, 31, 32},
    {3, 255, 988614.500000f, 44, 64, 33, 34},
    {3, 255, 787943.500000f, 40, 175, 35, 36},
    {8, 255, 0.085000f, 57, 61, 37, 38},
    {-1, 7, 0.000000f, 2, 3, -1, -1},
    {-1, 8, 0.000000f, 3, 3, -1, -1},
    {-1, 5, 0.000000f, 3, 3, -1, -1},
    {-1, 1, 0.000000f, 54, 54, -1, -1},
    {-1, 5, 0.000000f, 1, 1, -1, -1},
    {9, 255, 11.125000f, 5, 11, 39, 40},
    {-1, 8, 0.000000f, 25, 25, -1, -1},
    {-1, 8, 0.000000f, 1, 1, -1, -1},
    {-1, 7, 0.000000f, 41, 41, -1, -1},
    {2, 255, 976.489990f, 44, 49, 41, 42},
    {2, 255, 976.339966f, 9, 15, 43, 44},
    {2, 255, 976.830017f, 30, 41, 45, 46},
    {7, 255, 1875673.000000f, 40, 134, 47, 48},
    {-1, 8, 0.000000f, 1, 1, -1, -1},
    {8, 255, 1.550000f, 57, 60, 49, 50},
    {1, 255, 83.884995f, 4, 6, 51, 52},
    {-1, 11, 0.000000f, 5, 5, -1, -1},
    {10, 255, 1.140000f, 44, 46, 53, 54},
    {-1, 5, 0.000000f, 3, 3, -1, -1},
    {0, 255, 23.095001f, 6, 7, 55, 56},
    {-1, 4, 0.000000f, 8, 8, -1, -1},
    {4, 255, 28.360001f, 30, 37, 57, 58},
    {-1, 11, 0.000000f, 3, 4, -1, -1},
    {11, 255, -441429.500000f, 39, 87, 59, 60},
    {11, 255, -432004.000000f, 31, 47, 61, 62},
    {11, 255, -746381.000000f, 56, 58, 63, 64},
    {-1, 8, 0.000000f, 1, 2, -1, -1},
    {-1, 8, 0.000000f, 2, 2, -1, -1},
    {-1, 5, 0.000000f, 4, 4, -1, -1},
    {-1, 1, 0.000000f, 1, 2, -1, -1},
    {-1, 0, 0.000000f, 44, 44, -1, -1},
    {-1, 4, 0.000000f, 1, 1, -1, -1},
    {-1, 3, 0.000000f, 6, 6, -1, -1},
    {7, 255, 1854023.000000f, 26, 29, 65, 66},
    {1, 255, 74.544998f, 4, 8, 67, 68},
    {6, 255, 975.429993f, 24, 51, 69, 70},
    {6, 255, 975.125000f, 29, 36, 71, 72},
    {2, 255, 976.474976f, 31, 40, 73, 74},
    {-1, 4, 0.000000f, 7, 7, -1, -1},
    {-1, 8, 0.000000f, 1, 2, -1, -1},
    {8, 255, 0.335000f, 55, 56, 75, 76},
    {8, 255, 2.060000f, 26, 28, 77, 78},
    {-1, 11, 0.000000f, 1, 1, -1, -1},
    {-1, 11, 0.000000f, 3, 4, -1, -1},
    {-1, 5, 0.000000f, 4, 4, -1, -1},
    {9, 255, 4.015000f, 24, 44, 79, 80},
    {0, 255, 28.779999f, 6, 7, 81, 82},
    {0, 255, 30.005001f, 29, 31, 83, 84},
    {1, 255, 52.250000f, 2, 5, 85, 86},
    {1, 255, 44.985001f, 28, 29, 87, 88},
    {1, 255, 62.709999f, 4, 11, 89, 90},
    {0, 255, 30.080002f, 6, 7, 91, 92},
    {-1, 11, 0.000000f, 49, 49, -1, -1},
    {2, 255, 976.815002f, 26, 27, 93, 94},
    {-1, 10, 0.000000f, 1, 1, -1, -1},
    {2, 255, 976.319946f, 7, 12, 95, 96},
    {1, 255, 75.724998f, 22, 32, 97, 98},
    {-1, 10, 0.000000f, 6, 6, -1, -1},
    {-1, 8, 0.000000f, 1, 1, -1, -1},
    {2, 255, 976.400024f, 29, 30, 99, 100},
    {-1, 6, 0.000000f, 1, 1, -1, -1},
    {-1, 6, 0.000000f, 2, 2, -1, -1},
    {-1, 4, 0.000000f, 2, 3, -1, -1},
    {-1, 9, 0.000000f, 1, 1, -1, -1},
    {-1, 3, 0.000000f, 28, 28, -1, -1},
    {0, 255, 26.865000f, 4, 8, 101, 102},
    {-1, 3, 0.000000f, 3, 3, -1, -1},
    {-1, 11, 0.000000f, 6, 6, -1, -1},
    {-1, 8, 0.000000f, 1, 1, -1, -1},
    {-1, 5, 0.000000f, 25, 25, -1, -1},
    {-1, 5, 0.000000f, 1, 2, -1, -1},
    {-1, 9, 0.000000f, 7, 7, -1, -1},
    {3, 255, 874185.500000f, 2, 5, 103, 104},
    {6, 255, 974.839966f, 22, 29, 105, 106},
    {-1, 3, 0.000000f, 3, 3, -1, -1},
    {-1, 9, 0.000000f, 29, 29, -1, -1},
    {-1, 4, 0.000000f, 1, 1, -1, -1},
    {1, 255, 45.110001f, 4, 5, 107, 108},
    {-1, 4, 0.000000f, 2, 3, -1, -1},
    {-1, 5, 0.000000f, 2, 2, -1, -1},
    {-1, 4, 0.000000f, 2, 3, -1, -1},
    {-1, 3, 0.000000f, 2, 3, -1, -1},
    {7, 255, 1762567.000000f, 22, 26, 109, 110},
    {-1, 10, 0.000000f, 1, 1, -1, -1},
    {-1, 5, 0.000000f, 4, 4, -1, -1},
    {-1, 9, 0.000000f, 2, 2, -1, -1},
    {-1, 4, 0.000000f, 22, 24, -1, -1}
};

//===========================================================================
// DT Inference Functions
//===========================================================================

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
        // Laplace-smoothed leaf confidence with stronger prior and heavier support damping
        const float total = (float)node.totalSamples;
        const float majority = (float)node.majorityCount;
        const float laplace = (majority + 0.5f) / (total + (float)DT_NUM_CLASSES);
        const float support = total / (total + 20.0f);
        *confidence_out = laplace * support;
    }
    return node.label;
}

static inline const char* dt_get_class_name(uint8_t idx) {
    return (idx < 12) ? DT_CLASS_NAMES[idx] : "unknown";
}

#endif // DT_MODEL_DATA_H
