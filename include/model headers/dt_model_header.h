// Auto-generated C++ header file
// Model Type: Decision Tree
// Generated on: 2026-02-19 00:19:08
// Feature count: 8
// Classes: 12
// Info: Nodes: 303

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
#if defined(TOTAL_ML_FEATURES) && TOTAL_ML_FEATURES != 8
  #error "DT model was trained with 8 features but TOTAL_ML_FEATURES is different"
#endif

// Model Parameters
#define DT_FEATURE_COUNT 8
#define DT_TOTAL_NODES 303
#define DT_NUM_CLASSES 12
#define DT_MAX_DEPTH 10
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

static const dt_node_t DT_NODES[303] PROGMEM = {
    {1, 255, -0.276011f, 207, 1710, 1, 2},  // gas2_resp < -0.2760?
    {4, 255, -0.816986f, 180, 663, 3, 4},  // d_temp < -0.8170?
    {1, 255, 0.346295f, 200, 1047, 5, 6},  // gas2_resp < 0.3463?
    {1, 255, -0.879299f, 160, 175, 7, 8},  // gas2_resp < -0.8793?
    {6, 255, 0.010043f, 149, 488, 9, 10},  // d_pres < 0.0100?
    {3, 255, 2.003123f, 173, 288, 11, 12},  // gas_diff < 2.0031?
    {7, 255, -0.521252f, 144, 759, 13, 14},  // log_gas_cross < -0.5213?
    {-1, 2, 0.000000f, 151, 151, -1, -1},  // LEAF: berry burst (151/151)
    {3, 255, -0.821000f, 13, 24, 15, 16},  // gas_diff < -0.8210?
    {0, 255, -1.000534f, 108, 359, 17, 18},  // gas1_resp < -1.0005?
    {1, 255, -1.282981f, 105, 129, 19, 20},  // gas2_resp < -1.2830?
    {2, 255, 0.089589f, 173, 258, 21, 22},  // gas_cross < 0.0896?
    {-1, 5, 0.000000f, 30, 30, -1, -1},  // LEAF: earl grey (30/30)
    {1, 255, 1.035661f, 110, 316, 23, 24},  // gas2_resp < 1.0357?
    {3, 255, 1.339179f, 119, 443, 25, 26},  // gas_diff < 1.3392?
    {1, 255, -0.498534f, 9, 13, 27, 28},  // gas2_resp < -0.4985?
    {-1, 1, 0.000000f, 11, 11, -1, -1},  // LEAF: thoroughly minted infusion (11/11)
    {2, 255, -2.243852f, 80, 117, 29, 30},  // gas_cross < -2.2439?
    {1, 255, -0.927963f, 107, 242, 31, 32},  // gas2_resp < -0.9280?
    {0, 255, -0.832414f, 11, 17, 33, 34},  // gas1_resp < -0.8324?
    {5, 255, -1.044236f, 105, 112, 35, 36},  // d_hum < -1.0442?
    {5, 255, -0.104571f, 173, 223, 37, 38},  // d_hum < -0.1046?
    {1, 255, 0.031992f, 19, 35, 39, 40},  // gas2_resp < 0.0320?
    {5, 255, -0.050957f, 96, 143, 41, 42},  // d_hum < -0.0510?
    {7, 255, -1.094600f, 104, 173, 43, 44},  // log_gas_cross < -1.0946?
    {6, 255, 0.640332f, 119, 362, 45, 46},  // d_pres < 0.6403?
    {6, 255, 0.019815f, 55, 81, 47, 48},  // d_pres < 0.0198?
    {5, 255, -0.496972f, 9, 11, 49, 50},  // d_hum < -0.4970?
    {-1, 7, 0.000000f, 2, 2, -1, -1},  // LEAF: fresh orange (2/2)
    {-1, 5, 0.000000f, 18, 18, -1, -1},  // LEAF: earl grey (18/18)
    {5, 255, 1.728645f, 80, 99, 51, 52},  // d_hum < 1.7286?
    {5, 255, 0.972602f, 63, 81, 53, 54},  // d_hum < 0.9726?
    {2, 255, 0.864887f, 90, 161, 55, 56},  // gas_cross < 0.8649?
    {-1, 8, 0.000000f, 6, 6, -1, -1},  // LEAF: garden selection (lemon) (6/6)
    {-1, 2, 0.000000f, 11, 11, -1, -1},  // LEAF: berry burst (11/11)
    {0, 255, 0.460392f, 5, 7, 57, 58},  // gas1_resp < 0.4604?
    {1, 255, -0.337497f, 103, 105, 59, 60},  // gas2_resp < -0.3375?
    {2, 255, -0.583875f, 34, 66, 61, 62},  // gas_cross < -0.5839?
    {0, 255, -0.028836f, 141, 157, 63, 64},  // gas1_resp < -0.0288?
    {3, 255, -1.128597f, 19, 23, 65, 66},  // gas_diff < -1.1286?
    {4, 255, 0.884474f, 7, 12, 67, 68},  // d_temp < 0.8845?
    {7, 255, -0.727330f, 89, 108, 69, 70},  // log_gas_cross < -0.7273?
    {2, 255, 0.637584f, 20, 35, 71, 72},  // gas_cross < 0.6376?
    {4, 255, -0.396745f, 39, 54, 73, 74},  // d_temp < -0.3967?
    {5, 255, -0.992640f, 90, 119, 75, 76},  // d_hum < -0.9926?
    {6, 255, -0.010670f, 118, 260, 77, 78},  // d_pres < -0.0107?
    {2, 255, -0.537307f, 47, 102, 79, 80},  // gas_cross < -0.5373?
    {5, 255, -0.137483f, 13, 28, 81, 82},  // d_hum < -0.1375?
    {5, 255, 1.554067f, 43, 53, 83, 84},  // d_hum < 1.5541?
    {-1, 2, 0.000000f, 9, 9, -1, -1},  // LEAF: berry burst (9/9)
    {-1, 1, 0.000000f, 2, 2, -1, -1},  // LEAF: thoroughly minted infusion (2/2)
    {0, 255, -1.302165f, 80, 91, 85, 86},  // gas1_resp < -1.3022?
    {1, 255, -0.720313f, 4, 8, 87, 88},  // gas2_resp < -0.7203?
    {4, 255, -0.397002f, 60, 69, 89, 90},  // d_temp < -0.3970?
    {4, 255, 1.335085f, 7, 12, 91, 92},  // d_temp < 1.3351?
    {4, 255, 0.172956f, 87, 134, 93, 94},  // d_temp < 0.1730?
    {5, 255, -0.747009f, 22, 27, 95, 96},  // d_hum < -0.7470?
    {-1, 2, 0.000000f, 5, 5, -1, -1},  // LEAF: berry burst (5/5)
    {-1, 1, 0.000000f, 2, 2, -1, -1},  // LEAF: thoroughly minted infusion (2/2)
    {3, 255, -0.901067f, 103, 104, 97, 98},  // gas_diff < -0.9011?
    {-1, 11, 0.000000f, 1, 1, -1, -1},  // LEAF: sweet cherry (1/1)
    {1, 255, -0.013706f, 30, 37, 99, 100},  // gas2_resp < -0.0137?
    {6, 255, -0.279766f, 25, 29, 101, 102},  // d_pres < -0.2798?
    {5, 255, 0.683580f, 141, 155, 103, 104},  // d_hum < 0.6836?
    {-1, 9, 0.000000f, 2, 2, -1, -1},  // LEAF: green tea (2/2)
    {-1, 6, 0.000000f, 3, 4, -1, -1},  // LEAF: english breakfast tea (3/4)
    {6, 255, -0.324090f, 18, 19, 105, 106},  // d_pres < -0.3241?
    {-1, 9, 0.000000f, 7, 7, -1, -1},  // LEAF: green tea (7/7)
    {-1, 6, 0.000000f, 5, 5, -1, -1},  // LEAF: english breakfast tea (5/5)
    {1, 255, 0.962968f, 76, 80, 107, 108},  // gas2_resp < 0.9630?
    {4, 255, -0.603156f, 13, 28, 109, 110},  // d_temp < -0.6032?
    {-1, 4, 0.000000f, 12, 12, -1, -1},  // LEAF: decaf nutmeg and vanilla (12/12)
    {0, 255, 1.392920f, 8, 23, 111, 112},  // gas1_resp < 1.3929?
    {2, 255, 1.330217f, 16, 31, 113, 114},  // gas_cross < 1.3302?
    {-1, 4, 0.000000f, 23, 23, -1, -1},  // LEAF: decaf nutmeg and vanilla (23/23)
    {0, 255, 1.558431f, 8, 17, 115, 116},  // gas1_resp < 1.5584?
    {0, 255, 1.278726f, 88, 102, 117, 118},  // gas1_resp < 1.2787?
    {2, 255, -0.251001f, 34, 104, 119, 120},  // gas_cross < -0.2510?
    {4, 255, 0.403697f, 115, 156, 121, 122},  // d_temp < 0.4037?
    {2, 255, -0.993899f, 18, 34, 123, 124},  // gas_cross < -0.9939?
    {1, 255, 0.699763f, 34, 68, 125, 126},  // gas2_resp < 0.6998?
    {-1, 5, 0.000000f, 9, 9, -1, -1},  // LEAF: earl grey (9/9)
    {4, 255, 0.310355f, 13, 19, 127, 128},  // d_temp < 0.3104?
    {-1, 5, 0.000000f, 37, 37, -1, -1},  // LEAF: earl grey (37/37)
    {0, 255, -0.953253f, 7, 16, 129, 130},  // gas1_resp < -0.9533?
    {-1, 8, 0.000000f, 64, 64, -1, -1},  // LEAF: garden selection (lemon) (64/64)
    {4, 255, 0.653715f, 16, 27, 131, 132},  // d_temp < 0.6537?
    {-1, 1, 0.000000f, 4, 4, -1, -1},  // LEAF: thoroughly minted infusion (4/4)
    {-1, 11, 0.000000f, 4, 4, -1, -1},  // LEAF: sweet cherry (4/4)
    {-1, 2, 0.000000f, 2, 2, -1, -1},  // LEAF: berry burst (2/2)
    {0, 255, -0.797506f, 60, 67, 133, 134},  // gas1_resp < -0.7975?
    {3, 255, 0.301613f, 7, 8, 135, 136},  // gas_diff < 0.3016?
    {-1, 6, 0.000000f, 2, 4, -1, -1},  // LEAF: english breakfast tea (2/4)
    {1, 255, -0.568164f, 17, 39, 137, 138},  // gas2_resp < -0.5682?
    {2, 255, -0.443447f, 77, 95, 139, 140},  // gas_cross < -0.4434?
    {0, 255, 0.659463f, 21, 22, 141, 142},  // gas1_resp < 0.6595?
    {0, 255, 0.249648f, 3, 5, 143, 144},  // gas1_resp < 0.2496?
    {0, 255, -0.209285f, 13, 14, 145, 146},  // gas1_resp < -0.2093?
    {-1, 1, 0.000000f, 90, 90, -1, -1},  // LEAF: thoroughly minted infusion (90/90)
    {5, 255, -0.407941f, 7, 11, 147, 148},  // d_hum < -0.4079?
    {-1, 8, 0.000000f, 26, 26, -1, -1},  // LEAF: garden selection (lemon) (26/26)
    {1, 255, 0.318700f, 25, 26, 149, 150},  // gas2_resp < 0.3187?
    {-1, 8, 0.000000f, 3, 3, -1, -1},  // LEAF: garden selection (lemon) (3/3)
    {2, 255, -1.369211f, 57, 69, 151, 152},  // gas_cross < -1.3692?
    {4, 255, -0.988245f, 84, 86, 153, 154},  // d_temp < -0.9882?
    {-1, 7, 0.000000f, 18, 18, -1, -1},  // LEAF: fresh orange (18/18)
    {-1, 6, 0.000000f, 1, 1, -1, -1},  // LEAF: english breakfast tea (1/1)
    {0, 255, 1.719122f, 74, 75, 155, 156},  // gas1_resp < 1.7191?
    {2, 255, 1.443283f, 3, 5, 157, 158},  // gas_cross < 1.4433?
    {1, 255, 0.815521f, 8, 10, 159, 160},  // gas2_resp < 0.8155?
    {5, 255, -0.136130f, 12, 18, 161, 162},  // d_hum < -0.1361?
    {2, 255, 0.767045f, 7, 12, 163, 164},  // gas_cross < 0.7670?
    {4, 255, -0.350465f, 8, 11, 165, 166},  // d_temp < -0.3505?
    {1, 255, 1.050581f, 12, 13, 167, 168},  // gas2_resp < 1.0506?
    {5, 255, -0.741735f, 15, 18, 169, 170},  // d_hum < -0.7417?
    {4, 255, -1.284006f, 8, 9, 171, 172},  // d_temp < -1.2840?
    {4, 255, -0.835681f, 6, 8, 173, 174},  // d_temp < -0.8357?
    {5, 255, 0.041339f, 6, 13, 175, 176},  // d_hum < 0.0413?
    {1, 255, 1.985383f, 82, 89, 177, 178},  // gas2_resp < 1.9854?
    {1, 255, 0.531451f, 11, 39, 179, 180},  // gas2_resp < 0.5315?
    {1, 255, 0.627881f, 27, 65, 181, 182},  // gas2_resp < 0.6279?
    {1, 255, 1.048785f, 113, 139, 183, 184},  // gas2_resp < 1.0488?
    {5, 255, 0.061411f, 8, 17, 185, 186},  // d_hum < 0.0614?
    {4, 255, -0.801176f, 8, 10, 187, 188},  // d_temp < -0.8012?
    {5, 255, -0.651844f, 16, 24, 189, 190},  // d_hum < -0.6518?
    {0, 255, 0.228266f, 10, 12, 191, 192},  // gas1_resp < 0.2283?
    {5, 255, -0.385635f, 32, 56, 193, 194},  // d_hum < -0.3856?
    {-1, 11, 0.000000f, 13, 13, -1, -1},  // LEAF: sweet cherry (13/13)
    {0, 255, -1.162179f, 3, 6, 195, 196},  // gas1_resp < -1.1622?
    {0, 255, -1.197479f, 6, 9, 197, 198},  // gas1_resp < -1.1975?
    {-1, 10, 0.000000f, 7, 7, -1, -1},  // LEAF: raspberry (7/7)
    {2, 255, -0.596576f, 13, 16, 199, 200},  // gas_cross < -0.5966?
    {4, 255, 1.443235f, 8, 11, 201, 202},  // d_temp < 1.4432?
    {2, 255, -0.297266f, 8, 14, 203, 204},  // gas_cross < -0.2973?
    {1, 255, -0.938346f, 52, 53, 205, 206},  // gas2_resp < -0.9383?
    {-1, 1, 0.000000f, 7, 7, -1, -1},  // LEAF: thoroughly minted infusion (7/7)
    {-1, 6, 0.000000f, 1, 1, -1, -1},  // LEAF: english breakfast tea (1/1)
    {7, 255, -0.491594f, 17, 30, 207, 208},  // log_gas_cross < -0.4916?
    {-1, 7, 0.000000f, 9, 9, -1, -1},  // LEAF: fresh orange (9/9)
    {3, 255, 0.268160f, 6, 10, 209, 210},  // gas_diff < 0.2682?
    {1, 255, -0.875646f, 77, 85, 211, 212},  // gas2_resp < -0.8756?
    {-1, 6, 0.000000f, 21, 21, -1, -1},  // LEAF: english breakfast tea (21/21)
    {-1, 1, 0.000000f, 1, 1, -1, -1},  // LEAF: thoroughly minted infusion (1/1)
    {-1, 1, 0.000000f, 1, 2, -1, -1},  // LEAF: thoroughly minted infusion (1/2)
    {-1, 7, 0.000000f, 3, 3, -1, -1},  // LEAF: fresh orange (3/3)
    {-1, 2, 0.000000f, 1, 1, -1, -1},  // LEAF: berry burst (1/1)
    {-1, 1, 0.000000f, 13, 13, -1, -1},  // LEAF: thoroughly minted infusion (13/13)
    {-1, 11, 0.000000f, 6, 6, -1, -1},  // LEAF: sweet cherry (6/6)
    {0, 255, -0.510786f, 4, 5, 213, 214},  // gas1_resp < -0.5108?
    {-1, 11, 0.000000f, 25, 25, -1, -1},  // LEAF: sweet cherry (25/25)
    {-1, 8, 0.000000f, 1, 1, -1, -1},  // LEAF: garden selection (lemon) (1/1)
    {1, 255, 0.048777f, 6, 7, 215, 216},  // gas2_resp < 0.0488?
    {4, 255, 1.915272f, 57, 62, 217, 218},  // d_temp < 1.9153?
    {-1, 5, 0.000000f, 1, 1, -1, -1},  // LEAF: earl grey (1/1)
    {4, 255, -0.943861f, 84, 85, 219, 220},  // d_temp < -0.9439?
    {-1, 9, 0.000000f, 71, 71, -1, -1},  // LEAF: green tea (71/71)
    {-1, 9, 0.000000f, 3, 4, -1, -1},  // LEAF: green tea (3/4)
    {-1, 4, 0.000000f, 3, 3, -1, -1},  // LEAF: decaf nutmeg and vanilla (3/3)
    {-1, 9, 0.000000f, 2, 2, -1, -1},  // LEAF: green tea (2/2)
    {1, 255, 0.477875f, 8, 9, 221, 222},  // gas2_resp < 0.4779?
    {-1, 0, 0.000000f, 1, 1, -1, -1},  // LEAF: camomile (1/1)
    {0, 255, 0.744632f, 12, 16, 223, 224},  // gas1_resp < 0.7446?
    {-1, 4, 0.000000f, 2, 2, -1, -1},  // LEAF: decaf nutmeg and vanilla (2/2)
    {0, 255, 1.145462f, 3, 5, 225, 226},  // gas1_resp < 1.1455?
    {-1, 9, 0.000000f, 7, 7, -1, -1},  // LEAF: green tea (7/7)
    {-1, 3, 0.000000f, 3, 3, -1, -1},  // LEAF: darjeeling blend (3/3)
    {-1, 4, 0.000000f, 8, 8, -1, -1},  // LEAF: decaf nutmeg and vanilla (8/8)
    {-1, 4, 0.000000f, 1, 1, -1, -1},  // LEAF: decaf nutmeg and vanilla (1/1)
    {-1, 3, 0.000000f, 12, 12, -1, -1},  // LEAF: darjeeling blend (12/12)
    {4, 255, -1.150723f, 15, 16, 227, 228},  // d_temp < -1.1507?
    {-1, 3, 0.000000f, 2, 2, -1, -1},  // LEAF: darjeeling blend (2/2)
    {-1, 0, 0.000000f, 1, 1, -1, -1},  // LEAF: camomile (1/1)
    {-1, 10, 0.000000f, 8, 8, -1, -1},  // LEAF: raspberry (8/8)
    {-1, 3, 0.000000f, 2, 2, -1, -1},  // LEAF: darjeeling blend (2/2)
    {-1, 4, 0.000000f, 6, 6, -1, -1},  // LEAF: decaf nutmeg and vanilla (6/6)
    {1, 255, 1.129740f, 3, 8, 229, 230},  // gas2_resp < 1.1297?
    {-1, 3, 0.000000f, 5, 5, -1, -1},  // LEAF: darjeeling blend (5/5)
    {1, 255, 1.217836f, 78, 81, 231, 232},  // gas2_resp < 1.2178?
    {5, 255, -0.719797f, 4, 8, 233, 234},  // d_hum < -0.7198?
    {5, 255, -0.381567f, 11, 14, 235, 236},  // d_hum < -0.3816?
    {3, 255, 0.396836f, 8, 25, 237, 238},  // gas_diff < 0.3968?
    {4, 255, -0.302684f, 15, 19, 239, 240},  // d_temp < -0.3027?
    {0, 255, 0.281310f, 27, 46, 241, 242},  // gas1_resp < 0.2813?
    {6, 255, 0.605582f, 107, 122, 243, 244},  // d_pres < 0.6056?
    {4, 255, -0.742380f, 6, 17, 245, 246},  // d_temp < -0.7424?
    {0, 255, 0.412666f, 8, 10, 247, 248},  // gas1_resp < 0.4127?
    {1, 255, 0.643023f, 4, 7, 249, 250},  // gas2_resp < 0.6430?
    {-1, 5, 0.000000f, 2, 3, -1, -1},  // LEAF: earl grey (2/3)
    {-1, 10, 0.000000f, 7, 7, -1, -1},  // LEAF: raspberry (7/7)
    {-1, 3, 0.000000f, 2, 2, -1, -1},  // LEAF: darjeeling blend (2/2)
    {1, 255, 0.658060f, 16, 22, 251, 252},  // gas2_resp < 0.6581?
    {-1, 10, 0.000000f, 2, 2, -1, -1},  // LEAF: raspberry (2/2)
    {-1, 4, 0.000000f, 10, 10, -1, -1},  // LEAF: decaf nutmeg and vanilla (10/10)
    {6, 255, 2.127398f, 26, 33, 253, 254},  // d_pres < 2.1274?
    {1, 255, 0.908926f, 14, 23, 255, 256},  // gas2_resp < 0.9089?
    {-1, 5, 0.000000f, 3, 3, -1, -1},  // LEAF: earl grey (3/3)
    {-1, 10, 0.000000f, 3, 3, -1, -1},  // LEAF: raspberry (3/3)
    {-1, 5, 0.000000f, 6, 6, -1, -1},  // LEAF: earl grey (6/6)
    {-1, 11, 0.000000f, 3, 3, -1, -1},  // LEAF: sweet cherry (3/3)
    {6, 255, -0.146831f, 13, 14, 257, 258},  // d_pres < -0.1468?
    {-1, 1, 0.000000f, 1, 2, -1, -1},  // LEAF: thoroughly minted infusion (1/2)
    {-1, 1, 0.000000f, 8, 8, -1, -1},  // LEAF: thoroughly minted infusion (8/8)
    {-1, 8, 0.000000f, 3, 3, -1, -1},  // LEAF: garden selection (lemon) (3/3)
    {4, 255, 1.828089f, 7, 8, 259, 260},  // d_temp < 1.8281?
    {1, 255, -1.452145f, 5, 6, 261, 262},  // gas2_resp < -1.4521?
    {-1, 6, 0.000000f, 48, 48, -1, -1},  // LEAF: english breakfast tea (48/48)
    {1, 255, -0.935683f, 4, 5, 263, 264},  // gas2_resp < -0.9357?
    {3, 255, -1.182874f, 15, 16, 265, 266},  // gas_diff < -1.1829?
    {6, 255, -0.508961f, 11, 14, 267, 268},  // d_pres < -0.5090?
    {-1, 8, 0.000000f, 6, 6, -1, -1},  // LEAF: garden selection (lemon) (6/6)
    {-1, 1, 0.000000f, 2, 4, -1, -1},  // LEAF: thoroughly minted infusion (2/4)
    {0, 255, -0.205235f, 7, 12, 269, 270},  // gas1_resp < -0.2052?
    {6, 255, -0.429792f, 70, 73, 271, 272},  // d_pres < -0.4298?
    {-1, 8, 0.000000f, 4, 4, -1, -1},  // LEAF: garden selection (lemon) (4/4)
    {-1, 11, 0.000000f, 1, 1, -1, -1},  // LEAF: sweet cherry (1/1)
    {-1, 8, 0.000000f, 6, 6, -1, -1},  // LEAF: garden selection (lemon) (6/6)
    {-1, 5, 0.000000f, 1, 1, -1, -1},  // LEAF: earl grey (1/1)
    {5, 255, 0.672581f, 57, 61, 273, 274},  // d_hum < 0.6726?
    {-1, 8, 0.000000f, 1, 1, -1, -1},  // LEAF: garden selection (lemon) (1/1)
    {-1, 0, 0.000000f, 1, 2, -1, -1},  // LEAF: camomile (1/2)
    {-1, 11, 0.000000f, 83, 83, -1, -1},  // LEAF: sweet cherry (83/83)
    {-1, 9, 0.000000f, 1, 1, -1, -1},  // LEAF: green tea (1/1)
    {-1, 4, 0.000000f, 8, 8, -1, -1},  // LEAF: decaf nutmeg and vanilla (8/8)
    {-1, 6, 0.000000f, 3, 4, -1, -1},  // LEAF: english breakfast tea (3/4)
    {3, 255, -1.200994f, 11, 12, 275, 276},  // gas_diff < -1.2010?
    {-1, 3, 0.000000f, 3, 3, -1, -1},  // LEAF: darjeeling blend (3/3)
    {-1, 5, 0.000000f, 2, 2, -1, -1},  // LEAF: earl grey (2/2)
    {-1, 9, 0.000000f, 1, 1, -1, -1},  // LEAF: green tea (1/1)
    {-1, 4, 0.000000f, 15, 15, -1, -1},  // LEAF: decaf nutmeg and vanilla (15/15)
    {-1, 4, 0.000000f, 3, 3, -1, -1},  // LEAF: decaf nutmeg and vanilla (3/3)
    {7, 255, -0.672115f, 2, 5, 277, 278},  // log_gas_cross < -0.6721?
    {5, 255, -0.620953f, 11, 14, 279, 280},  // d_hum < -0.6210?
    {-1, 3, 0.000000f, 67, 67, -1, -1},  // LEAF: darjeeling blend (67/67)
    {-1, 3, 0.000000f, 4, 4, -1, -1},  // LEAF: darjeeling blend (4/4)
    {-1, 4, 0.000000f, 4, 4, -1, -1},  // LEAF: decaf nutmeg and vanilla (4/4)
    {-1, 8, 0.000000f, 3, 3, -1, -1},  // LEAF: garden selection (lemon) (3/3)
    {-1, 11, 0.000000f, 11, 11, -1, -1},  // LEAF: sweet cherry (11/11)
    {0, 255, 0.168718f, 6, 11, 281, 282},  // gas1_resp < 0.1687?
    {4, 255, 0.761293f, 8, 14, 283, 284},  // d_temp < 0.7613?
    {-1, 9, 0.000000f, 3, 3, -1, -1},  // LEAF: green tea (3/3)
    {1, 255, 0.382812f, 15, 16, 285, 286},  // gas2_resp < 0.3828?
    {-1, 4, 0.000000f, 4, 4, -1, -1},  // LEAF: decaf nutmeg and vanilla (4/4)
    {6, 255, -0.293061f, 27, 42, 287, 288},  // d_pres < -0.2931?
    {5, 255, -0.437126f, 107, 121, 289, 290},  // d_hum < -0.4371?
    {-1, 10, 0.000000f, 1, 1, -1, -1},  // LEAF: raspberry (1/1)
    {-1, 0, 0.000000f, 6, 6, -1, -1},  // LEAF: camomile (6/6)
    {1, 255, 1.182952f, 5, 11, 291, 292},  // gas2_resp < 1.1830?
    {-1, 10, 0.000000f, 8, 8, -1, -1},  // LEAF: raspberry (8/8)
    {-1, 6, 0.000000f, 1, 2, -1, -1},  // LEAF: english breakfast tea (1/2)
    {-1, 0, 0.000000f, 2, 3, -1, -1},  // LEAF: camomile (2/3)
    {-1, 5, 0.000000f, 4, 4, -1, -1},  // LEAF: earl grey (4/4)
    {0, 255, -0.254609f, 4, 8, 293, 294},  // gas1_resp < -0.2546?
    {6, 255, 1.886066f, 13, 14, 295, 296},  // d_pres < 1.8861?
    {2, 255, -0.019814f, 26, 30, 297, 298},  // gas_cross < -0.0198?
    {-1, 3, 0.000000f, 3, 3, -1, -1},  // LEAF: darjeeling blend (3/3)
    {0, 255, 0.808982f, 3, 6, 299, 300},  // gas1_resp < 0.8090?
    {1, 255, 1.166935f, 14, 17, 301, 302},  // gas2_resp < 1.1669?
    {-1, 8, 0.000000f, 12, 12, -1, -1},  // LEAF: garden selection (lemon) (12/12)
    {-1, 1, 0.000000f, 1, 2, -1, -1},  // LEAF: thoroughly minted infusion (1/2)
    {-1, 6, 0.000000f, 7, 7, -1, -1},  // LEAF: english breakfast tea (7/7)
    {-1, 8, 0.000000f, 1, 1, -1, -1},  // LEAF: garden selection (lemon) (1/1)
    {-1, 6, 0.000000f, 1, 1, -1, -1},  // LEAF: english breakfast tea (1/1)
    {-1, 1, 0.000000f, 5, 5, -1, -1},  // LEAF: thoroughly minted infusion (5/5)
    {-1, 7, 0.000000f, 1, 1, -1, -1},  // LEAF: fresh orange (1/1)
    {-1, 6, 0.000000f, 4, 4, -1, -1},  // LEAF: english breakfast tea (4/4)
    {-1, 7, 0.000000f, 1, 1, -1, -1},  // LEAF: fresh orange (1/1)
    {-1, 6, 0.000000f, 15, 15, -1, -1},  // LEAF: english breakfast tea (15/15)
    {-1, 1, 0.000000f, 11, 12, -1, -1},  // LEAF: thoroughly minted infusion (11/12)
    {-1, 2, 0.000000f, 1, 2, -1, -1},  // LEAF: berry burst (1/2)
    {-1, 7, 0.000000f, 7, 9, -1, -1},  // LEAF: fresh orange (7/9)
    {-1, 6, 0.000000f, 3, 3, -1, -1},  // LEAF: english breakfast tea (3/3)
    {-1, 7, 0.000000f, 63, 63, -1, -1},  // LEAF: fresh orange (63/63)
    {-1, 7, 0.000000f, 7, 10, -1, -1},  // LEAF: fresh orange (7/10)
    {-1, 11, 0.000000f, 57, 60, -1, -1},  // LEAF: sweet cherry (57/60)
    {-1, 8, 0.000000f, 1, 1, -1, -1},  // LEAF: garden selection (lemon) (1/1)
    {-1, 6, 0.000000f, 1, 1, -1, -1},  // LEAF: english breakfast tea (1/1)
    {-1, 9, 0.000000f, 11, 11, -1, -1},  // LEAF: green tea (11/11)
    {-1, 10, 0.000000f, 2, 2, -1, -1},  // LEAF: raspberry (2/2)
    {-1, 0, 0.000000f, 1, 3, -1, -1},  // LEAF: camomile (1/3)
    {-1, 5, 0.000000f, 3, 3, -1, -1},  // LEAF: earl grey (3/3)
    {-1, 3, 0.000000f, 11, 11, -1, -1},  // LEAF: darjeeling blend (11/11)
    {-1, 10, 0.000000f, 4, 5, -1, -1},  // LEAF: raspberry (4/5)
    {-1, 3, 0.000000f, 6, 6, -1, -1},  // LEAF: darjeeling blend (6/6)
    {-1, 5, 0.000000f, 7, 7, -1, -1},  // LEAF: earl grey (7/7)
    {-1, 8, 0.000000f, 3, 7, -1, -1},  // LEAF: garden selection (lemon) (3/7)
    {-1, 9, 0.000000f, 1, 1, -1, -1},  // LEAF: green tea (1/1)
    {-1, 4, 0.000000f, 15, 15, -1, -1},  // LEAF: decaf nutmeg and vanilla (15/15)
    {-1, 10, 0.000000f, 19, 22, -1, -1},  // LEAF: raspberry (19/22)
    {-1, 10, 0.000000f, 8, 20, -1, -1},  // LEAF: raspberry (8/20)
    {-1, 0, 0.000000f, 17, 23, -1, -1},  // LEAF: camomile (17/23)
    {-1, 0, 0.000000f, 90, 98, -1, -1},  // LEAF: camomile (90/98)
    {-1, 8, 0.000000f, 3, 5, -1, -1},  // LEAF: garden selection (lemon) (3/5)
    {-1, 3, 0.000000f, 5, 6, -1, -1},  // LEAF: darjeeling blend (5/6)
    {-1, 5, 0.000000f, 2, 3, -1, -1},  // LEAF: earl grey (2/3)
    {-1, 10, 0.000000f, 4, 5, -1, -1},  // LEAF: raspberry (4/5)
    {-1, 5, 0.000000f, 11, 11, -1, -1},  // LEAF: earl grey (11/11)
    {-1, 5, 0.000000f, 2, 3, -1, -1},  // LEAF: earl grey (2/3)
    {-1, 10, 0.000000f, 4, 8, -1, -1},  // LEAF: raspberry (4/8)
    {-1, 10, 0.000000f, 22, 22, -1, -1},  // LEAF: raspberry (22/22)
    {-1, 4, 0.000000f, 3, 4, -1, -1},  // LEAF: decaf nutmeg and vanilla (3/4)
    {-1, 10, 0.000000f, 2, 2, -1, -1},  // LEAF: raspberry (2/2)
    {-1, 3, 0.000000f, 4, 7, -1, -1},  // LEAF: darjeeling blend (4/7)
    {-1, 3, 0.000000f, 10, 10, -1, -1}  // LEAF: darjeeling blend (10/10)
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
        // Laplace-smoothed leaf confidence with support damping
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
