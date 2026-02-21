#ifndef CONFIG_H
#define CONFIG_H
#include <Arduino.h>

//===========================================================================================================
//sources
//===========================================================================================================

//https://stackoverflow.com/questions/53897723/mapping-float-values-into-a-new-value-within-a-specified-range-in-c-or-python
//https://docs.arduino.cc/language-reference/en/functions/math/map/
//https://github.com/boschsensortec/Bosch-BSEC2-Library/tree/master/examples/bme68x_demo_sample


//version info
#define SOFTWARE_VERSION "1.0.0"


//===========================================================================================================
//pin definitions
//===========================================================================================================
//i2c BUS pins
//wire 0 
#define I2C_SDA_PIN 4
#define I2C_SCL_PIN 5
//wire 1
#define I2C1_SDA_PIN 6
#define I2C1_SCL_PIN 7

//BME688 I2C addresses
#define BME688_PRIMARY_ADDR 0x76
#define BME688_SECONDARY_ADDR 0x77

#define I2C_FREQUENCY_HZ 400000L

//display
#define DISPLAY_WIDTH 128
#define DISPLAY_HEIGHT 64
#define DISPLAY_I2C_ADDR 0x3C
#define DISPLAY_RESET_PIN -1

//interaction buttons
#define BUTTON_SELECT_PIN 7 //GP7   black
#define BUTTON_DOWN_PIN 6 //GP6     red
#define BUTTON_DEBOUNCE_MS 50
#define BUTTON_LONG_PRESS_MS 2000

//led indicatios
#define LED_STATUS_PIN LED_BUILTIN
#define LED_ERROR_PIN 11

//SD card
#define SD_MISO_PIN 16
#define SD_CS_PIN 17
#define SD_SCK_PIN 18
#define SD_MOSI_PIN 19

//===========================================================================================================
//sensor config
//===========================================================================================================
#define BME688_NUM_HEATER_STEPS 10
#define BME688_HEATER_DURATION 100 //ms

//heater profile defaults tuned for tea VOC range (higher temps 360–400C)
static const uint16_t DEFAULT_HEATER_TEMPERATURES[BME688_NUM_HEATER_STEPS] = {200, 250, 280, 310, 340, 360, 380, 400, 430, 450};
static const uint16_t DEFAULT_HEATER_DURATIONS[BME688_NUM_HEATER_STEPS] = {150, 150, 140, 130, 120, 110, 100, 100, 100, 100};

static const uint16_t VOC_HEATER_TEMPERATURES[BME688_NUM_HEATER_STEPS] = {330, 340, 350, 360, 370, 380, 390, 395, 400, 400};
static const uint16_t FOOD_HEATER_TEMPERATURES[BME688_NUM_HEATER_STEPS] = {320, 340, 360, 380, 400, 420, 440, 460, 480, 480};

//sampling config
#define BME688_SAMPLE_RATE 3000
#define BME688_GAS_BASE_SAMPLES 30
#define BME688_WARMUP_SAMPLES 10
#define BME688_STABLE_MS 30000

typedef enum{
    MODE_SLEEP = 0,
    MODE_FORCED,
    MODE_PARALLEL,
    MODE_SEQUENTIAL
} bme688_mode_t;

//gas thresholds
#define GAS_RESISTANCE_MIN 1000.0f //ohms
#define GAS_RESISTANCE_MAX 1000000.0f //ohms
#define HUMIDITY_BASELINE 40.0f
#define TEMPERATURE_BASELINE 24.0f

//corrections
#define TEMP_CORRECTION_C (-15.0f)
#define HUM_CORRECTION_PCT (-6.0f)


//===========================================================================================================
//machine learning config
//===========================================================================================================
//paraeters for edge
#define ML_CONFIDENCE_THRESHOLD 0.60f
#define ML_INFERENCE_INTERVAL_MS 2000
#define ML_ANOMALY_THRESHOLD 0.30f
#define ML_SAMPLES 10

//feature extraction perams
#define ML_WINDOW_SIZE 30
#define ML_FEATURE_COUNT 4
#define ML_SENSOR_COUNT 2
#define ML_MODEL_PATH
#define ML_STRIDE 5
#define ML_HEATER_STEPS BME688_NUM_HEATER_STEPS

//total features
#define ML_RAW_FEATURES (ML_HEATER_STEPS * ML_FEATURE_COUNT) //per sensor
#define ML_DELTA_FEATURES 4 //delta_temp, delta_hum, delta_pres, log_gas_cross
#define TOTAL_ML_FEATURES 37

//tea classification labels
typedef enum {
    SCENT_CLASS_PURE_CAMOMILE,
    SCENT_CLASS_THOROUGHLY_MINTED_INFUSION,
    SCENT_CLASS_BERRY_BURST,
    SCENT_CLASS_DARJEELING_BLEND,
    SCENT_CLASS_DECAF_NUTMEG_VANILLA,
    SCENT_CLASS_EARL_GREY,
    SCENT_CLASS_ENGLISH_BREAKFAST_TEA,
    SCENT_CLASS_FRESH_ORANGE,
    SCENT_CLASS_GARDEN_SELECTION_LEMON,
    SCENT_CLASS_GREEN_TEA,
    SCENT_CLASS_RASPBERRY,
    SCENT_CLASS_SWEET_CHERRY,
    SCENT_CLASS_COUNT,
    SCENT_CLASS_UNKNOWN = 255
} scent_class_t;

static const char* SCENT_CLASS_NAMES[SCENT_CLASS_COUNT] = {
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
typedef enum{
    INFERENCE_MODE_SINGLE= 0,
    INFERENCE_MODE_ENSEMBLE,
    INFERENCE_MODE_TEMPORAL,
    INFERENCE_MODE_COUNT,
}inference_mode_t;

static const char* INFERENCE_MODE_NAMES[] ={
    "Single",
    "Temporal",
    "Ensemble"
};

#define TEMPOERAL_COLLECTION_INTERVAL_MS 200
#define TEMPORAL_TIMEOUT_MS 15000

//===========================================================================================================
//data logging config
//===========================================================================================================
#define DATA_LOG_ENABLED true
#define DATA_LOG_BUFFER_SIZE 100
#define DATA_LOG_FILENAME "/iscent_log"
#define DATA_LOG_MAX_FILE_SIZE 1024 * 1024 //1MB
#define DATA_LOG_FLUSH_INTERVAL_MS 5000
#define DATA_LOG_CSV_HEADER "Timestamp,Temp_Primary,Hum_Primary,Pres_Primary,Gas_Primary,Temp_Secondary,Hum_Secondary,Pres_Secondary,Gas_Secondary,Delta_Temp,Delta_Hum,Delta_Pres,Delta_Gas\n"


//===========================================================================================================
//display config
//===========================================================================================================
#define DISPLAY_UPDATE_INTERVAL_MS 1000
#define DISPLAY_TIMEOUT 60000
#define DISPLAY_CONTRAST 0xCF
#define DISPLAY_SPLASH_DURATION_MS 2000

typedef enum{
    DISPLAY_MODE_SPLASH =0,
    DISPLAY_MODE_STATUS,
    DISPLAY_MODE_SENSOR_DATA,
    DISPLAY_MODE_PREDICTION,
    DISPLAY_MODE_GRAPH,
    DISPLAY_MODE_SETTINGS,
    DISPLAY_MODE_LOGGING,
    DISPLAY_MODE_COUNT,
    DISPLAY_MODE_MENU,
    DISPLAY_MODE_ERROR,
    DISPLAY_MODE_CALIBRATION,
    DISPLAY_MODE_DATA_COLLECTION
}display_mode_t;


//===========================================================================================================
//ble config
//===========================================================================================================
#define BLE_ENABLED false   //set to true to enable BLE functionality
#define BLE_DEVICE_NAME             "iScent"
#define BLE_SERVICE_UUID            "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define BLE_CHAR_SENSOR_UUID        "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define BLE_CHAR_PREDICTION_UUID    "beb5483e-36e1-4688-b7f5-ea07361b26a9"
#define BLE_CHAR_CONFIG_UUID        "beb5483e-36e1-4688-b7f5-ea07361b26aa"
#define BLE_NOTIFY_INTERVAL_MS      2000



//===========================================================================================================
//States
//===========================================================================================================
typedef enum{
    STATE_INIT = 0,
    STATE_WARMUP,
    STATE_IDLE,
    STATE_CALIBRATING,
    STATE_ML_BASELINE_CALIB,
    STATE_SAMPLING,
    STATE_INFERENCING,
    STATE_ENSEMBLE_INFERENCING,
    STATE_TEMPORAL_COLLECTING,
    STATE_LOGGING,
    STATE_BLE_CONNECTED,
    STATE_SLEEP,
    STATE_ERROR
} system_state_t;

typedef enum{
    ERROR_NONE =0,
    ERROR_PRIMARY_SENSOR_INIT,
    ERROR_SECONDARY_SENSOR_INIT,
    ERROR_BOTH_SENSORS_INIT,
    ERROR_DISPLAY_INIT,
    ERROR_SD_INIT,
    ERROR_SENSOR_READ,
    ERROR_FILE_SYSTEM_INIT,
    ERROR_SENSOR_TIMEOUT,
    ERROR_FILE_WRITE,
    ERROR_MEMORY_ALLOC,
    ERROR_ML_INIT,
    ERROR_BLE_INIT,
    ERROR_CALIBRATION_TIMEOUT,
    ERROR_CALIBRATION_FAILED,
    ERROR_UNKNOWN
} error_code_t;

static const char* ERROR_CODE_NAMES[] = {
    "No Error",
    "Primary Sensor Init Failed",
    "Secondary Sensor Init Failed",
    "Both Sensors Init Failed",
    "Display Init Failed",
    "SD Card Init Failed",
    "Sensor Read Error",
    "File System Error",
    "Sensor Timeout",
    "File Write Error",
    "Memory Allocation Error",
    "ML Init Failed",
    "BLE Init Failed",
    "Calibration Timeout",
    "Calibration Failed",
    "Unknown Error"
};


//===========================================================================================================
//timings
//===========================================================================================================
#define MAIN_LOOP_INTERVAL_MS 100
#define SENSOR_READ_TIMEOUT_MS 2000
#define CALIBRATION_TIMEOUT_MS 300000
#define SENSOR_WARMUP_TIME_MS 30000
#define SENSOR_STABILIZATION_TIME_MS 60000
#define BLE_CONNECTION_TIMEOUT_MS 30000
#define STATUS_LED_BLINK_MS 500
#define ERROR_LED_BLINK_MS 200
#define WATCHDOG_TIMEOUT_MS 15000


//===========================================================================================================
//debug
//===========================================================================================================
#define DEBUG_ENABLED true
#define DEBUG_SERIAL_BAUD 115200
#define DEBUG_VERBOSE false

#if DEBUG_ENABLED
    #define DEBUG_PRINTLN(x)  Serial.println(x)
    #define DEBUG_PRINT(x) Serial.print(x)
    #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
    
    #if DEBUG_VERBOSE
        #define DEBUG_VERBOSE_PRINTLN(x) Serial.println(x)
        #define DEBUG_VERBOSE_PRINT(x) Serial.print(x)
        #define DEBUG_VERBOSE_PRINTF(...) Serial.printf(__VA_ARGS__)
    #else
        #define DEBUG_VERBOSE_PRINTLN(x)
        #define DEBUG_VERBOSE_PRINT(x)
        #define DEBUG_VERBOSE_PRINTF(...)
    #endif
#else
    #define DEBUG_PRINTLN(x)
    #define DEBUG_PRINT(x)
    #define DEBUG_PRINTF(...)
    #define DEBUG_VERBOSE_PRINTLN(x)
    #define DEBUG_VERBOSE_PRINT(x)
    #define DEBUG_VERBOSE_PRINTF(...)
#endif

//helper macro
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MAP_FLOAT(x, in_min, in_max, out_min, out_max) ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
#define CONSTRAIN_FLOAT(x, min, max) ((x < min) ? min : ((x > max) ? max : x))
#define CONSTRAIN_UINT8(x) ((x<0) ? 0 : ((x>255) ? 255 : x))

#endif