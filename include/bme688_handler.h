#ifndef BME688_HANDER_H
#define BME688_HANDER_H

#include <Arduino.h>
#include "bme68xLibrary.h"
#include "config.h"
#include <Wire.h>

typedef struct{
    float temperature;
    float pressure;
    float humidity;
    float gas_resistance;
    uint8_t gas_index;
    uint32_t timestamp;
    uint8_t status;
    bool valid;
} sensor_data_t;

typedef struct{
    float temperatures[BME688_NUM_HEATER_STEPS];
    float gas_resistances[BME688_NUM_HEATER_STEPS];
    float humidities[BME688_NUM_HEATER_STEPS];
    float pressures[BME688_NUM_HEATER_STEPS];
    uint32_t timestamp;
    uint8_t validReadings;
    bool complete;
} sensor_scan_t;

typedef struct {
    sensor_scan_t primary;
    sensor_scan_t secondary;
    float delta_temp;
    float delta_pres;
    float delta_hum;
    float delta_gas_avg;
    uint32_t timestamp;
    bool valid;
} dual_sensor_data_t;

typedef struct {
    uint16_t temperatures[BME688_NUM_HEATER_STEPS];
    uint16_t durations[BME688_NUM_HEATER_STEPS];
    uint8_t steps;
    const char* profile_name;
} heater_profile_t;

typedef struct {
    float temp_offset_primary;
    float temp_offset_secondary;
    float humidity_offset_primary;
    float humidity_offset_secondary;
    float gas_baseline_primary[BME688_NUM_HEATER_STEPS];
    float gas_baseline_secondary[BME688_NUM_HEATER_STEPS];
    float pressure_offset_primary;
    float pressure_offset_secondary;
    bool calibrated;
    uint32_t timestamp;
} sensor_calibration_t;

typedef struct{
    float temp_mean;
    float temp_stddev;
    float pres_mean;
    float pres_stddev;
    float hum_mean;
    float hum_stddev;
    float gas_mean[BME688_NUM_HEATER_STEPS];
    float gas_stddev[BME688_NUM_HEATER_STEPS];
    uint32_t samples;
} sensor_stats_t;

class BME688Handler {
public:
    BME688Handler();
    ~BME688Handler();

//===========================================================================================================
//init sensors
//===========================================================================================================
    bool begin(TwoWire* primary_wire = &Wire, TwoWire* secondary_wire = &Wire);
    bool beginSingle(TwoWire* wire = &Wire ,uint8_t addr = BME688_PRIMARY_ADDR);
    bool isReady() const;
    bool isPrimaryReady() const;
    bool isSecondaryReady() const;

//===========================================================================================================
//config
//===========================================================================================================
    //set heater profile
    bool setHeaterProfile(const heater_profile_t &profile);

    //set default optimised heater profile
    bool setDefaultHeaterProfile();
    //set IAQ optimised heater profile
    bool setVOCHeaterProfile();

    bool setFoodHeaterProfile();

    //set custom heater profile
    bool setCustomHeaterProfile(const uint16_t* temperatures, const uint16_t* durations, uint8_t steps);

    bool setOperatingMode(bme688_mode_t mode);
    bme688_mode_t getOperatingMode() const;

//===========================================================================================================
//data reading
//===========================================================================================================
    //parallel
    //start parallel measurement
    bool startParallelMode();
    bool readParallelScan(dual_sensor_data_t &data);
    bool performFullScan(dual_sensor_data_t &data);



    //sequential
    //read
    bool readPrimary(sensor_data_t &data);
    bool readSecondary(sensor_data_t &data);

    //scan
    bool readPrimaryScan(sensor_scan_t &data);
    bool readSecondaryScan(sensor_scan_t &data);


//===========================================================================================================
//calibration
//===========================================================================================================
    sensor_calibration_t getCalibrationData() const;
    void setCalibrationData(const sensor_calibration_t &calib);
    void applyCalibration(sensor_scan_t &scan, bool isPrimary);
    void clearCalibration();

    //save/load
    bool saveCalibration();
    bool loadCalibration();

    bool startCalibration(uint16_t samples = BME688_GAS_BASE_SAMPLES);
    bool isCalibrating() const;
    float getCalibrationProgress() const;
    bool finishCalibration();


//===========================================================================================================
//data
//===========================================================================================================

    //calculate gas resistance ratio
    static float getGasRatio(float gas_resistance, float baseline);

    //get IAQ
    static float calculateIAQ(float gas_resistance, float humidity);

    //compensation
    static float compensateTemperature(float gas, float temp, float baseline = TEMPERATURE_BASELINE);
    static float compensateHumidity(float gas, float hum, float baseline = HUMIDITY_BASELINE);
    static bool detectAnomaly(float current, float base, float thresh=ML_ANOMALY_THRESHOLD);

    //===========================================================================================================
    //statistics
    //===========================================================================================================

    //get sample count
    uint32_t getSampleCount() const;

    //reset sample count
    void resetSampleCount();

    //statistics
    sensor_stats_t getPrimaryStats() const;
    sensor_stats_t getSecondaryStats() const;
    void updateStats(const dual_sensor_data_t &data);

//===========================================================================================================
//utility
//===========================================================================================================

    //power management
    void sleep();
    bool isSleeping() const;
    void wake();

    //print
    void printSensorData();
    void printCalibrationData();
    String getScanJSON(const dual_sensor_data_t &data);

//===========================================================================================================
//error handling
//===========================================================================================================

    //get last error code
    error_code_t getLastError() const;
    const char* getLastErrorString() const;
    void clearError();


private:
    //instances
    Bme68x _sensor_primary;
    Bme68x _sensor_secondary;
    TwoWire* _wire_primary;
    TwoWire* _wire_secondary;

    //config
    sensor_calibration_t _calibration_data;
    heater_profile_t _current_profile;
    bme688_mode_t _current_mode;

    //stats
    sensor_stats_t _primary_stats;
    sensor_stats_t _secondary_stats;
    //states
    bool _primary_ready;
    bool _secondary_ready;
    bool _isCalibrating;
    bool _isSleeping;


    uint32_t _sample_count;
    uint16_t _calibration_samples;
    uint16_t _calibration_collected;

    error_code_t _last_error;

    //calibration acc
    float _calibration_temp_sum_p, _calibration_temp_sum_s;
    float _calibration_hum_sum_p, _calibration_hum_sum_s;
    float _calibration_pres_sum_p, _calibration_pres_sum_s;
    float _calibration_gas_sum_p[BME688_NUM_HEATER_STEPS];
    float _calibration_gas_sum_s[BME688_NUM_HEATER_STEPS];
//===========================================================================================================
//helpers
//===========================================================================================================
    //init sensor
    bool initSensor(Bme68x &sensor, TwoWire* wire, uint8_t addr);

    //parallel sensor
    bool configureParallelMode(Bme68x &sensor);
    bool configureForcedMode(Bme68x &sensor);

    //read
    bool readSensorScan(Bme68x &sensor, sensor_scan_t &data);
    bool readSingleReading(Bme68x &sensor, sensor_data_t &data);

    void calculateDeltas(dual_sensor_data_t &data);
    void accumulateCalibrationData(const dual_sensor_data_t &data);
    float getMean(const float *data, uint8_t count);
    float getStdDev(const float *data, uint8_t count, float mean);
};

//ccalc indoor air quality
uint16_t calculateIAQIndex(float gas_resistance, float humidity);  

//check sensor data for VOCs
bool detectVOCs(float current, float base, float threshold=0.2f);

//gas res to est co2
float estimateCO2EQ(float gas_resistance, float baseline);

//get air quality string
const char* getAirQualityString(float iaq);

#endif