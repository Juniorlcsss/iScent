#ifndef BASELINE_CALIBRATION_H
#define BASELINE_CALIBRATION_H

#include <Arduino.h>
#include "config.h"
#include "bme688_handler.h"

//stores per-sensor baseline medians for drift correction
typedef struct {
    float temp1_baseline;
    float hum1_baseline;
    float pres1_baseline;
    float temp2_baseline;
    float hum2_baseline;
    float pres2_baseline;

    float gas1_step_baselines[BME688_NUM_HEATER_STEPS];
    float gas2_step_baselines[BME688_NUM_HEATER_STEPS];

    //legacy single value
    float gas1_baseline;
    float gas2_baseline;
    
    uint32_t calibration_time;
    bool valid;
} baseline_t;

//calibration status
typedef enum {
    CALIB_STATE_IDLE = 0,
    CALIB_STATE_COLLECTING,
    CALIB_STATE_COMPLETE,
    CALIB_STATE_ERROR
} calibration_state_t;

class BaselineCalibration {
public:
    BaselineCalibration();
    ~BaselineCalibration();

    //===========================================================================================================
    //calibration
    //===========================================================================================================
    bool startCalibration(uint16_t sample_count = 10);
    bool update(const dual_sensor_data_t& sensor_data);
    bool isCalibrationComplete() const;
    float getProgress() const;
    calibration_state_t getState() const { return _calibration_state; }
    
    //===========================================================================================================
    //baselines
    //===========================================================================================================
    const baseline_t& getBaseline() const { return _baseline; }
    void setBaseline(const baseline_t& baseline) { _baseline = baseline; _valid = baseline.valid; }
    
    //===========================================================================================================
    //save
    //===========================================================================================================
    bool saveToEEPROM(uint16_t eeprom_offset = 0);
    bool loadFromEEPROM(uint16_t eeprom_offset = 0);
    
    // Reset calibration
    void reset();
    bool isValid() const { return _valid && _baseline.valid; }

private:
    //state machine
    calibration_state_t _calibration_state;
    baseline_t _baseline;
    bool _valid;
    
    //nuffers
    float _temp1_samples[64];
    float _hum1_samples[64];
    float _pres1_samples[64];
    float _gas1_step_samples[BME688_NUM_HEATER_STEPS][64];
    
    float _temp2_samples[64];
    float _hum2_samples[64];
    float _pres2_samples[64];
    float _gas2_step_samples[BME688_NUM_HEATER_STEPS][64];
    
    uint16_t _sample_index;
    uint16_t _target_samples;
    uint32_t _calibration_start_time;
    
    float computeMedian(float* samples, uint16_t count);
};

#endif
