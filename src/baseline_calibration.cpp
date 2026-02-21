#include "baseline_calibration.h"
#include <algorithm>
#include <EEPROM.h>

BaselineCalibration::BaselineCalibration():_calibration_state(CALIB_STATE_IDLE),_sample_index(0), _target_samples(0),_valid(false) 
    {
    memset(&_baseline, 0, sizeof(baseline_t));
    memset(_temp1_samples, 0, sizeof(_temp1_samples));
    memset(_hum1_samples, 0, sizeof(_hum1_samples));
    memset(_pres1_samples, 0, sizeof(_pres1_samples));
    memset(_gas1_step_samples,0,sizeof(_gas1_step_samples));
    memset(_temp2_samples, 0, sizeof(_temp2_samples));
    memset(_hum2_samples, 0, sizeof(_hum2_samples));
    memset(_pres2_samples, 0, sizeof(_pres2_samples));
    memset(_gas2_step_samples, 0, sizeof(_gas2_step_samples));

    _baseline.valid = false;
}

BaselineCalibration::~BaselineCalibration() {
}

bool BaselineCalibration::startCalibration(uint16_t sample_count) {
    if (sample_count > 64) {
        sample_count = 64;
    }
    
    _target_samples = sample_count;
    _sample_index = 0;
    _calibration_state = CALIB_STATE_COLLECTING;
    _calibration_start_time = millis();

    //clear
    memset(_temp1_samples, 0, sizeof(_temp1_samples));
    memset(_hum1_samples, 0, sizeof(_hum1_samples));
    memset(_pres1_samples, 0, sizeof(_pres1_samples));
    memset(_gas1_step_samples,0,sizeof(_gas1_step_samples));
    memset(_temp2_samples, 0, sizeof(_temp2_samples));
    memset(_hum2_samples, 0, sizeof(_hum2_samples));
    memset(_pres2_samples, 0, sizeof(_pres2_samples));
    memset(_gas2_step_samples, 0, sizeof(_gas2_step_samples));
    return true;
}

bool BaselineCalibration::update(const dual_sensor_data_t& sensor_data) {
    if (_calibration_state != CALIB_STATE_COLLECTING) {
        return false;
    }
    
    if (_sample_index >= _target_samples) {
        //got enough
        return false; 
    }
    
    //store
    _temp1_samples[_sample_index] = sensor_data.primary.temperatures[0];
    _hum1_samples[_sample_index] = sensor_data.primary.humidities[0];
    _pres1_samples[_sample_index] = sensor_data.primary.pressures[0];
    _temp2_samples[_sample_index] = sensor_data.secondary.temperatures[0];
    _hum2_samples[_sample_index] = sensor_data.secondary.humidities[0];
    _pres2_samples[_sample_index] = sensor_data.secondary.pressures[0];

    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS;i++){
        _gas1_step_samples[i][_sample_index]= sensor_data.primary.gas_resistances[i];
        _gas2_step_samples[i][_sample_index]= sensor_data.secondary.gas_resistances[i];
    }

    _sample_index++;
    
    //check
    if (_sample_index >= _target_samples) {
        //compute median
        _baseline.temp1_baseline=computeMedian(_temp1_samples,_target_samples);
        _baseline.hum1_baseline=computeMedian(_hum1_samples,_target_samples);
        _baseline.pres1_baseline=computeMedian(_pres1_samples, _target_samples);
        _baseline.temp2_baseline=computeMedian(_temp2_samples, _target_samples);
        _baseline.hum2_baseline=computeMedian(_hum2_samples, _target_samples);
        _baseline.pres2_baseline=computeMedian(_pres2_samples, _target_samples);

        //per step gas baselines
        float gas1_total=0.0f, gas2_total=0.0f;
        for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
            _baseline.gas1_step_baselines[i]= computeMedian(_gas1_step_samples[i], _target_samples);
            _baseline.gas2_step_baselines[i]= computeMedian(_gas2_step_samples[i], _target_samples);

            //protect against invalid baselines
            if(_baseline.gas1_step_baselines[i] < 1.0f){
                _baseline.gas1_step_baselines[i] = 1.0f;
            }
            if(_baseline.gas2_step_baselines[i] < 1.0f){
                _baseline.gas2_step_baselines[i] = 1.0f;
            }

            gas1_total += _baseline.gas1_step_baselines[i];
            gas2_total += _baseline.gas2_step_baselines[i];
        }

        //legacy fields
        _baseline.gas1_baseline = gas1_total/BME688_NUM_HEATER_STEPS;
        _baseline.gas2_baseline = gas2_total /BME688_NUM_HEATER_STEPS;
        
        _baseline.calibration_time = _calibration_start_time;
        _baseline.valid = true;
        _valid = true;
        _calibration_state = CALIB_STATE_COMPLETE;

        DEBUG_PRINTLN(F("[BaselineCalibration] Calibration complete"));

        return true;
    }
    
    return false;
}

bool BaselineCalibration::isCalibrationComplete() const {
    return _calibration_state == CALIB_STATE_COMPLETE;
}

float BaselineCalibration::getProgress() const {
    if (_target_samples == 0) return 0.0f;
    return (float)_sample_index / (float)_target_samples;
}

float BaselineCalibration::computeMedian(float* samples, uint16_t count) {
    if (count == 0){return 0.0f;}
    if (count == 1) return samples[0];
    
    //copy sort
    float temp[64];
    memcpy(temp,samples, count*sizeof(float));
    std::sort(temp,temp +count);
    
    // Return median
    if (count % 2 == 0) {
        return (temp[count/2 - 1] + temp[count/2])/2.0f;
    } else {
        return temp[count/2 ];
    }
}

void BaselineCalibration::reset() {
    _calibration_state =CALIB_STATE_IDLE;
    _sample_index=0;
    _target_samples=0;
    _valid=false;
    memset(&_baseline, 0, sizeof(baseline_t));
    _baseline.valid=false;
}

bool BaselineCalibration::saveToEEPROM(uint16_t eeprom_offset) {
    uint8_t* ptr = (uint8_t*)&_baseline;
    for (size_t i = 0; i<sizeof(baseline_t); i++) {
        EEPROM.write(eeprom_offset + i, ptr[i]);
    }


    EEPROM.commit();
    return true;
}

bool BaselineCalibration::loadFromEEPROM(uint16_t eeprom_offset) {
    //read byte by byte and reconstruct baseline
    uint8_t* ptr = (uint8_t*)&_baseline;
    for (size_t i = 0;i<sizeof(baseline_t);i++) {
        ptr[i] = EEPROM.read(eeprom_offset + i);
    }
    _valid = _baseline.valid;
    return _baseline.valid;
}
