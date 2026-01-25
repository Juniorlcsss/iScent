#include "bme688_handler.h"
#include <LittleFS.h>

#define CALIBRATION_FILE "/calibration.dat"
#define SCAN_PROGRESS_GUARD_MS 500U
#define SCAN_EXTRA_TIMEOUT_MS 200U

BME688Handler::BME688Handler():
    _wire_primary(nullptr),
    _wire_secondary(nullptr),
    _current_mode(MODE_PARALLEL),
    _primary_ready(false),
    _secondary_ready(false),
    _isSleeping(false),
    _isCalibrating(false),
    _sample_count(0),
    _calibration_samples(0),
    _calibration_temp_sum_p(0),
    _last_error(ERROR_NONE),
    _has_saved_profile(false),
    _scan_active(false)
{
    memset(&_calibration_data, 0 , sizeof(_calibration_data));
    memset(&_primary_stats, 0, sizeof(_primary_stats));
    memset(&_secondary_stats, 0, sizeof(_secondary_stats));
    memset(&_current_profile, 0, sizeof(_current_profile));

    resetScanContext(_primary_scan_ctx);
    resetScanContext(_secondary_scan_ctx);
    memset(&_scan_buffer, 0, sizeof(_scan_buffer));

    //init calib
    _calibration_temp_sum_p = _calibration_temp_sum_s = 0.0f;
    _calibration_hum_sum_p = _calibration_hum_sum_s = 0.0f;
    _calibration_pres_sum_p = _calibration_pres_sum_s = 0.0f;
    memset(_calibration_gas_sum_p,0,sizeof(_calibration_gas_sum_p));
    memset(_calibration_gas_sum_s,0,sizeof(_calibration_gas_sum_s));
}

BME688Handler::~BME688Handler() {
    sleep();
}

//===========================================================================================================
//init
//===========================================================================================================

bool BME688Handler::begin(TwoWire *primary, TwoWire *secondary){
    _wire_primary = primary;
    _wire_secondary = secondary;

    //init primary
    _primary_ready = initSensor(_sensor_primary, _wire_primary, BME688_PRIMARY_ADDR);
    if(!_primary_ready){
        DEBUG_PRINTLN("[BME688] Primary sensor init failed");
        _last_error = ERROR_PRIMARY_SENSOR_INIT;
    }

    //init secondary
    _secondary_ready = initSensor(_sensor_secondary, _wire_secondary, BME688_SECONDARY_ADDR);
    if(!_secondary_ready){
        DEBUG_PRINTLN("[BME688] Secondary sensor init failed");
        if(_last_error == ERROR_NONE){
            _last_error = ERROR_SECONDARY_SENSOR_INIT;
        } else {
            _last_error = ERROR_BOTH_SENSORS_INIT;
        }
    }

    applyHeaterOffProfile();
    _current_mode = MODE_FORCED;
    setOperatingMode(MODE_FORCED);

    //load
    loadCalibration();

    DEBUG_PRINTLN("[BME688] Initialization complete");
    DEBUG_PRINTLN("Primary: " + String(_primary_ready ? "OK" : "FAIL"));
    DEBUG_PRINTLN("Secondary: " + String(_secondary_ready ? "OK" : "FAIL"));

    return _primary_ready ||  _secondary_ready;
}

bool BME688Handler::beginSingle(TwoWire *wire, uint8_t addr){
    _wire_primary = wire;
    _primary_ready = initSensor(_sensor_primary, _wire_primary, addr);
    if(!_primary_ready){
        DEBUG_PRINTLN("[BME688] Single sensor init failed");
        _last_error = ERROR_PRIMARY_SENSOR_INIT;
        return _primary_ready;
    }
    setDefaultHeaterProfile();
    loadCalibration();
    DEBUG_PRINTLN("[BME688] Single sensor initialization complete");
    return _primary_ready;
}

bool BME688Handler::initSensor(Bme68x &sensor, TwoWire *wire, uint8_t addr){
    sensor.begin(addr, *wire);

    if(sensor.checkStatus() == BME68X_ERROR){
        DEBUG_PRINTLN("[BME688] Sensor not responding at address " + String(addr, HEX));
        return false;
    }

    DEBUG_PRINTF("[BME688] Sensor found at address 0x%X\n", addr);
    return true;
}

bool BME688Handler::isReady() const {
    return _primary_ready || _secondary_ready;
}

bool BME688Handler::isPrimaryReady() const {
    return _primary_ready;
}

bool BME688Handler::isSecondaryReady() const {
    return _secondary_ready;
}

//===========================================================================================================
//CFG
//===========================================================================================================

void BME688Handler::applyHeaterOffProfile(){
    uint16_t zeroTemp[1] = {0};
    uint16_t zeroDur[1] = {0};

    if(_primary_ready){
        _sensor_primary.setHeaterProf(zeroTemp, zeroDur, 1);
        _sensor_primary.setOpMode(BME68X_FORCED_MODE);
    }
    if(_secondary_ready){
        _sensor_secondary.setHeaterProf(zeroTemp, zeroDur, 1);
        _sensor_secondary.setOpMode(BME68X_FORCED_MODE);
    }
}

bool BME688Handler::setHeaterProfile(const heater_profile_t &profile){
    memcpy(&_current_profile, &profile, sizeof(heater_profile_t));

    bool success = true;

    if(_primary_ready){
        _sensor_primary.setHeaterProf(_current_profile.temperatures, _current_profile.durations, _current_profile.steps);
        success &= (_sensor_primary.checkStatus() != BME68X_ERROR);
    }

    if(_secondary_ready){
        _sensor_secondary.setHeaterProf(_current_profile.temperatures, _current_profile.durations, _current_profile.steps);
        success &= (_sensor_secondary.checkStatus() != BME68X_ERROR);
    }

    DEBUG_PRINTLN("[BME688] Heater profile set: " + String(_current_profile.profile_name));
    return success;
}

bool BME688Handler::setDefaultHeaterProfile(){
    heater_profile_t default_profile;
    memcpy(default_profile.temperatures, DEFAULT_HEATER_TEMPERATURES, sizeof(DEFAULT_HEATER_TEMPERATURES));
    memcpy(default_profile.durations, DEFAULT_HEATER_DURATIONS, sizeof(DEFAULT_HEATER_DURATIONS));

    default_profile.steps = BME688_NUM_HEATER_STEPS;
    default_profile.profile_name = "Default";

    return setHeaterProfile(default_profile);
}

bool BME688Handler::setVOCHeaterProfile(){
    heater_profile_t profile;
    memcpy(profile.temperatures, VOC_HEATER_TEMPERATURES, sizeof(VOC_HEATER_TEMPERATURES));
    memcpy(profile.durations, DEFAULT_HEATER_DURATIONS, sizeof(DEFAULT_HEATER_DURATIONS));

    profile.steps = BME688_NUM_HEATER_STEPS;
    profile.profile_name = "VOC";

    return setHeaterProfile(profile);
}

bool BME688Handler::setFoodHeaterProfile(){
    heater_profile_t profile;
    memcpy(profile.temperatures, FOOD_HEATER_TEMPERATURES, sizeof(FOOD_HEATER_TEMPERATURES));
    memcpy(profile.durations, DEFAULT_HEATER_DURATIONS, sizeof(DEFAULT_HEATER_DURATIONS));

    profile.steps = BME688_NUM_HEATER_STEPS;
    profile.profile_name = "Food";

    return setHeaterProfile(profile);
}

bool BME688Handler::setCustomHeaterProfile(const uint16_t *temperatures, const uint16_t *durations, uint8_t steps){
    if(steps == 0 || steps > BME688_NUM_HEATER_STEPS){
        DEBUG_PRINTLN("[BME688] steps out of range, setting to max");
        steps = BME688_NUM_HEATER_STEPS;
    }

    heater_profile_t profile;
    memcpy(profile.temperatures, temperatures, steps * sizeof(uint16_t));
    memcpy(profile.durations, durations, steps * sizeof(uint16_t));

    profile.steps = steps;
    profile.profile_name = "Custom";

    return setHeaterProfile(profile);
}

bool BME688Handler::setOperatingMode(bme688_mode_t mode){
    _current_mode = mode;

    bool success =true;

    switch(mode){

        case MODE_FORCED:
            DEBUG_PRINTLN("[BME688] Operating mode set to FORCED");
            if(_primary_ready){
                success &= configureForcedMode(_sensor_primary);
            }
            if(_secondary_ready){
                success &= configureForcedMode(_sensor_secondary);
            }
            break;

        case MODE_PARALLEL:
            DEBUG_PRINTLN("[BME688] Operating mode set to PARALLEL (overridden to FORCED heater-off for ambient)");
            mode = MODE_FORCED;
            _current_mode = MODE_FORCED;
            success &= setOperatingMode(MODE_FORCED);
            break;
        
        case MODE_SEQUENTIAL:
            DEBUG_PRINTLN("[BME688] Operating mode set to SEQUENTIAL");
            break;

        case MODE_SLEEP:
            DEBUG_PRINTLN("[BME688] Operating mode set to SLEEP");
            sleep();
            break;

        default:
            DEBUG_PRINTLN("[BME688] Unknown operating mode");
            success = false;
            break;
    }
    return success;
}

bme688_mode_t BME688Handler::getOperatingMode() const {
    return _current_mode;
}

bool BME688Handler::configureForcedMode(Bme68x &sensor){
    sensor.setTPH(BME68X_OS_2X, BME68X_OS_16X, BME68X_OS_1X);
    sensor.setOpMode(BME68X_FORCED_MODE);
    return !sensor.checkStatus();
}

bool BME688Handler::configureParallelMode(Bme68x &sensor){
    uint16_t zeroTemp[1] = {0};
    uint16_t zeroDur[1] = {0};
    sensor.setHeaterProf(zeroTemp, zeroDur, 1);
    sensor.setTPH(BME68X_OS_2X, BME68X_OS_16X, BME68X_OS_1X);
    sensor.setOpMode(BME68X_FORCED_MODE);
    return !sensor.checkStatus();
}

//===========================================================================================================
//data read
//===========================================================================================================

bool BME688Handler::startParallelMode(){
    bool success = true;
    if(_primary_ready){
        success &= configureParallelMode(_sensor_primary);
    }
    if(_secondary_ready){
        success &= configureParallelMode(_sensor_secondary);
    }
    return success;
}

bool BME688Handler::readParallelScan(dual_sensor_data_t &data){
    memset(&data,0,sizeof(data));
    data.timestamp = millis();

    bool primarySuccess = false;
    bool secondarySuccess = false;

    //primary
    if(_primary_ready){
        primarySuccess = readSensorScan(_sensor_primary, data.primary);
        if(!primarySuccess){
            DEBUG_PRINTLN("[BME688] Primary sensor read error");
        }
    }

    //secondary
    if(_secondary_ready){
        secondarySuccess = readSensorScan(_sensor_secondary, data.secondary);
        if(!secondarySuccess){
            DEBUG_PRINTLN("[BME688] Secondary sensor read error");
        }
    }

    //
    if(primarySuccess || secondarySuccess){
        if(primarySuccess && secondarySuccess){
            calculateDeltas(data);
        }
        else{
            data.delta_temp = data.delta_hum = data.delta_pres = data.delta_gas_avg = 0;
        }

        //apply any calibrations
        if(_calibration_data.calibrated){
            if(primarySuccess){
                applyCalibration(data.primary, true);
            }
            if(secondarySuccess){
                applyCalibration(data.secondary, false);
            }
        }

        if(primarySuccess || secondarySuccess){
            updateStats(data);
        }

        //accumuulate
        if(_isCalibrating){
            accumulateCalibrationData(data);
        }

        _sample_count++;
        data.valid = true;
    }
    else{
        _last_error = ERROR_SENSOR_READ;
    }
    return (primarySuccess || secondarySuccess);
}


bool BME688Handler::performFullScan(dual_sensor_data_t &data){
    memset(&data,0,sizeof(data));
    data.timestamp = millis();

    bool success = false;
    sensor_data_t singleData;

    if(_primary_ready && readSingleReading(_sensor_primary, singleData)){
        for(uint8_t i=0; i< BME688_NUM_HEATER_STEPS; i++){
            data.primary.temperatures[i] = singleData.temperature;
            data.primary.humidities[i] = singleData.humidity;
            data.primary.pressures[i] = singleData.pressure;
        }
        data.primary.validReadings = max<uint8_t>(data.primary.validReadings, 1);
        data.primary.complete = (data.primary.validReadings > 0);
        success = true;
    }

    if(_secondary_ready && readSingleReading(_sensor_secondary, singleData)){
        for(uint8_t i=0; i< BME688_NUM_HEATER_STEPS; i++){
            data.secondary.temperatures[i] = singleData.temperature;
            data.secondary.humidities[i] = singleData.humidity;
            data.secondary.pressures[i] = singleData.pressure;
        }
        data.secondary.validReadings = max<uint8_t>(data.secondary.validReadings, 1);
        data.secondary.complete = (data.secondary.validReadings > 0);
        success = true;
    }

    if(success){
        calculateDeltas(data);

        //apply any calibrations
        if(_calibration_data.calibrated){
            if(data.primary.validReadings > 0){
                applyCalibration(data.primary, true);
            }
            if(data.secondary.validReadings > 0){
                applyCalibration(data.secondary, false);
            }
        }

        updateStats(data);

        if(_isCalibrating){
            accumulateCalibrationData(data);
        }

        _sample_count++;
        data.valid = true;
    }
    else{
        _last_error = ERROR_SENSOR_READ;
    }
    return success;
}

bool BME688Handler::readSensorScan(Bme68x &sensor, sensor_scan_t &scan){
    memset(&scan,0,sizeof(scan));
    scan.timestamp = millis();

    bool seen[BME688_NUM_HEATER_STEPS];
    memset(seen, 0, sizeof(seen));    

    uint8_t readings=0;

    //set parallel mode
    sensor.setOpMode(BME68X_PARALLEL_MODE);

    uint32_t dur = 0;

    for(uint8_t i=0; i<_current_profile.steps; i++){
        dur += _current_profile.durations[i];
    }
    if(dur ==0){
        dur = (uint32_t)_current_profile.steps * BME688_HEATER_DURATION;
    }

    const uint32_t perStepUs = sensor.getMeasDur(BME68X_PARALLEL_MODE);
    const uint32_t perStepMs = (perStepUs + 999U) / 1000U; 
    const uint32_t sweepMs = (perStepMs + 1U) * _current_profile.steps; 
    const uint32_t timeout = millis() + dur + sweepMs + SCAN_EXTRA_TIMEOUT_MS;

    uint32_t lastProgressMs = millis();
    while(scan.validReadings < _current_profile.steps && millis() < timeout){
        delayMicroseconds(perStepUs);

        readings = sensor.fetchData();

        for(uint8_t i=0; i< readings; i++){
            bme68xData data;
            sensor.getData(data);
            uint8_t idx = data.gas_index;

            if(idx < BME688_NUM_HEATER_STEPS && !seen[idx]){
                scan.temperatures[idx] = data.temperature;
                scan.humidities[idx] = data.humidity;
                scan.pressures[idx] = data.pressure / 100.0f; //hPa
                scan.gas_resistances[idx] = data.gas_resistance;
                seen[idx] = true;
                scan.validReadings++;
                lastProgressMs = millis();
            }
        }

        if((millis() - lastProgressMs) >= SCAN_PROGRESS_GUARD_MS){
            DEBUG_PRINTLN("[BME688] Scan stalled, exiting early");
            break;
        }
    }

    scan.complete = (scan.validReadings >= _current_profile.steps);
    DEBUG_PRINTF("[BME688] Sensor scan complete. Valid readings: %d/%d\n", scan.validReadings, _current_profile.steps);
    return scan.validReadings > 0;
}

bool BME688Handler::readPrimaryScan(sensor_scan_t &data){
    if(!_primary_ready){
        DEBUG_PRINTLN("[BME688] Primary sensor not ready");
        return false;
    }
    return readSensorScan(_sensor_primary, data);
}

bool BME688Handler::readSecondaryScan(sensor_scan_t &data){
    if(!_secondary_ready){
        DEBUG_PRINTLN("[BME688] Secondary sensor not ready");
        return false;
    }
    return readSensorScan(_sensor_secondary, data);
}

bool BME688Handler::readPrimary(sensor_data_t &data){
    if(!_primary_ready){
        DEBUG_PRINTLN("[BME688] Primary sensor not ready");
        return false;
    }
    return readSingleReading(_sensor_primary, data);
}

bool BME688Handler::readSecondary(sensor_data_t &data){
    if(!_secondary_ready){
        DEBUG_PRINTLN("[BME688] Secondary sensor not ready");
        return false;
    }
    return readSingleReading(_sensor_secondary, data);
}

bool BME688Handler::readSingleReading(Bme68x &sensor, sensor_data_t &data){
    memset(&data, 0, sizeof(data));
    data.timestamp = millis();

    //
    uint16_t zeroTemp[1] = {0};
    uint16_t zeroDur[1] = {0};

    //disable heater
    sensor.setHeaterProf(zeroTemp, zeroDur, 1);
    sensor.setTPH(BME68X_OS_2X, BME68X_OS_16X, BME68X_OS_1X);

    sensor.setOpMode(BME68X_FORCED_MODE);
    delayMicroseconds(sensor.getMeasDur(BME68X_FORCED_MODE));

    if(sensor.fetchData()){
        bme68xData bmeData;
        sensor.getData(bmeData);
        data.temperature = bmeData.temperature;
        data.pressure = bmeData.pressure / 100.0f; //hPa
        data.humidity = bmeData.humidity;
        data.gas_resistance = bmeData.gas_resistance;
        data.gas_index = bmeData.gas_index;
        data.status = bmeData.status;
        data.valid = true;
        return true;
    }
    _last_error = ERROR_SENSOR_READ;
    return false;
}

void BME688Handler::resetScanContext(scan_context_t &ctx){
    ctx.active = false;
    memset(ctx.seen, 0, sizeof(ctx.seen));
    ctx.expectedSteps = 0;
    ctx.timeoutMs = 0;
    ctx.startMs = 0;
    ctx.lastProgressMs = 0;
}

bool BME688Handler::beginNonBlockingScan(){
    if(!isReady()){
        _last_error = ERROR_SENSOR_READ;
        return false;
    }

    resetScanContext(_primary_scan_ctx);
    resetScanContext(_secondary_scan_ctx);
    memset(&_scan_buffer, 0, sizeof(_scan_buffer));
    _scan_buffer.timestamp = millis();

    const uint8_t steps = (_current_profile.steps > 0) ? _current_profile.steps : BME688_NUM_HEATER_STEPS;

    auto computeTimeout = [&](Bme68x &sensor){
        uint32_t heaterDur = 0;
        for(uint8_t i=0; i<steps; i++){
            heaterDur += _current_profile.durations[i];
        }
        if(heaterDur == 0){
            heaterDur = (uint32_t)steps * BME688_HEATER_DURATION;
        }

        //getMeasDur returns microseconds
        uint32_t perStepUs = sensor.getMeasDur(BME68X_PARALLEL_MODE);
        uint32_t perStepMs = (perStepUs + 999U) / 1000U; //ceil to ms
        uint32_t sweepMs = steps * (perStepMs + 10U);
        return millis() + heaterDur + sweepMs + SCAN_EXTRA_TIMEOUT_MS;
    };

    if(_primary_ready){
        resetScanContext(_primary_scan_ctx);
        _primary_scan_ctx.active = true;
        _primary_scan_ctx.expectedSteps = steps;
        _primary_scan_ctx.startMs = millis();
        _primary_scan_ctx.timeoutMs = computeTimeout(_sensor_primary);
        _primary_scan_ctx.lastProgressMs = _primary_scan_ctx.startMs;
        configureParallelMode(_sensor_primary);
        _sensor_primary.setOpMode(BME68X_PARALLEL_MODE);
    }

    if(_secondary_ready){
        resetScanContext(_secondary_scan_ctx);
        _secondary_scan_ctx.active = true;
        _secondary_scan_ctx.expectedSteps = steps;
        _secondary_scan_ctx.startMs = millis();
        _secondary_scan_ctx.timeoutMs = computeTimeout(_sensor_secondary);
        _secondary_scan_ctx.lastProgressMs = _secondary_scan_ctx.startMs;
        configureParallelMode(_sensor_secondary);
        _sensor_secondary.setOpMode(BME68X_PARALLEL_MODE);
    }

    _scan_active = _primary_scan_ctx.active || _secondary_scan_ctx.active;

    if(!_scan_active){
        _last_error = ERROR_SENSOR_READ;
    }

    return _scan_active;
}

bool BME688Handler::pollScanContext(Bme68x &sensor, sensor_scan_t &scan, scan_context_t &ctx){
    if(!ctx.active){
        return false;
    }

    uint8_t prevReadings = scan.validReadings;
    uint8_t readings = sensor.fetchData();

    for(uint8_t i=0; i<readings; i++){
        bme68xData data;
        sensor.getData(data);
        uint8_t idx = data.gas_index;

        if(idx < BME688_NUM_HEATER_STEPS && !ctx.seen[idx]){
            scan.temperatures[idx] = data.temperature;
            scan.humidities[idx] = data.humidity;
            scan.pressures[idx] = data.pressure / 100.0f;
            scan.gas_resistances[idx] = data.gas_resistance;
            ctx.seen[idx] = true;
            scan.validReadings++;
        }
    }

    if(scan.validReadings > prevReadings){
        ctx.lastProgressMs = millis();
    }

    uint32_t nowMs = millis();
    bool timedOut = nowMs >= ctx.timeoutMs;
    bool stalled = (ctx.lastProgressMs > 0) && ((nowMs - ctx.lastProgressMs) >= SCAN_PROGRESS_GUARD_MS);

    if((scan.validReadings >= ctx.expectedSteps) || timedOut || stalled){
        scan.complete = (scan.validReadings >= ctx.expectedSteps);
        ctx.active = false;
        return true;
    }

    return false;
}

bool BME688Handler::isScanActive() const {
    return _scan_active;
}

void BME688Handler::calculateDeltas(dual_sensor_data_t &data){
    uint8_t count = min(data.primary.validReadings, data.secondary.validReadings);
    if(count==0){
        data.delta_temp = 0;
        data.delta_hum = 0;
        data.delta_pres = 0;
        data.delta_gas_avg = 0;
        return;
    }

    float temp_p = 0, temp_s = 0;
    float hum_p = 0, hum_s = 0;
    float pres_p = 0, pres_s = 0;
    float gas_p = 0, gas_s = 0;

    for(uint8_t i=0; i< count; i++){
        temp_p += data.primary.temperatures[i];
        temp_s += data.secondary.temperatures[i];
        hum_p += data.primary.humidities[i];
        hum_s += data.secondary.humidities[i];
        pres_p += data.primary.pressures[i];
        pres_s += data.secondary.pressures[i];
        gas_p += data.primary.gas_resistances[i];
        gas_s += data.secondary.gas_resistances[i];
    }
    data.delta_temp = (temp_p -temp_s) / count;
    data.delta_hum = (hum_p - hum_s) / count;
    data.delta_pres = (pres_p - pres_s) / count;
    data.delta_gas_avg = (gas_p - gas_s) / count;
}

//===========================================================================================================
//calibration
//===========================================================================================================

bool BME688Handler::startCalibration(uint16_t samples){
    if(!isReady()){
        DEBUG_PRINTLN("[BME688] Cannot start calibration, sensors not ready");
        _last_error = ERROR_CALIBRATION_FAILED;
        return false;
    }

    _isCalibrating = true;
    _calibration_samples = samples;
    _calibration_collected = 0;

    //reset vals
    _calibration_temp_sum_p = _calibration_temp_sum_s = 0.0f;
    _calibration_hum_sum_p = _calibration_hum_sum_s = 0.0f;
    _calibration_pres_sum_p = _calibration_pres_sum_s = 0.0f;
    memset(_calibration_gas_sum_p,0,sizeof(_calibration_gas_sum_p));
    memset(_calibration_gas_sum_s,0,sizeof(_calibration_gas_sum_s));

    //use a shorter heater sweep
    if(!_has_saved_profile){
        memcpy(&_saved_profile, &_current_profile, sizeof(_current_profile));
        _has_saved_profile = true;
    }

    heater_profile_t fast_profile;
    uint8_t steps = (_current_profile.steps > 0) ? min<uint8_t>(_current_profile.steps, 5) : 5;
    memcpy(fast_profile.temperatures, _current_profile.temperatures, steps * sizeof(uint16_t));
    for(uint8_t i=0; i<steps; i++){
        fast_profile.durations[i] = 80;
    }
    fast_profile.steps = steps;
    fast_profile.profile_name = "CalibFast";
    setHeaterProfile(fast_profile);

    DEBUG_PRINTLN("[BME688] Calibration started");
    return true;
}

bool BME688Handler::isCalibrating() const {
    return _isCalibrating;
}

float BME688Handler::getCalibrationProgress() const {
    if(!_isCalibrating || _calibration_samples ==0){
        return _calibration_data.calibrated ? 1.0f : 0.0f;
    }
    return (float)_calibration_collected / _calibration_samples;
}

void BME688Handler::accumulateCalibrationData(const dual_sensor_data_t &data){
    if(!_isCalibrating){
        DEBUG_PRINTLN("[BME688] Not calibrating, cannot accumulate data");
        return;
    }

    uint8_t steps = _current_profile.steps;

    //primary
    if(data.primary.complete){
        for(uint8_t i=0; i<steps; i++){
            _calibration_temp_sum_p += data.primary.temperatures[i];
            _calibration_hum_sum_p += data.primary.humidities[i];
            _calibration_pres_sum_p += data.primary.pressures[i];
            _calibration_gas_sum_p[i] += data.primary.gas_resistances[i];
        }
    }

    //secondary
    if(data.secondary.complete){
        for(uint8_t i=0; i<steps; i++){
            _calibration_temp_sum_s += data.secondary.temperatures[i];
            _calibration_hum_sum_s += data.secondary.humidities[i];
            _calibration_pres_sum_s += data.secondary.pressures[i];
            _calibration_gas_sum_s[i] += data.secondary.gas_resistances[i];
        }
    }

    _calibration_collected++;

    if(_calibration_collected >= _calibration_samples){
        finishCalibration();
    }
}

bool BME688Handler::finishCalibration(){
    if(!_isCalibrating || _calibration_collected ==0){
        DEBUG_PRINTLN("[BME688] Not calibrating, cannot finish calibration");
        return false;
    }

    uint8_t steps = _current_profile.steps;
    float n = (float)_calibration_collected * steps;

    //calculate offsets
    _calibration_data.temp_offset_primary = (_calibration_temp_sum_p / n) - TEMPERATURE_BASELINE;
    _calibration_data.temp_offset_secondary = (_calibration_temp_sum_s / n) - TEMPERATURE_BASELINE;
    _calibration_data.humidity_offset_primary = (_calibration_hum_sum_p / n) - HUMIDITY_BASELINE;
    _calibration_data.humidity_offset_secondary = (_calibration_hum_sum_s / n) - HUMIDITY_BASELINE;
    _calibration_data.pressure_offset_primary = 0;
    _calibration_data.pressure_offset_secondary = 0;

    //gas baselines
    float samples = (float)_calibration_collected;
    for(uint8_t i=0; i<steps; i++){
        _calibration_data.gas_baseline_primary[i] = _calibration_gas_sum_p[i] / samples;
        _calibration_data.gas_baseline_secondary[i] = _calibration_gas_sum_s[i] / samples;
    }

    _calibration_data.calibrated = true;
    _calibration_data.timestamp = millis();
    _isCalibrating = false;
    DEBUG_PRINTLN("[BME688] Calibration finished");
    printCalibrationData();
    saveCalibration();

    if(_has_saved_profile){
        setHeaterProfile(_saved_profile);
        _has_saved_profile = false;
    }
    return true;
}

void BME688Handler::applyCalibration(sensor_scan_t &scan, bool isPrimary){
    if(!_calibration_data.calibrated){
        DEBUG_PRINTLN("[BME688] No calibration data to apply");
        return;
    }

    float tempOffset = isPrimary ? _calibration_data.temp_offset_primary : _calibration_data.temp_offset_secondary;
    float humOffset = isPrimary ? _calibration_data.humidity_offset_primary : _calibration_data.humidity_offset_secondary;

    for(uint i=0; i<_current_profile.steps; i++){
        scan.temperatures[i] -= tempOffset;
        scan.humidities[i] -= humOffset;
    }
}

sensor_calibration_t BME688Handler::getCalibrationData() const {
    return _calibration_data;
}

void BME688Handler::setCalibrationData(const sensor_calibration_t &calib){
    memcpy(&_calibration_data, &calib, sizeof(sensor_calibration_t));
}

void BME688Handler::clearCalibration(){
    memset(&_calibration_data, 0, sizeof(_calibration_data));
    _calibration_data.calibrated = false;
    DEBUG_PRINTLN("[BME688] Calibration data cleared");
}

bool BME688Handler::saveCalibration(){
    if(!LittleFS.begin()){
        DEBUG_PRINTLN("[BME688] LittleFS mount failed, cannot save calibration");
        return false;
    }
    File file = LittleFS.open(CALIBRATION_FILE, "w");
    if(!file){
        DEBUG_PRINTLN("[BME688] Cannot open calibration file for writing");
        LittleFS.end();
        return false;
    }

    size_t written = file.write((uint8_t*)&_calibration_data, sizeof(_calibration_data));
    file.close();
    LittleFS.end();
    if(written != sizeof(_calibration_data)){
        DEBUG_PRINTLN("[BME688] Calibration data write failed");
        return false;
    }
    DEBUG_PRINTLN("[BME688] Calibration data saved");
    return true;
}

bool BME688Handler::loadCalibration(){
    if(!LittleFS.begin()){
        DEBUG_PRINTLN("[BME688] LittleFS mount failed, cannot load calibration");
        return false;
    }

    if(!LittleFS.exists(CALIBRATION_FILE)){
        DEBUG_PRINTLN("[BME688] Calibration file does not exist");
        LittleFS.end();
        return false;
    }

    File f = LittleFS.open(CALIBRATION_FILE, "r");
    if(!f){
        DEBUG_PRINTLN("[BME688] Cannot open calibration file for reading");
        LittleFS.end();
        return false;
    }

    size_t read = f.read((uint8_t*)&_calibration_data, sizeof(_calibration_data));
    f.close();

    if(read != sizeof(_calibration_data) && !_calibration_data.calibrated){
        DEBUG_PRINTLN("[BME688] Calibration data read failed or incomplete");
        clearCalibration();
        LittleFS.end();
        return false;
    }
    LittleFS.end();
    DEBUG_PRINTLN("[BME688] Calibration data loaded");
    return true;
}

//===========================================================================================================
//data
//===========================================================================================================

float BME688Handler::calculateIAQ(float gasRes, float hum){
    //scale: 0-500
    /*
    
    */

    float gScore = 0.0f;
    float hScore = 0.0f;

    //gas score contribution
    if(gasRes >= 50000.0f){
        gScore = 0.0f;
    }
    else if(gasRes <= 5000.0f){
        gScore = 375.0f;
    }
    else{
        gScore = 375 * (1.0f - (gasRes - 5000.0f) / 45000.0f);
    }

    //humidity score contribution
    if(hum >= 38.0f && hum <= 42.0f){
        hScore=0;
    }
    else if(hum < 38.0f){
        hScore = 125.0f * (38.0f - hum) / 38.0f;
    }
    else{
        hScore = 125.0f * (hum - 42.0f) / 58.0f;
    }
    float rawIAQ = CONSTRAIN_FLOAT(gScore + hScore, 0.0f, 500.0f);
    float daqi = CONSTRAIN_FLOAT(1.0f + (rawIAQ / 500.0f) * 9.0f, 1.0f, 10.0f);

    return daqi;
}

float BME688Handler::getGasRatio(float current, float base){
    if(base <= 0.0f){
        return 1.0f;
    }
    return current / base;
}

float BME688Handler::compensateHumidity(float gas, float hum, float humBase){
    float factor = 1.0f + ((hum - humBase) * 0.01f);
    return gas * factor;
}

float BME688Handler::compensateTemperature(float gas, float temp, float base){
    float factor = 1.0f + ((temp - base) * 0.02f);
    return gas * factor;
}

bool BME688Handler::detectAnomaly(float current, float base, float thresh){
    if(base <=0){
        return false;
    }

    float ratio = current / base;
    return (ratio < (1.0f - thresh)) || (ratio > (1.0f + thresh));

}

//===========================================================================================================
//stats
//===========================================================================================================

uint32_t BME688Handler::getSampleCount() const{
    return _sample_count;
}

void BME688Handler::resetSampleCount() {
    _sample_count = 0;
}

void BME688Handler::updateStats(const dual_sensor_data_t &data){
    float a = 0.1f;

    //primary
    if(_primary_stats.samples == 0){
        //init
        _primary_stats.temp_mean = data.primary.temperatures[0];
        _primary_stats.hum_mean = data.primary.humidities[0];
        _primary_stats.pres_mean = data.primary.pressures[0];

        for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
            _primary_stats.gas_mean[i] = data.primary.gas_resistances[i];
        }
    }
    else{
        _primary_stats.temp_mean = a * data.primary.temperatures[0] + (1-a) * _primary_stats.temp_mean;
        _primary_stats.hum_mean = a * data.primary.humidities[0] + (1-a) * _primary_stats.hum_mean;
        _primary_stats.pres_mean = a * data.primary.pressures[0] + (1-a) * _primary_stats.pres_mean;   
        for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
            _primary_stats.gas_mean[i] = a * data.primary.gas_resistances[i] + (1-a) * _primary_stats.gas_mean[i];

        }
    }

    _primary_stats.samples++;

    //secondary
    if(data.secondary.complete && _secondary_stats.samples == 0){
        //init
        _secondary_stats.temp_mean = data.secondary.temperatures[0];
        _secondary_stats.hum_mean = data.secondary.humidities[0];
        _secondary_stats.pres_mean = data.secondary.pressures[0];
        for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
            _secondary_stats.gas_mean[i] = data.secondary.gas_resistances[i];
        }
        _secondary_stats.samples++;
    }
    else if(data.secondary.complete){
        _secondary_stats.temp_mean = a * data.secondary.temperatures[0] + (1-a) * _secondary_stats.temp_mean;
        _secondary_stats.hum_mean = a * data.secondary.humidities[0] + (1-a) * _secondary_stats.hum_mean;
        _secondary_stats.pres_mean = a * data.secondary.pressures[0] + (1-a) * _secondary_stats.pres_mean;   
        for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
            _secondary_stats.gas_mean[i] = a * data.secondary.gas_resistances[i] + (1-a) * _secondary_stats.gas_mean[i];

        }
        _secondary_stats.samples++;
    }
    
}

sensor_stats_t BME688Handler::getPrimaryStats() const {
    return _primary_stats;
}

sensor_stats_t BME688Handler::getSecondaryStats() const {
    return _secondary_stats;
}


//===========================================================================================================
//utility
//===========================================================================================================

void BME688Handler::sleep(){
    if(_primary_ready){
        _sensor_primary.setOpMode(BME68X_SLEEP_MODE);
    }
    if(_secondary_ready){
        _sensor_secondary.setOpMode(BME68X_SLEEP_MODE);
    }
    _isSleeping = true;
    DEBUG_PRINTLN("[BME688] Sensors put to sleep");
}

void BME688Handler::wake(){
    _isSleeping = false;
    setOperatingMode(_current_mode);
    DEBUG_PRINTLN("[BME688] Sensors woken up");
}

bool BME688Handler::isSleeping() const {
    return _isSleeping;
}

void BME688Handler::printSensorData(){
    DEBUG_PRINTLN("---BME688 SENSOR INFO---");
    DEBUG_PRINTF("Primary Sensor: %s\n", _primary_ready ? "CONNECTED" : "NOT READY");
    DEBUG_PRINTF("Secondary Sensor: %s\n", _secondary_ready ? "CONNECTED" : "NOT READY");
    DEBUG_PRINTF("Operating Mode: %d\n", _current_mode);
    DEBUG_PRINTF("Heater Profile: %s\n", _current_profile.profile_name);
    DEBUG_PRINTF("STEPS: %d\n", _current_profile.steps);
    DEBUG_PRINTF("Samples Collected: %d\n", _sample_count);
    DEBUG_PRINTF("Calibrated: %s\n", _calibration_data.calibrated ? "YES" : "NO");
    DEBUG_PRINTLN("------------------------");
}

void BME688Handler::printCalibrationData(){
    if(!_calibration_data.calibrated){
        DEBUG_PRINTLN("[BME688] No calibration data to print");
        return;
    }

    DEBUG_PRINTLN("---BME688 CALIBRATION DATA---");
    DEBUG_PRINTF("Temp Offset Primary: %.2f C\n", _calibration_data.temp_offset_primary);
    DEBUG_PRINTF("Temp Offset Secondary: %.2f C\n", _calibration_data.temp_offset_secondary);
    DEBUG_PRINTF("Humidity Offset Primary: %.2f %%\n", _calibration_data.humidity_offset_primary);
    DEBUG_PRINTF("Humidity Offset Secondary: %.2f %%\n", _calibration_data.humidity_offset_secondary);
    DEBUG_PRINTF("Pressure Offset Primary: %.2f hPa\n", _calibration_data.pressure_offset_primary);
    DEBUG_PRINTF("Pressure Offset Secondary: %.2f hPa\n", _calibration_data.pressure_offset_secondary);
    DEBUG_PRINTLN("Gas Baselines Primary:");
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
        DEBUG_PRINTF("  Step %d: %.2f Ohms\n", i, _calibration_data.gas_baseline_primary[i]);
    }
    DEBUG_PRINTLN("Gas Baselines Secondary:");
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
        DEBUG_PRINTF("  Step %d: %.2f Ohms\n", i, _calibration_data.gas_baseline_secondary[i]);
    }
}

String BME688Handler::getScanJSON(const dual_sensor_data_t &data){
    //start of string
    String json = "{";

    //timestamp
    json += "\"timestamp\":" + String(data.timestamp) + ",";
    json += "\"valid\":" + String(data.valid ? "true" : "false") + ",";

    //primary
    json += "\"primary\":{";
    json += "\"temps\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.primary.temperatures[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"humidity\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.primary.humidities[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"pressures\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.primary.pressures[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"gas_resistances\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.primary.gas_resistances[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "]";
    json += "\"delta_temp\":" + String(data.delta_temp,2) + ",";
    json += "\"delta_hum\":" + String(data.delta_hum,2) + ",";
    json += "\"delta_pres\":" + String(data.delta_pres,2) + ",";
    json += "\"delta_gas_avg\":" + String(data.delta_gas_avg,2);
    json += "},";

    //secondary
    json += "\"secondary\":{";
    json += "\"temps\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.secondary.temperatures[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"humidity\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.secondary.humidities[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"pressures\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.secondary.pressures[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "],";
    json += "\"gas_resistances\":[";
    for(uint8_t i=0; i<_current_profile.steps;i++){
        json += String(data.secondary.gas_resistances[i], 2);
        if(i < _current_profile.steps -1){
            json += ",";
        }
    }
    json += "]";
    json += "\"delta_temp\":" + String(data.delta_temp,2) + ",";
    json += "\"delta_hum\":" + String(data.delta_hum,2) + ",";
    json += "\"delta_pres\":" + String(data.delta_pres,2) + ",";
    json += "\"delta_gas_avg\":" + String(data.delta_gas_avg,2);
    json += "},";

    return json;
}

//===========================================================================================================
//error
//===========================================================================================================

error_code_t BME688Handler::getLastError() const{
    return _last_error;
}

const char* BME688Handler::getLastErrorString() const{
    if(_last_error < ARRAY_SIZE(ERROR_CODE_NAMES)){
        return ERROR_CODE_NAMES[_last_error];
    }
    return "Invalid Error Code";
}

void BME688Handler::clearError(){
    _last_error = ERROR_NONE;
}

//===========================================================================================================
//calculate gas ratio
//===========================================================================================================

uint16_t calculateIAQIndex(float gas, float hum){
    return (uint16_t)BME688Handler::calculateIAQ(gas, hum);
}

bool detectVOCs(float current, float base, float threshold){
    if(base <=0) return false;

    float ratio = current / base;
    return ratio < (1.0f - threshold);
}

float estimateCO2EQ(float gas_resistance, float baseline){
    //simple estimation :PPP
    if(baseline <=0 || gas_resistance <=0) return 400.0f; //default return

    return 400.0f + (baseline / gas_resistance) * 2000.0f;
}

const char* getAirQualityString(float iaq){
    //Base on check-air-quality.service.gov.uk
    if(iaq >= 1.0f && iaq <= 3.0f){
        return "Low";
    }
    else if(iaq > 3.0f && iaq <= 6.0f){
        return "Moderate";
    }
    else if(iaq > 6.0f && iaq <= 9.0f){
        return "High";
    }
    else if(iaq > 9.0f){
        return "Very High";
    }
    return "Unknown";
}