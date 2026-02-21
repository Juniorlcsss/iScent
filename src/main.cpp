#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "bme688_handler.h"
#include "ml_inference.h"
#include "display_handler.h"
#include "data_logger.h"
#include "button_handler.h"
#include "ble_handler.h"

//===========================================================================================================
//global vars
//===========================================================================================================
BLEHandler ble;
BME688Handler sensors;
ButtonHandler buttons;
DataLogger logger;
DisplayHandler display;
MLInference ml;

//states
system_state_t currentState = STATE_INIT;
error_code_t lastError = ERROR_NONE;

uint32_t lastSampleTime = 0;
uint32_t lastInferenceTime = 0;
uint32_t lastDisplayUpdateTime = 0;
uint32_t warmupStartTime = 0;
uint32_t lastCalibDebugTime = 0;
float lastCalibLoggedProgress = -1.0f;
uint32_t lastCalibProgressMs = 0;
float lastCalibProgressVal = -1.0f;
uint32_t calibStartTime = 0;
const uint32_t CALIB_TIMEOUT_MARGIN_MS = 60000UL;
const uint32_t CALIB_MAX_DURATION_MS = (uint32_t)(BME688_GAS_BASE_SAMPLES * (BME688_SAMPLE_RATE + 500UL)) + CALIB_TIMEOUT_MARGIN_MS;

dual_sensor_data_t currentSensorData;
ml_prediction_t currentPrediction;
inference_mode_t currentInferenceMode = INFERENCE_MODE_SINGLE;

bool loggingActive = false;
bool continuousInference = true;
bool collectingLabeled = false;
int16_t currentLabelSelection = SCENT_CLASS_PURE_CAMOMILE;
uint32_t lastLoggingRetryMs = 0;
const uint32_t LOGGING_RETRY_INTERVAL_MS = 3000;
uint8_t predictionSelection = 0;
bool temporalInProgress=false;
uint32_t temporalStartTime=0;

//===========================================================================================================
//functions
//===========================================================================================================
void initialiseSystem();
void handleStateMachine();
void handleButtons();
void handleBLE();
void updateDisplay();
bool performSampling();
void performInference();
bool performTemporalEnsembleInference();
void enterState(system_state_t newState);
void handleError(error_code_t error);
void buttonCallback(button_id_t buttonId, button_event_t event);
void printStatus();
void ensureLoggingActive();
bool isMenuNavigationActive();

//menu actions
void menuActionShowStatus();
void menuActionShowSensorData();
void menuActionShowPrediction();
void menuActionShowGraph();
void menuActionShowError();
void menuActionDataCollection();
void menuActionCalibrate();
void menuActionSettings();

//settings menu actions
void settingsActionCalibrate();
void settingsActionCycleHeaterProfile();
void settingsActionCycleThresholdProfile();
void settingsActionToggleDisplayProfile();
void settingsActionExportData();
void settingsActionWipeSD();
void settingsActionBackToMain();
void refreshSettingsMenu();
void applyHeaterProfile(uint8_t index);
void settingsActionCycleInferenceMode();

//data collection menu actions
void dataCollectActionCycleLabel();
void dataCollectActionToggle();
void dataCollectActionBack();
void refreshDataCollectionMenu();

//menu items
menu_item_t main_menu_items[] = {
    {"Status", menuActionShowStatus, DISPLAY_MODE_STATUS},
    {"Sensor Data", menuActionShowSensorData, DISPLAY_MODE_SENSOR_DATA},
    {"Prediction", menuActionShowPrediction, DISPLAY_MODE_PREDICTION},
    {"Gas Graph", menuActionShowGraph, DISPLAY_MODE_GRAPH},
    {"Data Collect", menuActionDataCollection, DISPLAY_MODE_DATA_COLLECTION},
    {"Calibrate", menuActionCalibrate, DISPLAY_MODE_CALIBRATION},
    {"Last Error", menuActionShowError, DISPLAY_MODE_ERROR},
    {"Settings", menuActionSettings, DISPLAY_MODE_SETTINGS}
};
const uint8_t MAIN_MENU_COUNT = sizeof(main_menu_items)  /sizeof(menu_item_t);

//settings menu labels (mutable for inline status)
static char settingsLabelCalibrate[] = "Calibrate";
static char settingsLabelHeater[24] = "Heater: Default";
static char settingsLabelThreshold[28] = "ML Thr: Default";
static char settingsLabelDisplay[24] = "Display: Normal";
static char settingsLabelExport[] = "Export Last Log";
static char settingsLabelWipe[] = "Wipe SD Card";
static char settingsLabelBack[] = "Back";
static char settingsLabelInfMode[28]="Infer: Single";

//data collection labels
static char collectLabelSelect[24] = "Label: pure camomile";
static char collectLabelToggle[24] = "Start Collect";
static char collectLabelBack[] = "Back";

//heater profile cycling
static const char* HEATER_PROFILE_NAMES[] = {"Default", "VOC", "Food"};
static const uint8_t HEATER_PROFILE_COUNT = sizeof(HEATER_PROFILE_NAMES) / sizeof(HEATER_PROFILE_NAMES[0]);
uint8_t currentHeaterProfileIndex = 0;

//ml threshold presets
typedef struct {
    const char* name;
    float confidence;
    float anomaly;
} threshold_profile_t;

static const threshold_profile_t THRESHOLD_PROFILES[] = {
    {"Default", ML_CONFIDENCE_THRESHOLD, ML_ANOMALY_THRESHOLD},
    {"Sensitive", 0.55f, 0.25f},
    {"Strict", 0.85f, 0.40f}
};
static const uint8_t THRESHOLD_PROFILE_COUNT = sizeof(THRESHOLD_PROFILES) / sizeof(THRESHOLD_PROFILES[0]);
uint8_t currentThresholdProfileIndex = 0;

//display profiles
typedef struct {
    const char* name;
    uint8_t brightness;
    bool inverted;
} display_profile_t;

static const display_profile_t DISPLAY_PROFILES[] = {
    {"Normal", DISPLAY_CONTRAST, false},
    {"Dim", (uint8_t)(DISPLAY_CONTRAST * 0.6f), false},
    {"Invert", DISPLAY_CONTRAST, true}
};
static const uint8_t DISPLAY_PROFILE_COUNT = sizeof(DISPLAY_PROFILES) / sizeof(DISPLAY_PROFILES[0]);
uint8_t currentDisplayProfileIndex = 0;

//settings menu definitions
menu_item_t settings_menu_items[] = {
    {settingsLabelCalibrate,    settingsActionCalibrate,               DISPLAY_MODE_CALIBRATION},
    {settingsLabelHeater,       settingsActionCycleHeaterProfile,      DISPLAY_MODE_SETTINGS},
    {settingsLabelThreshold,    settingsActionCycleThresholdProfile,   DISPLAY_MODE_SETTINGS},
    {settingsLabelInfMode,      settingsActionCycleInferenceMode,      DISPLAY_MODE_SETTINGS},
    {settingsLabelDisplay,      settingsActionToggleDisplayProfile,    DISPLAY_MODE_SETTINGS},
    {settingsLabelExport,       settingsActionExportData,              DISPLAY_MODE_SETTINGS},
    {settingsLabelWipe,         settingsActionWipeSD,                  DISPLAY_MODE_SETTINGS},
    {settingsLabelBack,         settingsActionBackToMain,              DISPLAY_MODE_MENU}
};
const uint8_t SETTINGS_MENU_COUNT = sizeof(settings_menu_items) / sizeof(menu_item_t);

menu_item_t collect_menu_items[] = {
    {collectLabelSelect, dataCollectActionCycleLabel, DISPLAY_MODE_DATA_COLLECTION},
    {collectLabelToggle, dataCollectActionToggle, DISPLAY_MODE_DATA_COLLECTION},
    {collectLabelBack, dataCollectActionBack, DISPLAY_MODE_MENU}
};
const uint8_t COLLECT_MENU_COUNT = sizeof(collect_menu_items) / sizeof(menu_item_t);

//===========================================================================================================
//setup
//===========================================================================================================


void setup(){
    Serial.begin(DEBUG_SERIAL_BAUD);
    delay(1000);

    
    DEBUG_PRINTLN(F("\n\n"));
    DEBUG_PRINTLN(F("==== iScent ===="));
    DEBUG_PRINTF("Firmware Version: %s\n", SOFTWARE_VERSION);


    initialiseSystem();
}

void ensureLoggingActive(){
    if(loggingActive) return;
    if(currentState == STATE_WARMUP) return;

    uint32_t now = millis();
    if(now - lastLoggingRetryMs < LOGGING_RETRY_INTERVAL_MS) return;
    lastLoggingRetryMs = now;

    if(logger.startLogging()){
        loggingActive = true;
        DEBUG_PRINTLN(F("[AutoLog] Logging active"));
    } else {
        loggingActive = false;
    }
}

//===========================================================================================================
//MAIN LOOP
//===========================================================================================================

void loop(){
    //update button states
    buttons.update();
    handleButtons();

    //handle ble
    handleBLE();

    //state machine
    handleStateMachine();

    //ensure logging
    ensureLoggingActive();

    //update display regularly
    if(millis() - lastDisplayUpdateTime >= DISPLAY_UPDATE_INTERVAL_MS){
        updateDisplay();
        lastDisplayUpdateTime = millis();
    }

    delay(1);
}

bool isMenuNavigationActive(){
    display_mode_t mode = display.getMode();
    return (mode == DISPLAY_MODE_MENU || mode == DISPLAY_MODE_SETTINGS || mode == DISPLAY_MODE_DATA_COLLECTION);
}

//===========================================================================================================
//init
//===========================================================================================================

void initialiseSystem(){
    currentState = STATE_INIT;

    //init i2c
    DEBUG_PRINTLN(F("[INIT] Initializing I2C..."));
    Wire.setSDA(I2C_SDA_PIN);
    Wire.setSCL(I2C_SCL_PIN);
    Wire.begin();
    Wire.setClock(I2C_FREQUENCY_HZ);

    //init display
    DEBUG_PRINTLN(F("[INIT] Initializing Display..."));
    if(!display.begin(&Wire)){
        DEBUG_PRINTLN(F("[ERROR] Display initialization failed!"));
        lastError = ERROR_DISPLAY_INIT;
    }

    display.showSplashScreen();
    delay(DISPLAY_SPLASH_DURATION_MS);


    //init buttons
    DEBUG_PRINTLN(F("[INIT] Initializing Buttons..."));
    buttons.begin();
    //buttons.setCallback(buttonCallback);


    //init sensors
    DEBUG_PRINTLN(F("[INIT] Initializing Sensors..."));
    display.showStatusScreen(STATE_INIT, lastError);
    if(!sensors.begin(&Wire, &Wire)){
        handleError(sensors.getLastError());
        
        if(!sensors.isReady()){
            DEBUG_PRINTLN(F("[ERROR] Sensor initialization failed!"));
            lastError = ERROR_BOTH_SENSORS_INIT;
            enterState(STATE_ERROR);
            return;
        }
    }
    sensors.printSensorData();
    sensors.setDefaultHeaterProfile();


    //init ML
    DEBUG_PRINTLN(F("[INIT] Initializing ML Inference..."));
    if(!ml.begin()){
        DEBUG_PRINTLN(F("[ERROR] ML Inference initialization failed!"));
        lastError = ERROR_ML_INIT;
    }
    ml.printModelInfo();

    
    //init data logger
    DEBUG_PRINTLN(F("[INIT] Initializing Data Logger..."));
    if(!logger.begin()){
        DEBUG_PRINTLN(F("[ERROR] Data Logger initialization failed!"));
        lastError = ERROR_FILE_SYSTEM_INIT;
    }

    //init ble
    DEBUG_PRINTLN(F("[INIT] Initializing BLE..."));
    if(!ble.begin()){
        DEBUG_PRINTLN(F("[ERROR] BLE initialization failed! Continuing without BLE."));
        lastError = ERROR_BLE_INIT;
    }
    else{
        ble.startAdvertising();
    }

    DEBUG_PRINTLN(F("[INIT] Starting sensor warm-up..."));
    enterState(STATE_WARMUP);
}

//===========================================================================================================
//warmup
//==========================================================================================================
float getWarmupProgress(){
    if(currentState!=STATE_WARMUP){
        return 1.0f;
    }

    uint32_t elapsed = millis() - warmupStartTime;
    float progress = (float)elapsed / (float)BME688_STABLE_MS;
    return constrain(progress, 0.0f, 1.0f);
}

uint32_t getWarmupRemainingSeconds(){
    if(currentState!=STATE_WARMUP){
        return 0;
    }
    uint32_t elapsed = millis() - warmupStartTime;
    if(elapsed >= BME688_STABLE_MS){
        return 0;
    }
    return (BME688_STABLE_MS - elapsed)/1000;
}

bool isSensorWarmedUp(){
    return (currentState != STATE_WARMUP)&&(millis()- warmupStartTime >= BME688_STABLE_MS||warmupStartTime== 0);
}

//===========================================================================================================
//state machine
//===========================================================================================================

void handleStateMachine(){
    switch(currentState){
        case STATE_INIT:
            DEBUG_PRINTLN(F("[ERROR] In STATE_INIT in main loop!"));
            break;
        
        case STATE_WARMUP:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                if(performSampling()){
                    lastSampleTime = millis();
                }
            }

            if(millis() - warmupStartTime >= BME688_STABLE_MS){
                DEBUG_PRINTLN(F("[STATE] Warmup complete. Sensors stabilized."));
                DEBUG_PRINTF("[STATE] Warmup took %lu ms, %lu samples collected\n",
                    millis() - warmupStartTime, sensors.getSampleCount());

                if(!sensors.getCalibrationData().calibrated){
                    DEBUG_PRINTLN(F("[STATE] No sensor calibration — starting calibration..."));
                    enterState(STATE_CALIBRATING);
                }
                else{
                    DEBUG_PRINTLN(F("[STATE] Calibration loaded. Ready."));
                    display.setMode(DISPLAY_MODE_MENU);
                    enterState(STATE_IDLE);
                }
            }
            break;

        case STATE_CALIBRATING:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                if(performSampling()){
                    lastSampleTime = millis();
                }
            }

            if(sensors.isCalibrating()){
                float prog = sensors.getCalibrationProgress();
                uint16_t collected = sensors.getCalibrationCollected();
                uint16_t target = sensors.getCalibrationTarget();
                uint32_t now = millis();
                
                if((prog - lastCalibLoggedProgress) >= 0.05f || (now - lastCalibDebugTime) >= 2000){
                    char buf[96];
                    snprintf(buf, sizeof(buf), "prog=%.3f collected=%u/%u", prog, collected, target);
                    logger.logCalibDebug(String(buf));
                    lastCalibLoggedProgress = prog;
                    lastCalibDebugTime = now;
                }

                if(prog > lastCalibProgressVal + 0.001f){
                    lastCalibProgressVal = prog;
                    lastCalibProgressMs = now;
                }
                else if(target > 0 && collected + 1 >= target && (now - lastCalibProgressMs) > 3000UL){
                    logger.logCalibDebug("CALIB_N_MINUS_ONE_FORCE_FINISH");
                    sensors.finishCalibration();
                }
                else if(prog >= 0.95f && (now - lastCalibProgressMs) > 12000UL){
                    logger.logCalibDebug("CALIB_NEAR_DONE_FORCE_FINISH");
                    sensors.finishCalibration();
                }
                else if(now - lastCalibProgressMs > 45000UL){
                    char buf[128];
                    snprintf(buf, sizeof(buf), "CALIB_STALL prog=%.3f dt=%lu", 
                        prog, (unsigned long)(now - lastCalibProgressMs));
                    logger.logCalibDebug(String(buf));
                    sensors.finishCalibration();
                    logger.flushCalibDebug();
                    enterState(STATE_IDLE);
                }

                if(calibStartTime > 0 && (now - calibStartTime) > CALIB_MAX_DURATION_MS){
                    logger.logCalibDebug("CALIB_TIMEOUT_SAVE");
                    sensors.finishCalibration();
                    logger.flushCalibDebug();
                    enterState(STATE_IDLE);
                }
            }

            if(!sensors.isCalibrating()){
                DEBUG_PRINTLN(F("[STATE] Calibration complete."));
                logger.logCalibDebug("CALIB_DONE");
                logger.flushCalibDebug();
                logger.flush();
                display.setMode(DISPLAY_MODE_MENU);
                enterState(STATE_IDLE);
            }
            break;

        case STATE_IDLE:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                enterState(STATE_SAMPLING);
            }
            break;

        case STATE_SAMPLING:
            if(performSampling()){
                lastSampleTime = millis();

                //move to inference if ready
                if(continuousInference && ml.isFeatureBufferReady()){
                    //base off mode
                    switch(ml.getInferenceMode()){
                        case INFERENCE_MODE_TEMPORAL:
                            enterState(STATE_TEMPORAL_COLLECTING);
                            break;
                        
                        case INFERENCE_MODE_ENSEMBLE:
                        case INFERENCE_MODE_SINGLE:
                        default:
                            enterState(STATE_INFERENCING);
                            break;
                    }
                }
                else
                {
                    enterState(STATE_IDLE);
                }
            }
            break;

        case STATE_INFERENCING:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                if(performSampling()){
                    lastSampleTime = millis();
                }
            }

            if(millis() - lastInferenceTime >= ML_INFERENCE_INTERVAL_MS){
                //dispatch
                if(ml.runActiveInference(currentPrediction)){
                    DEBUG_PRINTF("[INFERENCE] Predicted: %s (conf=%.2f, anomaly=%.3f)\n", ml.getClassName(currentPrediction.predictedClass), currentPrediction.confidence, currentPrediction.anomalyScore);
                
                    if(ble.isConnected()){
                        ble.sendPrediction(currentPrediction);
                    }
                    if(currentPrediction.isAnomalous){
                        DEBUG_PRINTLN(F("[INFERENCE] Anomaly detected!"));
                    }
                }
                lastInferenceTime=millis();
                enterState(STATE_IDLE);
            }
            break;

        case STATE_TEMPORAL_COLLECTING:
            if(millis()-lastSampleTime>=BME688_SAMPLE_RATE){
                if(performSampling()){
                    lastSampleTime = millis();
                }
            }

            {
                bool complete = ml.updateTemporalCollection(currentPrediction);

                if(complete){
                    if(ble.isConnected()){
                        ble.sendPrediction(currentPrediction);
                    }
                    if(currentPrediction.isAnomalous){
                        DEBUG_PRINTLN(F("[INFERENCE] Anomaly detected!"));
                    }
                    lastInferenceTime=millis();
                    enterState(STATE_IDLE);
                }
            }
            break;

        case STATE_SLEEP:
            //not used
            break;

        case STATE_ERROR:
            //stay here
            break;

        default:
            DEBUG_PRINTLN(F("[ERROR] Unknown state in state machine!"));
            enterState(STATE_ERROR);
            break;
    }
}

void enterState(system_state_t newState){
    if(newState == currentState) return;

    DEBUG_PRINTF("[STATE] Transitioning from %d to %d\n", currentState, newState);

    //exit acts
    switch(currentState){
        case STATE_CALIBRATING:
            sensors.finishCalibration();
            calibStartTime = 0;
            break;

        case STATE_LOGGING:
            logger.stopLogging();
            loggingActive = false;
            break;

        case STATE_SLEEP:
            sensors.wake();
            break;

        case STATE_TEMPORAL_COLLECTING:
            if(ml.isTemporalCollectionActive()){
                ml.cancelTemporalCollection();
            }
            break;
        
        default:
            break;
    }

    //entry acts
    switch(newState){
        case STATE_WARMUP:
            warmupStartTime = millis();
            display.setMode(DISPLAY_MODE_CALIBRATION);
            DEBUG_PRINTF("[STATE] Warmup started. Duration: %lu seconds\n",BME688_STABLE_MS / 1000);
            break;

        case STATE_IDLE:
            {
                display_mode_t mode = display.getMode();
                if(mode == DISPLAY_MODE_SPLASH || mode == DISPLAY_MODE_MENU || mode == DISPLAY_MODE_CALIBRATION){
                    display.setMode(DISPLAY_MODE_MENU);
                    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
                }
                
                if(!collectingLabeled && display.getMode() != DISPLAY_MODE_PREDICTION){
                    logger.setActiveLabel(LOG_LABEL_AMBIENT);
                }
                if(!logger.isLogging()){
                    logger.startLogging();
                    loggingActive = true;
                }
            }
            break;

        case STATE_CALIBRATING:
            logger.setActiveLabel(LOG_LABEL_CALIBRATION);
            if(!logger.isLogging()){
                logger.startLogging();
            }
            loggingActive = true;

            lastCalibLoggedProgress = -1.0f;
            lastCalibDebugTime = 0;
            lastCalibProgressMs = millis();
            lastCalibProgressVal = -1.0f;
            calibStartTime = millis();
            logger.startCalibDebug();
            logger.logCalibDebug("CALIB_START");
            sensors.startCalibration(BME688_GAS_BASE_SAMPLES);
            break;

        case STATE_TEMPORAL_COLLECTING:
            ml.startTemporalCollection();
            temporalStartTime = millis();
            temporalInProgress = true;

            break;

        case STATE_LOGGING:
            //logging auto
            break;

        case STATE_SLEEP:
            sensors.sleep();
            display.clear();
            display.refresh();
            break;

        case STATE_ERROR:
            display.ShowErrorScreen(lastError);
            break;

        default:
            break;
    }

    currentState = newState;
}

//===========================================================================================================
//sampling
//===========================================================================================================

bool performSampling(){
    const bool needsAmbientRead = (currentState != STATE_CALIBRATING);

    dual_sensor_data_t ambientData;
    bool ambientOk = true;

    if(needsAmbientRead){
        ambientOk = sensors.performFullScan(ambientData);
        if(!ambientOk && sensors.getLastError() != ERROR_NONE){
            lastError = ERROR_SENSOR_READ;
        }
    }

    bool parallelOk = sensors.readParallelScan(currentSensorData);
    if(!parallelOk && sensors.getLastError() != ERROR_NONE){
        lastError = ERROR_SENSOR_READ;
    }

    if(!ambientOk || !parallelOk){
        return false;
    }

    if(needsAmbientRead){
        currentSensorData.primary.temperatures[0]   = ambientData.primary.temperatures[0];
        currentSensorData.primary.humidities[0]     = ambientData.primary.humidities[0];
        currentSensorData.primary.pressures[0]      = ambientData.primary.pressures[0];
        currentSensorData.primary.validReadings     = ambientData.primary.validReadings;

        currentSensorData.secondary.temperatures[0] = ambientData.secondary.temperatures[0];
        currentSensorData.secondary.humidities[0]   = ambientData.secondary.humidities[0];
        currentSensorData.secondary.pressures[0]    = ambientData.secondary.pressures[0];
        currentSensorData.secondary.validReadings   = ambientData.secondary.validReadings;

        currentSensorData.delta_temp = ambientData.primary.temperatures[0] - ambientData.secondary.temperatures[0];
        currentSensorData.delta_hum  = ambientData.primary.humidities[0]   - ambientData.secondary.humidities[0];
        currentSensorData.delta_pres = ambientData.primary.pressures[0]    - ambientData.secondary.pressures[0];
    }

    ml.addToWindow(currentSensorData);

    if(display.getMode() == DISPLAY_MODE_PREDICTION){
        logger.setActiveLabel(LOG_LABEL_PRED);
    }
    if(loggingActive){
        logger.logEntry(currentSensorData, currentPrediction.valid ? &currentPrediction : nullptr);
    }

    if(ble.isConnected()){
        ble.sendSensorData(currentSensorData);
    }

    DEBUG_VERBOSE_PRINTF("[Sample] T:%.1f H:%.1f G:%.0f\n", 
        currentSensorData.primary.temperatures[0],
        currentSensorData.primary.humidities[0],
        currentSensorData.primary.gas_resistances[0]
    );

    return true;
}

void performInference(){
    if(ml.runInference(currentPrediction)){     
        DEBUG_PRINTF("[Inference] Model=%s Prediction: %s (%.2f), Anomaly: %.2f\n", 
            ml.getActiveModelName(),
            ml.getClassName(currentPrediction.predictedClass),
            currentPrediction.confidence * 100.0f,
            currentPrediction.anomalyScore
        );

        //send via BLE
        if(ble.isConnected()){
            ble.sendPrediction(currentPrediction);
        }

        //check for anomalous
        if(currentPrediction.isAnomalous){
            DEBUG_PRINTLN(F("[ALERT] Anomalous scent detected!"));
        }
    }
}

bool performTemporalEnsembleInference(){
    if(currentState==STATE_TEMPORAL_COLLECTING){
        return false;

    }

    enterState(STATE_TEMPORAL_COLLECTING);
    return true;
}

//===========================================================================================================
//button handling
//===========================================================================================================
void handleButtons(){
    if(buttons.getLastEvent(BUTTON_DOWN) != BUTTON_EVENT_NONE || buttons.getLastEvent(BUTTON_SELECT) != BUTTON_EVENT_NONE){
        display.resetTimeout();
    }

    //menu nav
    if(display.getMode() == DISPLAY_MODE_MENU){
        if(buttons.wasPressed(BUTTON_DOWN)){
            display.menuDown();
            display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
            updateDisplay();
        }

        if(buttons.wasLongPressed(BUTTON_DOWN)){
            display.menuUp();
            display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
            updateDisplay();
        }
        

        if(buttons.wasPressed(BUTTON_SELECT)){
            display.menuSelect();
            updateDisplay();
        }
        return;
    }

    //settings menu
    if(display.getMode() == DISPLAY_MODE_SETTINGS){
        if(buttons.wasPressed(BUTTON_DOWN)){
            display.menuDown();
            display.showMenu(settings_menu_items, SETTINGS_MENU_COUNT, display.getSelectedMenuIndex(), "Settings");
            updateDisplay();
        }

        if(buttons.wasLongPressed(BUTTON_DOWN)){
            display.menuUp();
            display.showMenu(settings_menu_items, SETTINGS_MENU_COUNT, display.getSelectedMenuIndex(), "Settings");
            updateDisplay();
        }

        if(buttons.wasPressed(BUTTON_SELECT)){
            display.menuSelect();
            updateDisplay();
        }

        if(buttons.wasLongPressed(BUTTON_SELECT)){
            display.setMode(DISPLAY_MODE_MENU);
            display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
            updateDisplay();
        }
        return;
    }

    //data collection menu
    if(display.getMode() == DISPLAY_MODE_DATA_COLLECTION){
        if(buttons.wasPressed(BUTTON_DOWN)){
            display.menuDown();
            display.showMenu(collect_menu_items, COLLECT_MENU_COUNT, display.getSelectedMenuIndex(), "Data Collect");
            updateDisplay();
        }

        if(buttons.wasLongPressed(BUTTON_DOWN)){
            display.menuUp();
            display.showMenu(collect_menu_items, COLLECT_MENU_COUNT, display.getSelectedMenuIndex(), "Data Collect");
            updateDisplay();
        }

        if(buttons.wasPressed(BUTTON_SELECT)){
            display.menuSelect();
            updateDisplay();
        }

        if(buttons.wasLongPressed(BUTTON_SELECT)){
            display.setMode(DISPLAY_MODE_MENU);
            display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
            updateDisplay();
        }
        return;
    }

    //prediction screen nav
    if(display.getMode() == DISPLAY_MODE_PREDICTION){
        if(buttons.wasPressed(BUTTON_DOWN)){
            predictionSelection = (predictionSelection + 1) % 3;
            display.setPredictionSelection(predictionSelection);
            updateDisplay();
            return;
        }

        if(buttons.wasPressed(BUTTON_SELECT) || buttons.wasLongPressed(BUTTON_SELECT)){
            switch(predictionSelection){
                case 0:{
                    currentPrediction.valid = false;
                    currentPrediction.confidence = 0.0f;
                    currentPrediction.predictedClass = SCENT_CLASS_UNKNOWN;
                    display.showPredictionScreen(currentPrediction, currentSensorData, ml.getActiveModelName());
                    display.refresh();
                    logger.setActiveLabel(LOG_LABEL_PRED);

                    if(ml.getInferenceMode()==INFERENCE_MODE_TEMPORAL){
                        performTemporalEnsembleInference();
                    }
                    else{
                        ml.runActiveInference(currentPrediction);
                        updateDisplay();
                    }
                    break;
                }
                case 1:{
                    //cycle
                    if(ml.getInferenceMode()==INFERENCE_MODE_SINGLE){
                        ml.nextModel();
                    }
                    else{
                        ml.cycleInferenceMode();
                        snprintf(settingsLabelInfMode, sizeof(settingsLabelInfMode), "Infer: %s", ml.getInferenceModeName());
                    }
                    currentPrediction.valid=false;
                    display.showPredictionScreen(currentPrediction, currentSensorData, ml.getActiveModelName());
                    display.refresh();
                    break;
                }
                default:
                    //
                    if(!collectingLabeled){
                        logger.stopLogging();
                        loggingActive = false;
                        logger.setActiveLabel(-1);
                    }
                    else{
                        logger.setActiveLabel(currentLabelSelection);
                    }
                    display.setMode(DISPLAY_MODE_MENU);
                    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
                    updateDisplay();
                    break;
            }
            return;
        }
    }

    //outside menus: only allow long-press select to return to menu per controls
    if(buttons.wasPressed(BUTTON_DOWN)){
        return;
    }

    if(buttons.wasPressed(BUTTON_SELECT) || buttons.wasLongPressed(BUTTON_SELECT)){
        if(display.getMode()==DISPLAY_MODE_PREDICTION&&!collectingLabeled){
            logger.stopLogging();
            loggingActive = false;
            logger.setActiveLabel(-1);
        }
        display.setMode(DISPLAY_MODE_MENU);
        display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
        updateDisplay();
        return;
    }
}

void buttonCallback(button_id_t button, button_event_t event){
    display.resetTimeout();

    if(event != BUTTON_EVENT_SHORT_PRESS && event != BUTTON_EVENT_LONG_PRESS) return;

    DEBUG_PRINTF("[BUTTON] Button %d Event %d\n", button, event);

    //handle menu nav
    if(display.getMode() == DISPLAY_MODE_MENU || display.getMode() == DISPLAY_MODE_SETTINGS){
        const menu_item_t* activeItems = (display.getMode() == DISPLAY_MODE_MENU) ? main_menu_items : settings_menu_items;
        const uint8_t activeCount = (display.getMode() == DISPLAY_MODE_MENU) ? MAIN_MENU_COUNT : SETTINGS_MENU_COUNT;

        switch(button){
            case BUTTON_DOWN:
                if(event == BUTTON_EVENT_SHORT_PRESS){
                    display.menuDown();
                    display.showMenu(activeItems, activeCount, display.getSelectedMenuIndex(), display.getMode() == DISPLAY_MODE_MENU ? "Menu" : "Settings");
                } 
                else if(event == BUTTON_EVENT_LONG_PRESS){
                    display.menuUp();
                    display.showMenu(activeItems, activeCount, display.getSelectedMenuIndex(), display.getMode() == DISPLAY_MODE_MENU ? "Menu" : "Settings");
                }
                break;

            case BUTTON_SELECT:
                if(event == BUTTON_EVENT_SHORT_PRESS){
                    display.menuSelect();
                    updateDisplay();
                }
                else if(event == BUTTON_EVENT_LONG_PRESS){
                    display.setMode(DISPLAY_MODE_MENU);
                    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
                    updateDisplay();
                }
                break;
            
            default:
                break;
        }
        return;
    }

    //outside menus, only long-select returns to menu
    if(button == BUTTON_SELECT && event == BUTTON_EVENT_LONG_PRESS){
        display.setMode(DISPLAY_MODE_MENU);
        display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
        updateDisplay();
    }
}

//===========================================================================================================
//BLEHANDLER
//===========================================================================================================

void handleBLE(){
    ble.update();

    if(ble.hasNewConfig()){
        String cfg = ble.getReceivedConfig();
        DEBUG_PRINTF("[BLE] Received new config via BLE: %s", cfg.c_str());

        //apply config here
        if(cfg.startsWith("CALIB")){
            enterState(STATE_CALIBRATING);
        }
        else if(cfg.startsWith("LOG:ON")){
            if(!logger.isLogging()) logger.startLogging();
            loggingActive = true;
        }
        else if(cfg.startsWith("LOG:OFF")){
            logger.stopLogging();
            loggingActive = false;
        }
        else if(cfg.startsWith("LABEL:")){
            int l = cfg.substring(6).toInt();
            if(l >= 0 && l < SCENT_CLASS_COUNT){
                ml.setDataCollectionMode(true);
                ml.setCurrentLabel((scent_class_t)l);
            }
        }
        else if(cfg.startsWith("PROFILE:")){
            String pf = cfg.substring(8);
            DEBUG_PRINTF("[BLE] Received profile change command: %s\n", pf.c_str());
            if(pf == "VOC"){
                DEBUG_PRINTLN("[BLE] Setting VOC Heater Profile");
                sensors.setVOCHeaterProfile();
            }
            else if(pf=="FOOD"){
                DEBUG_PRINTLN("[BLE] Setting Food Heater Profile");
                sensors.setFoodHeaterProfile();
            }
            else{
                //assume default
                DEBUG_PRINTLN("[BLE] Setting Default Heater Profile");
                sensors.setDefaultHeaterProfile();
            }
        }
    }
}

//===========================================================================================================
//display
//===========================================================================================================

void updateDisplay(){
    switch(display.getMode()){
        case DISPLAY_MODE_STATUS:
            if(currentState==STATE_WARMUP){
                uint32_t remaining= getWarmupRemainingSeconds();
                char msg[32];
                snprintf(msg,sizeof(msg),"Warming up... %lum %02lus", remaining/60, remaining%60);
                display.showCalibrationScreen(getWarmupProgress(), msg);
            }
            else{
                display.showStatusScreen(currentState, lastError);
            }
            break;

        case DISPLAY_MODE_SENSOR_DATA:
            display.showSensorDataScreen(currentSensorData);
            break;

        case DISPLAY_MODE_PREDICTION:
            if(currentState==STATE_TEMPORAL_COLLECTING){
                //show live progress
                char buffer[32];
                snprintf(buffer, sizeof(buffer), "Collecting %d/%d...",ml.getTemporalCount(), ml.getTemporalBufferSize());display.showPredictionScreen(currentPrediction, currentSensorData, buffer);
            }
            else{
                display.showPredictionScreen(currentPrediction, currentSensorData, ml.getActiveModelName());
            }
            break;

        case DISPLAY_MODE_CALIBRATION:
            if(currentState == STATE_WARMUP){
                uint32_t remaining = getWarmupRemainingSeconds();
                char msg[32];
                snprintf(msg, sizeof(msg), "Warming up... %lum %02lus", remaining/60, remaining%60);
                display.showCalibrationScreen(getWarmupProgress(), msg);
            }
            else if(sensors.isCalibrating()){
                display.showCalibrationScreen(sensors.getCalibrationProgress(), "Calibrating...");
            }
            else{
                display.showCalibrationScreen(1.0f, "Calibration Complete");
            }
            break;

        case DISPLAY_MODE_GRAPH:
            display.showGraphScreen(currentSensorData.primary.gas_resistances, BME688_NUM_HEATER_STEPS, "Gas Resistance");
            break;


        case DISPLAY_MODE_DATA_COLLECTION:
            display.showMenu(collect_menu_items, COLLECT_MENU_COUNT, display.getSelectedMenuIndex(), "Data Collect");
            break;

        case DISPLAY_MODE_SETTINGS:
            display.showMenu(settings_menu_items, SETTINGS_MENU_COUNT, display.getSelectedMenuIndex(), "Settings");
            break;

        case DISPLAY_MODE_ERROR:
            display.ShowErrorScreen(lastError);
            break;

        case DISPLAY_MODE_MENU:
            display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
            break;

        default:
            display.showStatusScreen(currentState, lastError);
            break;
    }
    display.refresh();
}

void menuActionShowStatus(){
    display.setMode(DISPLAY_MODE_STATUS);
}

void menuActionShowSensorData(){
    display.setMode(DISPLAY_MODE_SENSOR_DATA);
}

void menuActionShowPrediction(){
    if(!isSensorWarmedUp()){
        DEBUG_PRINTLN(F("[MENU] Cannot enter prediction mode: Sensor is still warming up."));
        display.setMode(DISPLAY_MODE_CALIBRATION);
        return;
    }
    predictionSelection = 0;
    display.setPredictionSelection(predictionSelection);
    display.setMode(DISPLAY_MODE_PREDICTION);

    logger.setActiveLabel(LOG_LABEL_PRED);
    if(!logger.isLogging()){
        logger.startLogging();
        loggingActive = true;
    }
}

void menuActionShowGraph(){
    display.setMode(DISPLAY_MODE_GRAPH);
}

void menuActionShowError(){
    display.setMode(DISPLAY_MODE_ERROR);
}

void menuActionDataCollection(){
    if(!isSensorWarmedUp()){
        DEBUG_PRINTLN(F("[MENU] Cannot enter data collection mode: Sensor is still warming up."));
        display.setMode(DISPLAY_MODE_CALIBRATION);
        return;
    }
    display.setMode(DISPLAY_MODE_DATA_COLLECTION);
    refreshDataCollectionMenu();
}

void menuActionCalibrate(){
    if(currentState != STATE_ERROR){
        enterState(STATE_CALIBRATING);
        display.setMode(DISPLAY_MODE_CALIBRATION);
    }
}

void menuActionSettings(){
    display.setMode(DISPLAY_MODE_SETTINGS);
    refreshSettingsMenu();
}

//===========================================================================================================
//settings menu actions
//===========================================================================================================
void settingsActionCycleInferenceMode(){
    ml.cycleInferenceMode();
    currentInferenceMode= ml.getInferenceMode();
    snprintf(settingsLabelInfMode, sizeof(settingsLabelInfMode),
    "Inf: %s",ml.getInferenceModeName());
    
    DEBUG_PRINTF("[Settings] Inference mode set to %s\n", ml.getInferenceModeName());
    refreshSettingsMenu();
}

void refreshSettingsMenu(){
    display.showMenu(settings_menu_items, SETTINGS_MENU_COUNT, display.getSelectedMenuIndex(), "Settings");
}

void applyHeaterProfile(uint8_t index){
    switch(index % HEATER_PROFILE_COUNT){
        case 0:
            sensors.setDefaultHeaterProfile();
            break;
        case 1:
            sensors.setVOCHeaterProfile();
            break;
        case 2:
            sensors.setFoodHeaterProfile();
            break;
        default:
            sensors.setDefaultHeaterProfile();
            break;
    }
    DEBUG_PRINTF("[Settings] Heater profile set to %s\n", HEATER_PROFILE_NAMES[index % HEATER_PROFILE_COUNT]);
}

void settingsActionCalibrate(){
    menuActionCalibrate();
}

void settingsActionCycleHeaterProfile(){
    currentHeaterProfileIndex = (currentHeaterProfileIndex + 1) % HEATER_PROFILE_COUNT;
    applyHeaterProfile(currentHeaterProfileIndex);
    snprintf(settingsLabelHeater, sizeof(settingsLabelHeater), "Heater: %s", HEATER_PROFILE_NAMES[currentHeaterProfileIndex]);
    refreshSettingsMenu();
}

void settingsActionCycleThresholdProfile(){
    currentThresholdProfileIndex = (currentThresholdProfileIndex + 1) % THRESHOLD_PROFILE_COUNT;
    const threshold_profile_t &profile = THRESHOLD_PROFILES[currentThresholdProfileIndex];
    ml.setConfidenceThreshold(profile.confidence);
    ml.setAnomalyThreshold(profile.anomaly);
    snprintf(settingsLabelThreshold, sizeof(settingsLabelThreshold), "ML Thr: %s", profile.name);
    DEBUG_PRINTF("[Settings] ML thresholds set to %s (conf=%.2f, anom=%.2f)\n", profile.name, profile.confidence, profile.anomaly);
    refreshSettingsMenu();
}

void settingsActionToggleDisplayProfile(){
    currentDisplayProfileIndex = (currentDisplayProfileIndex + 1) % DISPLAY_PROFILE_COUNT;
    const display_profile_t &profile = DISPLAY_PROFILES[currentDisplayProfileIndex];
    display.setBrightness(profile.brightness);
    display.setInverted(profile.inverted);
    snprintf(settingsLabelDisplay, sizeof(settingsLabelDisplay), "Display: %s", profile.name);
    DEBUG_PRINTF("[Settings] Display profile set to %s (brightness=%u, inverted=%d)\n", profile.name, profile.brightness, profile.inverted);
    refreshSettingsMenu();
}

void settingsActionExportData(){
    String target = logger.getCurrentFilename();

    if(target.length() == 0){
        String files[1];
        uint8_t count = 0;
        if(logger.listLogFiles(files, 1, count) && count > 0){
            target = files[0];
        }
    }

    if(target.length() == 0){
        DEBUG_PRINTLN(F("[Settings] No log file to export."));
        return;
    }

    if(logger.exportToSerial(target.c_str())){
        DEBUG_PRINTF("[Settings] Exported log %s to serial.\n", target.c_str());
    } 
    else {
        DEBUG_PRINTF("[Settings] Failed to export log %s.\n", target.c_str());
    }
}

void settingsActionWipeSD(){
    DEBUG_PRINTLN(F("[Settings] Wiping log files..."));
    logger.deleteAllLogFiles();
    loggingActive = false;
    collectingLabeled = false;
    logger.setActiveLabel(-1);
    refreshSettingsMenu();
}

void settingsActionBackToMain(){
    display.setMode(DISPLAY_MODE_MENU);
    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
}

//===========================================================================================================
//data collection actions
//===========================================================================================================

void refreshDataCollectionMenu(){
    if(currentLabelSelection < SCENT_CLASS_PURE_CAMOMILE || currentLabelSelection >= SCENT_CLASS_COUNT){
        currentLabelSelection = SCENT_CLASS_PURE_CAMOMILE;
    }

    const char* labelName = (currentLabelSelection >=0 && currentLabelSelection < SCENT_CLASS_COUNT) ? SCENT_CLASS_NAMES[currentLabelSelection] : "Unknown";
    snprintf(collectLabelSelect, sizeof(collectLabelSelect), "Label: %s", labelName);
    snprintf(collectLabelToggle, sizeof(collectLabelToggle), collectingLabeled ? "Stop Collect" : "Start Collect");
    display.showMenu(collect_menu_items, COLLECT_MENU_COUNT, display.getSelectedMenuIndex(), "Data Collect");
}

void dataCollectActionCycleLabel(){
    currentLabelSelection++;
    if(currentLabelSelection >= SCENT_CLASS_COUNT || currentLabelSelection == SCENT_CLASS_UNKNOWN){
        currentLabelSelection = SCENT_CLASS_PURE_CAMOMILE;
    }
    refreshDataCollectionMenu();
}

void dataCollectActionToggle(){
    if(collectingLabeled){
        logger.setActiveLabel(-1);
        if(logger.isLogging()){
            logger.stopLogging();
        }
        loggingActive = false;
        collectingLabeled = false;
    } 
    else {
        if(currentLabelSelection < SCENT_CLASS_PURE_CAMOMILE || currentLabelSelection >= SCENT_CLASS_COUNT){
            currentLabelSelection = SCENT_CLASS_PURE_CAMOMILE;
        }
        logger.setActiveLabel(currentLabelSelection);
        if(!logger.isLogging()){
            logger.startLogging();
        }
        loggingActive = true;
        collectingLabeled = true;
    }
    refreshDataCollectionMenu();
}

void dataCollectActionBack(){
    display.setMode(DISPLAY_MODE_MENU);
    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
}

//===========================================================================================================
//errors
//===========================================================================================================

void handleError(error_code_t error){
    lastError = error;
    DEBUG_PRINTF("[ERROR] Handling error code %d\n", error);

    if(ble.isConnected()){
        ble.sendStatus(currentState, lastError);
    }
}

//===========================================================================================================
//debug
//===========================================================================================================
void printStatus(){
    DEBUG_PRINTLN("==== System Status ====");
    DEBUG_PRINTF("State: %d\n", currentState);
    DEBUG_PRINTF("Last Error: %d\n", lastError);
    DEBUG_PRINTF("Logging Active: %s\n", loggingActive ? "Yes" : "No");
    DEBUG_PRINTF("Samples: %lu\n", sensors.getSampleCount());
    DEBUG_PRINTF("Inferences: %lu\n", ml.getTotalInferences());
    DEBUG_PRINTF("FREE RAM: %d bytes\n", rp2040.getFreeHeap());
    DEBUG_PRINTLN("=======================\n");
}