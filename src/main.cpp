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

dual_sensor_data_t currentSensorData;
ml_prediction_t currentPrediction;

bool loggingActive = false;
bool continuousInference = true;

//===========================================================================================================
//functions
//===========================================================================================================
void initialiseSystem();
void handleStateMachine();
void handleButtons();
void handleBLE();
void updateDisplay();
void performSampling();
void performInference();
void enterState(system_state_t newState);
void handleError(error_code_t error);
void buttonCallback(button_id_t buttonId, button_event_t event);
void printStatus();

//menu actions
void menuActionCalibrate();
void menuActionToggleLogging();
void menuActionSettings();
void menuActionBack();

//menu items
menu_item_t main_menu_items[] = {
    {"Calibrate", menuActionCalibrate, DISPLAY_MODE_CALIBRATION},
    {"Toggle Logging", menuActionToggleLogging, DISPLAY_MODE_LOGGING},
    {"Settings", menuActionSettings, DISPLAY_MODE_SETTINGS},
    {"Back", menuActionBack, DISPLAY_MODE_SENSOR_DATA}
};
const uint8_t MAIN_MENU_COUNT = sizeof(main_menu_items)  /sizeof(menu_item_t);

//===========================================================================================================
//setup
//===========================================================================================================


void setup(){
    Serial.begin(DEBUG_SERIAL_BAUD);
    delay(1000);

    
    DEBUG_PRINTLN(F("\n\n"))
    DEBUG_PRINTLN(F("==== iScent ===="))
    DEBUG_PRINTF("Firmware Version: %s\n", SOFTWARE_VERSION);


    initialiseSystem();
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

    //update display regularly
    if(millis() - lastDisplayUpdateTime >= DISPLAY_UPDATE_INTERVAL_MS){
        updateDisplay();
        lastDisplayUpdateTime = millis();
    }

    delay(10);
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

    Wire1.setSDA(I2C1_SDA_PIN);
    Wire1.setSCL(I2C1_SCL_PIN);
    Wire1.begin();
    Wire1.setClock(I2C_FREQUENCY_HZ);


    //init display
    DEBUG_PRINTLN(F("[INIT] Initializing Display..."));
    if(!display.begin(&Wire)){
        DEBUG_PRINTLN(F("[ERROR] Display initialization failed!"));
        lastError = ERROR_DISPLAY_INIT;
    }

    display.showSplashScreen();
    delay(DISPLAY_SPASH_DURATION_MS);


    //init buttons
    DEBUG_PRINTLN(F("[INIT] Initializing Buttons..."));
    buttons.begin();
    buttons.setCallback(buttonCallback);


    //init sensors
    DEBUG_PRINTLN(F("[INIT] Initializing Sensors..."));
    display.showStatusScreen(STATE_INIT, lastError);
    if(!sensors.begin(&Wire, &Wire1)){
        handleError(sensors.getLastError());
        
        if(!sensors.isReady()){
            DEBUG_PRINTLN(F("[ERROR] Sensor initialization failed!"));
            lastError = ERROR_BOTH_SENSORS_INIT;
            enterState(STATE_ERROR);
            return;
        }
    }
    sensors.printSensorData();


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

    //warmup
    DEBUG_PRINTLN(F("[INIT] Warmup Phase..."));
    enterState(STATE_WARMUP);
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
            //wait for warmup
            if(millis() - warmupStartTime >= BME688_STABLE_MS){
                DEBUG_PRINTLN(F("[STATE] Warmup complete."));
                
                if(!sensors.getCalibrationData().calibrated){
                    DEBUG_PRINTLN(F("[STATE] Starting calibration..."));
                    enterState(STATE_CALIBRATING);
                }
                else{
                    DEBUG_PRINTLN(F("[STATE] Entering IDLE state."));
                    enterState(STATE_IDLE);
                }
            }
            //sample during warmup :)
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                sensors.performFullScan(currentSensorData);
                lastSampleTime = millis();
            }
            break;

        case STATE_CALIBRATING:
            //
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                performSampling();
                lastSampleTime = millis();
            }

            if(!sensors.isCalibrating()){
                DEBUG_PRINTLN(F("[STATE] Calibration complete."));
                enterState(STATE_IDLE);
            }
            break;

        case STATE_IDLE:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                enterState(STATE_SAMPLING);
            }
            break;

        case STATE_SAMPLING:
            performSampling();
            lastSampleTime = millis();

            //move to inference if ready
            if(continuousInference && ml.isFeatureBufferReady()){
                enterState(STATE_INFERENCING);
            }
            else{
                enterState(STATE_IDLE);
            }
            break;

        case STATE_INFERENCING:
            if(millis() - lastSampleTime >= BME688_SAMPLE_RATE){
                performSampling();
                lastSampleTime = millis();
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
            break;
        
        case STATE_LOGGING:
            logger.stopLogging();
            loggingActive = false;
            break;

        case STATE_SLEEP:
            sensors.wake();
            break;
        
        default:
            break;
    }

    //entry acts
    switch(newState){
        case STATE_WARMUP:
            warmupStartTime = millis();
            display.showStatusScreen(STATE_WARMUP, lastError);
            break;

        case STATE_CALIBRATING:
            sensors.startCalibration(NBME688_GAS_BASE_SAMPLES);
            break;

        case STATE_LOGGING:
            logger.startLogging();
            loggingActive = true;
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

void performSampling(){
    if(sensors.performFullScan(currentSensorData)){
        //add window
        ml.addToWindow(currentSensorData);

        //log
        if(loggingActive){
            logger.logEntry(currentSensorData, currentPrediction.valid ? &currentPrediction : nullptr);
        }

        //send via BLE
        if(ble.isConnected()){
            ble.sendSensorData(currentSensorData);
        }

        DEBUG_VERBOSE_PRINTF("[Sample] T:%.1f H:%.1f G:%.0f\n", 
            currentSensorData.primary.temperatures,
            currentSensorData.primary.humidities[0],
            currentSensorData.primary.gas_resistances[0]
        );
    }
    else{
        DEBUG_PRINTLN(F("[ERROR] Sensor read failed during sampling!"));
        lastError = ERROR_SENSOR_READ;
    }

}

void performInference(){
    if(ml.runInference(currentPrediction)){     
        DEBUG_PRINTF("[Inference] Prediction: %s (%.2f), Anomaly: %.2f\n", 
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

//===========================================================================================================
//button handling
//===========================================================================================================
void buttonCallback(button_id_t button, button_event_t event){
    display.resetTimeout();

    if(event != BUTTON_EVENT_SHORT_PRESS && event != BUTTON_EVENT_LONG_PRESS) return;

    DEBUG_PRINTF("[BUTTON] Button %d Event %d\n", button, event);

    //handle menu nav
    if(display.getMode() == DISPLAY_MODE_MENU){
        switch(button){
            case BUTTON_UP:
                if(event == BUTTON_EVENT_SHORT_PRESS){
                    display.menuUp();
                    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
                }
                break;

            case BUTTON_DOWN:
                if(event == BUTTON_EVENT_SHORT_PRESS){
                    display.menuDown();
                    display.showMenu(main_menu_items, MAIN_MENU_COUNT, display.getSelectedMenuIndex());
                }
                break;

            case BUTTON_SELECT:
                if(event == BUTTON_EVENT_SHORT_PRESS){
                    display.menuSelect();
                }
                else if(event == BUTTON_EVENT_LONG_PRESS){
                    //exit
                    display.setMode(DISPLAY_MODE_STATUS);
                }
                break;
            
            default:
                break;
        }
        return;
    }

    //normal button handling
    switch(button){
        case BUTTON_UP:
            if(event == BUTTON_EVENT_SHORT_PRESS){
                display.prevMode();
            }
            break;
        
        case BUTTON_DOWN:
            if(event == BUTTON_EVENT_SHORT_PRESS){
                display.nextMode();
            }
            break;

        case BUTTON_SELECT:
            if(event == BUTTON_EVENT_SHORT_PRESS){
                //toggle log
                if(currentState != STATE_LOGGING){
                    enterState(STATE_LOGGING);
                }
                else{
                    enterState(STATE_IDLE);
                }
            }
            else if(event == BUTTON_EVENT_LONG_PRESS){
                //open menu
                display.setMode(DISPLAY_MODE_MENU);
                display.showMenu(main_menu_items, MAIN_MENU_COUNT, 0);
            }
            break;

        default:
            break;
    }
}

//===========================================================================================================
//BLEHANDLER
//===========================================================================================================

void handleBLE(){
    ble.update();

    if(ble.hasNewConfig()){
        String cfg = ble.getRecievedConfig();
        DEBUG_PRINTLN("[BLE] Received new config via BLE: %s", cfg.c_str());

        //apply config here
        applyBLEConfig(cfg);
    }
}

void applyBLEConfig(String cfg){
    switch(cfg){
        case (cfg.startsWith("CALIB")):
            enterState(STATE_CALIBRATING);
            break;

        case (cfg.startsWith("LOG:ON")):
            enterState(STATE_LOGGING);
            break;

        case (cfg.startsWith("LOG:OFF")):
            if(currentState == STATE_LOGGING){
                enterState(STATE_IDLE);
            }
            break;

        case (cfg.startsWith("PROFILE:")): 
            String profile = cfg.substring(8);

            if(profile == "VOC"){
                sensors.setVOCHeaterProfile();
            }
            else if(profile == "FOOD"){
                sensors.setFoodHeaterProfile();
            }
            else{
                sensors.setDefaultHeaterProfile();
            }
            break;

        case (cfg.startsWith("LABEL:")):
            int l = cfg.substring(6).toInt();
            if(l >=0 && l < SCENT_CLASS_COUNT){
                ml.setDataCollectionMode(true);
                ml.setCurrentLabel((scent_class_t)l);
            }
            break;

        default:
            DEBUG_PRINTLN("[BLE] Undefined config command received: %s", cfg.c_str());
            break;
    }
}

//===========================================================================================================
//display
//===========================================================================================================

void updateDisplay(){
    switch(display.getMode()){
        case DISPLAY_MODE_STATUS:
            display.showStatusScreen(currentState, lastError);
            break;

        case DISPLAY_MODE_SENSOR_DATA:
            display.showSensorDataScreen(currentSensorData);
            break;

        case DISPLAY_MODE_PREDICTION:
            display.showPredictionScreen(currentPrediction, currentSensorData);
            break;

        case DISPLAY_MODE_CALIBRATION:
            if(sensors.isCalibrating()){    
                display.showCalibrationScreen(sensors.getCalibrationProgress(), "Calibrating...");
            }
            else{
                display.showCalibrationScreen(1.0f, "Calibration Complete");
            }
            break;

        case DISPLAY_MODE_GRAPH:
            display.showGraphScreen(currentSensorData.primary.gas_resistances, BME688_NUM_HEATER_STEPS, "Gas Resistance");
            break;

        case DISPLAY_MODE_LOGGING:
            display.showLoggingScreen(logger.getTotalLoggedEntries(), logger.isLogging());
            break;

        case DISPLAY_MODE_SETTINGS:
            display.showSettingsScreen();
            break;

        case DISPLAY_MODE_ERROR:
            display.ShowErrorScreen(lastError);
            break;

        default:
            display.showStatus(currentState, lastError);
            break;
    }
}

void menuActionCalibrate(){
    if(currentState != STATE_IDLE || currentState != STATE_WARMUP){
        enterState(STATE_CALIBRATING);
        display.setMode(DISPLAY_MODE_CALIBRATION);
    }
}

void menuActionToggleLogging(){
    if(currentState == STATE_LOGGING){
        enterState(STATE_IDLE);
    }
    else if(currentState == STATE_IDLE){
        enterState(STATE_LOGGING);
    }
    display.setMode(DISPLAY_MODE_LOGGING);
}

void menuActionSettings(){
    display.setMode(DISPLAY_MODE_SETTINGS);
}

void menuActionBack(){
    display.setMode(DISPLAY_MODE_STATUS);
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
    sensors.printSensorData();
    ml.printStatus();
    logger.printStatus();
    ble.isConnected() ? DEBUG_PRINTLN("BLE: Connected") : DEBUG_PRINTLN("BLE: Not Connected");
    DEBUG_PRINTF("FREE RAM: %d bytes\n", rp2040.getFreeHeap());
    DEBUG_PRINTLN("=======================\n");
}