#ifndef BLE_HANDLER_H
#define BLE_HANDLER_H

#include <Arduino.h>
#include "config.h"
#include "bme688_handler.h"
#include "ml_inference.h"

#if BLE_ENABLED
#include <NimBLEDevice.h>
#endif

class BLEHandler {
    BLEHandler();
    ~BLEHandler();

    //===========================================================================================================
    //initialize BLE
    //===========================================================================================================

    bool begin();
    bool isReady() const;

    //===========================================================================================================
    //connection
    //===========================================================================================================
    bool startAdvertising();
    bool stopAdvertising();
    bool isConnected() const;
    bool disconnect();
    uint8_t getConnectedCount() const;

    //===========================================================================================================
    //transmit data
    //===========================================================================================================
    bool sendSensorData(const dual_sensor_data_t &data);
    bool sendPrediction(const ml_prediction_t &prediction);
    bool sendStatus(system_state_t state, error_code_t error);

    //===========================================================================================================
    //reception
    //===========================================================================================================
    bool hasNewConfig() const;
    String getReceivedConfig();

    //===========================================================================================================
    //update
    //===========================================================================================================
    void update();

    //===========================================================================================================
    //config
    //===========================================================================================================
    void setDeviceName(const char* name);
    void setNotifyInterval(uint32_t interval_ms);

private:

#if BLE_ENABLED
    NimBLEServer* _server;
    NimBLEService* _service;
    NimBLECharacteristic* _char_sensor;
    NimBLECharacteristic* _char_prediction;
    NimBLECharacteristic* _char_config;
    NimBLEAdvertising* _advertising;
#endif

    bool _init;
    bool _advertising;
    uint8_t _connected_count;
    uint32_t _last_notify_time;
    uint32_t _notify_interval_ms;

    String _received_config;
    bool _new_config;
    String _device_name;

    //===========================================================================================================
    //callbacks
    //===========================================================================================================
    #if BLE_ENABLED
    class ServerCallbacks;
    class ConfigCallbacks;
    #endif

};

#endif