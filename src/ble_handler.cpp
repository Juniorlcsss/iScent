#include "ble_handler.h"

#if BLE_ENABLED

//Server callbacks
class BLEHandler::ServerCallbacks : public NimBLEServrCallbacks{
public:
    BLEHandler *handler;

    ServerCallbacks(BLEHandler *h) : Handler(h){}

    //connect
    void onConnect(NimBLEServer *s) override{
        DEBUG_PRINTLN(F("[BLE] Client connected!"));
        handler->_connection_count++;

        if(handler->_connection_count < 3){
            NimBLEDevice::startAdvertising();
        }
    }

    void onDisconnect(NimBLEServer *s) override{
        DEBUG_PRINTLN(F("[BLE] Client disconnected!"));
        if(handler->_connection_count > 0){
            handler->_connection_count--;
        }

        //restart advertising
        NimBLEDevice::startAdvertising();
    }
};

//Cfg
class BLEHandler::ConfigCallbacks : public NimBLECharacteristicCallbacks{
public:
    BLEHandler *handler;

    ConfigCallbacks(BLEHandler *h) : handler(h){}

    void onWrite(NimBLECharacteristic *c) override{
        std::string value = c->getValue();
        if(value.length() == 0){
            handler->_received_config = String(value.c_str());
            handler->_has_new_config = true;
            DEBUG_PRINTF("[BLE] Config received: %s\n", handler->_received_config.c_str());
        }
        
    }
};

#endif

BLEHandler::BLEHandler() :
#if BLE_ENABLED
    _server(nullptr),
    _service(nullptr),
    _char_sensor(nullptr),
    _char_prediction(nullptr),
    _char_config(nullptr),
    _advertising(nullptr),
#endif
    _init(false),
    _advertising(false),
    _connected_count(0),
    _last_notify_time(0),
    _notify_interval_ms(BLE_NOTIFY_INTERVAL_MS),
    _new_config(false),
    _device_name(BLE_DEVICE_NAME)
{}

BLEHandler::~BLEHandler(){
#if BLE_ENABLED
    if(_init){
        NimBLEDevice::deinit(true);
    }
#endif
}

//===========================================================================================================
//init
//===========================================================================================================

bool BLEHandler::begin(){
#if BLE_ENABLED
    DEBUG_PRINTLN(F("[BLE] Initializing BLE..."));

    //init nimble
    NimBLEDevice::init(_device_name.c_str());
    NimBLEDevice::setPower(ESP_PWR_LVL_P9); //max power
    NimBLEDevice::setSecurityAuth(false, false, false);

    //create server
    _server = NimBLEDevice::createServer();
    _server->setCallbacks(new ServerCallbacks(this));

    //service
    _service = _server->createService(BLE_SERVICE_UUID);

    //char
    _char_sensor = _service->createCharacteristic(BLE_CHAR_SENSOR_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
    _char_prediction = _service->createCharacteristic(BLE_CHAR_PREDICTION_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
    _char_config = _service->createCharacteristic(BLE_CHAR_CONFIG_UUID< NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::WRITE);
    _char_config->setCallbacks(new ConfigCallbacks(this));

    //start
    _service->start();

    //advertise
    _advertising = NimBLEDevice::getAdvertising();
    _advertising->addServiceUUID(_service->getUUID());
    _advertising->setScanResponse(true);
    _advertising->setMinPreferred(0x06);
    _advertising->setMinPreferred(0x12);
    _init = true;

    DEBUG_PRINTLN(F("[BLE] BLE initialized successfully"));
    return true;

#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

bool BLEHandler::isReady() const{
    return _init;
}


//===========================================================================================================
//connection management`
//===========================================================================================================

bool BLEHandler::startAdvertising(){
#if BLE_ENABLED
    if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot start advertising: BLE not initialized"));
        return false;
    }
    NimBLEDevice::startAdvertising();
    _advertising = true;
    DEBUG_PRINTLN(F("[BLE] Started advertising"));
    return true;
#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

bool BLEHandler::stopAdvertising(){
#if BLE_ENABLED
    if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot stop advertising: BLE not initialized"));
        return false;
    }
    NimBLEDevice::stopAdvertising();
    _advertising = false;
    DEBUG_PRINTLN(F("[BLE] Stopped advertising"));
    return true;
#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

bool BLEHandler::isConnected() const{
    return _connection_count > 0;
}

bool BLEHandler::disconnect(){
#if BLE_ENABLED
    if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot disconnect: BLE not initialized"));
        return false;
    }
    if(!_server){
        DEBUG_PRINTLN(F("[BLE] Server not created"));
        return false;
    }
    if(!isConnected()){
        DEBUG_PRINTLN(F("[BLE] No clients connected to disconnect"));
        return false;
    }

    //disconnect all clients
    for(int i = 0; i < _server->getConnectedCount(); i++){
        NimBLEDevice::getServer()->getPeerByIndex(i)->disconnect();
    }
    return true;
#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

uint8_t BLEHandler::getConnectedCount() const{
    return _connection_count;
}

//===========================================================================================================
//data
//===========================================================================================================

bool BLEHandler::sendSensorData(const dual_sensor_data_t &data){
#if BLE_ENABLED
    if(!_server){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: Server not initialized"));
        return false;
    }
    else if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: BLE not initialized"));
        return false;
    }
    else if(!isConnected()){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: No clients connected"));
        return false;       

    }

    //throttle
    if(millis() - _last_notify_time < _notify_interval_ms){
        return true;
    }

    //prepare data
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "{\"t\":%.1f,\"h\":%.1f,\"g\":%.0f,\"dt\":%.1f,\"dg\":%.0f}", data.primary.temperatures[0],
    data.primary.temperatures[0],data.primary.humidities[0],data.primary.gas_resistances[0],data.delta_temp,data.delta_gas_avg);

    _char_sensor->setValue((uint8_t*)buffer, strlen(buffer));
    _char_sensor->notify();
    _last_notify_time = millis();
    DEBUG_PRINTLN(F("[BLE] Sensor data sent"));
    return true;
#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

bool BLEHandler::sendPrediction(const ml_prediction_t &prediction){
#if BLE_ENABLED
    if(!_server){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: Server not initialized"));
        return false;
    }
    else if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: BLE not initialized"));
        return false;
    }
    else if(!isConnected()){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: No clients connected"));
        return false;       
    }

    //prepare
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "{\"c\":\"%s\",\"cf\":%.2f,\"an\":%d}",
    SCENT_CLASS_NAMES[prediction.predicted_class], prediction.confidence, prediction.is_anomaly ? 1 : 0);

    _char_prediction->setValue((uint8_t*)buffer, strlen(buffer));
    _char_prediction->notify();

    DEBUG_PRINTLN(F("[BLE] Prediction data sent"));
    return true;

#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

bool BLEHandler::sendStatus(system_state_t state, error_code_t error){
#if BLE_ENABLED
    if(!_server){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: Server not initialized"));
        return false;
    }
    else if(!_init){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: BLE not initialized"));
        return false;
    }
    else if(!isConnected()){
        DEBUG_PRINTLN(F("[BLE] Cannot send sensor data: No clients connected"));
        return false;       
    }

    char buffer[64];
    snprintf(buffer, sizeof(buffer), "{\"st\":%d,\"er\":%d}", state, error);

    _char_sensor->setValue((uint8_t*)buffer, strlen(buffer));
    _char_sensor->notify();
    DEBUG_PRINTLN(F("[BLE] Status data sent"));
    return true;
#else
    DEBUG_PRINTLN(F("[BLE] BLE not enabled in config.h"));
    return false;
#endif
}

//===========================================================================================================
//config
//===========================================================================================================

bool BLEHandler::hasNewConfig() const{
    return _new_config;
}

String BLEHandler::getRecievedConfig(){
    _new_config = false;
    return _received_config;
}

void BLEHandler::setDeviceName(const char* name){
    _device_name = String(name);
}

void BLEHandler::setNotifyInterval(uint32_t interval_ms){
    _notify_interval_ms = interval_ms;
}

//===========================================================================================================
//update
//===========================================================================================================

void BLEHandler::update(){
#if BLE_ENABLED
    if(_init){
        NimBLEDevice::poll();
    }
#endif
}