#include "data_logger.h"

DataLogger::DataLogger():
    _buffer(nullptr),
    _buffer_size(DATA_LOG_BUFFER_SIZE),
    _buffer_head(0),
    _buffer_count(0),
    _init(false),
    _is_logging(false),
    _total_entries(0),
    _file_entry_count(0),
    _file_index(0),
    _last_flush_time(0),
    _auto_flush_interval(DATA_LOG_FLUSH_INTERVAL_MS),
    _max_file_size(DATA_LOG_MAX_FILE_SIZE),
    _active_label(-1)
{}

DataLogger::~DataLogger() {
    stopLogging();
    if (_buffer) {
        delete[] _buffer;
        _buffer = nullptr;
    }
}

//===========================================================================================================
//init
//===========================================================================================================


bool DataLogger::begin(){
    DEBUG_PRINTLN(F("[DataLogger] Initialising..."));

    //init
    if(!LittleFS.begin()){
        DEBUG_PRINTLN("[DataLogger] Formatting...");
        if(!LittleFS.format()){
            DEBUG_PRINTLN(F("[DataLogger] Error: LittleFS format failed"));
            return false;
        }
        if(!LittleFS.begin()){
            DEBUG_PRINTLN(F("[DataLogger] Error: LittleFS mount failed after format"));
            return false;
        }
    }

    //allocate buffer
    _buffer = new log_entry_t[_buffer_size];
    if(_buffer == nullptr){
        DEBUG_PRINTLN(F("[DataLogger] Error: Buffer allocation failed"));
        return false;
    }
    memset(_buffer, 0, sizeof(log_entry_t)*_buffer_size);
    _init=true;

    //find highest existing file id
    Dir dir = LittleFS.openDir("/");

    while(dir.next()){
        String name = dir.fileName();
        if(name.startsWith(DATA_LOG_FILENAME)){
            int idx = name.substring(strlen(DATA_LOG_FILENAME) + 1, name.indexOf('.')).toInt();
            if(idx >= _file_index){
                _file_index = idx + 1;
            }
        }
    }
    DEBUG_PRINTF("[DataLogger] Initialization complete. Next log file index: %d\n", _file_index);
    return true;
}

bool DataLogger::isReady() const{
    return _init;
}


//===========================================================================================================
//logging
//===========================================================================================================

bool DataLogger::startLogging(const char *filename){
    if(!_init){
        DEBUG_PRINTLN(F("[DataLogger] Error: DataLogger not initialized. Call begin() first."));
        return false;
    }
    if(_is_logging){
        DEBUG_PRINTLN(F("[DataLogger] Warning: Logging already in progress."));
        return true;
    }
    if(filename != nullptr){
        _current_filename = String(filename);
    }
    else{
        _current_filename = generateFilename();
    }
    return createNewLogFile();
}

bool DataLogger::stopLogging(){
    if(!_is_logging){
        DEBUG_PRINTLN(F("[DataLogger] Warning: No active logging session to stop."));
        return true;
    }
    flush();
    closeLogFile();
    _is_logging = false;
    DEBUG_PRINTF("[DataLogger] Stopped. Total entries: %lu\n", _total_entries);
    return true;
}

bool DataLogger::isLogging() const{
    return _is_logging;
}

bool DataLogger::logEntry(const dual_sensor_data_t &data, const ml_prediction_t *pred){
    if(!_is_logging){
        return false;
    }

    //create log entry
    log_entry_t entry;
    entry.timestamp = millis();
    memcpy(&entry.sensor_data, &data, sizeof(dual_sensor_data_t));

    if(pred != nullptr){
        memcpy(&entry.ml_prediction, pred, sizeof(ml_prediction_t));
    }
    else{
        memset(&entry.ml_prediction, 0, sizeof(ml_prediction_t));
    }

    //iaq
    entry.iaq_index = calculateIAQIndex(data.primary.gas_resistances[0], data.primary.humidities[0]);
    entry.label = _active_label;

    //add to buffer
    _buffer[_buffer_head] = entry;
    _buffer_head = (_buffer_head +1) % _buffer_size;

    if(_buffer_count < _buffer_size){
        _buffer_count++;
    }

    _total_entries++;

    //autoflush
    if(millis() - _last_flush_time >= _auto_flush_interval || isBufferFull()){
        flush();
    }

    checkFileRotation();

    return true;
}   

bool DataLogger::logRawData(const dual_sensor_data_t &data){
    return logEntry(data, nullptr);
}

bool DataLogger::logPrediction(const ml_prediction_t &pred){
    if(!_is_logging || !_log_file){
        return false;
    }

    _log_file.printf("%lu,PRED,%s,%.4f,%.4f\n", pred.timestamp, SCENT_CLASS_NAMES[pred.predictedClass], pred.confidence, pred.anomalyScore);
    return true;
}

//===========================================================================================================
//buffer
//===========================================================================================================

bool DataLogger::flush(){
    if(!_is_logging || _buffer_count ==0){
        return true;
    }

    bool success = writeBufferToFile();

    if(success){
        _buffer_count=0;
        _buffer_head=0;
        _last_flush_time = millis();
    }
    return success;
}

uint16_t DataLogger::getBufferCount() const{
    return _buffer_count;
}

bool DataLogger::isBufferFull() const{
    return _buffer_count >= _buffer_size;
}

size_t getUsedSpace() {
    FSInfo info;
    if(LittleFS.info(info)){
        return info.usedBytes;
    }
    return 0;
}

size_t getFreeSpace() {
    FSInfo info;
    if(LittleFS.info(info)){
        return info.totalBytes - info.usedBytes;
    }
    return 0;
}

//===========================================================================================================
//files
//===========================================================================================================

bool DataLogger::createNewLogFile(){
    if(_log_file){
        closeLogFile();
    }

    _log_file = LittleFS.open(_current_filename, "w");
    if(!_log_file){
        DEBUG_PRINTF("[DataLogger] Error: Failed to create log file %s\n", _current_filename.c_str());
        return false;
    }

    //header
    writeHeader();

    _file_entry_count=0;
    _is_logging=true;
    _last_flush_time = millis();

    DEBUG_PRINTF("[DataLogger] Started logging to %s\n", _current_filename.c_str());
    return true;
}

bool DataLogger::closeLogFile(){
    if(_log_file){
        _log_file.flush();
        _log_file.close();
        DEBUG_PRINTF("[DataLogger] Closed log file %s\n", _current_filename.c_str());
    }
    return true;
}

String DataLogger::getCurrentFilename() const {
    return _current_filename;
}

uint32_t DataLogger::getCurrentFileSize() const{
    if(_log_file){
        return _log_file.size();
    }
    return 0;
}

uint32_t DataLogger::getTotalLoggedEntries() const{
    return _total_entries;
}

String DataLogger::generateFilename(){
    char filename[32];
    snprintf(filename, sizeof(filename), "%s_%04d.csv",DATA_LOG_FILENAME, _file_index++);
    return String(filename);
}

bool DataLogger::writeHeader(){
    if(!_log_file){
        return false;
    }

    //csv header
    _log_file.println(F("timestamp,label,temp1,hum1,pres1,gas1_0,gas1_1,gas1_2,gas1_3,gas1_4,"\
                        "gas1_5,gas1_6,gas1_7,gas1_8,gas1_9,"
                        "temp2,hum2,pres2,gas2_0,gas2_1,gas2_2,gas2_3,gas2_4,"
                        "gas2_5,gas2_6,gas2_7,gas2_8,gas2_9,"
                        "delta_temp,delta_hum,delta_pres,delta_gas,"
                        "pred_class,pred_conf,anomaly_score,iaq"));
    return true;
}

bool DataLogger::writeEntry(const log_entry_t& entry){
    if(!_log_file){
        return false;
    }

    //timestamp
    _log_file.printf("%lu,",entry.timestamp);

    //label
    if(entry.label >= 0 && entry.label < SCENT_CLASS_COUNT){
        _log_file.printf("%s,", SCENT_CLASS_NAMES[entry.label]);
    }else{
        _log_file.print("None,");
    }

    //sensor data
    _log_file.printf("%.2f,%.2f,%.2f,",
    entry.sensor_data.primary.temperatures[0],
    entry.sensor_data.primary.humidities[0],
    entry.sensor_data.primary.pressures[0]);

    //gas value
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
        _log_file.printf("%.0f,", entry.sensor_data.primary.gas_resistances[i]);
    }

    //secondary sensor
    _log_file.printf("%.2f,%.2f,%.2f,",
    entry.sensor_data.secondary.temperatures[0],
    entry.sensor_data.secondary.humidities[0],
    entry.sensor_data.secondary.pressures[0]);

    //gas value
    for(uint8_t i=0; i<BME688_NUM_HEATER_STEPS; i++){
        _log_file.printf("%.0f,", entry.sensor_data.secondary.gas_resistances[i]);
    }

    //deltas
    _log_file.printf("%.2f,%.2f,%.2f,%.0f,",
    entry.sensor_data.delta_temp,
    entry.sensor_data.delta_hum,
    entry.sensor_data.delta_pres,
    entry.sensor_data.delta_gas_avg);

    //prediction
    _log_file.printf("%d,%.4f,%.4f,",
    entry.ml_prediction.predictedClass,
    entry.ml_prediction.confidence,
    entry.ml_prediction.anomalyScore);

    //iaq
    _log_file.printf("%d\n", entry.iaq_index);

    _file_entry_count++;
    return true;
}

void DataLogger::setActiveLabel(int16_t label){
    _active_label = label;
}

int16_t DataLogger::getActiveLabel() const{
    return _active_label;
}

bool DataLogger::writeBufferToFile(){
    if(!_log_file || _buffer_count == 0){
        return false;
    }

    uint16_t start_idx = (_buffer_head >= _buffer_count) ? (_buffer_head - _buffer_count) :(_buffer_size - (_buffer_count - _buffer_head));

    for(uint16_t i=0; i<_buffer_count; i++){
        uint16_t idx = (start_idx +i) % _buffer_size;
        if(!writeEntry(_buffer[idx])){
            DEBUG_PRINTLN(F("[DataLogger] Error: Failed to write log entry to file."));
            return false;
        }
    }

    _log_file.flush();
    return true;
}

bool DataLogger::checkFileRotation(){
    if(!_log_file){
        return false;
    }

    if(_log_file.size() >= _max_file_size){
        DEBUG_PRINTLN(F("[Logger] File size limit reached, rotating..."));

        flush();
        closeLogFile();

        _current_filename = generateFilename();
        return createNewLogFile();
    }
    return true;
}

bool DataLogger::listLogFiles(String *files, uint8_t max, uint8_t &count){
    if(!_init){
        return false;
    }

    count=0;
    Dir dir = LittleFS.openDir("/");

    while(dir.next() && count < max){
        String name = dir.fileName();
        if(name.startsWith(DATA_LOG_FILENAME)){
            files[count++] = name;
        }
    }
    return true;
}

bool DataLogger::deleteLogFile(const char* filename){
    if(!_init){
        return false;
    }

    if(_is_logging && _current_filename == String(filename)){
        DEBUG_PRINTLN(F("[DataLogger] Error: Cannot delete active log file."));
        return false;
    }

    if(LittleFS.remove(filename)){
        DEBUG_PRINTF("[DataLogger] Deleted log file %s\n", filename);
        return true;
    }
    else{
        DEBUG_PRINTF("[DataLogger] Error: Failed to delete log file %s\n", filename);
        return false;
    }
}

bool DataLogger::deleteAllLogFiles(){
    if(!_init){
        return false;
    }

    stopLogging();

    Dir dir = LittleFS.openDir("/");
    while(dir.next()){
        String name = dir.fileName();
        if(name.startsWith(DATA_LOG_FILENAME)){
            LittleFS.remove(name);
            DEBUG_PRINTF("[DataLogger] Deleted log file %s\n", name.c_str());
        }
    }

    _file_index =0;
    _total_entries = 0;
    return true;
}

bool DataLogger::exportToSerial(const char* filename){
    if(!_init){
        return false;
    }

    File file = LittleFS.open(filename, "r");
    if(!file){
        DEBUG_PRINTF("[DataLogger] Error: Failed to open log file %s for export\n", filename);
        return false;
    }

    Serial.println(F("Begin log export:"));
    Serial.println(filename);
    while(file.available()){
        Serial.write(file.read());
    }

    file.close();
    Serial.println(F("End log export."));
    return true;
}

//===========================================================================================================
//config
//===========================================================================================================

void DataLogger::setAutoFlushInterval(uint32_t interval_ms){
    _auto_flush_interval = interval_ms;
}

void DataLogger::setMaxFileSize(uint32_t max_size) {
    _max_file_size = max_size;
}

void DataLogger::setBufferSize(uint16_t size){
    if(_is_logging){
        DEBUG_PRINTLN(F("[DataLogger] Error: Cannot change buffer size while logging is active."));
        return;
    }

    if(_buffer){
        delete[] _buffer;
    }

    _buffer = new log_entry_t[size];
    _buffer_size = size;
    _buffer_head =0;
    _buffer_count =0;
}