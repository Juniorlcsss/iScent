#include "data_logger.h"
#include <ctype.h>

static const char LOG_HEADER[] = "timestamp,label,temp1,hum1,pres1,delta_temp,delta_hum,delta_pres,delta_gas,pred_class,pred_conf,anomaly_score,iaq";

static const char CALIB_DEBUG_FILE[] = "/calib_debug.txt";
static const uint32_t CALIB_DEBUG_MIN_INTERVAL_MS = 500;

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
    _active_label(-1),
    _calib_debug_count(0),
    _calib_debug_last_ms(0),
    _calib_debug_active(false),
    _using_sd(false)
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

    //init sd
    SPI.setRX(SD_MISO_PIN);
    SPI.setSCK(SD_SCK_PIN);
    SPI.setTX(SD_MOSI_PIN);
    pinMode(SD_CS_PIN, OUTPUT);
    digitalWrite(SD_CS_PIN, HIGH);

    bool sdOk = SD.begin(SD_CS_PIN);
    if(!sdOk){
        DEBUG_PRINTLN(F("[DataLogger] SD init failed, retrying at alternate speed..."));
        sdOk = SD.begin(SD_CS_PIN, SPI_FULL_SPEED, SPI);
    }

    if(sdOk){
        _using_sd = true;
        DEBUG_PRINTLN(F("[DataLogger] SD card initialized."));
    }
    else{
        DEBUG_PRINTLN(F("[DataLogger] Warning: SD card initialization failed, falling back to LittleFS."));
        //init littlefs
        if(!LittleFS.begin()){
            DEBUG_PRINTLN("[DataLogger] mount failed");
            return false;
        }
        _using_sd = false;
        DEBUG_PRINTLN("[DataLogger] LittleFS initialized.");
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
    {
        File dir = _using_sd ? SD.open("/", FILE_READ) : LittleFS.open("/", "r");
        const char* base = (DATA_LOG_FILENAME[0] == '/') ? DATA_LOG_FILENAME + 1 : DATA_LOG_FILENAME;
        String baseFile = String(base) + ".csv";
        while(dir){
            File entry = dir.openNextFile();
            if(!entry){
                break;
            }

            String name = stripLeadingSlash(String(entry.name()));
            if(name.equalsIgnoreCase(baseFile)){
                if(_file_index < 1){
                    _file_index = 1;
                }
            } else if(name.startsWith(base)){
                int underscorePos = name.indexOf('_', strlen(base));
                int dot = name.indexOf('.');

                if(underscorePos > 0 && dot > underscorePos){
                    int idxVal = name.substring(underscorePos + 1, dot).toInt();
                    if(idxVal >= _file_index){
                        _file_index = idxVal + 1;
                    }
                }
            }
            entry.close();
        }
        dir.close();
    }
    DEBUG_PRINTF("[DataLogger] Initialization complete. Next log file index: %d\n", _file_index);

    //ensure a default log exists
    String defaultFile = String(DATA_LOG_FILENAME) + ".csv";
    ensureFileExists(defaultFile.c_str());
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

    if(!_using_sd){
        DEBUG_PRINTLN(F("[DataLogger] Attempting late SD init..."));
        SPI.setRX(SD_MISO_PIN);
        SPI.setSCK(SD_SCK_PIN);
        SPI.setTX(SD_MOSI_PIN);
        pinMode(SD_CS_PIN, OUTPUT);
        digitalWrite(SD_CS_PIN, HIGH);

        bool sdOk = SD.begin(SD_CS_PIN);
        if(!sdOk){
            DEBUG_PRINTLN(F("[DataLogger] SD init failed, retrying at alternate speed..."));
            sdOk = SD.begin(SD_CS_PIN, SPI_FULL_SPEED, SPI);
        }
        if(sdOk){
            _using_sd = true;
            DEBUG_PRINTLN(F("[DataLogger] SD card initialized (late init)."));
        } else {
            DEBUG_PRINTLN(F("[DataLogger] SD late init failed; continuing with LittleFS."));
        }
    }
    if(filename == nullptr){
        _file_index = 0;
    }

    if(filename != nullptr){
        _current_filename = normalisePath(String(filename));
    }
    else{
        _current_filename = generateFilename();
    }

    bool ok = createNewLogFile();
    if(!ok && _using_sd){
        DEBUG_PRINTF("[DataLogger] startLogging failed on SD for %s, falling back to LittleFS\n", _current_filename.c_str());
        _using_sd = false;
        if(!LittleFS.begin()){
            DEBUG_PRINTLN(F("[DataLogger] LittleFS mount failed during fallback"));
            return false;
        }
        ok = createNewLogFile();
    }

    if(!ok){
        DEBUG_PRINTF("[DataLogger] startLogging failed creating %s on %s\n", _current_filename.c_str(), _using_sd ? "SD" : "LittleFS");
        return false;
    }
    DEBUG_PRINTF("[DataLogger] startLogging target=%s fs=%s ok=%d\n", _current_filename.c_str(), _using_sd ? "SD" : "LittleFS", ok);
    return ok;
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
    if(!_is_logging || !_buffer){
        return false;
    }

    if(!_log_file){
        DEBUG_PRINTLN(F("[DataLogger] Warning: no open log file; stopping logging."));
        _is_logging = false;
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


void DataLogger::startCalibDebug(){
    _calib_debug_count = 0;
    _calib_debug_last_ms = 0;
    _calib_debug_active = true;
}

void DataLogger::logCalibDebug(const String &line){
    if(!_init || !_calib_debug_active){
        return;
    }

    uint32_t now = millis();
    if(_calib_debug_count > 0 && (now - _calib_debug_last_ms) < CALIB_DEBUG_MIN_INTERVAL_MS){
        return; //throttle 
    }

    if(_calib_debug_count >= CALIB_DEBUG_MAX_LINES){
        return;
    }

    _calib_debug_buffer[_calib_debug_count++] = line;
    _calib_debug_last_ms = now;
}

bool DataLogger::flushCalibDebug(){
    if(!_init){
        return false;
    }

    if(!_calib_debug_active || _calib_debug_count == 0){
        _calib_debug_active = false;
        _calib_debug_count = 0;
        return true;
    }

    if(!LittleFS.begin()){
        DEBUG_PRINTLN("[DataLogger] LittleFS mount failed for calib debug");
        return false;
    }

    String path = normalisePath(String(CALIB_DEBUG_FILE));
    File f = LittleFS.open(path, "a");
    if(!f){
        DEBUG_PRINTF("[DataLogger] Failed to open calib debug file %s\n", path.c_str());
        LittleFS.end();
        return false;
    }

    for(uint8_t i=0; i<_calib_debug_count; i++){
        f.println(_calib_debug_buffer[i]);
    }
    f.flush();
    f.close();
    LittleFS.end();

    _calib_debug_active = false;
    _calib_debug_count = 0;
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

size_t DataLogger::getUsedSpace() const{

    return 0;
}

size_t DataLogger::getFreeSpace() const{
    
    return 0;
}


//===========================================================================================================
//state
//===========================================================================================================
String DataLogger::normalisePath(const String &name) const{
    if(name.length()==0){
        return "/";
    }
    if(name.startsWith("/")){
        return name;
    }
        return "/" + name;
    }

String DataLogger::stripLeadingSlash(const String &name) const{
    if(name.startsWith("/")){
        return name.substring(1);
    }
    return name;

}

String DataLogger::toFsPath(const String &path) const{
    if(_using_sd){
        return stripLeadingSlash(path);
    }
    return path;
}

//===========================================================================================================
//files
//===========================================================================================================

bool DataLogger::createNewLogFile(){
    if(_log_file){
        closeLogFile();
    }

    String path = normalisePath(_current_filename);
    String fsPath = _using_sd ? toFsPath(path) : path;
    bool exists = _using_sd ? SD.exists(fsPath.c_str()) : LittleFS.exists(fsPath);

    if(_using_sd){
        _log_file = SD.open(fsPath.c_str(), FILE_WRITE);
    }
    else{
        _log_file = LittleFS.open(fsPath, exists ? "a" : "w");
    }

    if(!_log_file){
        DEBUG_PRINTF("[DataLogger] Error: Failed to open log file %s\n", _current_filename.c_str());
        return false;
    }

    //write header only for fresh files
    if(!exists || _log_file.size() == 0){
        if(!writeHeader()){
            DEBUG_PRINTLN(F("[DataLogger] Error: Failed to write header; closing file."));
            closeLogFile();
            return false;
        }
    }

    //append at end
    _log_file.seek(_log_file.size());

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
    char filename[64];
    const char* baseName = (DATA_LOG_FILENAME[0] == '/') ? DATA_LOG_FILENAME + 1 : DATA_LOG_FILENAME;

    String suffix;
    if(_active_label == LOG_LABEL_CALIBRATION){
        suffix = "calibration";
    }
    else if(_active_label == LOG_LABEL_AMBIENT){
        suffix = "ambient";
    }
    else if(_active_label >= SCENT_CLASS_PURE_CAMOMILE && _active_label < SCENT_CLASS_COUNT){
        suffix = sanitizeLabel(SCENT_CLASS_NAMES[_active_label]);
    }

    if(suffix.length() == 0){
        //unlabeled or unknown -> base name
        if(_file_index == 0){
            snprintf(filename, sizeof(filename), "/%s.csv", baseName);
        } else {
            snprintf(filename, sizeof(filename), "/%s_%04d.csv", baseName, _file_index);
        }
    } else {
        if(_file_index == 0){
            snprintf(filename, sizeof(filename), "/%s_%s.csv", baseName, suffix.c_str());
        } else {
            snprintf(filename, sizeof(filename), "/%s_%s_%04d.csv", baseName, suffix.c_str(), _file_index);
        }
    }

    return String(filename);
}

String DataLogger::sanitizeLabel(const char* label) const{
    if(label == nullptr){
        return String();
    }

    String out;
    for(size_t i = 0; label[i] != '\0'; ++i){
        char c = label[i];
        if(isalnum(static_cast<unsigned char>(c))){
            out += c;
        }
        else if(c == ' ' || c == '-' || c == '_'){
            out += '_';
        }
        //ignore other characters
    }
    return out;
}

bool DataLogger::writeHeader(){
    if(!_log_file){
        return false;
    }

    _log_file.print("timestamp,label,temp1,hum1,pres1");
    for(int i=0; i<BME688_NUM_HEATER_STEPS;i++){
        _log_file.printf(",gas1_%d", i);
    }

    _log_file.print(",temp2,hum2,pres2");
    for(int i=0; i<BME688_NUM_HEATER_STEPS;i++){
        _log_file.printf(",gas2_%d", i);
    }

    _log_file.println(",delta_temp,delta_hum,delta_pres,delta_gas,pred_class,pred_conf,anomaly_score,iaq");
    _log_file.flush();
    return true;
}

bool DataLogger::writeEntry(const log_entry_t& entry){
    if(!_log_file){
        return false;
    }

    //timestamp
    _log_file.printf("%lu,",entry.timestamp);

    //label
    if(entry.label == LOG_LABEL_CALIBRATION){
        _log_file.print("calibration,");
    }
    else if(entry.label == LOG_LABEL_AMBIENT){
        _log_file.print("ambient,");
    }
    else if(entry.label >= 0 && entry.label < SCENT_CLASS_COUNT){
        _log_file.printf("%s,", SCENT_CLASS_NAMES[entry.label]);
    }else{
        _log_file.print("None,");
    }

    //sensor data
    auto checkMean = [](const float* data, uint8_t count)->float{
        if(count==0){
            return 0.0f;
        }
        float sum=0.0f;
        for(uint8_t i=0; i<count; i++){
            sum += data[i];
        }
        return sum / (float)count;
    };

    //primary THP
    _log_file.printf("%.2f,%.2f,%.2f,",
    checkMean(entry.sensor_data.primary.temperatures, entry.sensor_data.primary.validReadings),
    checkMean(entry.sensor_data.primary.humidities, entry.sensor_data.primary.validReadings),
    checkMean(entry.sensor_data.primary.pressures, entry.sensor_data.primary.validReadings));


    //primary all gas steps
    for(int i=0; i<BME688_NUM_HEATER_STEPS; i++){
        _log_file.printf(",%.0f", entry.sensor_data.primary.gas_resistances[i]);
    }

    //secondary THP
    _log_file.printf("%.2f,%.2f,%.2f,",
    checkMean(entry.sensor_data.secondary.temperatures, entry.sensor_data.secondary.validReadings),
    checkMean(entry.sensor_data.secondary.humidities, entry.sensor_data.secondary.validReadings),
    checkMean(entry.sensor_data.secondary.pressures, entry.sensor_data.secondary.validReadings));

    //secondary all gas steps
    for(int i=0; i<BME688_NUM_HEATER_STEPS; i++){
        _log_file.printf(",%.0f", entry.sensor_data.secondary.gas_resistances[i]);
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

        _file_index++;
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
    File dir = _using_sd ? SD.open("/", FILE_READ) : LittleFS.open("/", "r");
    const char* base = (DATA_LOG_FILENAME[0] == '/') ? DATA_LOG_FILENAME +1 : DATA_LOG_FILENAME;

    while(dir && count < max){
        File file = dir.openNextFile();
        if(!file){
            break;
        }
        String name = stripLeadingSlash(String(file.name()));
        if(name.startsWith(base)){
            files[count++] = normalisePath(name);
        }
        file.close();
    }
    if(dir){
        dir.close();
    }
    return true;
}

bool DataLogger::deleteLogFile(const char* filename){
    if(!_init){
        return false;
    }

    String path = normalisePath(String(filename));
    String fsPath = _using_sd ? toFsPath(path) : path;

    if(_is_logging && _current_filename == path){
        DEBUG_PRINTLN(F("[DataLogger] Error: Cannot delete active log file."));
        return false;
    }

    bool removed =  _using_sd ? SD.remove(fsPath.c_str()) : LittleFS.remove(fsPath);
    if(removed){
        DEBUG_PRINTF("[DataLogger] Deleted log file %s\n", path.c_str());
        return true;
    }
    else{
        DEBUG_PRINTF("[DataLogger] Error: Failed to delete log file %s\n", path.c_str());
        return false;
    }
}

bool DataLogger::deleteAllLogFiles(){
    if(!_init){
        return false;
    }

    stopLogging();

    File dir = _using_sd ? SD.open("/", FILE_READ) : LittleFS.open("/", "r");
    const char* base = (DATA_LOG_FILENAME[0] == '/') ? DATA_LOG_FILENAME +1 : DATA_LOG_FILENAME;

    while(dir){
        File file = dir.openNextFile();
        if(!file){
            break;
        }
        String name = stripLeadingSlash(String(file.name()));

        // delete any log file matching the base, plus calib debug
        if(name.startsWith(base) ||name.equalsIgnoreCase(stripLeadingSlash(String(CALIB_DEBUG_FILE)))){
            String path = normalisePath(name);
            String fsPath = _using_sd ? toFsPath(path) : path;
            bool removed = _using_sd ? SD.remove(fsPath.c_str()) : LittleFS.remove(fsPath);

            DEBUG_PRINTF("[DataLogger] %s %s\n", removed ? "Deleted" : "Failed to delete", name.c_str());
        }
        file.close();
    }

    if(dir){
        dir.close();
    }

    _file_index = 0;
    _total_entries = 0;
    _current_filename = "";
    _buffer_head = 0;
    _buffer_count = 0;
    _is_logging = false;

    String defaultFile = String(DATA_LOG_FILENAME) + ".csv";
    ensureFileExists(defaultFile.c_str());
    return true;
}

bool DataLogger::exportToSerial(const char* filename){
    if(!_init){
        return false;
    }

    String name = normalisePath(String(filename));
    String fsPath = _using_sd ? toFsPath(name) : name;
    File file = _using_sd ? SD.open(fsPath.c_str(), FILE_READ) : LittleFS.open(fsPath.c_str(), "r");
    if(!file){
        DEBUG_PRINTF("[DataLogger] Error: Failed to open log file %s for export\n", name.c_str());
        return false;
    }

    Serial.println(F("Begin log export:"));
    Serial.println(name);
    while(file.available()){
        Serial.write(file.read());
    }

    file.close();
    Serial.println(F("End log export."));
    return true;
}

//===========================================================================================================
//utility
//===========================================================================================================

bool DataLogger::ensureFileExists(const char* filename){
    if(!_init){
        return false;
    }

    String path = normalisePath(String(filename));
    String fsPath = _using_sd ? toFsPath(path) : path;

    bool exists = _using_sd ? SD.exists(fsPath.c_str()) : LittleFS.exists(fsPath);
    if(exists){
        return true;
    }

    File f = _using_sd ? SD.open(fsPath.c_str(), FILE_WRITE) : LittleFS.open(fsPath.c_str(), "w");
    if(!f){
        DEBUG_PRINTF("[DataLogger] Error: Failed to create file %s\n", path.c_str());
        return false;
    }

    if(f.println(LOG_HEADER) == 0){
        DEBUG_PRINTF("[DataLogger] Error: Failed to write header to %s\n", path.c_str());
        f.close();
        return false;
    }
    f.flush();
    f.close();
    DEBUG_PRINTF("[DataLogger] Created default log file %s\n", path.c_str());
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