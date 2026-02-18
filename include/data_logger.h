#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <Arduino.h>
#include <FS.h>
#include <SPI.h>
#include <SD.h>
#include "config.h"
#include <LittleFS.h>
#include "ml_inference.h"
#include "bme688_handler.h"

//log struct
typedef struct{
    uint32_t timestamp;
    dual_sensor_data_t sensor_data;
    ml_prediction_t ml_prediction;
    uint16_t iaq_index;
    int16_t label; //-1 for none
} log_entry_t;

//labels
static const int16_t LOG_LABEL_CALIBRATION = -2;
static const int16_t LOG_LABEL_AMBIENT = -3;
static const int16_t LOG_LABEL_PRED = -4;

class DataLogger {
public:
    DataLogger();
    ~DataLogger();

    //===========================================================================================================
    //init
    //===========================================================================================================
    bool begin();//
    bool isReady() const;//

    //===========================================================================================================
    //log
    //===========================================================================================================
    bool startLogging(const char* filename = nullptr);//
    bool stopLogging();//
    bool isLogging() const;//
    String toFsPath(const String &path) const;

    bool logEntry(const dual_sensor_data_t &data, const ml_prediction_t *pred = nullptr);//
    bool logRawData(const dual_sensor_data_t &data);//
    bool logPrediction(const ml_prediction_t &pred);//

    //calib debug
    void startCalibDebug();
    void logCalibDebug(const String &line);
    bool flushCalibDebug();

    //===========================================================================================================
    //memory
    //===========================================================================================================
    size_t getUsedSpace() const;//
    size_t getFreeSpace() const;//
    bool flush();//
    bool isBufferFull() const;//
    uint16_t getBufferCount() const;//

    //===========================================================================================================
    //files
    //===========================================================================================================
    bool createNewLogFile();//
    bool closeLogFile();//
    String getCurrentFilename() const;//
    uint32_t getCurrentFileSize() const;//
    uint32_t getTotalLoggedEntries() const;//
    bool listLogFiles(String *files, uint8_t max, uint8_t &count);//

    bool deleteLogFile(const char *filename);//
    bool deleteAllLogFiles();//
    bool exportToSerial(const char* filename);//

    //utility
    bool ensureFileExists(const char* filename);
    bool isUsingSD() const { return _using_sd; }

    //labels
    void setActiveLabel(int16_t label);
    int16_t getActiveLabel() const;

    //===========================================================================================================
    //cfg
    //===========================================================================================================
    void setBufferSize(uint16_t size);//
    void setAutoFlushInterval(uint32_t interval_ms);//
    void setMaxFileSize(uint32_t size_bytes);//

private:
    File _log_file;
    String _current_filename;
    bool _using_sd;

    //===========================================================================================================
    //buffer
    //===========================================================================================================
    log_entry_t *_buffer;
    uint16_t _buffer_size;
    uint16_t _buffer_head;
    uint16_t _buffer_count;

    //===========================================================================================================
    //timing
    //===========================================================================================================
    uint32_t _last_flush_time;
    uint32_t _auto_flush_interval;
    uint32_t _max_file_size;

    //===========================================================================================================
    //state
    //===========================================================================================================
    bool _init;
    bool _is_logging;
    uint32_t _total_entries;
    uint32_t _file_entry_count;
    uint16_t _file_index;
    int16_t _active_label;
    String normalisePath(const String &path) const;
    String stripLeadingSlash(const String &path) const;

    //calib debug state
    static const uint8_t CALIB_DEBUG_MAX_LINES = 64;
    String _calib_debug_buffer[CALIB_DEBUG_MAX_LINES];
    uint8_t _calib_debug_count;
    uint32_t _calib_debug_last_ms;
    bool _calib_debug_active;

    //===========================================================================================================
    //methods
    //===========================================================================================================
    String generateFilename();
    String sanitizeLabel(const char* label) const;
    bool writeHeader();
    bool writeEntry(const log_entry_t &entry);
    bool writeBufferToFile();
    bool checkFileRotation();
};

#endif