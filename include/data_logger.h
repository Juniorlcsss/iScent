#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <Arduino.h>
#include "config.h"
#include <LittleFS.h>
#include "ml_inference.h"
#include "bme668_handler.h"

//log struct
typedef struct{
    uint32_t timestamp;
    dual_sensor_data_t sensor_data;
    ml_prediction_t ml_prediction;
    uint16_t iaq_index;
} log_entry_t;

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

    bool logEntry(const dual_sensor_data_t &data, const ml_prediction_t *pred = nullptr);//
    bool logRawData(const dual_sensor_data_t &data);//
    bool logPrediction(const ml_prediction_t &pred);//

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

    bool deleteLogFile(const String *filename);//
    bool deleteAllLogFiles();//
    bool exportToSerial(const char* filename);//

    //===========================================================================================================
    //cfg
    //===========================================================================================================
    void setBufferSize(uint16_t size);//
    void setAutoFlushInterval(uint32_t interval_ms);//
    void setMaxFileSize(uint32_t size_bytes);//

private:
    File _log_file;
    String _current_filename;

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

    //===========================================================================================================
    //methods
    //===========================================================================================================
    String generateFilename();
    bool writeHeader();
    bool writeEntry(const log_entry_t &entry);
    bool writeBufferToFile();
    bool checkFileRotation();
};

#endif