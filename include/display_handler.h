#ifndef DISPLAY_HANDLER_H
#define DISPLAY_HANDLER_H

#include "config.h"
#include "bme688_handler.h"
#include "ml_inference.h"
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

//menu item struct
typedef struct{
    const char *label;
    void (*action)(void);
    display_mode_t target_mode;
} menu_item_t;

class DisplayHandler {
public:
    DisplayHandler();
    ~DisplayHandler();

    //===========================================================================================================
    //init
    //===========================================================================================================
    bool begin(TwoWire *wire = &Wire);
    bool isReady() const;

    //===========================================================================================================
    //modes
    //===========================================================================================================
    void setMode(display_mode_t mode);
    display_mode_t getMode() const;
    void nextMode();
    void prevMode();

    //===========================================================================================================
    //update
    //===========================================================================================================
    void update();
    void clear();
    void refresh();

    //per mode  
    void showSplashScreen();
    void showStatusScreen(system_state_t state, error_code_t error);
    void showSensorDataScreen(const dual_sensor_data_t &data);
    void showPredictionScreen(const ml_prediction_t &prediction, const dual_sensor_data_t &data);
    void showGraphScreen(const float *data, uint16_t length, const char* title);
    void showSettingsScreen();
    void showLoggingScreen(uint32_t samples, bool active);
    void ShowErrorScreen(error_code_t error);
    void showCalibrationScreen(float progress, const char* msg);

    //===========================================================================================================
    //ui objects
    //===========================================================================================================
    void drawProgressBar(int16_t x, int16_t y, int16_t w, int16_t h, float progress, const char* label = nullptr);
    void drawBarGraph(int16_t x, int16_t y, int16_t h, int16_t w, const float* vals, uint8_t count, float maxVal);
    void drawStatusIcon(int16_t x, int16_t y, system_state_t state);
    void drawBatteryIcon(int16_t x, int16_t y, float volt);
    void drawSignalIcon(int16_t x, int16_t y, int8_t rssi);

    void printCentered(const char* txt, int16_t y);
    void printRight(const char* txt, int16_t y);

    //scalers
    int16_t scaleX(float ratio) const;
    int16_t scaleY(float ratio) const;
    int16_t scaleWidth(float ratio) const;
    int16_t scaleHeight(float ratio) const;

    //===========================================================================================================
    //menu nav
    //===========================================================================================================
    void showMenu(const menu_item_t *items, uint8_t itemCount, uint8_t selected, const char* title = "Menu");
    void menuUp();
    void menuDown();
    void menuSelect();
    uint8_t getSelectedMenuIndex() const;

    //prediction actions
    void setPredictionSelection(uint8_t idx);
    uint8_t getPredictionSelection() const;

    static const uint8_t MENU_VISIBLE_COUNT =4;

    //===========================================================================================================
    //cfg
    //===========================================================================================================
    void setBrightness(uint8_t lvl);
    void setInverted(bool inverted);
    void setTimeout(uint32_t timeout);
    void resetTimeout();
    bool isTimedOut() const;

private:
    //===========================================================================================================
    //vars
    //===========================================================================================================
    Adafruit_SSD1306* _dsp;
    TwoWire* _wire;
    display_mode_t _mode;
    display_mode_t _prevMode;

    //timings
    uint32_t _prevUpdate;
    uint32_t _prevActivity;
    uint32_t _timeout;

    uint8_t _brightness;
    bool _inverted;
    bool _ready;
    bool _timedOut;

    uint8_t _menuSelection;
    uint8_t _menuCount;
    const menu_item_t* _menuItems;
    uint8_t _predictionSelection;

    //buffer
    static const uint8_t GRAPH_BUFFER_SIZE = 64;
    float _graphBuffer[GRAPH_BUFFER_SIZE];
    uint8_t _graphIdx;

    //===========================================================================================================
    //methods
    //===========================================================================================================
    void drawHeader(const char* title);
    void drawFooter(const char* left, const char* right);
    void updateGraphBuffer(float val);
    void formatGasValue(float ohms, float &val, const char* &unit) const;
};

#endif