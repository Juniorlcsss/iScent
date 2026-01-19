#include "display_handler.h"

DisplayHandler::DisplayHandler():
    _dsp(nullptr),
    _wire(nullptr),
    _mode(DISPLAY_MODE_SPLASH),
    _prevMode(DISPLAY_MODE_SPLASH),
    _prevUpdate(0),
    _prevActivity(0),
    _timeout(DISPLAY_TIMEOUT),
    _brightness(DISPLAY_CONTRAST),
    _inverted(false),
    _ready(false),
    _timedOut(false),
    _menuSelection(0),
    _menuCount(0),
    _graphIdx(0),
    _menuItems(nullptr)

{
    memset(_graphBuffer, 0, sizeof(_graphBuffer));
}

DisplayHandler::~DisplayHandler() {
    if(_dsp) {
        delete _dsp;
        _dsp = nullptr;
    }
}

//===========================================================================================================
//init
//===========================================================================================================
bool DisplayHandler::begin(TwoWire *wire){
    DEBUG_PRINTLN(F("[DisplayHandler] Initialising display...")); 

    _wire = wire;
    _dsp = new Adafruit_SSD1306(DISPLAY_WIDTH, DISPLAY_HEIGHT, _wire, DISPLAY_RESET_PIN);

    if(!_dsp->begin(SSD1306_SWITCHCAPVCC, DISPLAY_I2C_ADDR)){
        DEBUG_PRINTLN(F("[DisplayHandler] Display init failed!"));
        delete _dsp;
        _dsp = nullptr;
        return false;
    }

    _dsp->setTextColor(SSD1306_WHITE);
    _dsp->setTextSize(1);
    _dsp->cp437();
    setBrightness(_brightness);

    _ready = true;
    _prevActivity = millis();
    showSplashScreen();
    DEBUG_PRINTLN(F("[DisplayHandler] Display initialised successfully."));
    return true;
}

bool DisplayHandler::isReady() const {
    return _ready;
}

//===========================================================================================================
//modes
//===========================================================================================================

void DisplayHandler::setMode(display_mode_t mode){
    if(mode != _mode){
        _prevMode = _mode;
        _mode = mode;
        resetTimeout();
        update();
    }
}

display_mode_t DisplayHandler::getMode() const {
    return _mode;
}

void DisplayHandler::nextMode(){
    uint8_t nextMode = (_mode +1) % DISPLAY_MODE_COUNT;
    if(nextMode == DISPLAY_MODE_SPLASH){
        nextMode++;
    }
    setMode((display_mode_t)nextMode);
}

void DisplayHandler::prevMode(){
    uint8_t prevMode = (_mode == 0) ? DISPLAY_MODE_COUNT -1 : _mode -1;
    if(prevMode == DISPLAY_MODE_SPLASH){
        prevMode = DISPLAY_MODE_COUNT -1;
    }
    setMode((display_mode_t)prevMode);
}

//===========================================================================================================
//update
//===========================================================================================================

void DisplayHandler::update(){
    if(!_ready) return;

    if(_timeout > 0 && (millis() - _prevActivity) >= _timeout){
        if(!_timedOut){
            _timedOut = true;
            _dsp->dim(true);
        }
        return;
    }
    _prevUpdate = millis();
}

void DisplayHandler::clear(){
    if(!_ready) return;
    _dsp->clearDisplay();
}

void DisplayHandler::refresh(){
    if(!_ready) return;
    _dsp->display();
}

void DisplayHandler::showSplashScreen(){
    if (!_ready) return;

    clear();

    //draw
    _dsp->setTextSize(2);
    printCentered("iScent", 5);

    _dsp->setTextSize(1);
    printCentered("Portable Sniffer", 30);
    String ver = String("v") + String(SOFTWARE_VERSION);
    printCentered(ver.c_str(), 40);
    printCentered("Initialising...", 50);
    refresh();
}

void DisplayHandler::showStatusScreen(system_state_t state, error_code_t error){
    if(!_ready) return;
    clear();
    drawHeader("Status");

    _dsp->setTextSize(1);
    int16_t y = 16;

    _dsp->setCursor(0,y);
    _dsp->print("State:");

    switch(state){
        case STATE_INIT:
            _dsp->println("Init");
            break;
        case STATE_WARMUP:
            _dsp->println("WarmUp");
            break;
        case STATE_IDLE:
            _dsp->println("Idle");
            break;
        case STATE_CALIBRATING:
            _dsp->println("Calibr");
            break;
        case STATE_SAMPLING:
            _dsp->println("Sampling");
            break;
        case STATE_INFERENCING:
            _dsp->println("Inferencing");
            break;
        case STATE_LOGGING:
            _dsp->println("Logging");
            break;
        case STATE_BLE_CONNECTED:
            _dsp->println("BLE Conn");
            break;
        case STATE_SLEEP:
            _dsp->println("Sleep");
            break;
        case STATE_ERROR:
            _dsp->println("Error!");
            break;
        default:
            _dsp->println("Unknown");
            break;
    }
    y+=9;

    //error display
    if(error != ERROR_NONE){
        _dsp->setCursor(0,y);
        _dsp->print("Err: ");
        _dsp->println(ERROR_CODE_NAMES[error]);
        y+=9;
    }

    //sys info
    _dsp->setCursor(0,y);
    _dsp->printf("Up: %lus", millis() / 1000);
    
    y+=9;
    
    _dsp->setCursor(0,y);
    _dsp->printf("Mem: %lukb", rp2040.getFreeHeap() / 1024);

    refresh();
}

void DisplayHandler::showSensorDataScreen(const dual_sensor_data_t &data){
    if (!_ready) return;
    clear();
    drawHeader("Sensor Data");

    if(!data.primary.complete) return;
    //primary data
    _dsp->setCursor(0, 16);
    _dsp->print("Primary: ");
    _dsp->printf("T:%.1fC, H:%.1f%%\n", data.primary.temperatures[0], data.primary.humidities[0]);
    _dsp->printf("P:%.1fPa, G:%.1fOhm\n", data.primary.pressures[0], data.primary.gas_resistances[0]);

    //secondary data
    if(data.secondary.complete){
        _dsp->setCursor(0, 40);
        _dsp->print("Secondary: ");
        _dsp->printf("T:%.1fC, H:%.1f%%\n", data.secondary.temperatures[0], data.secondary.humidities[0]);
        _dsp->printf("P:%.1fPa, G:%.1fOhm\n", data.secondary.pressures[0], data.secondary.gas_resistances[0]);
    }

    //deltas
    _dsp->setCursor(0, 64);
    _dsp->printf("dT:%.1f dG:%.0f", data.delta_temp, data.delta_gas_avg);
}

void DisplayHandler::showPredictionScreen(const ml_prediction_t &pred, const dual_sensor_data_t &data){
    if(!_ready) return;

    clear();
    drawHeader("Detection");

    //prediction
    _dsp->setTextSize(1);
    _dsp->setCursor(0,16);
    _dsp->print(F("Prediction: "));

    if(pred.valid){
        _dsp->setTextSize(1);
        _dsp->setCursor(0,26);
        if(pred.predictedClass < SCENT_CLASS_COUNT){
            _dsp->printf(SCENT_CLASS_NAMES[pred.predictedClass]);
        }
        else{
            _dsp->print("Unknown");
        }
        //draw bar
        drawProgressBar(0,36,100,8,pred.confidence, "Confidence");

        //anomaly
        if(pred.isAnomalous){
            _dsp->setCursor(0,48);
            _dsp->print("Anomalous!");
        }
    }
    else{
        _dsp->setCursor(0,26);
        _dsp->println("Analysing...");

        //display current gas sensor readings
        _dsp->setCursor(0,48);
        _dsp->printf("Gas: %.1fOhm", data.primary.gas_resistances[0]);
    }

    //mini sensor bar graph
    drawBarGraph(102, 14, 24, 48, data.primary.gas_resistances, BME688_NUM_HEATER_STEPS, 500000.0f);

    refresh();
}

void DisplayHandler::showGraphScreen(const float* data, uint16_t count, const char* title){
    if(!_ready) return;

    clear();
    drawHeader(title);

    //bounds check
    if(count ==0) return;
    if(count > GRAPH_BUFFER_SIZE) count = GRAPH_BUFFER_SIZE;
    if(data==nullptr) return;

    //find min and max
    float minVal = data[0], maxVal = data[0];
    for(uint16_t i=1; i<count; i++){
        if(data[i] < minVal) minVal = data[i];
        if(data[i] > maxVal) maxVal = data[i];
    }

    //scale graph
    int16_t graphX = scaleX(0.15f);
    int16_t graphY= scaleY(0.22f);
    int16_t graphW = scaleWidth(0.85f);
    int16_t graphH = scaleHeight(0.70f);

    //draw border
    _dsp->drawRect(graphX-1, graphY-1, graphW+2, graphH+2, SSD1306_WHITE);
    //draw data
    float range = maxVal - minVal;
    if(range < 1.0f){
        range = 1.0f;
    }

    float pixelPerPoint = (float)graphW / count;
    for(uint16_t i=0; i<count; i++){
        uint16_t x1 = graphX+ (int16_t)(i * pixelPerPoint - pixelPerPoint);
        uint16_t x2 = graphX+ (int16_t)(i*pixelPerPoint);
        uint16_t y1= graphY + graphH - (int16_t)((data[i-1]-minVal) / range * graphH);
        uint16_t y2= graphY + graphH - (int16_t)((data[i]-minVal) / range * graphH);
        
        //bounds
        if(x2 < DISPLAY_WIDTH){
            _dsp->drawLine(x1, y1, x2, y2, SSD1306_WHITE);
        }

    }

    //y labels
    _dsp->setTextSize(1);
    _dsp->setCursor(0, graphY);
    _dsp->printf("%.1f", maxVal);
    _dsp->setCursor(0, graphY + graphH -6);
    _dsp->printf("%.1f", minVal);
    refresh();
}

void DisplayHandler::showCalibrationScreen(float progress, const char* msg){
    if(!_ready) return;
    clear();
    drawHeader("Calibration");

    _dsp->setCursor(0,16);
    _dsp->printf(msg);

    drawProgressBar(10,35,DISPLAY_WIDTH -20, 12, progress, nullptr);

    _dsp->setCursor(0, 60);
    _dsp->printf("Progress: %.1f%%", progress *100.0f);
    refresh();
}


void DisplayHandler::showSettingsScreen(){
    if(!_ready) return;
    clear();
    drawHeader("Settings");

    _dsp->setCursor(0,16);
    _dsp->println("1. Calibrate");
    _dsp->println("2. Heater Profile");
    _dsp->println("3. ML Thresholds");
    _dsp->println("4. Display");
    _dsp->println("5. Export Data");
    refresh();
}

void DisplayHandler::showLoggingScreen(uint32_t samples, bool active){
    if(!_ready) return;

    clear();
    drawHeader("Data Logging");

    _dsp->setCursor(0,16);
    _dsp->printf("Status: %s", active ? "Active" : "Inactive");
    _dsp->printf("Samples Logged: %lu", samples);

    if(active){
        if((millis()/500) %2){
            _dsp->fillCircle(120,10,4,SSD1306_WHITE);
        }
    }
    refresh();
}

void DisplayHandler::ShowErrorScreen(error_code_t error){
    if(!_ready) return;

    clear();
    drawHeader("Error");
    _dsp->setTextSize(2);
    printCentered("Error!", 20);
    _dsp->println(ERROR_CODE_NAMES[error]);
    _dsp->setCursor(0, 50);
    _dsp->print(F("Press any button..."));
    refresh();
}

//===========================================================================================================
//ui
//===========================================================================================================

void DisplayHandler::drawProgressBar(int16_t x, int16_t y, int16_t w, int16_t h, float progress, const char *label){
    if(!_ready) return;

    if(w <= 2 || h <= 2){
        DEBUG_VERBOSE_PRINTLN("[DisplayHandler] drawProgressBar: w or h too small");
        return;
    }
    if(x <0 || y<0 || x+w > DISPLAY_WIDTH || y+h > DISPLAY_HEIGHT){
        DEBUG_VERBOSE_PRINTLN("[DisplayHandler] drawProgressBar: out of bounds");
        return;
    }

    progress = CONSTRAIN_FLOAT(progress, 0.0f, 1.0f);

    //border
    _dsp->drawRect(x, y, w, h, SSD1306_WHITE);
    //fill
    int16_t fillW = (int16_t)(progress * (w-2));

    if(fillW > 0){
        _dsp->fillRect(x+1, y+1, fillW, h-2, SSD1306_WHITE);
    }
    else{
        DEBUG_VERBOSE_PRINTLN("[DisplayHandler] drawProgressBar: fillW <= 0");
        return;
    }

    //label
    if(label != nullptr && (x+w+4) < DISPLAY_WIDTH){
        _dsp->setCursor(x+w+4, y+(h-8)/2);
        _dsp->print(label);
    }
}

void DisplayHandler::drawBarGraph(int16_t x, int16_t y, int16_t h, int16_t w, const float* vals, uint8_t count, float maxVal){
    if(!_ready) return;
    if(count ==0 || maxVal == 0) return;
    if(vals == nullptr) return;

    int16_t barW = w / count;
    if(barW <= 0){
        DEBUG_VERBOSE_PRINTLN("[DisplayHandler] drawBarGraph: barW <= 0");
        return;
    }

    for(uint8_t i=0; i<count; i++){
        float v = CONSTRAIN_FLOAT((vals[i]/maxVal), 0.0f, 1.0f);

        int16_t barH = (int16_t)(v*h);
        int16_t barX = x+i*barW;
        int16_t barY = y + h - barH;
        
        if(barX >= 0 && barX+barW <= DISPLAY_WIDTH && barY >=0 && barY+barH <= DISPLAY_HEIGHT){
            _dsp->fillRect(barX, barY, barW-1, barH, SSD1306_WHITE);
        }
        else{
            DEBUG_VERBOSE_PRINTLN("[DisplayHandler] drawBarGraph: bar out of bounds");
        }
    }

}

void DisplayHandler::drawStatusIcon(int16_t x, int16_t y, system_state_t state){
    if(!_ready) return;

    switch(state){
        case STATE_IDLE:
            //draw idle icon
            _dsp->drawCircle(x+4, y+4, 4, SSD1306_WHITE);
            break;

        case STATE_SAMPLING:
            //draw sampling icon
            _dsp->drawRect(x, y, 8, 8, SSD1306_WHITE);
            break;

        case STATE_LOGGING:
            //draw logging icon
            _dsp->drawTriangle(x, y, x+8, y+4, x, y+8, SSD1306_WHITE);
            break;

        case STATE_INFERENCING:
            for(int i=0; i<3; i++){
                if(((millis()/300 ) %3) >=i){
                    _dsp->fillCircle(x+2+i*3, y+4, 1, SSD1306_WHITE);
                }
            }
            break;

        case STATE_ERROR:
            //draw error icon
            _dsp->drawLine(x, y, x+8, y+8, SSD1306_WHITE);
            _dsp->drawLine(x+8, y, x, y+8, SSD1306_WHITE);
            break;

        default:
            //unknown state
            _dsp->drawRect(x, y, 8, 8, SSD1306_WHITE);
            _dsp->drawLine(x, y, x+8, y+8, SSD1306_WHITE);
            _dsp->drawLine(x+8, y, x, y+8, SSD1306_WHITE);
            break;
    }
}

void DisplayHandler::printCentered(const char* text, int16_t y){
    if(!_ready) return;

    int16_t x1,y1;
    uint16_t w,h;
    _dsp->getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
    _dsp->setCursor((DISPLAY_WIDTH - w)/2, y);
    _dsp->print(text);
}

void DisplayHandler::printRight(const char* txt, int16_t y){
    if (!_ready) return;

    int16_t x1, y1;
    uint16_t w, h;
    _dsp->getTextBounds(txt,0,0,&x1,&y1,&w,&h);
    _dsp->setCursor(DISPLAY_WIDTH - w, y);
    _dsp->print(txt);
}

void DisplayHandler::drawHeader(const char* title){
    if (!_ready)return;

    _dsp->setTextSize(1);
    _dsp->setCursor(0,0);
    _dsp->print(title);
    _dsp->drawLine(0,10, DISPLAY_WIDTH,10, SSD1306_WHITE);
}

void DisplayHandler::drawFooter(const char* l, const char *r){
    if(!_ready)return;

    _dsp->drawLine(0,54,DISPLAY_WIDTH,54,SSD1306_WHITE);
    _dsp->setCursor(0, 56);
    _dsp->print(l);
    printRight(r, 56);
}

void DisplayHandler::updateGraphBuffer(float val){
    if(!_ready) return;

    //shift buffer
    for(uint16_t i=1; i<GRAPH_BUFFER_SIZE; i++){
        _graphBuffer[i-1] = _graphBuffer[i];
    }
    //add new value
    _graphBuffer[GRAPH_BUFFER_SIZE -1] = val;
}

void DisplayHandler::drawBatteryIcon(int16_t x, int16_t y, float volt){
    if(!_ready) return;

    _dsp->drawRect(x, y, 14, 8, SSD1306_WHITE);
    _dsp->drawRect(x+14, y+2, 2, 4, SSD1306_WHITE);

    //fill based on voltage
    float displayVoltage = CONSTRAIN_FLOAT((volt - 3.0f) / (4.2f - 3.0f), 0.0f, 1.0f);

    int16_t fillW = (int16_t)(displayVoltage * 12.0f);
    if(fillW >0){
        _dsp->fillRect(x+1, y+1, fillW, 6, SSD1306_WHITE);
    }
    else{
        DEBUG_VERBOSE_PRINTLN("[DisplayHandler]  Looks like someone forgot to change again");
    }

}
void DisplayHandler::drawSignalIcon(int16_t x, int16_t y, int8_t rssi){
    if(!_ready) return;

    int8_t bars = 0;
    if(rssi >= -50){
        bars = 4;
    }
    else if(rssi >= -60){
        bars = 3;
    }
    else if(rssi >= -70){
        bars = 2;
    }
    else if(rssi >= -80){
        bars = 1;
    }
    else{
        bars = 0;
    }

    for(int8_t i=0; i<4; i++){
        if(i<bars){
            _dsp->fillRect(x+i*2, y+6-i*2, 1, i*2+1, SSD1306_WHITE);
        }
        else{
            _dsp->drawRect(x+i*2, y+6-i*2, 1, i*2+1, SSD1306_WHITE);
        }
    }
}

int16_t DisplayHandler::scaleX(float percentage) const{
    return (int16_t)(percentage * DISPLAY_WIDTH);
}

int16_t DisplayHandler::scaleY(float percentage) const{
    return (int16_t)(percentage * DISPLAY_HEIGHT);
}

int16_t DisplayHandler::scaleWidth(float percentage) const{
    return (int16_t)(percentage * DISPLAY_WIDTH);
}

int16_t DisplayHandler::scaleHeight(float percentage) const{
    return (int16_t)(percentage * DISPLAY_HEIGHT);
}


//===========================================================================================================
//menu nav
//===========================================================================================================

void DisplayHandler::showMenu(const menu_item_t *items, uint8_t itemCount, uint8_t selected){
    if(!_ready || itemCount == 0) return;
    if(selected >= itemCount) selected = 0;

    _menuCount = itemCount;
    _menuSelection = selected;
    _menuItems = items;

    clear();
    drawHeader("Menu");

    int16_t y=12;
    for(uint8_t i=0; i<itemCount && i<5; i++){
        if(i==selected){
            _dsp->setCursor(0, y);
            _dsp->print("> ");
        }
        _dsp->setCursor(8, y);
        _dsp->print(items[i].label);
        y += 10;
    }
    drawFooter("Up/Down: Nav", "Select: OK");
    refresh();
}

void DisplayHandler::menuUp(){
    if(_menuCount ==0) return;

    if(_menuSelection >0){
        _menuSelection--;
    }
    else{
        _menuSelection = _menuCount -1;
    }
    resetTimeout();
}

void DisplayHandler::menuDown(){
    if(_menuCount ==0) return;

    if(_menuSelection < _menuCount -1){
        _menuSelection++;
    }
    else{
        _menuSelection = 0;
    }
    resetTimeout();
}

void DisplayHandler::menuSelect(){
    if(_menuItems !=nullptr && _menuSelection < _menuCount && _menuCount > 0){
        //call
        if(_menuItems[_menuSelection].action != nullptr){
            _menuItems[_menuSelection].action();
        }
    }
    resetTimeout();
}

uint8_t DisplayHandler::getSelectedMenuIndex() const{
    return _menuSelection;
}



//===========================================================================================================
//cfg
//===========================================================================================================

void DisplayHandler::setBrightness(uint8_t brightness){
    if(!_ready) return;
    _brightness = CONSTRAIN_UINT8(brightness);
    _dsp->ssd1306_command(SSD1306_SETCONTRAST);
    _dsp->ssd1306_command(_brightness);
}

void DisplayHandler::setInverted(bool inverted){
    if(!_ready) return;
    _inverted = inverted;
    _dsp->invertDisplay(_inverted);
}

void DisplayHandler::setTimeout(uint32_t timeout){
    _timeout = timeout;
}

void DisplayHandler::resetTimeout(){
    _prevActivity = millis();
    if(_timedOut){
        _timedOut = false;
        _dsp->dim(false);
    }
}

bool DisplayHandler::isTimedOut() const {
    return _timedOut;
}

