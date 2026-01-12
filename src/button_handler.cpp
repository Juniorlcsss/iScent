#include "button_handler.h"

ButtonHandler::ButtonHandler():
    _callback(nullptr),
    _long_press_time(BUTTON_LONG_PRESS_MS),
    _double_press_time(300),
    _debounce_time(BUTTON_DEBOUNCE_MS),
    _init(false),
    _active_low(false)
{
    memset(_buttons,0,sizeof(_buttons));
}

ButtonHandler::~ButtonHandler(){
}


//===========================================================================================================
//init
//===========================================================================================================

bool ButtonHandler::begin(){
    DEBUG_PRINTLN(F("[ButtonHandler] Initializing button handler"));

    addButton(BUTTON_SELECT, BUTTON_SELECT_PIN);
    addButton(BUTTON_DOWN, BUTTON_DOWN_PIN);

    _init = true;
    DEBUG_PRINTLN(F("[ButtonHandler] Button handler initialized successfully"));
    return true;
}

bool ButtonHandler::addButton(button_id_t id, uint8_t pin, bool active_low){
    if(id >= BUTTON_COUNT || (int)pin <0){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d or pin %d\n", id, pin);
        return false;
    }
    _buttons[id].pin = pin;
    _active_low = active_low;

    if(active_low){
        pinMode(pin, INPUT_PULLUP);
    }
    else{
        pinMode(pin, INPUT_PULLDOWN);
    }
        
    //initialize state
    _buttons[id].current_state = readButton(id);
    _buttons[id].last_state = _buttons[id].current_state;
    _buttons[id].pressed = false;
    _buttons[id].press_start_time = 0;
    _buttons[id].last_press_time = 0;
    _buttons[id].press_count = 0;
    _buttons[id].last_event = BUTTON_EVENT_NONE;
    DEBUG_PRINTF("[ButtonHandler] Added button ID %d on pin %d (active_low=%d)\n", id, pin, active_low);

    return true;
}

//===========================================================================================================
//update
//===========================================================================================================

void ButtonHandler::update(){
    if(!_init){
        DEBUG_PRINTLN(F("[ButtonHandler] Warning: ButtonHandler not initialized. Call begin() first."));
        return;
    }
    for(uint8_t i=0; i<BUTTON_COUNT; i++){
        processButton((button_id_t)i);
    }
}

void ButtonHandler::processButton(button_id_t id){
    button_state_t &button = _buttons[id];

    bool current = readButton(id);
    uint32_t now = millis();

    //debounce logic
    static uint32_t last_change_time[BUTTON_COUNT] = {0};
    if(current != button.last_state){
        last_change_time[id] = now;
    }

    if((now-last_change_time[id]) < _debounce_time){
        button.last_state = current;
        return; //debouncing
    }

    //state change
    if(current != button.current_state){
        button.current_state = current;

        if(current){
            //pressed
            button.pressed = true;
            button.press_start_time = now;
            button.last_event = BUTTON_EVENT_PRESS;

            //check for double press
            if((now-button.last_press_time) <_double_press_time){
                button.press_count++;
            }
            else{
                button.press_count = 1;
            }
            if(_callback){
                _callback(id, BUTTON_EVENT_PRESS);
            }
        }
        else{
            //released
            uint32_t press_duration = now - button.press_start_time;

            if(press_duration >= _long_press_time){
                button.last_event = BUTTON_EVENT_LONG_PRESS;
                if(_callback){
                    _callback(id, BUTTON_EVENT_LONG_PRESS);
                }
            }

            else if(button.press_count == 2){
                button.last_event = BUTTON_EVENT_DOUBLE_PRESS;
                if(_callback){
                    _callback(id, BUTTON_EVENT_DOUBLE_PRESS);
                }
                button.press_count = 0; //reset count after double press
            }

            else{
                button.last_event = BUTTON_EVENT_SHORT_PRESS;
                if(_callback){
                    _callback(id, BUTTON_EVENT_SHORT_PRESS);
                }
            }

            button.pressed = false;
            button.last_press_time = now;
            if(_callback){
                _callback(id, BUTTON_EVENT_RELEASE);
            }
        }
    }

    //check longpress while held
    if(button.pressed & (now - button.press_start_time) >= _long_press_time){
        if(button.last_event != BUTTON_EVENT_LONG_PRESS){
            button.last_event = BUTTON_EVENT_LONG_PRESS;
            if(_callback){
                _callback(id, BUTTON_EVENT_LONG_PRESS);
            }
        }
    }

    button.last_state = current;
}

bool ButtonHandler::readButton(button_id_t id){
    if(id >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in readButton\n", id);
        return false;
    }

    bool raw_state = digitalRead(_buttons[id].pin);

    //invert if active low
    if(_active_low){
        return !raw_state;
    }
    else{
        return raw_state;
    }
}

//===========================================================================================================
//state queries
//===========================================================================================================

bool ButtonHandler::isPressed(button_id_t button) const{
    if(button >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in isPressed\n", button);
        return false;
    }
    return _buttons[button].pressed;
}

bool ButtonHandler::wasPressed(button_id_t button){
    if(button >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in wasPressed\n", button);
        return false;
    }
    if(_buttons[button].last_event == BUTTON_EVENT_SHORT_PRESS){
        _buttons[button].last_event = BUTTON_EVENT_NONE; //reset event after query
        return true;
    }
    return false;
}

bool ButtonHandler::wasLongPressed(button_id_t button){
    if(button >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in wasLongPressed\n", button);
        return false;
    }
    if(_buttons[button].last_event == BUTTON_EVENT_LONG_PRESS){
        _buttons[button].last_event = BUTTON_EVENT_NONE; //reset event after query
        return true;
    }
    return false;
}

button_event_t ButtonHandler::getLastEvent(button_id_t button) const{
    if(button >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in getLastEvent\n", button);
        return BUTTON_EVENT_NONE;
    }
    return _buttons[button].last_event;
}

uint32_t ButtonHandler::getPressDuration(button_id_t button)const{
    if(button >= BUTTON_COUNT){
        DEBUG_PRINTF("[ButtonHandler] Invalid button ID %d in getPressDuration\n", button);
        return 0;
    }
    if(_buttons[button].pressed){
        return millis() - _buttons[button].press_start_time;
    }
    return 0;
}

//===========================================================================================================
//cfg
//===========================================================================================================

void ButtonHandler::setCallback(button_callback_t callback){
    _callback = callback;
}

void ButtonHandler::setLongPressTime(uint32_t duration_ms){
    _long_press_time = duration_ms;
}

void ButtonHandler::setDoublePressTime(uint32_t duration_ms){
    _double_press_time = duration_ms;
}

void ButtonHandler::setDebounceTime(uint32_t duration_ms){
    _debounce_time = duration_ms;
}