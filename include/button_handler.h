#ifndef BUTTON_HANDLER_H
#define BUTTON_HANDLER_H

#include <Arduino.h>
#include "config.h"

//button event types
typedef enum{
    BUTTON_EVENT_NONE =0,
    BUTTON_EVENT_PRESS,
    BUTTON_EVENT_RELEASE,
    BUTTON_EVENT_SHORT_PRESS,
    BUTTON_EVENT_LONG_PRESS,
    BUTTON_EVENT_DOUBLE_PRESS,
} button_event_t;

//button identifier types
typedef enum{
    BUTTON_DOWN,
    BUTTON_SELECT,
    BUTTON_COUNT
} button_id_t;

//button state
typedef struct{
    uint8_t pin;
    bool current_state;
    bool last_state;
    bool pressed;
    uint32_t press_start_time;
    uint32_t last_press_time;
    uint16_t press_count;
    button_event_t last_event;
    bool active_low;
} button_state_t;

typedef void (*button_callback_t)(button_id_t button, button_event_t event);

class ButtonHandler {
public:
    ButtonHandler();
    ~ButtonHandler();

    //===========================================================================================================
    //init
    //===========================================================================================================

    bool begin();
    bool addButton(button_id_t id, uint8_t pin, bool active_low=true);

    //===========================================================================================================
    //update
    //===========================================================================================================
    void update();

    //===========================================================================================================
    //queries
    //===========================================================================================================
    bool isPressed(button_id_t id) const;
    bool wasPressed(button_id_t id) ;
    bool wasLongPressed(button_id_t id);
    button_event_t getLastEvent(button_id_t id) const;
    uint32_t getPressDuration(button_id_t id) const;

    //===========================================================================================================
    //callback
    //===========================================================================================================
    void setCallback(button_callback_t callback);
    void setLongPressTime(uint32_t duration_ms);
    void setDoublePressTime(uint32_t duration_ms);
    void setDebounceTime(uint32_t duration_ms);

private:
    button_state_t _buttons[BUTTON_COUNT];
    button_callback_t _callback;

    uint32_t _long_press_time;
    uint32_t _double_press_time;
    uint32_t _debounce_time;

    bool _init;
    bool _active_low;

    void processButton(button_id_t id);
    bool readButton(button_id_t id);


};

#endif