#ifndef UINPUT_MOUSE_H
#define UINPUT_MOUSE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct uinput_mouse uinput_mouse_t;

bool uinput_mouse_init(uinput_mouse_t **m, int screen_w, int screen_h, int poll_hz);
void uinput_mouse_destroy(uinput_mouse_t *m);

void uinput_mouse_move(uinput_mouse_t *m, int x, int y);
void uinput_mouse_move_smooth(uinput_mouse_t *m, int x, int y, int duration_ms);
bool uinput_mouse_move_done(uinput_mouse_t *m);

void uinput_mouse_click(uinput_mouse_t *m, int button, int duration_us);
void uinput_mouse_press(uinput_mouse_t *m, int button);
void uinput_mouse_release(uinput_mouse_t *m, int button);
void uinput_mouse_wheel(uinput_mouse_t *m, int dy);
int  uinput_mouse_get_screen_w(uinput_mouse_t *m);
int  uinput_mouse_get_screen_h(uinput_mouse_t *m);

uint64_t uinput_mouse_report_count(uinput_mouse_t *m);

#endif
