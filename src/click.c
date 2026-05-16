/*
 * click.c — XTest-based click for Xvfb isolation
 *
 * DISPLAY=:99 olduğunda sadece Xvfb'ye tıklar, gerçek masaüstüne dokunmaz.
 * XTest extension kullanır — isTrusted=true, Xvfb'nin sanal input'una gider.
 *
 * Fallback: DISPLAY yoksa veya X11 bağlantısı kurulamazsa uinput kullanır.
 *
 * SOURCE: man 3 XTestFakeMotionEvent — XTest extension
 * SOURCE: man 3 XTestFakeButtonEvent — XTest button simulation
 * SOURCE: man 3 XOpenDisplay — X11 display connection
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* X11 + XTest */
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>

/* Fallback: uinput */
#include "uinput_mouse.h"
#include <linux/input.h>

static int click_xtest(int x, int y) {
    const char *display_env = getenv("DISPLAY");
    if (!display_env || strlen(display_env) == 0) {
        return -1; /* No DISPLAY, can't use XTest */
    }

    Display *dpy = XOpenDisplay(display_env);
    if (!dpy) {
        fprintf(stderr, "[click] XOpenDisplay(%s) failed\n", display_env);
        return -1;
    }

    /* Check XTest extension availability */
    int event_base, error_base, major, minor;
    if (!XTestQueryExtension(dpy, &event_base, &error_base, &major, &minor)) {
        fprintf(stderr, "[click] XTest extension not available on %s\n", display_env);
        XCloseDisplay(dpy);
        return -1;
    }

    /* Move mouse to (x, y) on screen 0 */
    XTestFakeMotionEvent(dpy, 0, x, y, 0);
    XFlush(dpy);
    usleep(50000); /* 50ms settle time — human-like */

    /* Click: press + release BTN_LEFT */
    XTestFakeButtonEvent(dpy, 1, True, 0);  /* press */
    XFlush(dpy);
    usleep(80000); /* 80ms hold — realistic click duration */
    XTestFakeButtonEvent(dpy, 1, False, 0); /* release */
    XFlush(dpy);
    usleep(30000); /* 30ms post-release settle */

    fprintf(stdout, "[click] XTest click at (%d, %d) on %s\n", x, y, display_env);

    XCloseDisplay(dpy);
    return 0;
}

static int click_uinput(int x, int y) {
    uinput_mouse_t *mouse = NULL;
    if (!uinput_mouse_init(&mouse, 1920, 941, 500)) {
        fprintf(stderr, "uinput_mouse_init failed\n");
        return 1;
    }
    uinput_mouse_move(mouse, x, y);
    usleep(5000);
    uinput_mouse_click(mouse, BTN_LEFT, 80000);
    usleep(50000);
    uinput_mouse_destroy(mouse);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "KULLANIM: %s <x> <y>\n", argv[0]);
        return 1;
    }
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);

    /* Try XTest first (Xvfb-safe, no real desktop impact) */
    if (click_xtest(x, y) == 0) {
        return 0;
    }

    /* Fallback to uinput (kernel-level) */
    fprintf(stderr, "[click] XTest failed, falling back to uinput\n");
    return click_uinput(x, y);
}
