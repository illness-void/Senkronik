#include "uinput_mouse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#include <linux/uinput.h>
#include <linux/input.h>
#include <sys/ioctl.h>

#define UINPUT_DEV     "/dev/uinput"
#define FAKE_VID       0x046d
#define FAKE_PID       0xc077
#define FAKE_VERSION   0x0110
#define FAKE_NAME      "Logitech G Pro Wireless"

#define BTN_MASK(btn)  (1U << ((btn) - BTN_MOUSE))

struct uinput_mouse {
    int           fd;
    int           screen_w;
    int           screen_h;

    pthread_t     thread;
    pthread_mutex_t lock;
    bool          running;

    int           poll_hz;
    long          interval_ns;

    int           target_x;
    int           target_y;
    int           current_x;
    int           current_y;

    unsigned      btn_mask;

    int           scroll_dy;

    bool          smooth_active;
    int           smooth_from_x;
    int           smooth_from_y;
    int           smooth_to_x;
    int           smooth_to_y;
    int64_t       smooth_dur_ns;
    struct timespec smooth_start;

    uint64_t      report_count;
};

static void emit(int fd, int type, int code, int val) {
    struct input_event ie = {0};
    ie.type  = type;
    ie.code  = code;
    ie.value = val;
    write(fd, &ie, sizeof(ie));
}

static void emit_report(struct uinput_mouse *m) {
    emit(m->fd, EV_ABS, ABS_X,     m->current_x);
    emit(m->fd, EV_ABS, ABS_Y,     m->current_y);

    emit(m->fd, EV_KEY, BTN_LEFT,   (m->btn_mask & BTN_MASK(BTN_LEFT))   ? 1 : 0);
    emit(m->fd, EV_KEY, BTN_RIGHT,  (m->btn_mask & BTN_MASK(BTN_RIGHT))  ? 1 : 0);
    emit(m->fd, EV_KEY, BTN_MIDDLE, (m->btn_mask & BTN_MASK(BTN_MIDDLE)) ? 1 : 0);

    if (m->scroll_dy != 0) {
        emit(m->fd, EV_REL, REL_WHEEL, m->scroll_dy);
        m->scroll_dy = 0;
    }

    emit(m->fd, EV_SYN, SYN_REPORT, 0);
    m->report_count++;
}

static double ease_in_out_cubic(double t) {
    if (t < 0.5) return 4.0 * t * t * t;
    return 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0;
}

static int64_t timespec_to_ns(struct timespec ts) {
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void *timer_thread(void *arg) {
    struct uinput_mouse *m = arg;
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    while (true) {
        pthread_mutex_lock(&m->lock);
        if (!m->running) {
            pthread_mutex_unlock(&m->lock);
            break;
        }

        if (m->smooth_active) {
            int64_t now_ns = timespec_to_ns(next);
            int64_t elapsed = now_ns - timespec_to_ns(m->smooth_start);
            double t = (double)elapsed / (double)m->smooth_dur_ns;

            if (t >= 1.0) {
                m->current_x = m->smooth_to_x;
                m->current_y = m->smooth_to_y;
                m->smooth_active = false;
            } else {
                double e = ease_in_out_cubic(t);
                m->current_x = (int)(m->smooth_from_x + (m->smooth_to_x - m->smooth_from_x) * e);
                m->current_y = (int)(m->smooth_from_y + (m->smooth_to_y - m->smooth_from_y) * e);
            }
        } else {
            m->current_x = m->target_x;
            m->current_y = m->target_y;
        }

        emit_report(m);
        pthread_mutex_unlock(&m->lock);

        next.tv_nsec += m->interval_ns;
        while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec++;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
    return NULL;
}

bool uinput_mouse_init(uinput_mouse_t **out, int screen_w, int screen_h, int poll_hz) {
    if (!out) return false;

    struct uinput_mouse *m = calloc(1, sizeof(*m));
    if (!m) return false;

    m->fd       = -1;
    m->screen_w = screen_w;
    m->screen_h = screen_h;
    m->poll_hz  = poll_hz;

    if (poll_hz < 125)  poll_hz = 125;
    if (poll_hz > 1000) poll_hz = 1000;
    m->interval_ns = 1000000000L / poll_hz;

    pthread_mutex_init(&m->lock, NULL);

    int fd = open(UINPUT_DEV, O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "[uinput] open(%s) basarisiz: %s\n", UINPUT_DEV, strerror(errno));
        free(m);
        return false;
    }

    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_EVBIT, EV_ABS);
    ioctl(fd, UI_SET_EVBIT, EV_REL);
    ioctl(fd, UI_SET_EVBIT, EV_SYN);

    ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    ioctl(fd, UI_SET_KEYBIT, BTN_MIDDLE);

    ioctl(fd, UI_SET_RELBIT, REL_WHEEL);
    ioctl(fd, UI_SET_RELBIT, REL_HWHEEL);

    ioctl(fd, UI_SET_ABSBIT, ABS_X);
    ioctl(fd, UI_SET_ABSBIT, ABS_Y);

    struct uinput_abs_setup absx = {0};
    absx.code = ABS_X;
    absx.absinfo.minimum    = 0;
    absx.absinfo.maximum    = screen_w - 1;
    absx.absinfo.fuzz       = 0;
    absx.absinfo.flat       = 0;
    absx.absinfo.resolution = 0;
    if (ioctl(fd, UI_ABS_SETUP, &absx) < 0) {
        fprintf(stderr, "[uinput] UI_ABS_SETUP ABS_X basarisiz: %s\n", strerror(errno));
        close(fd); free(m);
        return false;
    }

    struct uinput_abs_setup absy = {0};
    absy.code = ABS_Y;
    absy.absinfo.minimum    = 0;
    absy.absinfo.maximum    = screen_h - 1;
    absy.absinfo.fuzz       = 0;
    absy.absinfo.flat       = 0;
    absy.absinfo.resolution = 0;
    if (ioctl(fd, UI_ABS_SETUP, &absy) < 0) {
        fprintf(stderr, "[uinput] UI_ABS_SETUP ABS_Y basarisiz: %s\n", strerror(errno));
        close(fd); free(m);
        return false;
    }

    struct uinput_setup usetup = {0};
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor  = FAKE_VID;
    usetup.id.product = FAKE_PID;
    usetup.id.version = FAKE_VERSION;
    strncpy(usetup.name, FAKE_NAME, UINPUT_MAX_NAME_SIZE - 1);
    usetup.name[UINPUT_MAX_NAME_SIZE - 1] = '\0';

    if (ioctl(fd, UI_DEV_SETUP, &usetup) < 0) {
        fprintf(stderr, "[uinput] UI_DEV_SETUP basarisiz: %s\n", strerror(errno));
        close(fd); free(m);
        return false;
    }

    if (ioctl(fd, UI_DEV_CREATE) < 0) {
        fprintf(stderr, "[uinput] UI_DEV_CREATE basarisiz: %s\n", strerror(errno));
        close(fd); free(m);
        return false;
    }

    m->fd      = fd;
    m->running = true;

    if (pthread_create(&m->thread, NULL, timer_thread, m) != 0) {
        fprintf(stderr, "[uinput] pthread_create basarisiz\n");
        ioctl(fd, UI_DEV_DESTROY);
        close(fd); free(m);
        return false;
    }

    sleep(1);
    fprintf(stdout, "[uinput] Sanal fare yaratildi: %s (%dx%d) @ %dHz\n",
            FAKE_NAME, screen_w, screen_h, poll_hz);

    *out = m;
    return true;
}

void uinput_mouse_destroy(uinput_mouse_t *m) {
    if (!m) return;

    pthread_mutex_lock(&m->lock);
    m->running = false;
    pthread_mutex_unlock(&m->lock);

    pthread_join(m->thread, NULL);

    if (m->fd >= 0) {
        ioctl(m->fd, UI_DEV_DESTROY);
        close(m->fd);
    }

    pthread_mutex_destroy(&m->lock);
    fprintf(stdout, "[uinput] Sanal fare yok edildi. Toplam report: %lu\n",
            (unsigned long)m->report_count);
    free(m);
}

static int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

void uinput_mouse_move(uinput_mouse_t *m, int x, int y) {
    if (!m) return;
    x = clamp(x, 0, m->screen_w - 1);
    y = clamp(y, 0, m->screen_h - 1);
    pthread_mutex_lock(&m->lock);
    m->target_x      = x;
    m->target_y      = y;
    m->smooth_active = false;
    pthread_mutex_unlock(&m->lock);
}

void uinput_mouse_move_smooth(uinput_mouse_t *m, int x, int y, int duration_ms) {
    if (!m) return;
    x = clamp(x, 0, m->screen_w - 1);
    y = clamp(y, 0, m->screen_h - 1);
    pthread_mutex_lock(&m->lock);
    m->smooth_from_x  = m->current_x;
    m->smooth_from_y  = m->current_y;
    m->smooth_to_x    = x;
    m->smooth_to_y    = y;
    m->smooth_dur_ns  = (int64_t)duration_ms * 1000000LL;
    clock_gettime(CLOCK_MONOTONIC, &m->smooth_start);
    m->smooth_active  = true;
    m->target_x       = x;
    m->target_y       = y;
    pthread_mutex_unlock(&m->lock);
}

bool uinput_mouse_move_done(uinput_mouse_t *m) {
    if (!m) return true;
    pthread_mutex_lock(&m->lock);
    bool done = !m->smooth_active;
    pthread_mutex_unlock(&m->lock);
    return done;
}

void uinput_mouse_click(uinput_mouse_t *m, int button, int duration_us) {
    if (!m) return;
    uinput_mouse_press(m, button);
    if (duration_us > 0)
        usleep(duration_us);
    uinput_mouse_release(m, button);
}

void uinput_mouse_press(uinput_mouse_t *m, int button) {
    if (!m) return;
    pthread_mutex_lock(&m->lock);
    m->btn_mask |= BTN_MASK(button);
    pthread_mutex_unlock(&m->lock);
}

void uinput_mouse_release(uinput_mouse_t *m, int button) {
    if (!m) return;
    pthread_mutex_lock(&m->lock);
    m->btn_mask &= ~BTN_MASK(button);
    pthread_mutex_unlock(&m->lock);
}

void uinput_mouse_wheel(uinput_mouse_t *m, int dy) {
    if (!m) return;
    pthread_mutex_lock(&m->lock);
    m->scroll_dy += dy;
    pthread_mutex_unlock(&m->lock);
}

int uinput_mouse_get_screen_w(uinput_mouse_t *m) {
    return m ? m->screen_w : 0;
}

int uinput_mouse_get_screen_h(uinput_mouse_t *m) {
    return m ? m->screen_h : 0;
}

uint64_t uinput_mouse_report_count(uinput_mouse_t *m) {
    if (!m) return 0;
    pthread_mutex_lock(&m->lock);
    uint64_t c = m->report_count;
    pthread_mutex_unlock(&m->lock);
    return c;
}
