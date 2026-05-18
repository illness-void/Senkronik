#include "uinput_mouse.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <linux/input.h>

int main(int argc, char **argv) {
    uinput_mouse_t *mouse = NULL;
    if (!uinput_mouse_init(&mouse, 1920, 1080, 500)) {
        fprintf(stderr, "uinput mouse init basarisiz.\n");
        return 1;
    }

    if (argc >= 3) {
        // CLI mode: uinput_mouse <x> <y> [screen_w] [screen_h]
        int x = atoi(argv[1]);
        int y = atoi(argv[2]);
        int sw = argc > 3 ? atoi(argv[3]) : 1920;
        int sh = argc > 4 ? atoi(argv[4]) : 1080;
        if (sw != 1920 || sh != 1080) {
            // Re-init with custom resolution if provided
            uinput_mouse_destroy(mouse);
            if (!uinput_mouse_init(&mouse, sw, sh, 500)) {
                fprintf(stderr, "uinput mouse init basarisiz (%dx%d).\n", sw, sh);
                return 1;
            }
        }
        uinput_mouse_move(mouse, x, y);
        usleep(15000);
        uinput_mouse_click(mouse, BTN_LEFT, 80000);
        usleep(50000);
        uinput_mouse_destroy(mouse);
        return 0;
    }

    // Test mode (no args) — wrappers below still needed for compilation
    // but this path is only hit when no CLI arguments given
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    srand((unsigned)(ts.tv_nsec ^ ts.tv_sec));

    uint64_t base_reports = uinput_mouse_report_count(mouse);
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("=== Sabit pozisyonda 500Hz testi (1 sn) ===\n");
    uinput_mouse_move(mouse, 400, 300);
    sleep(1);

    printf("=== Smooth hareket: (400,300) -> (1500,800) 2sn ===\n");
    uinput_mouse_move_smooth(mouse, 1500, 800, 2000);
    while (!uinput_mouse_move_done(mouse))
        usleep(5000);

    printf("=== 3 rastgele tik ===\n");
    for (int i = 0; i < 3; i++) {
        int x = rand() % uinput_mouse_get_screen_w(mouse);
        int y = rand() % uinput_mouse_get_screen_h(mouse);
        printf("  -> (%d, %d)\n", x, y);
        uinput_mouse_move(mouse, x, y);
        usleep(10000);
        uinput_mouse_click(mouse, BTN_LEFT, 50000);
        usleep(200000);
    }

    printf("=== 1 sn daha 500Hz'de bekle ===\n");
    sleep(1);

    struct timespec t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                     (t_now.tv_nsec - t_start.tv_nsec) / 1e9;

    uint64_t total_reports = uinput_mouse_report_count(mouse);
    uint64_t measured_reports = total_reports - base_reports;
    double actual_hz = (double)measured_reports / elapsed;

    printf("\n=== SONUC ===\n");
    printf("Bu periyotta report: %lu\n", (unsigned long)measured_reports);
    printf("Gecen sure:          %.3f sn\n", elapsed);
    printf("Gercek Hz:           %.1f Hz (hedef: 500 Hz)\n", actual_hz);

    uinput_mouse_destroy(mouse);
    return 0;
}
