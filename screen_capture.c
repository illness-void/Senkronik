#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <drm/drm_fourcc.h>
#include <linux/dma-buf.h>
#include <pipewire/pipewire.h>
#include <spa/buffer/buffer.h>
#include <spa/param/buffers.h>
#include <spa/param/format.h>
#include <spa/param/video/raw-utils.h>
#include <spa/pod/builder.h>
#include <spa/pod/vararg.h>
#include <systemd/sd-bus.h>

#include "frame_shm.h"

#define FRAME_CROP_X 1466
#define FRAME_CROP_Y 0
#define FRAME_CROP_W 454
#define FRAME_CROP_H 1080

static struct pw_main_loop *loop;
static struct pw_context *context;
static struct pw_core *core;
static struct pw_stream *stream;
static struct spa_hook stream_listener;
static sd_bus *bus;
static int pw_fd = -1;
static char *session_handle;

static uint32_t node_id;
static uint64_t pipewire_serial;
static uint32_t frame_width;
static uint32_t frame_height;
static int32_t frame_stride;
static int capture_started_printed;
static volatile sig_atomic_t stop_requested;
static volatile uint8_t frame_sink;
static struct FrameSHM *g_fshm;

static struct FrameSHM *frame_shm_get(void) { return g_fshm; }

struct portal_response {
    int done;
    uint32_t code;
    sd_bus_message *message;
};

static void on_signal(int signum) {
    (void)signum;
    stop_requested = 1;
    if (loop)
        pw_main_loop_quit(loop);
}

static int dma_buf_sync_fd(int fd, uint64_t flags) {
    struct dma_buf_sync sync = { .flags = flags };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) < 0)
        return -errno;
    return 0;
}

static void process_frame(const void *pixels, int32_t stride, uint32_t width, uint32_t height) {
    const uint8_t *p = pixels;
    struct FrameSHM *fshm = NULL;

    if (!p || stride <= 0 || width == 0 || height == 0)
        return;

    /* AI ajanının okuyacağı ham XRGB8888 alanı: pixels + stride + width + height. */
    frame_sink ^= p[0];
    frame_sink ^= p[(height - 1) * (size_t)stride + (width - 1) * 4];

    // DOGGYSTYLE BRIDGE: 454x1080 @ x=1466 BGR -> FrameSHM
    fshm = frame_shm_get();
    if (fshm && (width == 1920 || width >= (uint32_t)(FRAME_CROP_X + FRAME_CROP_W))) {
        uint8_t *dst = fshm->data;
        for (uint32_t y = 0; y < FRAME_CROP_H; y++) {
            const uint32_t *src_row = (const uint32_t *)(p + y * (size_t)stride) + FRAME_CROP_X;
            for (uint32_t x = 0; x < FRAME_CROP_W; x++) {
                uint32_t px = src_row[x];
                *dst++ = (uint8_t)(px >> 16);  // B
                *dst++ = (uint8_t)(px >> 8);   // G
                *dst++ = (uint8_t)px;          // R
            }
        }
        fshm->frame_id++;
        fshm->ready = 1;
    }
}

static void on_process(void *userdata) {
    struct pw_buffer *buf;
    struct spa_buffer *sbuf;
    struct spa_data *data;

    (void)userdata;

    buf = pw_stream_dequeue_buffer(stream);
    if (!buf)
        return;

    sbuf = buf->buffer;
    if (!sbuf || sbuf->n_datas == 0)
        goto queue;

    data = &sbuf->datas[0];
    if (data->type == SPA_DATA_DmaBuf && data->fd >= 0 && data->chunk) {
        uint32_t offset = data->chunk->offset;
        uint32_t size = data->chunk->size;
        uint32_t maxsize = data->maxsize ? data->maxsize : offset + size;
        int32_t stride = data->chunk->stride > 0 ? data->chunk->stride : frame_stride;

        if (size > 0 && maxsize >= offset + size && stride > 0) {
            int synced = dma_buf_sync_fd(data->fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ) == 0;
            void *base = mmap(NULL, maxsize, PROT_READ, MAP_SHARED, data->fd, data->mapoffset);
            if (base != MAP_FAILED) {
                if (!capture_started_printed) {
                    printf("Yakalama başladı - DMA-BUF aktif - 60 FPS\n");
                    fflush(stdout);
                    capture_started_printed = 1;
                }
                process_frame((uint8_t *)base + offset, stride, frame_width, frame_height);
                munmap(base, maxsize);
            }
            if (synced)
                dma_buf_sync_fd(data->fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        }
    }

queue:
    pw_stream_queue_buffer(stream, buf);
}

static void on_param_changed(void *userdata, uint32_t id, const struct spa_pod *param) {
    uint8_t buffer[8192];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
    const struct spa_pod *params[1];
    struct spa_video_info_raw info;
    uint32_t stride;

    (void)userdata;

    if (!param || id != SPA_PARAM_Format)
        return;

    memset(&info, 0, sizeof(info));
    if (spa_format_video_raw_parse(param, &info) < 0)
        return;

    frame_width = info.size.width;
    frame_height = info.size.height;
    stride = SPA_ROUND_UP_N(frame_width * 4, 4);
    frame_stride = (int32_t)stride;

    params[0] = spa_pod_builder_add_object(&b,
        SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
        SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(8, 2, 16),
        SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
        SPA_PARAM_BUFFERS_size, SPA_POD_Int(stride * frame_height),
        SPA_PARAM_BUFFERS_stride, SPA_POD_Int(stride),
        SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1 << SPA_DATA_DmaBuf));

    pw_stream_update_params(stream, params, 1);
}

static void on_state_changed(void *userdata, enum pw_stream_state old,
                             enum pw_stream_state state, const char *error) {
    (void)userdata;
    if (state == PW_STREAM_STATE_ERROR)
        fprintf(stderr, "PipeWire stream hatası: %s -> %s: %s\n",
            pw_stream_state_as_string(old), pw_stream_state_as_string(state), error ? error : "bilinmiyor");
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_state_changed,
    .param_changed = on_param_changed,
    .process = on_process,
};

static void make_token(char *buf, size_t len) {
    snprintf(buf, len, "capture_%lu_%u", (unsigned long)time(NULL), (unsigned)rand());
}

static int make_sender_path_element(char *buf, size_t len) {
    const char *unique;
    size_t n = 0;
    int r = sd_bus_get_unique_name(bus, &unique);

    if (r < 0)
        return r;
    if (unique[0] == ':')
        unique++;
    while (*unique && n + 1 < len) {
        buf[n++] = *unique == '.' ? '_' : *unique;
        unique++;
    }
    buf[n] = '\0';
    return n > 0 ? 0 : -EINVAL;
}

static int request_response(sd_bus_message *m, void *userdata, sd_bus_error *ret_error) {
    struct portal_response *response = userdata;
    uint32_t code;
    int r;

    (void)ret_error;
    r = sd_bus_message_read(m, "u", &code);
    if (r < 0)
        return r;
    if (response->message)
        sd_bus_message_unref(response->message);
    response->message = sd_bus_message_ref(m);
    response->code = code;
    response->done = 1;
    return 0;
}

static int wait_for_response(struct portal_response *response) {
    while (!response->done && !stop_requested) {
        int r = sd_bus_process(bus, NULL);
        if (r < 0)
            return r;
        if (r > 0)
            continue;
        r = sd_bus_wait(bus, UINT64_MAX);
        if (r < 0 && r != -EINTR)
            return r;
    }
    return stop_requested ? -EINTR : 0;
}

static int add_request_match(sd_bus_slot **slot, const char *request_path, struct portal_response *response) {
    char match[1024];

    snprintf(match, sizeof(match),
        "type='signal',sender='org.freedesktop.portal.Desktop',path='%s',interface='org.freedesktop.portal.Request',member='Response'",
        request_path);
    return sd_bus_add_match(bus, slot, match, request_response, response);
}

static int append_option_string(sd_bus_message *m, const char *key, const char *value) {
    int r;

    r = sd_bus_message_open_container(m, 'e', "sv");
    if (r < 0) return r;
    r = sd_bus_message_append(m, "s", key);
    if (r < 0) return r;
    r = sd_bus_message_open_container(m, 'v', "s");
    if (r < 0) return r;
    r = sd_bus_message_append(m, "s", value);
    if (r < 0) return r;
    r = sd_bus_message_close_container(m);
    if (r < 0) return r;
    return sd_bus_message_close_container(m);
}

static int append_option_uint32(sd_bus_message *m, const char *key, uint32_t value) {
    int r;

    r = sd_bus_message_open_container(m, 'e', "sv");
    if (r < 0) return r;
    r = sd_bus_message_append(m, "s", key);
    if (r < 0) return r;
    r = sd_bus_message_open_container(m, 'v', "u");
    if (r < 0) return r;
    r = sd_bus_message_append(m, "u", value);
    if (r < 0) return r;
    r = sd_bus_message_close_container(m);
    if (r < 0) return r;
    return sd_bus_message_close_container(m);
}

static int read_session_handle(sd_bus_message *m, char *out, size_t out_len) {
    int r = sd_bus_message_enter_container(m, 'a', "{sv}");

    if (r < 0) return r;
    for (;;) {
        const char *key;
        r = sd_bus_message_enter_container(m, 'e', "sv");
        if (r <= 0) break;
        r = sd_bus_message_read(m, "s", &key);
        if (r < 0) return r;
        if (strcmp(key, "session_handle") == 0) {
            const char *value;
            r = sd_bus_message_enter_container(m, 'v', "s");
            if (r < 0) return r;
            r = sd_bus_message_read(m, "s", &value);
            if (r < 0) return r;
            snprintf(out, out_len, "%s", value);
            r = sd_bus_message_exit_container(m);
            if (r < 0) return r;
        } else {
            r = sd_bus_message_skip(m, "v");
            if (r < 0) return r;
        }
        r = sd_bus_message_exit_container(m);
        if (r < 0) return r;
    }
    if (r < 0) return r;
    r = sd_bus_message_exit_container(m);
    if (r < 0) return r;
    return out[0] ? 0 : -ENOENT;
}

static int portal_request_call(sd_bus_message *call, const char *sender, const char *handle_token, struct portal_response *response) {
    char request_path[512];
    sd_bus_slot *slot = NULL;
    sd_bus_message *reply = NULL;
    sd_bus_error error = SD_BUS_ERROR_NULL;
    int r;

    snprintf(request_path, sizeof(request_path), "/org/freedesktop/portal/desktop/request/%s/%s", sender, handle_token);
    r = add_request_match(&slot, request_path, response);
    if (r < 0) goto finish;
    r = sd_bus_call(bus, call, 0, &error, &reply);
    if (r < 0) goto finish;
    r = wait_for_response(response);
    if (r < 0) goto finish;
    r = response->code == 0 ? 0 : -ECANCELED;

finish:
    if (r < 0 && error.message)
        fprintf(stderr, "Portal hata: %s\n", error.message);
    sd_bus_error_free(&error);
    sd_bus_message_unref(reply);
    sd_bus_slot_unref(slot);
    return r;
}

static int call_create_session(const char *sender, char *session_out, size_t session_out_len, int *used_restore) {
    char handle_token[64], session_token[64];
    struct portal_response response = {0};
    sd_bus_message *call = NULL;
    int r;
    int has_restore = 0;

    make_token(handle_token, sizeof(handle_token));
    make_token(session_token, sizeof(session_token));

    // Daha once kaydedilmis restore token'i kontrol et
    char restore_token[128] = {0};
    FILE *rf = fopen("/tmp/senkronik_restore_token", "r");
    if (rf) {
        if (fgets(restore_token, sizeof(restore_token), rf)) {
            size_t len = strlen(restore_token);
            if (len > 0 && restore_token[len-1] == '\n') restore_token[len-1] = '\0';
            if (restore_token[0]) has_restore = 1;
        }
        fclose(rf);
    }

    r = sd_bus_message_new_method_call(bus, &call, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.ScreenCast", "CreateSession");
    if (r < 0) goto finish;
    r = sd_bus_message_open_container(call, 'a', "{sv}");
    if (r < 0) goto finish;
    r = append_option_string(call, "handle_token", handle_token);
    if (r < 0) goto finish;
    r = append_option_string(call, "session_handle_token", session_token);
    if (r < 0) goto finish;
    if (has_restore) {
        r = append_option_string(call, "restore_token", restore_token);
        if (r < 0) goto finish;
        printf("Restore token kullaniliyor (diyalog atlaniyor)...\n");
    }
    r = sd_bus_message_close_container(call);
    if (r < 0) goto finish;
    r = portal_request_call(call, sender, handle_token, &response);
    if (r < 0) goto finish;
    r = read_session_handle(response.message, session_out, session_out_len);

finish:
    if (used_restore) *used_restore = has_restore;
    sd_bus_message_unref(call);
    sd_bus_message_unref(response.message);
    return r;
}

static int call_select_sources(const char *sender, const char *session) {
    char handle_token[64];
    struct portal_response response = {0};
    sd_bus_message *call = NULL;
    int r;

    make_token(handle_token, sizeof(handle_token));
    r = sd_bus_message_new_method_call(bus, &call, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.ScreenCast", "SelectSources");
    if (r < 0) goto finish;
    r = sd_bus_message_append(call, "o", session);
    if (r < 0) goto finish;
    r = sd_bus_message_open_container(call, 'a', "{sv}");
    if (r < 0) goto finish;
    r = append_option_string(call, "handle_token", handle_token);
    if (r < 0) goto finish;
    r = append_option_uint32(call, "types", 1);
    if (r < 0) goto finish;
    r = append_option_uint32(call, "cursor_mode", 1);
    if (r < 0) goto finish;
    r = sd_bus_message_close_container(call);
    if (r < 0) goto finish;
    r = portal_request_call(call, sender, handle_token, &response);

finish:
    sd_bus_message_unref(call);
    sd_bus_message_unref(response.message);
    return r;
}

static int parse_stream_properties(sd_bus_message *m) {
    int r = sd_bus_message_enter_container(m, 'a', "{sv}");

    if (r < 0) return r;
    for (;;) {
        const char *key;
        r = sd_bus_message_enter_container(m, 'e', "sv");
        if (r <= 0) break;
        r = sd_bus_message_read(m, "s", &key);
        if (r < 0) return r;
        if (strcmp(key, "pipewire-serial") == 0) {
            r = sd_bus_message_enter_container(m, 'v', "t");
            if (r < 0) return r;
            r = sd_bus_message_read(m, "t", &pipewire_serial);
            if (r < 0) return r;
            r = sd_bus_message_exit_container(m);
        } else if (strcmp(key, "size") == 0) {
            int32_t w, h;
            r = sd_bus_message_enter_container(m, 'v', "(ii)");
            if (r < 0) return r;
            r = sd_bus_message_read(m, "(ii)", &w, &h);
            if (r < 0) return r;
            if (w > 0 && h > 0) {
                frame_width = (uint32_t)w;
                frame_height = (uint32_t)h;
                frame_stride = (int32_t)SPA_ROUND_UP_N(frame_width * 4, 4);
            }
            r = sd_bus_message_exit_container(m);
        } else {
            r = sd_bus_message_skip(m, "v");
        }
        if (r < 0) return r;
        r = sd_bus_message_exit_container(m);
        if (r < 0) return r;
    }
    if (r < 0) return r;
    return sd_bus_message_exit_container(m);
}

static int parse_start_streams(sd_bus_message *m) {
    int r = sd_bus_message_enter_container(m, 'a', "{sv}");

    if (r < 0) return r;
    for (;;) {
        const char *key;
        r = sd_bus_message_enter_container(m, 'e', "sv");
        if (r <= 0) break;
        r = sd_bus_message_read(m, "s", &key);
        if (r < 0) return r;
        if (strcmp(key, "streams") == 0) {
            r = sd_bus_message_enter_container(m, 'v', "a(ua{sv})");
            if (r < 0) return r;
            r = sd_bus_message_enter_container(m, 'a', "(ua{sv})");
            if (r < 0) return r;
            for (;;) {
                r = sd_bus_message_enter_container(m, 'r', "ua{sv}");
                if (r <= 0) break;
                r = sd_bus_message_read(m, "u", &node_id);
                if (r < 0) return r;
                r = parse_stream_properties(m);
                if (r < 0) return r;
                r = sd_bus_message_exit_container(m);
                if (r < 0) return r;
            }
            if (r < 0) return r;
            r = sd_bus_message_exit_container(m);
            if (r < 0) return r;
            r = sd_bus_message_exit_container(m);
        } else {
            r = sd_bus_message_skip(m, "v");
        }
        if (r < 0) return r;
        r = sd_bus_message_exit_container(m);
        if (r < 0) return r;
    }
    if (r < 0) return r;
    r = sd_bus_message_exit_container(m);
    if (r < 0) return r;
    return node_id || pipewire_serial ? 0 : -ENOENT;
}

static int call_start(const char *sender, const char *session) {
    char handle_token[64];
    struct portal_response response = {0};
    sd_bus_message *call = NULL;
    int r;

    make_token(handle_token, sizeof(handle_token));
    r = sd_bus_message_new_method_call(bus, &call, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.ScreenCast", "Start");
    if (r < 0) goto finish;
    r = sd_bus_message_append(call, "os", session, "");
    if (r < 0) goto finish;
    r = sd_bus_message_open_container(call, 'a', "{sv}");
    if (r < 0) goto finish;
    r = append_option_string(call, "handle_token", handle_token);
    if (r < 0) goto finish;
    r = sd_bus_message_close_container(call);
    if (r < 0) goto finish;
    r = portal_request_call(call, sender, handle_token, &response);
    if (r < 0) goto finish;
    r = parse_start_streams(response.message);

finish:
    sd_bus_message_unref(call);
    sd_bus_message_unref(response.message);
    return r;
}

static int call_open_pipewire_remote(const char *session) {
    sd_bus_message *call = NULL;
    sd_bus_message *reply = NULL;
    sd_bus_error error = SD_BUS_ERROR_NULL;
    int received_fd = -1;
    int r;

    r = sd_bus_message_new_method_call(bus, &call, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.ScreenCast", "OpenPipeWireRemote");
    if (r < 0) goto finish;
    r = sd_bus_message_append(call, "o", session);
    if (r < 0) goto finish;
    r = sd_bus_message_open_container(call, 'a', "{sv}");
    if (r < 0) goto finish;
    r = sd_bus_message_close_container(call);
    if (r < 0) goto finish;
    r = sd_bus_call(bus, call, 0, &error, &reply);
    if (r < 0) goto finish;
    r = sd_bus_message_read(reply, "h", &received_fd);
    if (r < 0) goto finish;
    pw_fd = fcntl(received_fd, F_DUPFD_CLOEXEC, 3);
    if (pw_fd < 0)
        r = -errno;

finish:
    if (r < 0 && error.message)
        fprintf(stderr, "OpenPipeWireRemote hata: %s\n", error.message);
    sd_bus_error_free(&error);
    sd_bus_message_unref(call);
    sd_bus_message_unref(reply);
    return r;
}

static const struct spa_pod *build_format_param(struct spa_pod_builder *b, enum spa_video_format format) {
    struct spa_pod_frame object_frame;
    struct spa_rectangle def_size = SPA_RECTANGLE(1920, 1080);
    struct spa_rectangle min_size = SPA_RECTANGLE(1, 1);
    struct spa_rectangle max_size = SPA_RECTANGLE(3840, 2160);
    struct spa_fraction fps = SPA_FRACTION(60, 1);

    spa_pod_builder_push_object(b, &object_frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
    spa_pod_builder_add(b,
        SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
        SPA_FORMAT_VIDEO_format, SPA_POD_Id(format),
        SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(&def_size, &min_size, &max_size),
        SPA_FORMAT_VIDEO_maxFramerate, SPA_POD_Fraction(&fps),
        0);
    spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY);
    spa_pod_builder_long(b, DRM_FORMAT_MOD_LINEAR);
    return (const struct spa_pod *)spa_pod_builder_pop(b, &object_frame);
}

static int start_pipewire(void) {
    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
    const struct spa_pod *params[6];
    struct pw_properties *props;
    uint32_t target_id = node_id;
    char serial_str[32];
    int r;

    loop = pw_main_loop_new(NULL);
    if (!loop) return -ENOMEM;
    context = pw_context_new(pw_main_loop_get_loop(loop), NULL, 0);
    if (!context) return -errno;
    core = pw_context_connect_fd(context, pw_fd, NULL, 0);
    pw_fd = -1;
    if (!core) return -errno;

    props = pw_properties_new(PW_KEY_MEDIA_TYPE, "Video", PW_KEY_MEDIA_CATEGORY, "Capture", PW_KEY_MEDIA_ROLE, "Screen", NULL);
    if (!props) return -ENOMEM;
    if (pipewire_serial)
        snprintf(serial_str, sizeof(serial_str), "%llu", (unsigned long long)pipewire_serial);
    (void)serial_str;

    stream = pw_stream_new(core, "screen-capture", props);
    if (!stream) return -errno;
    pw_stream_add_listener(stream, &stream_listener, &stream_events, NULL);

    params[0] = build_format_param(&b, SPA_VIDEO_FORMAT_BGRA);
    params[1] = build_format_param(&b, SPA_VIDEO_FORMAT_BGRx);
    params[2] = build_format_param(&b, SPA_VIDEO_FORMAT_RGBx);
    params[3] = build_format_param(&b, SPA_VIDEO_FORMAT_RGBA);
    params[4] = build_format_param(&b, SPA_VIDEO_FORMAT_xRGB);
    params[5] = build_format_param(&b, SPA_VIDEO_FORMAT_ARGB);
    r = pw_stream_connect(stream, PW_DIRECTION_INPUT, target_id,
        PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_RT_PROCESS | PW_STREAM_FLAG_MAP_BUFFERS,
        params, 6);
    return r < 0 ? r : 0;
}

static void cleanup(void) {
    if (stream) pw_stream_destroy(stream);
    if (core) pw_core_disconnect(core);
    if (context) pw_context_destroy(context);
    if (loop) pw_main_loop_destroy(loop);
    if (pw_fd >= 0) close(pw_fd);
    free(session_handle);
    if (g_fshm) { frame_shm_close(g_fshm, 1); frame_shm_destroy(FRAME_SHM_NAME); }
    if (bus) sd_bus_flush_close_unref(bus);
    pw_deinit();
}

int main(int argc, char **argv) {
    char sender[256];
    char session_buf[512];
    int r;

    srand((unsigned)time(NULL) ^ (unsigned)getpid());
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    printf("Portal başlatılıyor...\n");
    fflush(stdout);

    pw_init(&argc, &argv);
    r = sd_bus_open_user(&bus);
    if (r < 0) {
        fprintf(stderr, "DBus açılamadı: %s\n", strerror(-r));
        cleanup();
        return 1;
    }
    r = make_sender_path_element(sender, sizeof(sender));
    if (r < 0) {
        fprintf(stderr, "DBus benzersiz isim alınamadı: %s\n", strerror(-r));
        cleanup();
        return 1;
    }

    memset(session_buf, 0, sizeof(session_buf));
    int used_restore = 0;
    r = call_create_session(sender, session_buf, sizeof(session_buf), &used_restore);
    if (r < 0) {
        fprintf(stderr, "CreateSession başarısız: %s\n", strerror(-r));
        cleanup();
        return 1;
    }
    session_handle = strdup(session_buf);
    if (!session_handle) {
        cleanup();
        return 1;
    }

    // Eger restore token yoksa SelectSources ile ekran secimi yap
    if (!used_restore) {
        r = call_select_sources(sender, session_handle);
        if (r < 0) {
            fprintf(stderr, "SelectSources başarısız: %s\n", strerror(-r));
            cleanup();
            return 1;
        }
    } else {
        printf("Restore edilen session, SelectSources atlaniyor.\n");
    }

    r = call_start(sender, session_handle);
    if (r < 0) {
        fprintf(stderr, "Start başarısız: %s\n", strerror(-r));
        cleanup();
        return 1;
    }

    // Session handle'i kaydet (sonraki calistirmalarda diyalog atlamak icin)
    FILE *rf = fopen("/tmp/senkronik_restore_token", "w");
    if (rf) {
        fprintf(rf, "%s\n", session_handle);
        fclose(rf);
    }

    r = call_open_pipewire_remote(session_handle);
    if (r < 0) {
        fprintf(stderr, "PipeWire FD alınamadı: %s\n", strerror(-r));
        cleanup();
        return 1;
    }

    r = start_pipewire();
    if (r < 0) {
        fprintf(stderr, "PipeWire stream başlatılamadı: %s\n", strerror(-r));
        cleanup();
        return 1;
    }

    // Doggystyle koprusu: frame SHM olustur
    if (frame_shm_create(FRAME_SHM_NAME, &g_fshm) == 0) {
        printf("Frame SHM olusturuldu: %s (%dx%dx%d)\n", FRAME_SHM_NAME, FRAME_CROP_W, FRAME_CROP_H, FRAME_CHANNELS);
    } else {
        fprintf(stderr, "Frame SHM olusturulamadi (doggystyle calismiyor olabilir)\n");
        g_fshm = NULL;
    }

    pw_main_loop_run(loop);

    cleanup();
    printf("Yakalama durduruldu\n");
    return 0;
}
