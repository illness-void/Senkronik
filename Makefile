CC      := gcc
CFLAGS  := -Wall -Wextra -O2 -D_GNU_SOURCE
LDFLAGS := -lpthread -lm

# Doggystyle bridge include (frame_shm.h)
DOGGYSTYLE_DIR := ../doggystyle/src/bridge

# PipeWire + D-Bus + DRM (screen capture)
CAP_CFLAGS  := $(shell pkg-config --cflags libpipewire-0.3 2>/dev/null) $(shell pkg-config --cflags libspa-0.2 2>/dev/null) $(shell pkg-config --cflags libsystemd 2>/dev/null)
CAP_LDFLAGS := $(shell pkg-config --libs libpipewire-0.3 2>/dev/null) $(shell pkg-config --libs libspa-0.2 2>/dev/null) $(shell pkg-config --libs libsystemd 2>/dev/null)

SRC_DIR := src
OBJ_DIR := build

TARGET_UINPUT := uinput_mouse
TARGET_CAP    := screen_capture

.PHONY: all clean test

all: $(TARGET_UINPUT) $(TARGET_CAP)

# uinput_mouse (existing)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET_UINPUT): $(OBJ_DIR)/main.o $(OBJ_DIR)/uinput_mouse.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# screen_capture + frame_shm (Doggystyle bridge)
$(OBJ_DIR)/screen_capture.o: screen_capture.c $(DOGGYSTYLE_DIR)/frame_shm.h
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(CAP_CFLAGS) -I$(DOGGYSTYLE_DIR) -c -o $@ screen_capture.c

$(OBJ_DIR)/frame_shm.o: $(DOGGYSTYLE_DIR)/frame_shm.c $(DOGGYSTYLE_DIR)/frame_shm.h
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(DOGGYSTYLE_DIR)/frame_shm.c

$(TARGET_CAP): $(OBJ_DIR)/screen_capture.o $(OBJ_DIR)/frame_shm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(CAP_LDFLAGS) -lrt

test: $(TARGET_UINPUT)
	sg input -c ./$(TARGET_UINPUT) 2>&1

clean:
	rm -rf $(OBJ_DIR) $(TARGET_UINPUT) $(TARGET_CAP)
