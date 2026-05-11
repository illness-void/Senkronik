CC      := gcc
CFLAGS  := -Wall -Wextra -O2
LDFLAGS := -lpthread -lm
TARGET  := uinput_mouse

SRC_DIR := src
OBJ_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TARGET)
	sg input -c ./$(TARGET) 2>&1

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
