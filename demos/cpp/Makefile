# Makefile for Entangle C/C++ Demos
#
# This Makefile supports building on Linux, macOS, and Windows (with MinGW)
#
# Usage:
#   make              - Build all demos
#   make basic_usage  - Build C basic usage demo
#   make raii_wrapper - Build C++ RAII wrapper demo
#   make embedded     - Build embedded minimal demo
#   make clean        - Remove build artifacts
#   make run          - Build and run all demos
#
# Prerequisites:
#   - Build the Entangle library first: cargo build --release
#   - GCC/Clang for C demos
#   - G++/Clang++ for C++ demos

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11 -O2
CXXFLAGS = -Wall -Wextra -std=c++17 -O2

# Include paths
INCLUDE_DIR = ../../include

# Library paths - detect OS
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(UNAME_S),Linux)
    LIB_DIR = ../../target/release
    LIB_NAME = entangle_db
    LDFLAGS = -L$(LIB_DIR) -l$(LIB_NAME) -Wl,-rpath,$(LIB_DIR) -lpthread -ldl -lm
    LIB_FILE = $(LIB_DIR)/lib$(LIB_NAME).so
    EXE_EXT =
endif

ifeq ($(UNAME_S),Darwin)
    LIB_DIR = ../../target/release
    LIB_NAME = entangle_db
    LDFLAGS = -L$(LIB_DIR) -l$(LIB_NAME) -Wl,-rpath,@executable_path/$(LIB_DIR)
    LIB_FILE = $(LIB_DIR)/lib$(LIB_NAME).dylib
    EXE_EXT =
endif

# Windows (MinGW)
ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
    LIB_DIR = ../../target/release
    LIB_NAME = entangle_db
    LDFLAGS = -L$(LIB_DIR) -l$(LIB_NAME)
    LIB_FILE = $(LIB_DIR)/$(LIB_NAME).dll
    EXE_EXT = .exe
endif

ifeq ($(findstring MSYS,$(UNAME_S)),MSYS)
    LIB_DIR = ../../target/release
    LIB_NAME = entangle_db
    LDFLAGS = -L$(LIB_DIR) -l$(LIB_NAME)
    LIB_FILE = $(LIB_DIR)/$(LIB_NAME).dll
    EXE_EXT = .exe
endif

# Windows detection fallback
ifeq ($(OS),Windows_NT)
    LIB_DIR = ../../target/release
    LIB_NAME = entangle_db
    LDFLAGS = -L$(LIB_DIR) -l$(LIB_NAME)
    LIB_FILE = $(LIB_DIR)/$(LIB_NAME).dll
    EXE_EXT = .exe
endif

# Targets
TARGETS = basic_usage$(EXE_EXT) raii_wrapper$(EXE_EXT) embedded_minimal$(EXE_EXT)

.PHONY: all clean run check-lib help

all: check-lib $(TARGETS)

help:
	@echo "Entangle C/C++ Demos Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build all demos (default)"
	@echo "  basic_usage   - Build C basic usage demo"
	@echo "  raii_wrapper  - Build C++ RAII wrapper demo"
	@echo "  embedded      - Build embedded minimal demo"
	@echo "  clean         - Remove build artifacts"
	@echo "  run           - Build and run all demos"
	@echo "  check-lib     - Verify library is built"
	@echo ""
	@echo "Prerequisites:"
	@echo "  Build the Entangle library first:"
	@echo "    cd ../.. && cargo build --release"

check-lib:
	@if [ ! -f "$(LIB_FILE)" ]; then \
		echo "Error: Entangle library not found at $(LIB_FILE)"; \
		echo "Please build the library first:"; \
		echo "  cd ../.. && cargo build --release"; \
		exit 1; \
	fi

# C demos
basic_usage$(EXE_EXT): basic_usage.c check-lib
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

embedded_minimal$(EXE_EXT): embedded_minimal.c check-lib
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# C++ demos
raii_wrapper$(EXE_EXT): raii_wrapper.cpp check-lib
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(TARGETS)
	rm -f *.db
	rm -f *.o
	@echo "Cleaned build artifacts"

run: all
	@echo ""
	@echo "=== Running basic_usage demo ==="
	./basic_usage$(EXE_EXT)
	@echo ""
	@echo "=== Running raii_wrapper demo ==="
	./raii_wrapper$(EXE_EXT)
	@echo ""
	@echo "=== Running embedded_minimal demo ==="
	./embedded_minimal$(EXE_EXT)
