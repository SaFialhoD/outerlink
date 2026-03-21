# OuterLink CUDA Driver API Test Application
#
# Builds a minimal vector-add program that uses only the CUDA Driver API.
# Can run natively (against real libcuda.so) or under LD_PRELOAD with
# the OuterLink interception library.

# ---------- Configuration ----------

# CUDA toolkit path (override: make CUDA_PATH=/opt/cuda-12.x)
CUDA_PATH ?= /usr/local/cuda

# OuterLink client library (override: make OUTERLINK_LIB=/path/to/libouterlink_client.so)
OUTERLINK_LIB ?= ../../target/release/libouterlink_client.so

# OuterLink server address for interposed mode
OUTERLINK_SERVER ?= 127.0.0.1:14833

# ---------- Compiler ----------

CC       = gcc
CFLAGS   = -Wall -Wextra -std=c11 -O2
LDFLAGS  = -L$(CUDA_PATH)/lib64 -lcuda

# Include paths: prefer real CUDA headers if available, fall back to our stubs
CUDA_INC = $(CUDA_PATH)/include
STUB_INC = ../../cuda-stubs

# Check if real CUDA headers exist
REAL_CUDA_EXISTS := $(shell test -f $(CUDA_INC)/cuda.h && echo yes || echo no)

ifeq ($(REAL_CUDA_EXISTS),yes)
    CFLAGS += -I$(CUDA_INC)
else
    CFLAGS += -I$(STUB_INC)
    $(info NOTE: Using OuterLink CUDA stubs from $(STUB_INC))
endif

# ---------- Targets ----------

.PHONY: native interpose clean help

BINARY = test_vector_add

help:
	@echo "OuterLink CUDA Test Application"
	@echo ""
	@echo "Targets:"
	@echo "  make native       Build and link against real CUDA driver"
	@echo "  make interpose    Run under LD_PRELOAD with OuterLink"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH         CUDA toolkit location  (default: /usr/local/cuda)"
	@echo "  OUTERLINK_LIB     Path to libouterlink_client.so"
	@echo "  OUTERLINK_SERVER  Server address          (default: 127.0.0.1:14833)"

# Build the test binary linking against the real CUDA driver library.
# This produces a native executable for baseline verification.
native: $(BINARY)

$(BINARY): test_vector_add.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo ""
	@echo "Built: $(BINARY)"
	@echo "Run:   ./$(BINARY)"

# Run the (already built) binary under LD_PRELOAD so OuterLink intercepts
# all CUDA Driver API calls. The OuterLink server must be running at
# OUTERLINK_SERVER before invoking this target.
interpose: $(BINARY)
	@echo "Running with LD_PRELOAD interception..."
	@echo "  Library: $(OUTERLINK_LIB)"
	@echo "  Server:  $(OUTERLINK_SERVER)"
	@echo ""
	OUTERLINK_SERVER=$(OUTERLINK_SERVER) \
	LD_PRELOAD=$(OUTERLINK_LIB) \
	./$(BINARY)

clean:
	rm -f $(BINARY)
