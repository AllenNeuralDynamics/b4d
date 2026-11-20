# Build the B4D Blosc2 codec as a static C library, its example and the C unit tests (src/c, tests/c).
#
#   make BLOSC2_DIR=/path/to/c-blosc2        (built in $(BLOSC2_DIR)/build)
# or, with a system-wide install (pkg-config blosc2):
#   make SYSTEM_BLOSC=1
# The Python package is built by pip (see README.md); this Makefile is for the C library alone.

CC      ?= cc
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
# Apple clang on arm64 does not accept -march=native; NEON is baseline there.
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
  ARCHFLAGS ?= -mcpu=native
else
  ARCHFLAGS ?= -march=native
endif
CFLAGS  ?= -O3 $(ARCHFLAGS) -fno-math-errno -std=gnu11 -Wall -Wextra -Wno-unused-parameter
BLOSC2_DIR ?= ../../c-blosc2
SRC  := src/c
TEST := tests/c

ifdef SYSTEM_BLOSC
  # e.g. Homebrew: brew install c-blosc2 pkg-config
  BLOSC_CFLAGS := $(shell pkg-config --cflags blosc2)
  BLOSC_LIBS   := $(shell pkg-config --libs blosc2)
else
  BLOSC_CFLAGS := -I$(BLOSC2_DIR)/include
  BLOSC_LIBS   := $(BLOSC2_DIR)/build/blosc/libblosc2.a -lpthread
endif

all: libb4d_blosc2.a b4d_example b4d_test

b4d_blosc2.o: $(SRC)/b4d_blosc2.c $(SRC)/b4d_blosc2.h
	$(CC) $(CFLAGS) $(BLOSC_CFLAGS) -c $< -o $@

b4d_zarr.o: $(SRC)/b4d_zarr.c $(SRC)/b4d_blosc2.h
	$(CC) $(CFLAGS) $(BLOSC_CFLAGS) -c $< -o $@

b4d_original.o: $(SRC)/b4d_original.c $(SRC)/b4d_blosc2.h
	$(CC) $(CFLAGS) $(BLOSC_CFLAGS) -c $< -o $@

libb4d_blosc2.a: b4d_blosc2.o b4d_zarr.o b4d_original.o
	$(AR) rcs $@ $^

b4d_example: $(SRC)/b4d_example.c libb4d_blosc2.a
	$(CC) $(CFLAGS) -I$(SRC) $(BLOSC_CFLAGS) $< libb4d_blosc2.a $(BLOSC_LIBS) -lm -o $@

b4d_test: $(TEST)/b4d_test.c $(SRC)/b4d_blosc2.c $(SRC)/b4d_original.c $(SRC)/b4d_blosc2.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-misleading-indentation -I$(SRC) $(BLOSC_CFLAGS) $< $(SRC)/b4d_original.c $(BLOSC_LIBS) -lm -o $@

test: b4d_test
	./b4d_test

bench: b4d_example
	./b4d_example

# unit tests under AddressSanitizer / UBSan
test-asan: $(TEST)/b4d_test.c $(SRC)/b4d_blosc2.c $(SRC)/b4d_blosc2.h
	$(CC) -O1 -g -fsanitize=address,undefined -fno-math-errno -std=gnu11 -w -I$(SRC) $(BLOSC_CFLAGS) $< $(SRC)/b4d_original.c $(BLOSC_LIBS) -lm -o b4d_test_asan
	./b4d_test_asan

clean:
	rm -f *.o *.a b4d_example b4d_test b4d_test_asan

.PHONY: all test bench test-asan clean
