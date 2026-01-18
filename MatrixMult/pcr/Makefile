CC := gcc
CFLAGS := -O2 -Wall -Wextra
OPENMP_FLAGS := -fopenmp
LIBS := -lm
LINK_FLAGS := $(OPENMP_FLAGS) $(LIBS)

# Library source files
LIB_FILES := sle.c diagonal.c pcr.c
LIB_OBJS := $(LIB_FILES:.c=.o)

# Executables
TARGETS := pcrsolve

# Default target
.PHONY: all
all: $(TARGETS)

# Build the object files from library sources
$(LIB_OBJS): %.o: %.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Build pcrsolve executable
pcrsolve: main.c $(LIB_OBJS)
	$(CC) $(CFLAGS) -DPCR_MAIN $(OPENMP_FLAGS) $^ $(LIBS) -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(LIB_OBJS) $(TARGETS)

# Clean everything including executables
.PHONY: distclean
distclean: clean
	rm -f *.o $(TARGETS)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Build pcrsolve and thomassolve (default)"
	@echo "  pcrsolve     - Build pcrsolve executable"
	@echo "  clean        - Remove object files and executables"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help message"
