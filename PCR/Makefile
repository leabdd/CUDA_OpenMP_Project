# Compiler
NVCC = nvcc
SRC = card_details.cu
TARGET = $(SRC:.cu=)


# Compilation flags
CFLAGS = -O2

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET)