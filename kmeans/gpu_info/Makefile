NVCC        = /usr/local/cuda-12.1/bin/nvcc
CFLAGS = -arch=sm_60 # Adjust to match your GPU's architecture

# Target and source files
TARGET = gpu_info
SRC = gpu_info.cu

# Build the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean up generated files
clean:
	rm -f $(TARGET)
