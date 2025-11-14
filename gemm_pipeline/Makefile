NVCC       := nvcc
NVCC_FLAGS := -O3 -std=c++14

TARGET     := gemm_pipeline
SRC        := gemm_pipeline.cu

all: $(TARGET)

$(TARGET): $(SRC) utils.h
	$(NVCC) $(NVCC_FLAGS) $(SRC) -lcublas -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean

