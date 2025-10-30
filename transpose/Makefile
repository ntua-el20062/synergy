NVCC ?= nvcc
SM   ?= 90a                 
GEN  ?= -arch=sm_$(SM)
OPT  ?= -O3 -std=c++17 -Xcompiler -fPIC

TARGET = transpose_bench

all: transpose_bench transpose_bench_hybrid

transpose_bench: transpose_benchmark.cu
	$(NVCC) $(GEN) $(OPT) -o $@ $<

transpose_bench_hybrid: transpose_benchmark_hybrid.cu
	$(NVCC) $(GEN) $(OPT) -o $@ $<

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

