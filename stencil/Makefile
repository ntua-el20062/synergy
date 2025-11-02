NVCC        ?= nvcc

SM          ?= 90a

CXXSTD      ?= c++17
OPTFLAGS    ?= -O3
WARNFLAGS   ?= -Xcompiler -Wall -Xcompiler -fopenmp

ifeq ($(DEBUG),1)
  OPTFLAGS := -O0 -g -G
endif

ifeq ($(OMP_EXPLICIT_LINK),1)
  OMP_LINK := -Xlinker -lgomp
else
  OMP_LINK :=
endif

NVFLAGS = -std=$(CXXSTD) $(OPTFLAGS) $(WARNFLAGS) -arch=sm_$(SM)

STENCIL_SRC     := stencil_benchmark_synergy.cu

STENCIL_BIN     := stencil_bench_synergy

.PHONY: all
all: $(STENCIL_BIN)

stencil_bench_synergy: stencil_benchmark_synergy.cu
	$(NVCC) $(NVFLAGS) $< -o $@



.PHONY: clean
clean:
	@rm -f $(STENCIL_BIN) 


