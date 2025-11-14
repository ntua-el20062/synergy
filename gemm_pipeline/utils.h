// cublas_utils.h
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));          \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st__ = (call);                                             \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %s:%d: status = %d\n",             \
                         __FILE__, __LINE__, (int)st__);                          \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

