/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ctime>
#include "cublas_utils.h"


double wtime(void)
{
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              // in seconds
               ((double)etstart.tv_usec) / 1000000.0;  // in microseconds
    return now_time;
}

using data_type = double;

int main(int argc, char *argv[]) {
    printf("EXPLICIT VERSION\n");	
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    int iters = 10;
    int m = 2;
    int n = 2;
    int k = 2;

    if (argc == 4 || argc == 5) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
        if (argc == 5) {
            iters = std::atoi(argv[4]);
            if (iters < 1) iters = 1;
        }
    } else {
        std::printf("Usage: %s M N K [ITERS]\n", argv[0]);
        std::printf("No valid sizes given, using default M=N=K=2, ITERS=1\n");
    }
    
    const size_t lda = m;
    const size_t ldb = k;
    const size_t ldc = m;

    const size_t sizeA = (size_t)m * (size_t)k;
    const size_t sizeB = (size_t)k * (size_t)n;
    const size_t sizeC = (size_t)m * (size_t)n;
    
    std::printf("Running GEMM with m=%d, n=%d, k=%d\n", m, n, k);

    double t_end2end = wtime();
    double t_h2d=0.0, t_d2h = 0.0;


    //host pointers
    double t_malloc = wtime();
     
    data_type *h_A = nullptr;
    data_type *h_B = nullptr;
    data_type *h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, sizeA * sizeof(data_type)));   // pinned memory
    CUDA_CHECK(cudaMallocHost(&h_B, sizeB * sizeof(data_type)));   // pinned memory
    CUDA_CHECK(cudaMallocHost(&h_C, sizeC * sizeof(data_type)));   // pinned memory
    
    t_malloc = 1e3*(wtime() - t_malloc);

   
    /*std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < sizeA; ++i) {
        h_A[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (size_t i = 0; i < sizeB; ++i) {
        h_B[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (size_t i = 0; i < sizeC; ++i) {
        h_C[i] = 0.0;
    }*/


    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    printf("cudamalloc starting\n");
    double t_cuda_alloc = wtime();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * sizeA));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * sizeB));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * sizeC));
    t_cuda_alloc = 1e3*(wtime() - t_cuda_alloc);
    printf("cudamalloc end\n");

        std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < sizeA; ++i) {
        h_A[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (size_t i = 0; i < sizeB; ++i) {
        h_B[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (size_t i = 0; i < sizeC; ++i) {
        h_C[i] = 0.0;
    }

    double t1 = wtime(); 

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(data_type) * m*k, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(data_type) * k*n, cudaMemcpyHostToDevice,
                               stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    t_h2d += 1e3 * (wtime() - t1);
    
    double t_compute_total = 0.0;   
    for (int it = 0; it < iters; ++it) {
        double t0 = wtime();
        CUBLAS_CHECK(
            cublasDgemm(
                cublasH,
                transa, transb,
                m, n, k,
                &alpha,
                d_A, lda,
                d_B, ldb,
                &beta,
                d_C, ldc
            )
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t_iter = 1e3*(wtime() - t0);
        std::cout << "iter:" << it << ", t_iter: " << t_iter << "ms" << std::endl;
        t_compute_total += t_iter;
    }
    
    printf("---------------------------------\n");
    
    t1 = wtime();
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(data_type) * sizeC, cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    t_d2h += 1e3*(wtime() - t1);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    t_end2end = wtime() - t_end2end;
    
    double t_compute_avg = t_compute_total / iters;

    printf("t_compute_total = %.3f ms\n", t_compute_total);
    printf("t_compute_avg   = %.3f ms\n", t_compute_avg);
    printf("t_cuda_alloc   = %.3f ms\n", t_cuda_alloc);
    printf("t_malloc   = %.3f ms\n", t_malloc);
    printf("t_h2d   = %.3f ms, t_d2h   = %.3f ms\n", t_h2d, t_d2h);
    printf("t_end2end       = %.3f ms \n ============================================= \n", 1e3 * t_end2end);

    
    return EXIT_SUCCESS;
}
