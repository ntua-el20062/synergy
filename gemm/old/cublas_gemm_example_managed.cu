#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
    printf("MANAGED VERSION\n");

    cublasHandle_t cublasH = NULL;
    cudaStream_t   stream  = NULL;

    int m  = 2;
    int n  = 2;
    int k  = 2;
    
    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else {
        std::printf("Usage: %s M N K\n", argv[0]);
        std::printf("No valid sizes given, using default M=N=K=2\n");
    }


    size_t lda = m;
    size_t ldb = k;
    size_t ldc = m;

    const size_t sizeA = (size_t)m * (size_t)k;
    const size_t sizeB = (size_t)k * (size_t)n;
    const size_t sizeC = (size_t)m * (size_t)n;

    double t_end2end = wtime();

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    // Unified memory pointers (accessible on host + device)
    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // step 1: create cublas handle, bind a stream
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // step 2: allocate unified memory (managed)
    CUDA_CHECK(cudaMallocManaged(&d_A, sizeA * sizeof(data_type)));
    CUDA_CHECK(cudaMallocManaged(&d_B, sizeB * sizeof(data_type)));
    CUDA_CHECK(cudaMallocManaged(&d_C, sizeC * sizeof(data_type)));

    std::srand((unsigned)std::time(nullptr));

    for (int i = 0; i < sizeA; ++i) {
        d_A[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (int i = 0; i < sizeB; ++i) {
        d_B[i] = (data_type)std::rand() / (data_type)RAND_MAX;  // [0,1)
    }

    for (int i = 0; i < sizeC; ++i) {
        d_C[i] = 0.0;
    }

    /*printf("A\n");
    print_matrix(m, k, d_A, lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, d_B, ldb);
    printf("=====\n");
    */
    // step 3: compute C = alpha * A * B + beta * C on the GPU
    double t_compute = wtime();
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

    // Wait for GPU work + unified memory migrations to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t_compute = wtime() - t_compute;
    // Now d_C is valid on the host, no memcpy needed
    //printf("C\n");
    //print_matrix(m, n, d_C, ldc);
    //printf("=====\n");

    // free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    t_end2end = wtime() - t_end2end;
    printf("t_compute = %.3f ms\n", 1e3*t_compute);
    printf("t_end2end = %.3f ms \n ================== \n", 1e3 * t_end2end);


    return EXIT_SUCCESS;
}

