#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>

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
    printf("MALLOC VERSION\n");
    cublasHandle_t cublasH = NULL;
    cudaStream_t   stream  = NULL;

    int m   = 2;
    int n   = 2;
    int k   = 2;

    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else {
        std::printf("Usage: %s m n k\n", argv[0]);
        std::printf("No valid sizes given, using default m=n=k=2\n");
    }

    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const int sizeA = m * k;
    const int sizeB = k * n;
    const int sizeC = m * n;

    std::printf("Running GEMM with m=%d, n=%d, k=%d\n", m, n, k);

    double t_end2end = wtime();

    
    data_type alpha = 1.0;
    data_type beta  = 0.0;

    // ----- Host pointers (malloc) -----
    data_type *A = (data_type*)malloc(sizeA * sizeof(data_type));
    data_type *B = (data_type*)malloc(sizeB * sizeof(data_type));
    data_type *C = (data_type*)malloc(sizeC * sizeof(data_type));

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices in column-major order
    std::srand((unsigned)std::time(nullptr));

    for (int i = 0; i < sizeA; ++i) {
        A[i] = (data_type)std::rand() / (data_type)RAND_MAX; 
    }

    for (int i = 0; i < sizeB; ++i) {
        B[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }

    for (int i = 0; i < sizeC; ++i) {
        C[i] = 0.0;
    }

    /*printf("A (host)\n");
    print_matrix(m, k, A, lda);
    printf("=====\n");

    printf("B (host)\n");
    print_matrix(k, n, B, ldb);
    printf("=====\n");
    */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    double t_compute = wtime();
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH,
            transa, transb,
            m, n, k,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc
        )
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t_compute = wtime() - t_compute;

    //printf("C (host)\n");
    //print_matrix(m, n, C, ldc);
    //printf("=====\n");

    // Cleanup

    free(A);
    free(B);
    free(C);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    t_end2end = wtime() - t_end2end;
    printf("t_compute = %.3f ms\n", 1e3*t_compute);
    printf("t_end2end = %.3f ms \n ================== \n", 1e3 * t_end2end);

    return EXIT_SUCCESS;
}

