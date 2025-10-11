//modes: explicit, um_migrate, gh_hbm_shared, gh_cpu_shared, gh_hmm_pageable

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <unistd.h>   
#include <sys/time.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
#endif

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

//bank-conflict-free tiled transpose
template <typename T, int TILE=32, int BLOCK_ROWS=8>
__global__ void transpose_tiled(const T* __restrict__ A, T* __restrict__ B, int N) {
  __shared__ T tile[TILE][TILE+1]; // +1 to avoid bank conflicts on column reads

  int x = blockIdx.x * TILE + threadIdx.x; // column in A
  int y = blockIdx.y * TILE + threadIdx.y; // row in A

  //load A -> shared tile (coalesced)
  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    int yy = y + i;
    if (x < N && yy < N)
      tile[threadIdx.y + i][threadIdx.x] = A[yy * (size_t)N + x];
  }
  __syncthreads();

  //write shared^T -> B (coalesced)
  int xt = blockIdx.y * TILE + threadIdx.x; // column in B
  int yt = blockIdx.x * TILE + threadIdx.y; // row in B
  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    int yy = yt + i;
    if (xt < N && yy < N)
      B[yy * (size_t)N + xt] = tile[threadIdx.x][threadIdx.y + i];
  }
}


enum class Mode {
  EXPLICIT,        //device malloc + memcpy on pinned host
  UM_MIGRATE,      //managed + prefetch to GPU; prefetch result back to CPU
  GH_HBM_SHARED,   //managed preferred on GPU; CPU reads GPU HBM coherently (no post-prefetch)
  GH_CPU_SHARED,   //pinned host + device pointer (zero-copy: GPU accesses host)
  GH_HMM_PAGEABLE  //plain malloc() pageable host; GPU accesses via HMM/page faulting
};

static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")        return Mode::EXPLICIT;
  if (s == "um_migrate")      return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")   return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")   return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable") return Mode::GH_HMM_PAGEABLE;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str());
  std::exit(EXIT_FAILURE);
}


struct Args {
  int iters = 10;
  int N = 4096;
  std::string mode = "explicit";
  int prefetch = 1;     // only for UM_MIGRATE or GH_HBM_SHARED
  uint64_t seed = 12345;
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -i INT    iterations (default 10)\n"
    "  -n INT    matrix order N (default 4096)\n"
    "  -m STR    mode: explicit | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable (default explicit)\n"
    "  -p 0|1    prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -r INT    RNG seed (default 12345)\n"
    "  -h        help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "i:n:m:p:r:h")) != -1) {
    switch (opt) {
      case 'i': a.iters = std::atoi(optarg); break;
      case 'n': a.N     = std::atoi(optarg); break;
      case 'm': a.mode  = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'r': a.seed  = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  if (a.iters <= 0 || a.N <= 0) {
    fprintf(stderr, "Error: iterations and N must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

//validation helpers
static double max_abs_diff_transpose(const double* A, const double* B, int N) {
  // max |B[i,j] - A[j,i]|
  double m = 0.0;
  for (int i = 0; i < N; ++i) {
    size_t baseB = (size_t)i * N;
    for (int j = 0; j < N; ++j) {
      double diff = std::fabs(B[baseB + j] - A[(size_t)j * N + i]);
      if (diff > m) m = diff;
    }
  }
  return m;
}

template <typename T>
static double checksum(const T* p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += (double)p[i];
  return s;
}


int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);

  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));

  //HMM capability queried for pageable mode
  int pageable = 0, uses_host_pt = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev));

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  const size_t bytes = elems * sizeof(double);

  printf("iters=%d N=%d (%.3f MiB) mode=%s prefetch=%d seed=%llu\n",
         args.iters, N, double(bytes)/(1024.0*1024.0),
         args.mode.c_str(), args.prefetch, (unsigned long long)args.seed);

  //Host and device-visible pointers
  double *hA = nullptr, *hB = nullptr;
  double *dA = nullptr, *dB = nullptr;

  //allocation by mode
  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaMallocHost(&hA, bytes));
      CHECK_CUDA(cudaMallocHost(&hB, bytes));
      CHECK_CUDA(cudaMalloc(&dA, bytes));
      CHECK_CUDA(cudaMalloc(&dB, bytes));
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
      CHECK_CUDA(cudaMallocManaged(&dA, bytes));
      CHECK_CUDA(cudaMallocManaged(&dB, bytes));
      hA = dA; hB = dB; // same pointer usable on CPU
      CHECK_CUDA(cudaMemAdvise(dA, bytes, cudaMemAdviseSetPreferredLocation, dev));
      CHECK_CUDA(cudaMemAdvise(dB, bytes, cudaMemAdviseSetPreferredLocation, dev));
      CHECK_CUDA(cudaMemAdvise(dA, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
      CHECK_CUDA(cudaMemAdvise(dB, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaMallocHost(&hA, bytes));
      CHECK_CUDA(cudaMallocHost(&hB, bytes));
      CHECK_CUDA(cudaHostGetDevicePointer(&dA, hA, 0));
      CHECK_CUDA(cudaHostGetDevicePointer(&dB, hB, 0));
      //CHECK_CUDA(cudaMemAdvise(hA, bytes, cudaMemAdviseSetAccessedBy, dev));
      //CHECK_CUDA(cudaMemAdvise(hB, bytes, cudaMemAdviseSetAccessedBy, dev));
      break;
    case Mode::GH_HMM_PAGEABLE:
      if (!pageable) {
        fprintf(stderr, "ERROR: Device lacks cudaDevAttrPageableMemoryAccess; HMM pageable not supported.\n");
        return 1;
      }
      hA = (double*)std::malloc(bytes);
      hB = (double*)std::malloc(bytes);
      if (!hA || !hB) { fprintf(stderr, "malloc failed\n"); return 1; }
      dA = hA; dB = hB; // same raw pointer
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", uses_host_pt);
      break;
  }

  //random init on cpu
  {
    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) hA[i] = dist(rng);
    std::fill(hB, hB + elems, 0.0);
  }

  //prefetch / copies before kernel
  if (mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) {
    if (args.prefetch) {
      CHECK_CUDA(cudaMemPrefetchAsync(dA, bytes, dev));
      CHECK_CUDA(cudaMemPrefetchAsync(dB, bytes, dev));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  } else if (mode == Mode::EXPLICIT) {
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dB, 0, bytes));
  }

  
  constexpr int TILE = 32;
  constexpr int BLOCK_ROWS = 8;
  dim3 block(TILE, BLOCK_ROWS);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  //warm-up
  transpose_tiled<double, TILE, BLOCK_ROWS><<<grid, block>>>(dA, dB, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  //timed loop
  std::vector<float> times(args.iters);
  for (int it = 0; it < args.iters; ++it) {
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    transpose_tiled<double, TILE, BLOCK_ROWS><<<grid, block>>>(dA, dB, N);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    times[it] = ms;
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
  }
  double avg_ms = 0.0;
  for (float t : times) avg_ms += t;
  avg_ms /= times.size();
  const double bytes_moved = 2.0 * double(bytes); // read + write
  const double gbps = bytes_moved / (1.0e6 * avg_ms);
  printf("Kernel avg: %.3f ms, effective bandwidth: %.2f GB/s\n", avg_ms, gbps);

  //UM migrate: prefetch back to CPU to time migration
  if (mode == Mode::UM_MIGRATE) {
    auto t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(dB, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = wtime();
    double ms = (t1 - t0)*1e3;
    printf("UM prefetch-to-CPU: %.3f ms (%.2f GB/s logical)\n", ms, bytes / 1e6 / ms);
  }

  
  auto r0 = wtime();
  double sumB = checksum(hB, elems);
  auto r1 = wtime();
  double read_ms = (r1 - r0) * 1e3;
  printf("CPU checksum(B)=%.6e, CPU read time: %.3f ms (%.2f GB/s logical)\n",
         sumB, read_ms, bytes / 1e6 / read_ms);

  if (mode == Mode::EXPLICIT) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
    double t1 = wtime();
    double ms = (t1 - t0) * 1e3;
    printf("D2H memcpy: %.3f ms (%.2f GB/s)\n", ms, bytes / 1e6 / ms);
  }

  // Full correctness check: max |B - A^T|
  double maxerr = max_abs_diff_transpose(hA, hB, N);
  printf("Max |B - A^T| = %.3e  => %s\n", maxerr, (maxerr < 1e-9 ? "OK" : "MISMATCH"));

  // ---- Cleanup ----
  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFreeHost(hA));
      CHECK_CUDA(cudaFreeHost(hB));
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(hA));
      CHECK_CUDA(cudaFreeHost(hB));
      break;
    case Mode::GH_HMM_PAGEABLE:
      std::free(hA);
      std::free(hB);
      break;
  }

  return 0;
}

