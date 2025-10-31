// transpose_hybrid_benchmark.cu
// Hybrid CPU+GPU tiled transpose with 7 memory modes (explicit, UM variants, GH_*).
// CPU handles columns [0:k), GPU handles columns [k:N). Both run concurrently.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <unistd.h>     // getopt
#include <sys/time.h>   // gettimeofday

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

// 64-bit SplitMix: small, fast, repeatable per index
__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Convert 64-bit to uniform [0,1) as double using top 53 bits
__device__ inline double u01_from_u64(uint64_t x) {
  const double inv2_53 = 1.0 / 9007199254740992.0; // 2^53
  return double(x >> 11) * inv2_53;
}

// Fill array with random doubles in [-1, 1]
__global__ void init_random_double(double* a, size_t n, uint64_t seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) {
    uint64_t r = splitmix64(seed + i);
    double u = u01_from_u64(r);   // [0,1)
    a[i] = 2.0 * u - 1.0;         // [-1,1)
  }
}

// Set array to zero
__global__ void set_zero_double(double* a, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 0.0;
}


// ======================= Tiled transpose kernel (rect support via bounds) =======================
template <typename T, int TILE=32, int BLOCK_ROWS=8>
__global__ void transpose_tiled_rect(const T* __restrict__ A, T* __restrict__ B,
                                     int N,       // full leading dimension (square N x N)
                                     int x0,      // start column in A (and row in B)
                                     int y0,      // start row in A (and column in B)
                                     int W,       // sub-rect width  (columns in A)
                                     int H)       // sub-rect height (rows    in A)
{
  // We process a logical rectangle A[y0 : y0+H, x0 : x0+W].
  // Output goes to B[x0 : x0+W, y0 : y0+H] (rows/cols swapped).
  __shared__ T tile[TILE][TILE+1];

  int local_x = blockIdx.x * TILE + threadIdx.x; // 0..W-1
  int local_y = blockIdx.y * TILE + threadIdx.y; // 0..H-1
  int ax = x0 + local_x; // column in A
  int ay = y0 + local_y; // row in A

  // Load A -> shared tile (coalesced along rows)
  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    int ly = local_y + i;
    int ayy = y0 + ly;
    if (local_x < W && ly < H) {
      tile[threadIdx.y + i][threadIdx.x] = A[(size_t)ayy * N + ax];
    }
  }
  __syncthreads();

// Write transposed tile -> B (coalesced)
// Output sub-rect top-left in B is (row = x0, col = y0)
int bx_base = y0 + local_y;   // column base in B (depends on local_y)
int by      = x0 + local_x;   // row in B (depends on local_x)

#pragma unroll
for (int i = 0; i < TILE; i += BLOCK_ROWS) {
  int ly = local_y + i;
  if (local_x < W && ly < H) {
    // col advances with i (because ly advances)
    B[(size_t)by * N + (bx_base + i)] = tile[threadIdx.x][threadIdx.y + i];
  }
}
}

// =============================== CPU blocked transpose for a column range ======================
static void transpose_cpu_blocked(const double* __restrict__ A,
                                  double* __restrict__ B,
                                  int N, int c0, int c1, int BS=32)
{
  // Transpose columns [c0, c1) of A into rows [c0, c1) of B.
  for (int jj = c0; jj < c1; jj += BS) {
    int Jmax = std::min(jj + BS, c1);
    for (int ii = 0; ii < N; ii += BS) {
      int Imax = std::min(ii + BS, N);
      for (int j = jj; j < Jmax; ++j) {
        const size_t jN = (size_t)j * N;
        for (int i = ii; i < Imax; ++i) {
          B[jN + i] = A[(size_t)i * N + j];
        }
      }
    }
  }
}

// =============================== Modes / CLI ===================================================
enum class Mode {
  EXPLICIT,
  UM_MIGRATE,
  GH_HBM_SHARED,
  GH_CPU_SHARED,
  GH_HMM_PAGEABLE,
  GH_HMM_PAGEABLE_CUDA_INIT,
  UM_MIGRATE_NO_PREFETCH,
  GH_HBM_SHARED_NO_PREFETCH
};
static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")                  return Mode::EXPLICIT;
  if (s == "um_migrate")                return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")             return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")             return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable")           return Mode::GH_HMM_PAGEABLE;
  if (s == "gh_hmm_pageable_cuda_init")  return Mode::GH_HMM_PAGEABLE_CUDA_INIT;
  if (s == "um_migrate_no_prefetch")    return Mode::UM_MIGRATE_NO_PREFETCH;
  if (s == "gh_hbm_shared_no_prefetch") return Mode::GH_HBM_SHARED_NO_PREFETCH;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str()); std::exit(EXIT_FAILURE);
}

struct Args {
  int N = 8192;
  int iters = 10;         // kernel timing iterations (hybrid performed each iter)
  std::string mode = "explicit";
  int prefetch = 1;       // for UM_MIGRATE / GH_HBM_SHARED
  uint64_t seed = 12345;
  double frac = 0.5;      // GPU fraction f, CPU gets (1-f)
  int tile = 32;
  int block_rows = 8;
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -n INT     matrix order N (default 8192)\n"
    "  -i INT     iterations (default 10)\n"
    "  -m STR     mode: explicit | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable |\n"
    "             um_migrate_no_prefetch | gh_hbm_shared_no_prefetch (default explicit)\n"
    "  -p 0|1     prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -f FLOAT   GPU fraction in [0,1] (default 0.5) — GPU handles columns [f*N, N)\n"
    "  -r UINT64  RNG seed (default 12345)\n"
    "  -h         help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "n:i:m:p:f:r:h")) != -1) {
    switch (opt) {
      case 'n': a.N = std::atoi(optarg); break;
      case 'i': a.iters = std::atoi(optarg); break;
      case 'm': a.mode = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'f': a.frac = std::atof(optarg); break;
      case 'r': a.seed = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  if (a.N <= 0 || a.iters <= 0 || a.frac < 0.0 || a.frac > 1.0) {
    fprintf(stderr, "Error: invalid arguments.\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

// =============================== Capabilities / buffers ========================================
struct DeviceCaps {
  int pageable=0, uses_host_pt=0, dev=0;
};
static DeviceCaps query_caps() {
  DeviceCaps c{};
  CHECK_CUDA(cudaGetDevice(&c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.pageable, cudaDevAttrPageableMemoryAccess, c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, c.dev));
  return c;
}

template <typename T>
struct Buffers {
  T *hA=nullptr, *hB=nullptr;
  T *dA=nullptr, *dB=nullptr;
  size_t nbytes=0;
};

template <typename T>
static void allocate_buffers(Mode mode, size_t nElems, Buffers<T>& buf, const DeviceCaps& caps) {
  buf.nbytes = nElems * sizeof(T);
  switch (mode) {
    case Mode::EXPLICIT: {
      CHECK_CUDA(cudaMallocHost(&buf.hA, buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hB, buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dB, buf.nbytes));
      break;
    }
    case Mode::UM_MIGRATE: // the same allocation style with GH_HBM_SHARED(managed memory)
    case Mode::UM_MIGRATE_NO_PREFETCH: {
      CHECK_CUDA(cudaMallocManaged(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dB, buf.nbytes));
      buf.hA = buf.dA; buf.hB = buf.dB; //same pointer for CPU and GPU
      break;
    }
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
    case Mode::GH_HBM_SHARED: {
      CHECK_CUDA(cudaMallocManaged(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dB, buf.nbytes));
      buf.hA = buf.dA; buf.hB = buf.dB; //same pointer for CPU and GPU
      cudaMemLocation loc_dev{}; loc_dev.type = cudaMemLocationTypeDevice; loc_dev.id = caps.dev;

      CHECK_CUDA(cudaMemAdvise(buf.dA, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev)); //hint to UM pager:If you have a choice, keep these pages resident on GPU dev,it’s a policy hint, 
                                                                                    //not a hard pin. If the CPU (or another GPU) starts accessing those pages a lot, the runtime may 
                                                                                    //still migrate or replicate them. But with this hint, after GPU use, the pager will try to keep or return the pages 
                                                                                    //to GPU memory (HBM) rather than drifting back to system RAM.
      CHECK_CUDA(cudaMemAdvise(buf.dB, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      cudaMemLocation loc_cpu{}; loc_cpu.type = cudaMemLocationTypeHost; loc_cpu.id = 0;

      CHECK_CUDA(cudaMemAdvise(buf.dA, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu)); //This can reduce first-touch penalties when the CPU later reads managed memory.
      CHECK_CUDA(cudaMemAdvise(buf.dB, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu)); //  --//--
      
      break;
    }
    case Mode::GH_CPU_SHARED: {
      CHECK_CUDA(cudaMallocHost(&buf.hA, buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hB, buf.nbytes));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dA, buf.hA, 0));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dB, buf.hB, 0));
      break;
    }
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE: {
      if (!caps.pageable) {
        fprintf(stderr, "ERROR: Device lacks cudaDevAttrPageableMemoryAccess; HMM pageable not supported.\n");
        std::exit(EXIT_FAILURE);
      }
      buf.hA = (T*)std::malloc(buf.nbytes);
      buf.hB = (T*)std::malloc(buf.nbytes);
      if (!buf.hA || !buf.hB) { perror("malloc"); std::exit(EXIT_FAILURE); }
      buf.dA = buf.hA; buf.dB = buf.hB; // same raw pointers
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", caps.uses_host_pt);
      break;
    }
  }
}

template <typename T>
static void pre_kernel_setup(Mode mode, const Buffers<T>& buf, const DeviceCaps& caps, int prefetch) {
  if (mode == Mode::EXPLICIT) {
    CHECK_CUDA(cudaMemcpy(buf.dA, buf.hA, buf.nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(buf.dB, 0, buf.nbytes));
  } else if ((mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) && prefetch) {
    cudaMemLocation dst_dev{}; dst_dev.type = cudaMemLocationTypeDevice; dst_dev.id = caps.dev;

    CHECK_CUDA(cudaMemPrefetchAsync(buf.dA, buf.nbytes, dst_dev, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dB, buf.nbytes, dst_dev, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

// =============================== CPU checksum ===================================================
template <typename T>
static double checksum(const T* p, size_t n) {
  double s=0.0; for (size_t i=0;i<n;++i) s += (double)p[i]; return s;
}

// =============================== Main ===========================================================
int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);
  DeviceCaps caps = query_caps();

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  Buffers<double> buf;
  allocate_buffers<double>(mode, elems, buf, caps);

  
  if(mode != Mode::GH_HMM_PAGEABLE_CUDA_INIT)
  {
    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) buf.hA[i] = dist(rng);
    std::fill(buf.hB, buf.hB + elems, 0.0);
  }
  else {
        // GPU init into pageable host memory via HMM
        const int threads = 256;
        const int blocksA = int((elems + threads - 1) / threads);
        const int blocksB = blocksA;

        init_random_double<<<blocksA, threads>>>(buf.dA, elems, args.seed);
        set_zero_double<<<blocksB, threads>>>(buf.dB, elems);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

  } // Init A with RNG, B zeros (on CPU except HMM we could do GPU init, but keep simple)

  // Pre-placement
  pre_kernel_setup<double>(mode, buf, caps, args.prefetch);

  // Split
  int k = std::max(0, std::min(N, int(std::llround(args.frac * N))));
  printf("Hybrid split: CPU columns [0, %d), GPU columns [%d, %d)\n", k, k, N);

  // Launch config for GPU sub-rect
  constexpr int TILE = 32;
  constexpr int BLOCK_ROWS = 8;

  // warming (optional) — perform one hybrid step outside timing to absorb JIT/first-touch
  {
    // CPU half async via thread
    std::thread cpu_thr([&](){
      transpose_cpu_blocked(buf.hA, buf.hB, N, 0, k);
    });

    // GPU half
    int W = N - k;
    if (W > 0) {
      int H = N;
      dim3 block(TILE, BLOCK_ROWS);
      dim3 grid((W + TILE - 1)/TILE, (H + TILE - 1)/TILE);

      if (mode == Mode::EXPLICIT) {
        // use device buffers directly (already whole A copied H2D and B zeroed)
        transpose_tiled_rect<double, TILE, BLOCK_ROWS><<<grid, block>>>(buf.dA, buf.dB, N,
                                                                        /*x0=*/k, /*y0=*/0, /*W=*/W, /*H=*/H);
        CHECK_CUDA(cudaDeviceSynchronize());
      } else {
        // shared-pointer modes
        transpose_tiled_rect<double, TILE, BLOCK_ROWS><<<grid, block>>>(buf.dA, buf.dB, N, k, 0, W, H);
        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }

    cpu_thr.join();

    // explicit: copy back B fully (or only rows [k:N, :]); we’ll do it in timed loop anyway
    if (mode == Mode::EXPLICIT) {
      CHECK_CUDA(cudaMemcpy(buf.hB, buf.dB, buf.nbytes, cudaMemcpyDeviceToHost));
    }
  }

  // Timed loop
  std::vector<float> times(args.iters);
  for (int it = 0; it < args.iters; ++it) {
    cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));

    std::thread cpu_thr([&](){
      transpose_cpu_blocked(buf.hA, buf.hB, N, 0, k);
    });

    int W = N - k;
    if (W > 0) {
      int H = N;
      dim3 block(TILE, BLOCK_ROWS);
      dim3 grid((W + TILE - 1)/TILE, (H + TILE - 1)/TILE);

      if (mode == Mode::EXPLICIT) {
        // Whole A is already on device; run kernel and copy back after CPU finishes
        transpose_tiled_rect<double, TILE, BLOCK_ROWS><<<grid, block>>>(buf.dA, buf.dB, N, k, 0, W, H);
        CHECK_CUDA(cudaGetLastError());
      } else {
        // Shared-pointer modes: kernel runs directly on shared arrays
        transpose_tiled_rect<double, TILE, BLOCK_ROWS><<<grid, block>>>(buf.dA, buf.dB, N, k, 0, W, H);
        CHECK_CUDA(cudaGetLastError());
      }
    }

    CHECK_CUDA(cudaEventRecord(e));
    // Wait CPU
    cpu_thr.join();
    // Wait GPU
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    times[it] = ms;

    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));

    // explicit: pull back device B (entire matrix) so host has complete result
    if (mode == Mode::EXPLICIT) {
      double t0 = wtime();
      CHECK_CUDA(cudaMemcpy(buf.hB, buf.dB, buf.nbytes, cudaMemcpyDeviceToHost));
      double ms_copy = (wtime() - t0)*1e3;
      printf("  [iter %d] explicit D2H: %.3f ms\n", it, ms_copy);
    }
  }

  double avg_ms = 0.0; for (float t : times) avg_ms += t; avg_ms /= times.size();
  const double bytes = 2.0 * double(elems) * sizeof(double); // read + write
  const double gbps = bytes / (1.0e6 * avg_ms);
  printf("Hybrid Kernel avg: %.3f ms, effective bandwidth: %.2f GB/s (logical)\n", avg_ms, gbps);

  // Managed migrate back (UM_MIGRATE only): not needed for hybrid semantics, but we can time it
  if (mode == Mode::UM_MIGRATE) {
    cudaMemLocation dst_cpu{}; dst_cpu.type = cudaMemLocationTypeHost; dst_cpu.id = 0;
    double t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dB, buf.nbytes, dst_cpu, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    double ms = (wtime() - t0)*1e3;
    printf("UM prefetch-to-CPU: %.3f ms (%.2f GB/s logical)\n", ms, double(buf.nbytes)/1e6/ms);
  }

  // CPU checksum read timing (captures passive migration cost for no-prefetch / GH_* / HMM)
  double t0 = wtime();
  double sumB = checksum(buf.hB, elems);
  double read_ms = (wtime() - t0)*1e3;
  printf("CPU checksum(B)=%.6e, CPU read time: %.3f ms (%.2f GB/s logical)\n",
         sumB, read_ms, double(buf.nbytes)/1e6/read_ms);

  // Cleanup
  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(buf.dA)); CHECK_CUDA(cudaFree(buf.dB));
      CHECK_CUDA(cudaFreeHost(buf.hA)); CHECK_CUDA(cudaFreeHost(buf.hB));
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
      CHECK_CUDA(cudaFree(buf.dA)); CHECK_CUDA(cudaFree(buf.dB));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(buf.hA)); CHECK_CUDA(cudaFreeHost(buf.hB));
      break;
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE:
      std::free(buf.hA); std::free(buf.hB);
      break;
  }
  return 0;
}

