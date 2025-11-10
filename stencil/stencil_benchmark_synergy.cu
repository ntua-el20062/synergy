//jacobi style --> in each time step i use values from the previous step
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <unistd.h>   
#include <sys/time.h> 
#include <omp.h>
#include <numeric>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
#endif

static inline double wtime(void)
{
  double now_time;
  struct timeval  etstart;
  struct timezone tzp;
  if (gettimeofday(&etstart, &tzp) == -1)
    perror("Error: calling gettimeofday() not successful.\n");
  now_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1e6;
  return now_time;
}

enum class Mode {
  EXPLICIT,
  UM_MIGRATE,
  GH_HBM_SHARED,
  GH_CPU_SHARED,
  GH_HMM_PAGEABLE,
  GH_HMM_PAGEABLE_GPU_INIT,  
  UM_MIGRATE_NO_PREFETCH,
  GH_HBM_SHARED_NO_PREFETCH
};

static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")                  return Mode::EXPLICIT;
  if (s == "um_migrate")                return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")             return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")             return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable")           return Mode::GH_HMM_PAGEABLE;
  if (s == "gh_hmm_pageable_gpu_init")  return Mode::GH_HMM_PAGEABLE_GPU_INIT;
  if (s == "um_migrate_no_prefetch")    return Mode::UM_MIGRATE_NO_PREFETCH;
  if (s == "gh_hbm_shared_no_prefetch") return Mode::GH_HBM_SHARED_NO_PREFETCH;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str());
  std::exit(EXIT_FAILURE);
}

struct Args {
  int N = 4096;           // grid dimension N x N
  int iters = 10;         // iterations
  std::string mode = "explicit";
  int prefetch = 1;       // only relevant to UM_MIGRATE / GH_HBM_SHARED
  uint64_t seed = 12345;
  int steps = 1;          // time-steps per iteration
  int autoTune = 0;       // enable autotuner if 1
  int threads = 0;        // CPU threads (0 => OpenMP default)
  int kPct = 20;          // K as percent of N
  double amp = 1e-3;      // init amplitude: values in [-amp, +amp]
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -n INT      problem size N (default 4096; domain is N x N)\n"
    "  -i INT      iterations (default 10)\n"
    "  -t INT      stencil time-steps per iteration (default 1)\n"
    "  -m STR      mode: explicit | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable |\n"
    "              gh_hmm_pageable_gpu_init | um_migrate_no_prefetch | gh_hbm_shared_no_prefetch\n"
    "  -p 0|1      prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -r UINT64   RNG seed (default 12345)\n"
    "  -a FLOAT    init amplitude (default 1e-3) => values in [-a,+a]\n"
    "  --auto      enable autotuning for K and CPU threads\n"
    "  --threads T set CPU threads (ignored by --auto)\n"
    "  --kPct P    set seam as percent of N (0..95, default 20)\n"
    "  -h          help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "n:i:t:m:p:r:a:h")) != -1) {
    switch (opt) {
      case 'n': a.N = std::atoi(optarg); break;
      case 'i': a.iters = std::atoi(optarg); break;
      case 't': a.steps = std::atoi(optarg); break;
      case 'm': a.mode = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'r': a.seed = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'a': a.amp  = std::atof(optarg); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  for (int i = optind; i < argc; ++i) {
    if       (std::strcmp(argv[i], "--auto") == 0) a.autoTune = 1;
    else if (std::strncmp(argv[i], "--threads", 9) == 0) {
      const char* eq = std::strchr(argv[i], '=');
      if (eq) a.threads = std::atoi(eq+1);
      else if (i+1 < argc) a.threads = std::atoi(argv[++i]);
      else { fprintf(stderr, "Missing value for --threads\n"); std::exit(EXIT_FAILURE); }
    }
    else if (std::strncmp(argv[i], "--kPct", 6) == 0) {
      const char* eq = std::strchr(argv[i], '=');
      if (eq) a.kPct = std::atoi(eq+1);
      else if (i+1 < argc) a.kPct = std::atoi(argv[++i]);
      else { fprintf(stderr, "Missing value for --kPct\n"); std::exit(EXIT_FAILURE); }
    }
    else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); std::exit(EXIT_FAILURE); }
  }
  if (a.N <= 0 || a.iters <= 0 || a.steps <= 0) {
    fprintf(stderr, "Error: N, iters, and steps must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  if (a.amp <= 0) a.amp = 1e-3;
  return a;
}

//initiallizing on gpu from hmm_pageable_gpu_init
// A simple stateless per-index RNG (SplitMix64-ish) -> uniform [0,1)
__device__ inline double index_random(uint64_t seed, uint64_t i) {
    uint64_t z = seed + i + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    // map to [0,1) using 53 high bits to a double
    return (z >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

__global__ void k_fill_random(double* __restrict__ x,
                              size_t n,
                              uint64_t seed,
                              double amp)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double u = index_random(seed, i);      // [0,1)
        x[i] = (2.0 * u - 1.0) * amp;          // [-amp, amp]
    }
}

__global__ void k_memset_zero(double* __restrict__ x, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}



//first stencil kernel: all is done by the gpu, no cpu compute --> gpu also the borders
__global__ void stencil5_2d(const double* __restrict__ in, double* __restrict__ out, int N)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= N) return;
  const size_t idx = (size_t)y * N + x;

  if (x == 0 || y == 0 || x == N-1 || y == N-1) { out[idx] = in[idx]; return; }

  const size_t idxN = (size_t)(y-1)*N + x;
  const size_t idxS = (size_t)(y+1)*N + x;
  const size_t idxW = (size_t)y*N + (x-1);
  const size_t idxE = (size_t)y*N + (x+1);
  out[idx] = 0.25 * (in[idxN] + in[idxS] + in[idxW] + in[idxE]);
}

//second gpu stencil kernel: when splitted, interior rows that get assigned to GPU 
__global__ void stencil5_2d_yspan(const double* __restrict__ in, double* __restrict__ out,
                                  int N, int yStart, int yEnd)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= N) return;
  if (y < yStart || y > yEnd || x == 0 || x == N-1) return;

  const size_t idx  = (size_t)y * N + x;
  const size_t idxN = (size_t)(y-1)*N + x;
  const size_t idxS = (size_t)(y+1)*N + x;
  const size_t idxW = (size_t)y*N + (x-1);
  const size_t idxE = (size_t)y*N + (x+1);
  out[idx] = 0.25 * (in[idxN] + in[idxS] + in[idxW] + in[idxE]);
}

struct DeviceCaps {
  int pageable = 0;
  int uses_host_pt = 0;
  int dev = 0;
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
  T *hIn=nullptr, *hOut=nullptr;   // host view
  T *dIn=nullptr, *dOut=nullptr;   // device view
  size_t nbytes=0;
  cudaStream_t stream = 0;
};

template <typename T>
static void allocate_buffers(Mode mode, size_t nElems, Buffers<T>& buf, const DeviceCaps& caps) {
  buf.nbytes = nElems * sizeof(T);
  switch (mode) {
    case Mode::EXPLICIT: {
      CHECK_CUDA(cudaMallocHost(&buf.hIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hOut, buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dOut, buf.nbytes));
      break;
    }
    case Mode::UM_MIGRATE:
    case Mode::UM_MIGRATE_NO_PREFETCH:{
      CHECK_CUDA(cudaMallocManaged(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dOut, buf.nbytes));
      buf.hIn  = buf.dIn;
      buf.hOut = buf.dOut;
      break;
    }
    case Mode::GH_HBM_SHARED:
    case Mode::GH_HBM_SHARED_NO_PREFETCH: {
      CHECK_CUDA(cudaMallocManaged(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dOut, buf.nbytes));
      buf.hIn  = buf.dIn;
      buf.hOut = buf.dOut;
      cudaMemLocation loc_dev{}; loc_dev.type = cudaMemLocationTypeDevice; loc_dev.id = caps.dev;
      CHECK_CUDA(cudaMemAdvise(buf.dIn, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      CHECK_CUDA(cudaMemAdvise(buf.dOut, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      cudaMemLocation loc_cpu{}; loc_cpu.type = cudaMemLocationTypeHost; loc_cpu.id = 0;
      CHECK_CUDA(cudaMemAdvise(buf.dIn, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu));
      CHECK_CUDA(cudaMemAdvise(buf.dOut, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu));
      break;
    }
    case Mode::GH_CPU_SHARED: {
      CHECK_CUDA(cudaMallocHost(&buf.hIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hOut, buf.nbytes));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dIn,  buf.hIn,  0));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dOut, buf.hOut, 0));
      break;
    }
    case Mode::GH_HMM_PAGEABLE_GPU_INIT:
    case Mode::GH_HMM_PAGEABLE: {
      if (!caps.pageable) {
        fprintf(stderr, "ERROR: Device lacks pageable HMM support.\n");
        std::exit(EXIT_FAILURE);
      }
      buf.hIn  = (T*)std::malloc(buf.nbytes);
      buf.hOut = (T*)std::malloc(buf.nbytes);
      if (!buf.hIn || !buf.hOut) { perror("malloc"); std::exit(EXIT_FAILURE); }
      buf.dIn  = buf.hIn;
      buf.dOut = buf.hOut;
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", caps.uses_host_pt);
      break;
    }
  }
}

//measure timings
struct IterTimings {
  double end2end_ms = 0.0;
  double gpu_ms     = 0.0;   //sum over steps
  double cpu_ms     = 0.0;   //sum over steps
  double h2d_ms     = 0.0;   //explicit region exchanges + initial copy
  double d2h_ms     = 0.0;   //explicit region exchanges + final copy
  double um_to_dev_ms  = 0.0; //UM/GH_HBM_SHARED prefetch to device
  double um_to_host_ms = 0.0; //UM prefetch back to host
};

//prefetching - migrations for some of the modes
template <typename T>
static double pre_kernel_setup(Mode mode, const Buffers<T>& buf, const DeviceCaps& caps, int do_prefetch) {
  double ms = 0.0;
  if (mode == Mode::EXPLICIT) { //memcpy
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(buf.dIn, buf.hIn, buf.nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(buf.dOut, 0, buf.nbytes));
    ms = (wtime() - t0) * 1e3;
    printf("H2D memcpy + D memset: %.3f ms (%.2f GB/s memcpy only)\n",
           ms, (double(buf.nbytes)/1e6) / std::max(1e-9, ms));
  } else if ((mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) && do_prefetch) { //prefetching to device
    double t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dIn,  buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0)); //the managed buffer got initiallized by the cpu, so we prefetch them to the gpu before computation
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    ms = (wtime() - t0) * 1e3;
    printf("UM prefetch-to-Device (2x buffers): %.3f ms (%.2f GB/s logical)\n",
           ms, (double(buf.nbytes*2)/1e6) / std::max(1e-9, ms));
  }
  return ms;
}

template <typename T>
static double post_kernel_maybe_prefetch_back(Mode mode, const Buffers<T>& buf) { 
  double ms = 0.0;
  if (mode == Mode::UM_MIGRATE) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeHost, 0}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    ms = (wtime() - t0) * 1e3;
    printf("UM prefetch-to-Host (dOut): %.3f ms (%.2f GB/s logical)\n",
           ms, (double(buf.nbytes)/1e6) / std::max(1e-9, ms));
  }
  return ms;
}

template <typename T>
static double post_kernel_copy_back_if_needed(Mode mode, const Buffers<T>& buf) { //memcpy back to host, need it for EXPLICIT
  double ms = 0.0;
  if (mode == Mode::EXPLICIT) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(buf.hOut, buf.dOut, buf.nbytes, cudaMemcpyDeviceToHost));
    ms = (wtime() - t0) * 1e3;
    printf("D2H memcpy: %.3f ms (%.2f GB/s)\n",
           ms, (double(buf.nbytes)/1e6) / std::max(1e-9, ms));
  }
  return ms;
}

template <typename T>
static void free_buffers(Mode mode, Buffers<T>& buf) {
  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(buf.dIn));
      CHECK_CUDA(cudaFree(buf.dOut));
      CHECK_CUDA(cudaFreeHost(buf.hIn));
      CHECK_CUDA(cudaFreeHost(buf.hOut));
      break;
    case Mode::UM_MIGRATE:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
      CHECK_CUDA(cudaFree(buf.dIn));
      CHECK_CUDA(cudaFree(buf.dOut));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(buf.hIn));
      CHECK_CUDA(cudaFreeHost(buf.hOut));
      break;
    case Mode::GH_HMM_PAGEABLE_GPU_INIT:
    case Mode::GH_HMM_PAGEABLE:
      std::free(buf.hIn);
      std::free(buf.hOut);
      break;
  }
}


static inline void copy_borders(const double* in, double* out, int N) { //????????????????????????????????????
  if (N <= 1) return;
  std::memcpy(out + 0*size_t(N), in + 0*size_t(N), sizeof(double)*N);            // top row
  std::memcpy(out + (size_t)(N-1)*N, in + (size_t)(N-1)*N, sizeof(double)*N);    // bottom row
  for (int y = 1; y < N-1; ++y) { // left/right columns
    out[(size_t)y*N + 0]    = in[(size_t)y*N + 0];
    out[(size_t)y*N + N-1]  = in[(size_t)y*N + N-1];
  }
}


//omp parallel implementation of the stencil to the rows that are assigned to the cpu
static void cpu_stencil_rows_omp(const double* in, double* out, int N, int yStart, int yEnd, int threads)
{
  if (yEnd < yStart) return;
  if (threads > 0) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
  }
#pragma omp parallel for schedule(static)
  for (int y = yStart; y <= yEnd; ++y) {
    #pragma omp simd
    for (int x = 1; x <= N-2; ++x) {
      const size_t idx  = (size_t)y*N + x;
      const size_t idxN = (size_t)(y-1)*N + x;
      const size_t idxS = (size_t)(y+1)*N + x;
      const size_t idxW = (size_t)y*N + (x-1);
      const size_t idxE = (size_t)y*N + (x+1);
      out[idx] = 0.25 * (in[idxN] + in[idxS] + in[idxW] + in[idxE]);
    }
  }
}

//in explicit, to have consistent view of the data, we exchange regions(cudaMemcpyH2D & D2H), so the computations are correct for the next step and each of the CPU and the GPU has the whole interior
static void explicit_exchange_regions_after_step(
    int N, int K,
    const double* hOut, double* dOut,
    const double* dOut_dev, double* hOut_host,
    cudaStream_t stream,
    double& h2d_ms_acc, double& d2h_ms_acc)
{
  if (N <= 1) return;
  const size_t pitch = size_t(N) * sizeof(double);
  const size_t elem  = sizeof(double);

  double t0, ms;

  //CPU rows [1..K-1], interior cols [1..N-2] --> GPU
  if (K > 1 && N > 2) {
    const int width  = (N - 2) * int(elem);
    const int height = K - 1;
    const void* src  = hOut + size_t(1) * N + 1; //pass the first row and the first element of the second row(border)
    void*       dst  = dOut + size_t(1) * N + 1;
    t0 = wtime();
    CHECK_CUDA(cudaMemcpy2DAsync(dst, pitch, src, pitch, width, height, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    ms = (wtime() - t0) * 1e3;
    h2d_ms_acc += ms;
  }

  //GPU rows [K..N-2], interior cols [1..N-2] --> CPU
  if (K <= N - 2 && N > 2) {
    const int start  = std::max(K, 1);
    const int height = (N - 2) - start + 1;
    if (height > 0) {
      const int width = (N - 2) * int(elem);
      const void* src = dOut_dev + size_t(start) * N + 1;
      void*       dst = hOut_host + size_t(start) * N + 1;
      t0 = wtime();
      CHECK_CUDA(cudaMemcpy2DAsync(dst, pitch, src, pitch, width, height, cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      ms = (wtime() - t0) * 1e3;
      d2h_ms_acc += ms;
    }
  }

  //borders host -> device
  {
    t0 = wtime();
    //top row
    CHECK_CUDA(cudaMemcpyAsync(dOut + 0*size_t(N), hOut + 0*size_t(N), N*elem, cudaMemcpyHostToDevice, stream));
    //bottom row
    CHECK_CUDA(cudaMemcpyAsync(dOut + size_t(N-1)*N, hOut + size_t(N-1)*N, N*elem, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    ms = (wtime() - t0) * 1e3;
    h2d_ms_acc += ms;
  }

  //left/right columns host -> device (the corners were computed above)
  if (N > 2) {
    const void* srcL = hOut + size_t(1)*N + 0;
    void*       dstL = dOut + size_t(1)*N + 0;
    const void* srcR = hOut + size_t(1)*N + (N-1);
    void*       dstR = dOut + size_t(1)*N + (N-1);
    t0 = wtime();
    CHECK_CUDA(cudaMemcpy2DAsync(dstL, pitch, srcL, pitch, elem, N-2, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpy2DAsync(dstR, pitch, srcR, pitch, elem, N-2, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    ms = (wtime() - t0) * 1e3;
    h2d_ms_acc += ms;
  }
}

static void run_one_step_hybrid_umlike_timed(
    double* &hIn, double* &hOut,
    double* &dIn, double* &dOut,
    int N, int K, int threads,
    dim3 grid, dim3 block, cudaStream_t stream,
    double& gpu_ms_acc, double& cpu_ms_acc) {
  cudaEvent_t evStart, evStop;
  CHECK_CUDA(cudaEventCreate(&evStart));
  CHECK_CUDA(cudaEventCreate(&evStop));

  if (K == 0) { //CPU percentage is 0, work done only by the GPU, so do appripriate kernel(with borders)
    CHECK_CUDA(cudaEventRecord(evStart, stream));
    stencil5_2d<<<grid, block, 0, stream>>>(dIn, dOut, N);
    CHECK_CUDA(cudaEventRecord(evStop, stream));
  } else { //synergy: CPU will touch rows 1..K-1 and borders, GPU does rows K..N-2
    copy_borders(hIn, hOut, N);  //borders computed by the CPU
    const int yStart = std::max(K, 1);
    const int yEnd   = std::max(0, N-2);
    CHECK_CUDA(cudaEventRecord(evStart, stream));
    if (yStart <= yEnd) {
      stencil5_2d_yspan<<<grid, block, 0, stream>>>(dIn, dOut, N, yStart, yEnd);
    }
    CHECK_CUDA(cudaEventRecord(evStop, stream));

    const int cpuStart = 1;
    const int cpuEnd   = std::max(K-1, 0);
    if (cpuEnd >= cpuStart) {
      double t0 = wtime();
      cpu_stencil_rows_omp(hIn, hOut, N, cpuStart, cpuEnd, threads);
      cpu_ms_acc += (wtime() - t0) * 1e3;
    }

    /*if (mode == Mode::EXPLICIT) {
            explicit_exchange_regions_after_step(N, K, hOut, dOut, dOut, hOut, stream, h2d_ms_acc, d2h_ms_acc);
    }*/
  }

  //finish kernel
  CHECK_CUDA(cudaEventSynchronize(evStop));
  float step_gpu_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&step_gpu_ms, evStart, evStop));
  gpu_ms_acc += double(step_gpu_ms);

  CHECK_CUDA(cudaEventDestroy(evStart));
  CHECK_CUDA(cudaEventDestroy(evStop));
  //swap
  double* tmpIn = hIn;  hIn  = hOut;  hOut = tmpIn;
  tmpIn = dIn; dIn = dOut; dOut = tmpIn;
}


static void run_one_step_hybrid_explicit_timed(double* &hIn, double* &hOut, double* &dIn, double* &dOut, int N, int K, int threads, dim3 grid, dim3 block, cudaStream_t stream, double& gpu_ms_acc, double& cpu_ms_acc, double& h2d_ms_acc, double& d2h_ms_acc) {
  cudaEvent_t evStart, evStop;
  CHECK_CUDA(cudaEventCreate(&evStart));
  CHECK_CUDA(cudaEventCreate(&evStop));

  if (K == 0) {
    CHECK_CUDA(cudaEventRecord(evStart, stream));
    stencil5_2d<<<grid, block, 0, stream>>>(dIn, dOut, N);
    CHECK_CUDA(cudaEventRecord(evStop, stream));
  } else {
    copy_borders(hIn, hOut, N);

    const int yStart = std::max(K, 1);
    const int yEnd   = std::max(0, N-2);

    CHECK_CUDA(cudaEventRecord(evStart, stream));
    if (yStart <= yEnd) {
      stencil5_2d_yspan<<<grid, block, 0, stream>>>(dIn, dOut, N, yStart, yEnd); //launched on a stream, queued on the gpu, so cpu and gpu run concurrently
    }
    CHECK_CUDA(cudaEventRecord(evStop, stream));

    const int cpuStart = 1;
    const int cpuEnd   = std::max(K-1, 0);
    if (cpuEnd >= cpuStart) {
      double t0 = wtime();
      cpu_stencil_rows_omp(hIn, hOut, N, cpuStart, cpuEnd, threads);
      cpu_ms_acc += (wtime() - t0) * 1e3;
    }

    //stitch results across memories so both views are complete
    explicit_exchange_regions_after_step(
        N, K, hOut, dOut, dOut, hOut, stream, h2d_ms_acc, d2h_ms_acc);
  }

  //finish kernel
  CHECK_CUDA(cudaEventSynchronize(evStop)); //wait until everything up unti evStop is finished --> GPU synchonization!
  float step_gpu_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&step_gpu_ms, evStart, evStop));
  gpu_ms_acc += double(step_gpu_ms);

  CHECK_CUDA(cudaEventDestroy(evStart));
  CHECK_CUDA(cudaEventDestroy(evStop));

  //swap
  double* tmpIn = hIn;  hIn  = hOut;  hOut = tmpIn;
  tmpIn = dIn; dIn = dOut; dOut = tmpIn;
}


//main
int main(int argc, char** argv) {

  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);
  DeviceCaps caps = query_caps();

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  const size_t bytes = elems * sizeof(double);

  printf("stencil 5-point | N=%d (%.3f MiB), iters=%d, steps/iter=%d, mode=%s, prefetch=%d, seed=%llu, amp=%g\n",
         N, double(bytes)/(1024.0*1024.0), args.iters, args.steps,
         args.mode.c_str(), args.prefetch, (unsigned long long)args.seed, args.amp);

  Buffers<double> buf;
  allocate_buffers<double>(mode, elems, buf, caps);

  if (mode == Mode::GH_HMM_PAGEABLE_GPU_INIT) {
    const size_t elems = buf.nbytes / sizeof(double);
    const int block = 256;
    const int grid  = int((elems + block - 1) / block);
    k_fill_random<<<grid, block, 0, 0>>>(reinterpret_cast<double*>(buf.dIn),
                                         elems, uint64_t(args.seed), args.amp);
    k_memset_zero<<<grid, block, 0, 0>>>(reinterpret_cast<double*>(buf.dOut),
                                         elems);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  else {
        std::mt19937_64 rng(args.seed);
        std::uniform_real_distribution<double> dist(-args.amp, args.amp);
        for (size_t i = 0; i < elems; ++i) buf.hIn[i] = dist(rng);
        std::fill(buf.hOut, buf.hOut + elems, 0.0);
  }

 
  //prefetch / copies before kernel (measure here, added at the end to the total time)
  IterTimings totals{};
  double pre_ms = pre_kernel_setup<double>(mode, buf, caps, args.prefetch);
  if (mode == Mode::EXPLICIT) totals.h2d_ms = pre_ms;
  if (mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) totals.um_to_dev_ms = pre_ms;

  //launch config
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  cudaStream_t stream = nullptr; //to have cpu and gpu work overlap

  int K = (args.kPct * N) / 100;
  int cpuThreads = args.threads > 0 ? args.threads : omp_get_max_threads();

  fflush(stdout);

  auto gbps = [](double bytes_, double ms) { 
	  return (ms > 0.0) ? (bytes_ / 1e6) / ms : 0.0; 
  };

  double t0_full = wtime();
  double full_ms_no_checksum = 0.0;

  //warmup iterations
  int warmup_iters = 2;

  for (int w = 0; w < warmup_iters; ++w) {
  double dummy_gpu = 0.0, dummy_cpu = 0.0;
  for (int s = 0; s < args.steps; ++s) {
    if (mode == Mode::EXPLICIT) {
      double dummy_h2d = 0.0, dummy_d2h = 0.0;
      run_one_step_hybrid_explicit_timed(
        buf.hIn, buf.hOut, buf.dIn, buf.dOut,
        N, K, cpuThreads, grid, block, stream,
        dummy_gpu, dummy_cpu, dummy_h2d, dummy_d2h);
    } else {
      double dummy_h2d = 0.0, dummy_d2h = 0.0;

      run_one_step_hybrid_umlike_timed(
        buf.hIn, buf.hOut, buf.dIn, buf.dOut,
        N, K, cpuThreads, grid, block, stream,
        dummy_gpu, dummy_cpu);
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  printf("Warm-up done.\n\n");


  full_ms_no_checksum = (wtime() - t0_full) * 1e3;
  printf("WARMUP: %.3f ms\n", full_ms_no_checksum);
  
  //per-iteration accumulators
  double sum_end2end_ms = 0.0, sum_gpu_ms = 0.0, sum_cpu_ms = 0.0;

  for (int it = 0; it < args.iters; ++it) {
    double iter_gpu_ms = 0.0, iter_cpu_ms = 0.0;
    double t0_iter = wtime();

    for (int s = 0; s < args.steps; ++s) {
      if (mode == Mode::EXPLICIT) {
        run_one_step_hybrid_explicit_timed(
          buf.hIn, buf.hOut,
          buf.dIn, buf.dOut,
          N, K, cpuThreads, grid, block, stream,
          iter_gpu_ms, iter_cpu_ms, totals.h2d_ms, totals.d2h_ms);
      } else {
        run_one_step_hybrid_umlike_timed(
          buf.hIn, buf.hOut,
          buf.dIn, buf.dOut,
          N, K, cpuThreads, grid, block, stream,
          iter_gpu_ms, iter_cpu_ms);
      }
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    double end2end_ms = (wtime() - t0_iter) * 1e3;

    printf("Iter %d: end2end=%.3f ms, gpu=%.3f ms, cpu=%.3f ms\n",
           it, end2end_ms, iter_gpu_ms, iter_cpu_ms);

    sum_end2end_ms += end2end_ms;
    sum_gpu_ms     += iter_gpu_ms;
    sum_cpu_ms     += iter_cpu_ms;
  }


  //managed migrate back (only UM_MIGRATE)
  totals.um_to_host_ms = post_kernel_maybe_prefetch_back<double>(mode, buf);

  //explicit copy back if needed (final full buffer)
  totals.d2h_ms += post_kernel_copy_back_if_needed<double>(mode, buf);

  // final summary 
  const double avg_end2end_ms = (sum_end2end_ms / args.iters);
  const double avg_gpu_ms     = sum_gpu_ms     / args.iters; //avg computation per iteration
  const double avg_cpu_ms     = sum_cpu_ms     / args.iters;

  full_ms_no_checksum += (avg_end2end_ms + totals.um_to_host_ms + totals.d2h_ms);

  printf("\n==== Total time over %d iteration(s) (steps/iter=%d) ====\n", args.iters, args.steps);
  printf("End-to-end (compute loop only): %.3f ms\n", avg_end2end_ms);
  printf("GPU compute: %.3f ms (pure GPU compute, not cuda device synchronize stream overhead measured)\n", avg_gpu_ms);
  printf("CPU compute: %.3f ms\n", avg_cpu_ms);

  if (totals.h2d_ms > 0.0)      printf("H2D memcpy (total): %.3f ms\n", totals.h2d_ms);
  if (totals.d2h_ms > 0.0)      printf("D2H memcpy (total): %.3f ms\n", totals.d2h_ms);
  if (totals.um_to_dev_ms > 0)  printf("UM->Device prefetch (2x buffers): %.3f ms \n", totals.um_to_dev_ms);
  if (totals.um_to_host_ms > 0) printf("UM->Host prefetch (dOut): %.3f ms\n",
                                       totals.um_to_host_ms);

/*
  //stencil bandwidth model (4 reads + 1 write per interior cell)
  const double bytes_per_cell = 5.0 * sizeof(double);
  const int interior_cols = std::max(0, N - 2);
  const int cpu_rows = std::max(0, K - 1);
  const int gpu_rows = std::max(0, (N - 2) - cpu_rows);
  const double cpu_cells = double(interior_cols) * double(cpu_rows) * args.steps;
  const double gpu_cells = double(interior_cols) * double(gpu_rows) * args.steps;
  const double cpu_bytes = cpu_cells * bytes_per_cell;
  const double gpu_bytes = gpu_cells * bytes_per_cell;

  auto gbps_cells = [&](double bytes_moved, double ms) {
    return (ms > 0.0) ? (bytes_moved / 1e6) / ms : 0.0;
  };

  if (avg_cpu_ms > 0.0)
    printf("CPU stencil BW: %.2f GB/s\n", gbps_cells(cpu_bytes, avg_cpu_ms));
  if (avg_gpu_ms > 0.0)
    printf("GPU stencil BW: %.2f GB/s\n", gbps_cells(gpu_bytes, avg_gpu_ms));
*/

  // Final full end-to-end summary (mode total)
  printf("\n==== Full end-to-end (mode total, with pre kernel setup costs(added)) ====\n");
  printf("Full end-to-end: %.3f ms\n", full_ms_no_checksum + pre_ms);

  free_buffers<double>(mode, buf);
  return 0;
}

