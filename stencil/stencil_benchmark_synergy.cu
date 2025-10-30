//Jacobi style --> in each time step i only read from IN and write to OUT

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <unistd.h>   // getopt
#include <sys/time.h> // gettimeofday
#include <omp.h>
#include <numeric>    // Added for std::accumulate

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
  int N = 4096;           //grid dimension N x N
  int iters = 10;         //iterations
  std::string mode = "explicit";
  int prefetch = 1;       //only relevant to UM_MIGRATE / GH_HBM_SHARED
  uint64_t seed = 12345;
  int steps = 1;          //time-steps per iteration
  int autoTune = 0;       //enable autotuner if 1
  int threads = 0;        //CPU threads (0 => OpenMP default)
  int kPct = 20;          //K as percent of N (used if !autoTune)
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
    "  --auto      enable autotuning for K and CPU threads\n"
    "  --threads T set CPU threads (ignored by --auto)\n"
    "  --kPct P    set seam as percent of N (5..95, default 20; ignored by --auto)\n"
    "  -h          help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "n:i:t:m:p:r:h")) != -1) {
    switch (opt) {
      case 'n': a.N = std::atoi(optarg); break;
      case 'i': a.iters = std::atoi(optarg); break;
      case 't': a.steps = std::atoi(optarg); break;
      case 'm': a.mode = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'r': a.seed = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
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
  a.kPct = std::max(5, std::min(95, a.kPct));
  return a;
}

//device init kernels
__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__device__ inline double u01_from_u64(uint64_t x) {
  const double inv2_53 = 1.0 / 9007199254740992.0; // 2^53
  return double(x >> 11) * inv2_53;
}

__global__ void init_random_double(double* a, size_t n, uint64_t seed) { //init dIn on GPU, used by HMM_PAGABLE_GPU_INIT mode
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) {
    uint64_t r = splitmix64(seed + i);
    double u = u01_from_u64(r);
    a[i] = 2.0 * u - 1.0; // [-1,1)
  }
}
__global__ void set_zero_double(double* a, size_t n) { //init dOut on GPU, used by HMM_PAGABLE_GPU_INIT mode
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 0.0;
}


//-------------------------------------- 2 stencil kernels ----------------------------------------------
//stencil5_2d --> gpu does everything
__global__ void stencil5_2d(const double* __restrict__ in, double* __restrict__ out,
                             int N)
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

/*
    stencil5_2d_yspan: hybrid approach, only computes rows y in [yStart, yEnd] and columns x in [1, N-2], it does not touch the top/bottom borders or the CPU’s rows. 
    In hybrid mode, borders are copied on the CPU (copy_borders), the CPU computes rows 1..K-1, the GPU computes K+1..N-2, and the row K(where CPU and GPU computations meet) is computed once 
    afterward to avoid races.
  */
__global__ void stencil5_2d_yspan(const double* __restrict__ in, double* __restrict__ out,
                                 int N, int yStart, int yEnd)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= N) return;
  if (y < yStart || y > yEnd || x == 0 || x == N-1) return; //return for rows-columns ouside the GPU part

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
  T *hIn=nullptr, *hOut=nullptr;
  T *dIn=nullptr, *dOut=nullptr;
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
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH: {
      CHECK_CUDA(cudaMallocManaged(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dOut, buf.nbytes));
      buf.hIn  = buf.dIn;
      buf.hOut = buf.dOut;
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
      buf.dIn  = buf.hIn;  //HMM: same raw pointer
      buf.dOut = buf.hOut;
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", caps.uses_host_pt);
      break;
    }
  }
}

template <typename T>
static void pre_kernel_setup(Mode mode, const Buffers<T>& buf, const DeviceCaps& caps, int do_prefetch) {
  if (mode == Mode::EXPLICIT) {
    CHECK_CUDA(cudaMemcpy(buf.dIn, buf.hIn, buf.nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(buf.dOut, 0, buf.nbytes));
  } else if ((mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) && do_prefetch) {
    // FIX: Updated to use cudaMemLocation and stream argument (0)
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dIn,  buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

template <typename T>
static void post_kernel_maybe_prefetch_back(Mode mode, const Buffers<T>& buf) {
  if (mode == Mode::UM_MIGRATE) {
    double t0 = wtime();
    // FIX: Updated to use cudaMemLocationTypeHost and stream argument (0)
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeHost, 0}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    double ms = (wtime() - t0) * 1e3;
    printf("UM prefetch-to-CPU: %.3f ms (%.2f GB/s logical)\n", ms, double(buf.nbytes) / 1e6 / ms);
  }
}

template <typename T>
static void post_kernel_copy_back_if_needed(Mode mode, const Buffers<T>& buf) {
  if (mode == Mode::EXPLICIT) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(buf.hOut, buf.dOut, buf.nbytes, cudaMemcpyDeviceToHost));
    double ms = (wtime() - t0) * 1e3;
    printf("D2H memcpy: %.3f ms (%.2f GB/s)\n", ms, double(buf.nbytes) / 1e6 / ms);
  }
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
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
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


template <typename T>
static double checksum(const T* p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += (double)p[i];
  return s;
}

static void copy_borders(const double* in, double* out, int N) {
  std::memcpy(out + 0*size_t(N), in + 0*size_t(N), sizeof(double)*N); //copy top row row
  std::memcpy(out + (size_t)(N-1)*N, in + (size_t)(N-1)*N, sizeof(double)*N); //copy bottom
  for (int y = 1; y < N-1; ++y) { //copy each interior row's leftmost and righmost element 
    out[(size_t)y*N + 0]    = in[(size_t)y*N + 0];
    out[(size_t)y*N + N-1] = in[(size_t)y*N + N-1];
  }
}

static void cpu_stencil_rows_omp(const double* in, double* out, int N, int yStart, int yEnd, int threads)
{
  if (yEnd < yStart) return;
  if (threads > 0) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
  }
#pragma omp parallel for schedule(static) //static split of thw for loop across threads
  for (int y = yStart; y <= yEnd; ++y) { //CPU chunk of rows
    #pragma omp simd              //idk if this would help
    for (int x = 1; x <= N-2; ++x) { //only interior columns
      const size_t idx  = (size_t)y*N + x;
      const size_t idxN = (size_t)(y-1)*N + x;
      const size_t idxS = (size_t)(y+1)*N + x;
      const size_t idxW = (size_t)y*N + (x-1);
      const size_t idxE = (size_t)y*N + (x+1);
      out[idx] = 0.25 * (in[idxN] + in[idxS] + in[idxW] + in[idxE]);
    }
  }
}


//one time-step with CPU+GPU overlap
static void run_one_step_hybrid(const double* &in, double* &out, int N, int K, int threads, dim3 grid, dim3 block, cudaStream_t stream)
{
  copy_borders(in, out, N);

  const int yStart = std::max(K, 1);
  const int yEnd   = std::max(0, N-2);
  if (yStart <= yEnd) {
    stencil5_2d_yspan<<<grid, block, 0, stream>>>(in, out, N, yStart, yEnd);
  }

  const int cpuStart = 1;
  const int cpuEnd   = std::max(K-1, 0);
  cpu_stencil_rows_omp(in, out, N, cpuStart, cpuEnd, threads);

  CHECK_CUDA(cudaStreamSynchronize(stream));

  const double* tmp = in;
  in  = out;
  out = (double*)tmp;
}

//micro-benchmark one iteration with given K/threads
static double time_one_iteration_hybrid(const double* baseIn, double* baseOut,
                                         double* scratchIn, double* scratchOut,
                                         int N, int steps, int K, int threads,
                                         dim3 grid, dim3 block, cudaStream_t stream)
{
  std::memcpy(scratchIn, baseIn, sizeof(double)*size_t(N)*N);
  std::memset(scratchOut, 0, sizeof(double)*size_t(N)*N);

  const double* in  = scratchIn;
  double* out = scratchOut;

  double t0 = wtime();
  for (int s = 0; s < steps; ++s) {
    run_one_step_hybrid(in, out, N, K, threads, grid, block, stream);
  }
  double ms = (wtime() - t0) * 1e3;
  return ms;
}

//autotune K (% of N) and threads
static void autotune(const double* baseIn, double* baseOut,
                     double* scratchIn, double* scratchOut,
                     int N, int steps, int& bestK, int& bestThreads,
                     dim3 grid, dim3 block, cudaStream_t stream)
{
  std::vector<int> kPercents;
  for (int p = 5; p <= 45; p += 5) kPercents.push_back(p);

  const int maxT = std::max(1, omp_get_max_threads());
  std::vector<int> threadCands = {1,2,4,8,16,32};
  threadCands.erase(std::remove_if(threadCands.begin(), threadCands.end(),
                   [maxT](int t){ return t > maxT; }), threadCands.end());
  if (threadCands.empty()) threadCands.push_back(1);

  double bestMs = 1e300;
  bestK = std::max(1, (N * 20) / 100);
  bestThreads = std::min(maxT, 8);

  const int stepsAT = std::max(1, steps/2);

  for (int p : kPercents) {
    int K = std::max(1, (N * p) / 100);
    for (int th : threadCands) {
      double ms = time_one_iteration_hybrid(baseIn, baseOut, scratchIn, scratchOut,
                                            N, stepsAT, K, th, grid, block, stream);
      if (ms < bestMs) {
        bestMs = ms;
        bestK = K;
        bestThreads = th;
      }
    }
  }
  printf("Autotune picked: K=%d (%.1f%% of N), threads=%d, est=%.3f ms for %d step(s)\n",
         bestK, 100.0*bestK/N, bestThreads, bestMs, stepsAT);
  fflush(stdout);
}

// Function to calculate average of vector<float>
auto average_float = [](const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += x;
  return (v.empty() ? 0.0 : s / double(v.size()));
};


//----------------------------- main -------------------------------
int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);
  DeviceCaps caps = query_caps();

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  const size_t bytes = elems * sizeof(double);

  printf("stencil 5-point | N=%d (%.3f MiB), iters=%d, steps/iter=%d, mode=%s, prefetch=%d, seed=%llu\n",
         N, double(bytes)/(1024.0*1024.0), args.iters, args.steps,
         args.mode.c_str(), args.prefetch, (unsigned long long)args.seed);

  Buffers<double> buf;
  allocate_buffers<double>(mode, elems, buf, caps);

  //initialization
  if (mode == Mode::GH_HMM_PAGEABLE_GPU_INIT) {
    const int threads = 256;
    const int blocks  = int((elems + threads - 1) / threads);
    init_random_double<<<blocks, threads>>>(buf.dIn, elems, args.seed);
    set_zero_double<<<blocks, threads>>>(buf.dOut, elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) buf.hIn[i] = dist(rng);
    std::fill(buf.hOut, buf.hOut + elems, 0.0);
  }

  //prefetch / copies before kernel 
  pre_kernel_setup<double>(mode, buf, caps, args.prefetch);

  //launch config
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  cudaStream_t stream = nullptr; // default stream

  
  //base copies (host-side) for autotune reproducibility
  std::vector<double> baseIn(elems), baseOut(elems, 0.0);
  std::memcpy(baseIn.data(), buf.hIn, bytes);

  int K = std::max(1, (args.kPct * N) / 100);
  int cpuThreads = args.threads > 0 ? args.threads : omp_get_max_threads();

  if (args.autoTune) {
    autotune(baseIn.data(), baseOut.data(), buf.hIn, buf.hOut,
             N, args.steps, K, cpuThreads, grid, block, stream);
  } else {
    printf("Hybrid params: K=%d (%.1f%% of N), threads=%d\n", K, 100.0*K/N, cpuThreads);
  } 
  fflush(stdout);

// -----------------------------------------------------------------
// START OF NEW PER-STEP TIMING LOGIC (COLD/WARM Separation)
// -----------------------------------------------------------------
  std::vector<float> total_times(args.iters, 0.0f);
  std::vector<float> step1_times(args.iters, 0.0f);
  std::vector<float> warm_avg_times(args.iters, 0.0f); 

  for (int it = 0; it < args.iters; ++it) {
    // set starting endpoints for this iteration
    const double* in  = buf.dIn;
    double* out = buf.dOut;
    const int S = args.steps;

    // Create a chain of events: ev[0] before step 1, ev[k] after step k
    std::vector<cudaEvent_t> ev(S + 1);
    for (auto& e : ev) CHECK_CUDA(cudaEventCreate(&e));

    // Record start event (before step 1)
    CHECK_CUDA(cudaEventRecord(ev[0], stream));

    // Perform S steps, recording after each step; swap in/out each time (ping-pong)
    for (int t = 0; t < S; ++t) {
        run_one_step_hybrid(in, out, N, K, cpuThreads, grid, block, stream);
        // Note: run_one_step_hybrid already swaps in/out
        CHECK_CUDA(cudaEventRecord(ev[t + 1], stream)); // after step t+1
    }

    // Wait for the stream (which contains all the steps) and the final event
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventSynchronize(ev[S]));

    // Per-step times: Δt_k = ev[k] - ev[k-1]
    std::vector<float> step_ms(S, 0.0f);
    for (int k = 1; k <= S; ++k) {
      CHECK_CUDA(cudaEventElapsedTime(&step_ms[k - 1], ev[k - 1], ev[k]));
    }

    for (auto& e : ev) CHECK_CUDA(cudaEventDestroy(e));

    // Save per-iteration stats
    float total = std::accumulate(step_ms.begin(), step_ms.end(), 0.0f);
    total_times[it] = total;
    step1_times[it] = step_ms[0];

    if (S > 1) {
      float warm_sum = std::accumulate(step_ms.begin() + 1, step_ms.end(), 0.0f);
      warm_avg_times[it] = warm_sum / float(S - 1);
    } else {
      warm_avg_times[it] = 0.0f;
    }
    
    // If steps is odd, ensure buf.dOut points to the latest result (must align with run_one_step_hybrid swap logic)
    if ((S & 1) == 1) {
      // swap buffer roles back to original meaning if needed (since run_one_step_hybrid swaps at the end)
      double* tmp = buf.dOut;
      buf.dOut = (double*)buf.dIn;
      buf.dIn  = tmp;
    }
  }

  const double avg_total_per_iter_ms = average_float(total_times);
  const double avg_step1_ms          = average_float(step1_times);
  const double avg_warm_ms           = average_float(warm_avg_times);

  // Use the robust bracketed format for BASH script parsing
  printf("Kernel avg (total per iteration over %d step(s)): %.3f ms\n",
         args.steps, avg_total_per_iter_ms);
  printf("  [Step 1 cold]: %.3f ms\n", avg_step1_ms);
  if (args.steps > 1)
    printf("  [Steps 2..%d warm avg]: %.3f ms\n", args.steps, avg_warm_ms);
  else
    printf("  [No warm steps (steps=1)]\n");
  fflush(stdout);
// -----------------------------------------------------------------
// END OF NEW PER-STEP TIMING LOGIC
// -----------------------------------------------------------------

  //managed migrate back (only UM_MIGRATE)
  post_kernel_maybe_prefetch_back<double>(mode, buf);

  //explicit copy back if needed
  post_kernel_copy_back_if_needed<double>(mode, buf);

  //CPU checksum timing
  double t0 = wtime();
  double sum = checksum<double>(buf.hOut, elems);
  double read_ms = (wtime() - t0) * 1e3;
  printf("CPU checksum(out)=%.6e, CPU read time: %.3f ms (%.2f GB/s logical)\n",
         sum, read_ms, double(bytes) / 1e6 / read_ms);

  free_buffers<double>(mode, buf);
  return 0;
}
